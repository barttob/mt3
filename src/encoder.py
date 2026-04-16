"""Spectrogram encoder for MT3 music transcription model."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    Args:
        d_model: Model dimension.
        max_len: Maximum sequence length to precompute.
    """

    def __init__(self, d_model: int, max_len: int = 2048) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.

        Args:
            x: Input tensor of shape (B, T, d_model).

        Returns:
            Tensor of shape (B, T, d_model) with positional encoding added.
        """
        return x + self.pe[:, : x.size(1)]


def _build_sinusoidal_pe(length: int, dim: int) -> torch.Tensor:
    """Build a sinusoidal PE matrix of shape (length, dim).

    Args:
        length: Number of positions.
        dim: Embedding dimension (must be even).

    Returns:
        Tensor of shape (length, dim).
    """
    pe = torch.zeros(length, dim)
    position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """Rotary position embeddings (RoPE) from Su et al. (2021).

    Encodes position by rotating query and key vectors inside each attention
    layer rather than adding a fixed bias to input embeddings. This enables
    relative position sensitivity without modifying the value path.

    Args:
        head_dim: Per-head attention dimension (d_model // nhead).
        base: Base for the inverse-frequency computation (default 10000).
    """

    def __init__(self, head_dim: int, base: int = 10000) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq)
        self.head_dim = head_dim
        self._seq_len_cached: int = 0
        self._cos_cached: Optional[torch.Tensor] = None
        self._sin_cached: Optional[torch.Tensor] = None

    def _build_cache(self, seq_len: int, device: torch.device) -> None:
        """Lazily build (and extend) the cos/sin cache."""
        if seq_len <= self._seq_len_cached and self._cos_cached is not None:
            return
        self._seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (T, head_dim//2)
        emb = torch.cat([freqs, freqs], dim=-1)             # (T, head_dim)
        self._cos_cached = emb.cos()[None, None, :, :]      # (1, 1, T, head_dim)
        self._sin_cached = emb.sin()[None, None, :, :]      # (1, 1, T, head_dim)

    def apply(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate queries and keys with RoPE.

        Args:
            q: Query tensor of shape (B, H, T_q, head_dim).
            k: Key tensor of shape (B, H, T_k, head_dim).

        Returns:
            Tuple (q_rotated, k_rotated) with the same shapes.
        """
        T = max(q.shape[2], k.shape[2])
        self._build_cache(T, q.device)

        def _rotate(x: torch.Tensor) -> torch.Tensor:
            T_x = x.shape[2]
            cos = self._cos_cached[:, :, :T_x, :].to(x.dtype)  # (1,1,T_x,D)
            sin = self._sin_cached[:, :, :T_x, :].to(x.dtype)
            half = self.head_dim // 2
            x1, x2 = x[..., :half], x[..., half:]
            rotated = torch.cat([-x2, x1], dim=-1)
            return x * cos + rotated * sin

        return _rotate(q), _rotate(k)


class RoPEMultiheadAttention(nn.Module):
    """Multi-head attention with rotary position embeddings.

    Drop-in replacement for ``nn.MultiheadAttention(batch_first=True)``
    for self-attention paths.  Cross-attention should still use the standard
    module because encoder and decoder positions are independent sequences.

    Args:
        d_model: Model/embedding dimension.
        nhead: Number of attention heads.
        dropout: Dropout probability on attention weights.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self._dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, None]:
        """Multi-head attention with RoPE applied to Q and K.

        Args:
            query: Shape (B, T_q, d_model).
            key: Shape (B, T_k, d_model).
            value: Shape (B, T_k, d_model).
            attn_mask: Optional float additive mask of shape (T_q, T_k) or
                (B, T_q, T_k) or (B, H, T_q, T_k).
            key_padding_mask: Optional boolean mask of shape (B, T_k); True
                where keys should be ignored.
            is_causal: If True, apply a causal mask (overrides attn_mask).

        Returns:
            Tuple (output, None) where output has shape (B, T_q, d_model).
            The second element is None to match ``nn.MultiheadAttention``.
        """
        B, T_q, _ = query.shape
        T_k = key.shape[1]

        q = self.q_proj(query).view(B, T_q, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, T_k, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, T_k, self.nhead, self.head_dim).transpose(1, 2)

        q, k = self.rope.apply(q, k)

        # Merge attn_mask and key_padding_mask into a single float bias tensor
        combined_mask: Optional[torch.Tensor] = None
        if attn_mask is not None:
            am = attn_mask.to(q.dtype)
            # Ensure at least 4-D so it broadcasts against (B, H, T_q, T_k)
            if am.dim() == 2:
                am = am.unsqueeze(0).unsqueeze(0)  # (1, 1, T_q, T_k)
            elif am.dim() == 3:
                am = am.unsqueeze(1)               # (B, 1, T_q, T_k)
            combined_mask = am
        if key_padding_mask is not None:
            kpm = torch.zeros(B, 1, T_q, T_k, device=query.device, dtype=q.dtype)
            kpm = kpm.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
            combined_mask = kpm if combined_mask is None else combined_mask + kpm

        dropout_p = self._dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=combined_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        return self.out_proj(out), None


class RoPETransformerEncoderLayer(nn.Module):
    """Pre-norm Transformer encoder layer with RoPE self-attention.

    Identical in structure to ``nn.TransformerEncoderLayer(norm_first=True)``
    except that ``nn.MultiheadAttention`` is replaced with
    :class:`RoPEMultiheadAttention`.

    Args:
        d_model: Model dimension.
        nhead: Number of attention heads.
        dim_feedforward: FFN hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attn = RoPEMultiheadAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop_ff = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            src: Shape (B, T, d_model).
            src_mask: Optional additive attention mask.
            src_key_padding_mask: Optional boolean padding mask (B, T).

        Returns:
            Shape (B, T, d_model).
        """
        x = src
        x2 = self.norm1(x)
        x2, _ = self.self_attn(
            x2, x2, x2,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        x = x + self.drop1(x2)
        x2 = self.norm2(x)
        x2 = self.linear2(self.drop_ff(F.gelu(self.linear1(x2))))
        x = x + self.drop2(x2)
        return x


# ---------------------------------------------------------------------------
# Convolutional Front-End
# ---------------------------------------------------------------------------

class ConvFrontend(nn.Module):
    """Convolutional front-end replacing the per-frame linear projection.

    A stack of ``Conv1d`` blocks with GELU activations operating along the
    time axis of the spectrogram.  Captures local temporal onset/offset
    patterns before the Transformer encoder at stride=1 (sequence length is
    preserved). Acts as a drop-in replacement for ``nn.Linear(n_mels, d_model)``
    after transposing.

    Args:
        n_mels: Number of input mel frequency bins (input channels).
        d_model: Output channel dimension (and all intermediate channels).
        num_layers: Number of Conv1d + GELU blocks (default 2).
        kernel_size: Convolution kernel size (default 3; padding preserves T).
    """

    def __init__(
        self,
        n_mels: int,
        d_model: int,
        num_layers: int = 2,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        layers: list[nn.Module] = []
        in_ch = n_mels
        for _ in range(num_layers):
            layers.append(nn.Conv1d(in_ch, d_model, kernel_size, padding=padding))
            layers.append(nn.GELU())
            in_ch = d_model
        self.net = nn.Sequential(*layers)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Project spectrogram frames via convolution.

        Args:
            spec: Log-mel spectrogram of shape (B, n_mels, T).

        Returns:
            Projected features of shape (B, T, d_model).
        """
        x = self.net(spec)          # (B, d_model, T)
        return x.transpose(1, 2)    # (B, T, d_model)


# ---------------------------------------------------------------------------
# 2D Patch Embedding
# ---------------------------------------------------------------------------

class PatchEmbedding2D(nn.Module):
    """2D frequency-time patch embedding for log-mel spectrograms.

    Divides the spectrogram (n_mels × T) into non-overlapping 2D patches
    of size (patch_f × patch_t), projects each patch to d_model, and adds
    a 2D sinusoidal positional encoding that separately encodes the frequency
    axis and the time axis. This gives the encoder explicit pitch-band awareness.

    The resulting sequence length is (n_mels // patch_f) × (T // patch_t),
    which equals the standard per-frame sequence length when
    patch_f × patch_t == n_mels / T_frames (e.g. patch_f=64, patch_t=8 with
    n_mels=512 and T=256 yields 256 patches — same as the default encoder).

    Args:
        n_mels: Number of mel frequency bins.
        patch_f: Number of mel bins per patch (frequency axis patch size).
        patch_t: Number of time frames per patch (time axis patch size).
        d_model: Output embedding dimension.
        max_time_patches: Maximum number of time patches (for PE precomputation).
        max_freq_patches: Maximum number of frequency patches (for PE precomputation).
    """

    def __init__(
        self,
        n_mels: int,
        patch_f: int,
        patch_t: int,
        d_model: int,
        max_time_patches: int = 512,
        max_freq_patches: int = 64,
    ) -> None:
        super().__init__()
        assert n_mels % patch_f == 0, (
            f"n_mels={n_mels} must be divisible by patch_f={patch_f}"
        )
        assert d_model % 2 == 0, "d_model must be even for 2D sinusoidal PE"

        self.patch_f = patch_f
        self.patch_t = patch_t
        self.n_freq_patches = n_mels // patch_f

        self.patch_proj = nn.Linear(patch_f * patch_t, d_model)

        # Build 2D PE: time half (d_model//2) + freq half (d_model//2)
        half = d_model // 2
        time_pe = _build_sinusoidal_pe(max_time_patches, half)  # (max_tp, half)
        freq_pe = _build_sinusoidal_pe(max_freq_patches, half)  # (max_fp, half)
        self.register_buffer("time_pe", time_pe)  # (max_tp, half)
        self.register_buffer("freq_pe", freq_pe)  # (max_fp, half)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Embed a log-mel spectrogram as a sequence of 2D patch tokens.

        Args:
            spec: Log-mel spectrogram of shape (B, n_mels, T).
                  T must be divisible by patch_t.

        Returns:
            Patch embedding tensor of shape (B, n_freq_patches * n_time_patches, d_model).
        """
        B, n_mels, T = spec.shape
        # Trim to the largest multiple of patch_t (torchaudio may add one extra frame)
        T = (T // self.patch_t) * self.patch_t
        spec = spec[:, :, :T]
        n_tp = T // self.patch_t                 # number of time patches
        n_fp = self.n_freq_patches               # number of freq patches

        # Reshape into patches: (B, n_fp, patch_f, n_tp, patch_t)
        x = spec.view(B, n_fp, self.patch_f, n_tp, self.patch_t)
        # Permute to (B, n_fp, n_tp, patch_f, patch_t)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        # Flatten patch content: (B, n_fp, n_tp, patch_f * patch_t)
        x = x.view(B, n_fp, n_tp, self.patch_f * self.patch_t)

        # Linear projection: (B, n_fp, n_tp, d_model)
        x = self.patch_proj(x)

        # Build 2D PE: concat time PE and freq PE for each (i_fp, i_tp) pair
        # time_pe[i_tp]: (n_tp, half) — same for all freq positions
        # freq_pe[i_fp]: (n_fp, half) — same for all time positions
        t_pe = self.time_pe[:n_tp]               # (n_tp, half)
        f_pe = self.freq_pe[:n_fp]               # (n_fp, half)

        # Broadcast to (n_fp, n_tp, half) each
        t_pe = t_pe.unsqueeze(0).expand(n_fp, -1, -1)   # (n_fp, n_tp, half)
        f_pe = f_pe.unsqueeze(1).expand(-1, n_tp, -1)   # (n_fp, n_tp, half)

        pe_2d = torch.cat([t_pe, f_pe], dim=-1)  # (n_fp, n_tp, d_model)
        x = x + pe_2d.unsqueeze(0)              # (B, n_fp, n_tp, d_model)

        # Flatten to sequence: (B, n_fp * n_tp, d_model)
        x = x.view(B, n_fp * n_tp, -1)
        return x


# ---------------------------------------------------------------------------
# Spectrogram Encoder
# ---------------------------------------------------------------------------

class SpectrogramEncoder(nn.Module):
    """Transformer encoder that operates on log-mel spectrogram frames.

    Supports four combinations of input embedding and position encoding:

    - Default: per-frame linear projection + 1D sinusoidal PE.
    - ``use_conv_frontend=True``: Conv1d stack + 1D sinusoidal PE (or RoPE).
    - ``use_2d_patches=True``: ViT-style 2D patch projection with baked-in 2D
      sinusoidal PE.  ``use_conv_frontend`` is ignored in this mode.
    - ``use_rope=True``: replaces additive sinusoidal PE with RoPE inside each
      attention layer.  When ``use_2d_patches=True`` the 2D patch PE and RoPE
      are complementary: patches carry frequency-band position and RoPE encodes
      sequence position in attention.

    All combinations feed into the same Transformer encoder stack (standard
    ``nn.TransformerEncoderLayer`` or ``RoPETransformerEncoderLayer`` when
    ``use_rope=True``), using pre-layer-norm (T5 convention).

    Args:
        n_mels: Number of mel frequency bins (input feature dimension).
        d_model: Model/embedding dimension.
        nhead: Number of attention heads.
        num_layers: Number of Transformer encoder layers.
        dim_feedforward: Feed-forward sublayer hidden dimension.
        dropout: Dropout probability.
        use_2d_patches: If True, use PatchEmbedding2D instead of per-frame projection.
        patch_f: Mel bins per patch (only used when use_2d_patches=True).
        patch_t: Time frames per patch (only used when use_2d_patches=True).
        use_rope: If True, replace sinusoidal additive PE with RoPE inside
            each encoder self-attention layer.
        use_conv_frontend: If True, replace the per-frame linear projection
            with a stack of Conv1d blocks.  Ignored when use_2d_patches=True.
        conv_layers: Number of Conv1d + GELU blocks in the conv frontend.
    """

    def __init__(
        self,
        n_mels: int = 512,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        use_2d_patches: bool = False,
        patch_f: int = 64,
        patch_t: int = 8,
        use_rope: bool = False,
        use_conv_frontend: bool = False,
        conv_layers: int = 2,
    ) -> None:
        super().__init__()
        self.use_2d_patches = use_2d_patches
        self.use_rope = use_rope
        self.use_conv_frontend = use_conv_frontend

        # --- Input projection / patch embedding ----------------------------
        if use_2d_patches:
            self.patch_embed = PatchEmbedding2D(
                n_mels=n_mels,
                patch_f=patch_f,
                patch_t=patch_t,
                d_model=d_model,
            )
            # 2D PE is baked into PatchEmbedding2D; no extra pos_enc needed.
        elif use_conv_frontend:
            self.conv_frontend = ConvFrontend(n_mels, d_model, num_layers=conv_layers)
            if not use_rope:
                self.pos_enc = SinusoidalPositionalEncoding(d_model)
        else:
            self.input_proj = nn.Linear(n_mels, d_model)
            if not use_rope:
                self.pos_enc = SinusoidalPositionalEncoding(d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        # --- Transformer encoder stack -------------------------------------
        if use_rope:
            self.rope_enc_layers = nn.ModuleList([
                RoPETransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ])
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,  # Pre-LN for training stability (T5 convention)
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers, enable_nested_tensor=False
            )

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Encode a log-mel spectrogram into contextual representations.

        Args:
            spectrogram: Log-mel spectrogram of shape (B, n_mels, T).

        Returns:
            Encoder hidden states of shape (B, N, d_model), where N is the
            number of time frames (default mode) or patches (2D mode).
        """
        if self.use_2d_patches:
            x = self.patch_embed(spectrogram)           # (B, n_patches, d_model)
            # RoPE (if enabled) applies inside rope_enc_layers; no additive PE here.
        elif self.use_conv_frontend:
            x = self.conv_frontend(spectrogram)         # (B, T, d_model)
            if not self.use_rope:
                x = self.pos_enc(x)
        else:
            x = spectrogram.transpose(1, 2)             # (B, T, n_mels)
            x = self.input_proj(x)                      # (B, T, d_model)
            if not self.use_rope:
                x = self.pos_enc(x)

        x = self.dropout(x)

        if self.use_rope:
            for layer in self.rope_enc_layers:
                x = layer(x)
        else:
            x = self.encoder(x)                         # (B, N, d_model)

        enc_out = self.layer_norm(x)
        return enc_out


if __name__ == "__main__":
    # --- Default (per-frame) encoder ---
    encoder = SpectrogramEncoder(
        n_mels=512, d_model=512, nhead=8, num_layers=8,
        dim_feedforward=2048, dropout=0.1,
    )
    encoder.eval()
    x = torch.randn(2, 512, 256)
    with torch.no_grad():
        out = encoder(x)
    print(f"[default]       input: {tuple(x.shape)}  output: {tuple(out.shape)}")
    print(f"[default]       params: {sum(p.numel() for p in encoder.parameters()):,}")

    # --- 2D patch encoder (patch_f=64, patch_t=8 → 256 patches) ---
    encoder2d = SpectrogramEncoder(
        n_mels=512, d_model=512, nhead=8, num_layers=8,
        dim_feedforward=2048, dropout=0.1,
        use_2d_patches=True, patch_f=64, patch_t=8,
    )
    encoder2d.eval()
    with torch.no_grad():
        out2d = encoder2d(x)
    print(f"[2d patches]    input: {tuple(x.shape)}  output: {tuple(out2d.shape)}")
    print(f"[2d patches]    params: {sum(p.numel() for p in encoder2d.parameters()):,}")

    # --- RoPE encoder ---
    enc_rope = SpectrogramEncoder(
        n_mels=512, d_model=512, nhead=8, num_layers=8,
        dim_feedforward=2048, dropout=0.1, use_rope=True,
    )
    enc_rope.eval()
    with torch.no_grad():
        out_rope = enc_rope(x)
    print(f"[rope]          input: {tuple(x.shape)}  output: {tuple(out_rope.shape)}")
    print(f"[rope]          params: {sum(p.numel() for p in enc_rope.parameters()):,}")

    # --- Conv frontend encoder ---
    enc_conv = SpectrogramEncoder(
        n_mels=512, d_model=512, nhead=8, num_layers=8,
        dim_feedforward=2048, dropout=0.1, use_conv_frontend=True, conv_layers=2,
    )
    enc_conv.eval()
    with torch.no_grad():
        out_conv = enc_conv(x)
    print(f"[conv frontend] input: {tuple(x.shape)}  output: {tuple(out_conv.shape)}")
    print(f"[conv frontend] params: {sum(p.numel() for p in enc_conv.parameters()):,}")

    # --- All features combined ---
    enc_all = SpectrogramEncoder(
        n_mels=512, d_model=512, nhead=8, num_layers=8,
        dim_feedforward=2048, dropout=0.1,
        use_2d_patches=True, patch_f=64, patch_t=8, use_rope=True,
    )
    enc_all.eval()
    with torch.no_grad():
        out_all = enc_all(x)
    print(f"[2d+rope]       input: {tuple(x.shape)}  output: {tuple(out_all.shape)}")
    assert out.shape == out2d.shape == out_rope.shape == out_conv.shape == out_all.shape
    print("All encoder smoke tests passed.")
