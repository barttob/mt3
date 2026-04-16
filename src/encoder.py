"""Spectrogram encoder for MT3 music transcription model."""

import math

import torch
import torch.nn as nn


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


class SpectrogramEncoder(nn.Module):
    """Transformer encoder that operates on log-mel spectrogram frames.

    Supports two input embedding modes:
    - Default (use_2d_patches=False): each mel frame linearly projected to d_model,
      followed by 1D sinusoidal time positional encoding.
    - 2D patch mode (use_2d_patches=True): the spectrogram is divided into
      (patch_f × patch_t) patches, each projected to d_model, with 2D sinusoidal
      positional encoding that separately encodes frequency and time axes.

    Both modes feed into the same stack of bidirectional Transformer encoder layers
    with pre-layer-norm (T5 convention).

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
    ) -> None:
        super().__init__()
        self.use_2d_patches = use_2d_patches

        if use_2d_patches:
            self.patch_embed = PatchEmbedding2D(
                n_mels=n_mels,
                patch_f=patch_f,
                patch_t=patch_t,
                d_model=d_model,
            )
        else:
            self.input_proj = nn.Linear(n_mels, d_model)
            self.pos_enc = SinusoidalPositionalEncoding(d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

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
            x = self.patch_embed(spectrogram)   # (B, n_patches, d_model)
        else:
            x = spectrogram.transpose(1, 2)     # (B, T, n_mels)
            x = self.input_proj(x)              # (B, T, d_model)
            x = self.pos_enc(x)                 # adds sinusoidal time PE

        x = self.dropout(x)
        enc_out = self.encoder(x)               # (B, N, d_model)
        enc_out = self.layer_norm(enc_out)
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
    print(f"[default] input: {tuple(x.shape)}  output: {tuple(out.shape)}")
    print(f"[default] params: {sum(p.numel() for p in encoder.parameters()):,}")

    # --- 2D patch encoder (patch_f=64, patch_t=8 → 256 patches) ---
    encoder2d = SpectrogramEncoder(
        n_mels=512, d_model=512, nhead=8, num_layers=8,
        dim_feedforward=2048, dropout=0.1,
        use_2d_patches=True, patch_f=64, patch_t=8,
    )
    encoder2d.eval()
    with torch.no_grad():
        out2d = encoder2d(x)
    print(f"[2d]     input: {tuple(x.shape)}  output: {tuple(out2d.shape)}")
    print(f"[2d]     params: {sum(p.numel() for p in encoder2d.parameters()):,}")
