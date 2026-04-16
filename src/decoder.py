"""Autoregressive event decoder for MT3 music transcription model."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoder import RoPEMultiheadAttention, SinusoidalPositionalEncoding


class PitchAwareDecoderLayer(nn.Module):
    """Decoder layer that injects pitch-frequency bias before cross-attention.

    Architecturally identical to ``nn.TransformerDecoderLayer(norm_first=True)``
    with one addition: a learned ``pitch_embedding`` (129 entries — 128 MIDI
    pitches + 1 null) is added to the query input of the cross-attention
    sublayer. This biases the decoder's attention toward encoder representations
    characteristic of the most recently emitted note pitch, providing an
    explicit pitch-frequency inductive bias without changing any other
    component.

    The null embedding (index 128) is initialised to zero so that positions
    with no pitch context contribute no bias.

    When ``use_rope=True`` the causal self-attention sub-layer uses
    :class:`~src.encoder.RoPEMultiheadAttention` instead of the standard
    ``nn.MultiheadAttention``, replacing additive positional encoding with
    rotary position embeddings on Q and K.  Cross-attention is always left as
    standard ``nn.MultiheadAttention`` because encoder and decoder positions
    are independent sequences.

    Args:
        d_model: Model dimension.
        nhead: Number of attention heads.
        dim_feedforward: Hidden dimension of the FFN sublayer.
        dropout: Dropout probability applied throughout.
        use_rope: If True, use RoPEMultiheadAttention for self-attention.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        use_rope: bool = False,
    ) -> None:
        super().__init__()
        if use_rope:
            self.self_attn: nn.Module = RoPEMultiheadAttention(d_model, nhead, dropout)
        else:
            self.self_attn = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=True
            )
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)
        self.drop_ff = nn.Dropout(dropout)

        # 128 MIDI pitch embeddings + 1 null (index 128, initialised to 0)
        self.pitch_embedding = nn.Embedding(129, d_model)
        nn.init.normal_(self.pitch_embedding.weight, std=0.02)
        with torch.no_grad():
            self.pitch_embedding.weight[128].zero_()

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        pitch_ids: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with pitch-conditioned cross-attention queries.

        Args:
            tgt: Decoder hidden states, shape (B, S, d_model).
            memory: Encoder output (cross-attention keys/values), shape (B, T, d_model).
            pitch_ids: MIDI pitch context at each decoder position, shape (B, S).
                Values 0–127 are MIDI pitches; 128 means no pitch context yet.
            tgt_mask: Causal additive mask, shape (S, S).
            tgt_key_padding_mask: Padding mask for target, shape (B, S).
            memory_key_padding_mask: Padding mask for memory, shape (B, T).

        Returns:
            Updated decoder hidden states, shape (B, S, d_model).
        """
        x = tgt

        # Pre-norm self-attention
        x2 = self.norm1(x)
        x2, _ = self.self_attn(
            x2, x2, x2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        x = x + self.drop1(x2)

        # Pre-norm cross-attention with pitch bias injected into query
        x2 = self.norm2(x)
        pitch_bias = self.pitch_embedding(pitch_ids)  # (B, S, d_model)
        x2_q = x2 + pitch_bias                       # pitch-conditioned query
        x2, _ = self.cross_attn(
            x2_q, memory, memory,
            key_padding_mask=memory_key_padding_mask,
        )
        x = x + self.drop2(x2)

        # Pre-norm feed-forward
        x2 = self.norm3(x)
        x2 = self.linear2(self.drop_ff(F.gelu(self.linear1(x2))))
        x = x + self.drop3(x2)
        return x


class RoPETransformerDecoderLayer(nn.Module):
    """Pre-norm decoder layer with RoPE self-attention and standard cross-attention.

    Used when ``use_rope=True`` and ``use_pitch_aware_attention=False``.
    Self-attention uses :class:`~src.encoder.RoPEMultiheadAttention`; cross-
    attention remains a standard ``nn.MultiheadAttention`` because encoder and
    decoder positions are independent sequences and rotating cross-attention
    Q/K would be semantically incorrect.

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
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)
        self.drop_ff = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            tgt: Decoder hidden states, shape (B, S, d_model).
            memory: Encoder output, shape (B, T, d_model).
            tgt_mask: Causal additive mask, shape (S, S).
            tgt_key_padding_mask: Padding mask for target, shape (B, S).
            memory_key_padding_mask: Padding mask for memory, shape (B, T).

        Returns:
            Updated decoder hidden states, shape (B, S, d_model).
        """
        x = tgt

        # Pre-norm self-attention with RoPE
        x2 = self.norm1(x)
        x2, _ = self.self_attn(
            x2, x2, x2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        x = x + self.drop1(x2)

        # Pre-norm cross-attention (standard — no RoPE on cross-attention)
        x2 = self.norm2(x)
        x2, _ = self.cross_attn(
            x2, memory, memory,
            key_padding_mask=memory_key_padding_mask,
        )
        x = x + self.drop2(x2)

        # Pre-norm feed-forward
        x2 = self.norm3(x)
        x2 = self.linear2(self.drop_ff(F.gelu(self.linear1(x2))))
        x = x + self.drop3(x2)
        return x


class EventDecoder(nn.Module):
    """Autoregressive Transformer decoder that generates MIDI event token sequences.

    Cross-attends to encoder hidden states (spectrogram representations) and
    autoregressively predicts the next token at each step. Uses pre-layer-norm
    (norm_first=True) for training stability, matching the T5 convention.

    The output projection weight is tied to the token embedding weight, which
    reduces parameter count and typically improves performance.

    The decoder supports four combinations of two orthogonal flags:

    =====================  =========  =======================================
    use_pitch_aware        use_rope   Decoder stack
    =====================  =========  =======================================
    False                  False      ``nn.TransformerDecoder`` (default)
    False                  True       ``nn.ModuleList[RoPETransformerDecoderLayer]``
    True                   False      ``nn.ModuleList[PitchAwareDecoderLayer]``
    True                   True       ``nn.ModuleList[PitchAwareDecoderLayer(use_rope=True)]``
    =====================  =========  =======================================

    When ``use_rope=True`` the sinusoidal positional encoding is not applied
    to the token embeddings; RoPE encodes position inside each self-attention
    layer instead.

    Args:
        vocab_size: Total number of tokens in the MIDI event vocabulary.
        d_model: Model/embedding dimension.
        nhead: Number of attention heads.
        num_layers: Number of Transformer decoder layers.
        dim_feedforward: Feed-forward sublayer hidden dimension.
        dropout: Dropout probability.
        max_seq_len: Maximum target sequence length for positional encoding.
        use_pitch_aware_attention: If True, use PitchAwareDecoderLayer.
        use_rope: If True, use RoPE in decoder self-attention and skip the
            sinusoidal positional encoding on token embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        use_pitch_aware_attention: bool = False,
        use_rope: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.use_pitch_aware_attention = use_pitch_aware_attention
        self.use_rope = use_rope

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # pos_enc is created even when use_rope=True for the non-RoPE paths;
        # it simply won't be called in the forward pass when use_rope=True.
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len)
        self.dropout = nn.Dropout(dropout)

        if use_pitch_aware_attention:
            # Pitch-aware layers with optional RoPE in self-attention
            self.pitch_layers = nn.ModuleList([
                PitchAwareDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                       use_rope=use_rope)
                for _ in range(num_layers)
            ])
        elif use_rope:
            # RoPE-only (no pitch bias)
            self.rope_dec_layers = nn.ModuleList([
                RoPETransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ])
        else:
            # Standard nn.TransformerDecoder (original behaviour)
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,  # Pre-LN for training stability (T5 convention)
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.layer_norm = nn.LayerNorm(d_model)

        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying: share parameters between embedding and output projection
        self.output_proj.weight = self.token_embedding.weight

        # Initialize token embeddings with scaled variance to account for LayerNorm,
        # preventing massive initial CE loss from unscaled logits.
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=d_model**-0.5)

    def forward(
        self,
        tgt_tokens: torch.Tensor,
        enc_out: torch.Tensor,
        pitch_ids: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode target token sequences conditioned on encoder hidden states.

        Args:
            tgt_tokens: Target token IDs of shape (B, S), right-shifted for
                teacher forcing (i.e. starts with <sos>).
            enc_out: Encoder hidden states of shape (B, T, d_model) from the
                spectrogram encoder.
            pitch_ids: MIDI pitch context at each decoder position, shape (B, S).
                Values 0–127 are pitches; 128 = no pitch context. Required when
                ``use_pitch_aware_attention=True``; ignored otherwise.
            tgt_mask: Causal attention mask of shape (S, S). True/−inf values
                prevent attending to future positions.
            tgt_padding_mask: Boolean padding mask of shape (B, S). True where
                positions are padding and should be ignored.

        Returns:
            logits: Token logits of shape (B, S, vocab_size).
        """
        x = self.token_embedding(tgt_tokens) * math.sqrt(self.d_model)  # (B, S, d_model)
        if not self.use_rope:
            x = self.pos_enc(x)
        x = self.dropout(x)

        if self.use_pitch_aware_attention:
            if pitch_ids is None:
                pitch_ids = torch.full(
                    tgt_tokens.shape, 128,
                    dtype=torch.long, device=tgt_tokens.device
                )
            for layer in self.pitch_layers:
                x = layer(
                    tgt=x,
                    memory=enc_out,
                    pitch_ids=pitch_ids,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_padding_mask,
                )
        elif self.use_rope:
            for layer in self.rope_dec_layers:
                x = layer(
                    tgt=x,
                    memory=enc_out,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_padding_mask,
                )
        else:
            # Ensure both masks use the same dtype to avoid deprecation warnings.
            if tgt_mask is not None and tgt_padding_mask is not None:
                if tgt_padding_mask.dtype == torch.bool and tgt_mask.is_floating_point():
                    float_pad = torch.zeros(
                        tgt_padding_mask.shape,
                        dtype=tgt_mask.dtype,
                        device=tgt_padding_mask.device,
                    )
                    tgt_padding_mask = float_pad.masked_fill(tgt_padding_mask, float("-inf"))

            x = self.decoder(
                tgt=x,
                memory=enc_out,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_padding_mask,
            )

        x = self.layer_norm(x)
        logits = self.output_proj(x)  # (B, S, vocab_size)
        return logits


if __name__ == "__main__":
    VOCAB_SIZE = 988  # piano-only vocabulary size
    B, T_enc, S = 2, 256, 100
    enc_out = torch.randn(B, T_enc, 512)
    tgt_tokens = torch.randint(0, VOCAB_SIZE, (B, S))
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(S)

    # --- Default decoder ---
    decoder = EventDecoder(
        vocab_size=VOCAB_SIZE, d_model=512, nhead=8, num_layers=8,
        dim_feedforward=2048, dropout=0.1, max_seq_len=1024,
    )
    decoder.eval()
    with torch.no_grad():
        logits = decoder(tgt_tokens, enc_out, tgt_mask=tgt_mask)
    assert logits.shape == (B, S, VOCAB_SIZE)
    num_unique = sum(p.numel() for p in set(decoder.parameters()))
    print(f"[default]            logits: {tuple(logits.shape)}  params: {num_unique:,}")

    # --- Pitch-aware decoder ---
    decoder_pa = EventDecoder(
        vocab_size=VOCAB_SIZE, d_model=512, nhead=8, num_layers=8,
        dim_feedforward=2048, dropout=0.1, max_seq_len=1024,
        use_pitch_aware_attention=True,
    )
    decoder_pa.eval()
    pitch_ids = torch.randint(0, 129, (B, S))
    with torch.no_grad():
        logits_pa = decoder_pa(tgt_tokens, enc_out, pitch_ids=pitch_ids, tgt_mask=tgt_mask)
    assert logits_pa.shape == (B, S, VOCAB_SIZE)
    num_unique_pa = sum(p.numel() for p in set(decoder_pa.parameters()))
    print(f"[pitch-aware]        logits: {tuple(logits_pa.shape)}  params: {num_unique_pa:,}")

    # --- RoPE decoder ---
    decoder_rope = EventDecoder(
        vocab_size=VOCAB_SIZE, d_model=512, nhead=8, num_layers=8,
        dim_feedforward=2048, dropout=0.1, max_seq_len=1024,
        use_rope=True,
    )
    decoder_rope.eval()
    with torch.no_grad():
        logits_rope = decoder_rope(tgt_tokens, enc_out, tgt_mask=tgt_mask)
    assert logits_rope.shape == (B, S, VOCAB_SIZE)
    num_unique_rope = sum(p.numel() for p in set(decoder_rope.parameters()))
    print(f"[rope]               logits: {tuple(logits_rope.shape)}  params: {num_unique_rope:,}")

    # --- Pitch-aware + RoPE decoder ---
    decoder_pa_rope = EventDecoder(
        vocab_size=VOCAB_SIZE, d_model=512, nhead=8, num_layers=8,
        dim_feedforward=2048, dropout=0.1, max_seq_len=1024,
        use_pitch_aware_attention=True, use_rope=True,
    )
    decoder_pa_rope.eval()
    with torch.no_grad():
        logits_par = decoder_pa_rope(tgt_tokens, enc_out, pitch_ids=pitch_ids, tgt_mask=tgt_mask)
    assert logits_par.shape == (B, S, VOCAB_SIZE)
    num_unique_par = sum(p.numel() for p in set(decoder_pa_rope.parameters()))
    print(f"[pitch-aware + rope] logits: {tuple(logits_par.shape)}  params: {num_unique_par:,}")

    print("Smoke tests passed.")
