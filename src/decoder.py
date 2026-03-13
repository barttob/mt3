"""Autoregressive event decoder for MT3 music transcription model."""

import math
from typing import Optional

import torch
import torch.nn as nn

from src.encoder import SinusoidalPositionalEncoding


class EventDecoder(nn.Module):
    """Autoregressive Transformer decoder that generates MIDI event token sequences.

    Cross-attends to encoder hidden states (spectrogram representations) and
    autoregressively predicts the next token at each step. Uses pre-layer-norm
    (norm_first=True) for training stability, matching the T5 convention.

    The output projection weight is tied to the token embedding weight, which
    reduces parameter count and typically improves performance.

    Args:
        vocab_size: Total number of tokens in the MIDI event vocabulary.
        d_model: Model/embedding dimension.
        nhead: Number of attention heads.
        num_layers: Number of Transformer decoder layers.
        dim_feedforward: Feed-forward sublayer hidden dimension.
        dropout: Dropout probability.
        max_seq_len: Maximum target sequence length for positional encoding.
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
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len)
        self.dropout = nn.Dropout(dropout)

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

    def forward(
        self,
        tgt_tokens: torch.Tensor,
        enc_out: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode target token sequences conditioned on encoder hidden states.

        Args:
            tgt_tokens: Target token IDs of shape (B, S), right-shifted for
                teacher forcing (i.e. starts with <sos>).
            enc_out: Encoder hidden states of shape (B, T, d_model) from the
                spectrogram encoder.
            tgt_mask: Causal attention mask of shape (S, S). True/−inf values
                prevent attending to future positions.
            tgt_padding_mask: Boolean padding mask of shape (B, S). True where
                positions are padding and should be ignored.

        Returns:
            logits: Token logits of shape (B, S, vocab_size).
        """
        x = self.token_embedding(tgt_tokens) * math.sqrt(self.d_model)  # (B, S, d_model)
        x = self.pos_enc(x)
        x = self.dropout(x)

        # Ensure both masks use the same dtype to avoid deprecation warnings.
        # generate_square_subsequent_mask returns a float additive mask; if the
        # padding mask is bool, convert it to a matching additive float mask.
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

    decoder = EventDecoder(
        vocab_size=VOCAB_SIZE,
        d_model=512,
        nhead=8,
        num_layers=8,
        dim_feedforward=2048,
        dropout=0.1,
        max_seq_len=1024,
    )
    decoder.eval()

    B, T_enc, S = 2, 256, 100
    enc_out = torch.randn(B, T_enc, 512)         # encoder output (B, T, d_model)
    tgt_tokens = torch.randint(0, VOCAB_SIZE, (B, S))  # random token IDs

    # Causal mask: upper-triangular −inf matrix so position i cannot attend to j > i
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(S)

    with torch.no_grad():
        logits = decoder(tgt_tokens, enc_out, tgt_mask=tgt_mask)

    print(f"enc_out shape:   {tuple(enc_out.shape)}")
    print(f"tgt_tokens shape:{tuple(tgt_tokens.shape)}")
    print(f"logits shape:    {tuple(logits.shape)}")   # expected: (2, 100, 988)
    assert logits.shape == (B, S, VOCAB_SIZE), f"Unexpected shape: {logits.shape}"
    print("Smoke test passed.")

    num_params = sum(p.numel() for p in decoder.parameters())
    # Weight tying means embedding & output_proj share the same tensor;
    # count unique parameters to avoid double-counting.
    num_unique = sum(p.numel() for p in set(decoder.parameters()))
    print(f"Parameters (total / unique): {num_params:,} / {num_unique:,}")
