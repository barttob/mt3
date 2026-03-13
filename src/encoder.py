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


class SpectrogramEncoder(nn.Module):
    """Transformer encoder that operates on log-mel spectrogram frames.

    Each frame (a vector of mel-bin values) is linearly projected to d_model,
    positional encoding is added, then processed by a stack of bidirectional
    (non-causal) Transformer encoder layers with pre-layer-norm.

    Args:
        n_mels: Number of mel frequency bins (input feature dimension).
        d_model: Model/embedding dimension.
        nhead: Number of attention heads.
        num_layers: Number of Transformer encoder layers.
        dim_feedforward: Feed-forward sublayer hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        n_mels: int = 512,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
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
        """Encode a log-mel spectrogram into contextual frame representations.

        Args:
            spectrogram: Log-mel spectrogram of shape (B, n_mels, T).

        Returns:
            Encoder hidden states of shape (B, T, d_model).
        """
        x = spectrogram.transpose(1, 2)   # (B, T, n_mels)
        x = self.input_proj(x)             # (B, T, d_model)
        x = self.pos_enc(x)
        x = self.dropout(x)
        enc_out = self.encoder(x)          # (B, T, d_model)
        enc_out = self.layer_norm(enc_out)
        return enc_out


if __name__ == "__main__":
    encoder = SpectrogramEncoder(
        n_mels=512,
        d_model=512,
        nhead=8,
        num_layers=8,
        dim_feedforward=2048,
        dropout=0.1,
    )
    encoder.eval()

    x = torch.randn(2, 512, 256)  # (B=2, n_mels=512, T=256)
    with torch.no_grad():
        out = encoder(x)

    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(out.shape)}")  # expected: (2, 256, 512)

    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"Parameters:   {num_params:,}")
