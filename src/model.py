"""MT3Model: full encoder-decoder wrapper for music transcription.

Combines SpectrogramFrontend, SpectrogramEncoder, and EventDecoder into a
single nn.Module with teacher-forced training and autoregressive inference.
Also provides a ``build_model`` factory that reads a YAML config file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import yaml

from src.decoder import EventDecoder
from src.encoder import SpectrogramEncoder
from src.frontend import SpectrogramFrontend
from src.tokenizer import MidiTokenizer


class MT3Model(nn.Module):
    """Full MT3 encoder-decoder model for music transcription.

    Wraps a spectrogram frontend, a Transformer encoder, an autoregressive
    Transformer decoder, and the MIDI tokenizer into a single module.

    Args:
        frontend: :class:`SpectrogramFrontend` that converts waveforms to
            log-mel spectrograms.
        encoder: :class:`SpectrogramEncoder` that encodes spectrogram frames
            into contextual representations.
        decoder: :class:`EventDecoder` that autoregressively generates MIDI
            event tokens conditioned on encoder outputs.
        tokenizer: :class:`MidiTokenizer` used for special-token IDs and
            vocabulary size.
    """

    def __init__(
        self,
        frontend: SpectrogramFrontend,
        encoder: SpectrogramEncoder,
        decoder: EventDecoder,
        tokenizer: MidiTokenizer,
    ) -> None:
        super().__init__()
        self.frontend = frontend
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer

    def forward(
        self,
        waveform: torch.Tensor,
        tgt_tokens: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Teacher-forced forward pass.

        Args:
            waveform: Raw audio of shape (B, num_samples).
            tgt_tokens: Right-shifted target token IDs of shape (B, S),
                starting with ``<sos>``.
            tgt_mask: Causal attention mask of shape (S, S).
            tgt_padding_mask: Boolean padding mask of shape (B, S); True
                where positions are ``<pad>`` and should be ignored.

        Returns:
            logits: Token logits of shape (B, S, vocab_size).
        """
        spec = self.frontend(waveform)                              # (B, n_mels, T)
        enc_out = self.encoder(spec)                                # (B, T, d_model)
        logits = self.decoder(
            tgt_tokens, enc_out, tgt_mask, tgt_padding_mask
        )                                                           # (B, S, vocab_size)
        return logits

    @torch.no_grad()
    def transcribe(
        self,
        waveform: torch.Tensor,
        max_len: int = 1024,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        """Autoregressive greedy / temperature-sampled decoding.

        Encodes the waveform once, then generates tokens one at a time until
        every sequence in the batch has produced ``<eos>`` or ``max_len`` is
        reached.

        Args:
            waveform: Raw audio of shape (B, num_samples).
            max_len: Maximum number of tokens to generate (including the
                initial ``<sos>``).
            temperature: Sampling temperature. ``0.0`` (default) uses greedy
                argmax; positive values sample from the softmax distribution.

        Returns:
            generated: Token ID tensor of shape (B, L) where L ≤ max_len.
                Each row starts with ``<sos>``.
        """
        spec = self.frontend(waveform)                              # (B, n_mels, T)
        enc_out = self.encoder(spec)                                # (B, T, d_model)

        B = enc_out.size(0)
        device = enc_out.device
        sos_id = self.tokenizer.special["<sos>"]
        eos_id = self.tokenizer.special["<eos>"]

        generated = torch.full((B, 1), sos_id, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            S = generated.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(S, device=device)
            logits = self.decoder(generated, enc_out, tgt_mask=tgt_mask)
            next_logits = logits[:, -1, :]  # (B, vocab_size)

            if temperature <= 0.0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)  # (B, 1)
            else:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)   # (B, 1)

            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == eos_id).all():
                break

        return generated


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(config: Union[str, Path, dict]) -> MT3Model:
    """Instantiate an :class:`MT3Model` from a YAML config file or dict.

    Reads the ``audio``, ``model``, and ``tokenizer`` sections of the config
    and constructs all sub-components with the specified hyperparameters.

    Args:
        config: Path to a YAML config file (str or :class:`~pathlib.Path`) or
            a pre-loaded config dictionary.

    Returns:
        Fully initialised :class:`MT3Model` (weights are randomly initialised).
    """
    if not isinstance(config, dict):
        with open(config) as fh:
            config = yaml.safe_load(fh)

    audio_cfg = config["audio"]
    model_cfg = config["model"]
    tok_cfg = config["tokenizer"]
    data_cfg = config.get("data", {})

    tokenizer = MidiTokenizer(
        time_step_ms=tok_cfg.get("time_step_ms", 8),
        max_time_steps=tok_cfg.get("max_time_steps", 600),
        multi_instrument=tok_cfg.get("multi_instrument", False),
    )

    frontend = SpectrogramFrontend(
        sample_rate=data_cfg.get("sample_rate", 16000),
        n_fft=audio_cfg["n_fft"],
        hop_length=audio_cfg["hop_length"],
        n_mels=audio_cfg["n_mels"],
    )

    encoder = SpectrogramEncoder(
        n_mels=audio_cfg["n_mels"],
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_layers=model_cfg["enc_layers"],
        dim_feedforward=model_cfg["d_ff"],
        dropout=model_cfg["dropout"],
    )

    decoder = EventDecoder(
        vocab_size=tokenizer.vocab_size,
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_layers=model_cfg["dec_layers"],
        dim_feedforward=model_cfg["d_ff"],
        dropout=model_cfg["dropout"],
        max_seq_len=model_cfg["max_token_len"],
    )

    return MT3Model(frontend, encoder, decoder, tokenizer)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    CONFIG_PATH = Path("configs/maestro_piano.yaml")

    print("=" * 60)
    print("MT3Model smoke test")
    print("=" * 60)

    # --- Build model --------------------------------------------------------
    model = build_model(CONFIG_PATH)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    unique_params = sum(p.numel() for p in set(model.parameters()))
    print(f"  Config:         {CONFIG_PATH}")
    print(f"  Vocab size:     {model.tokenizer.vocab_size}")
    print(f"  Total params:   {total_params:,}")
    print(f"  Unique params:  {unique_params:,}  (weight tying deduped)")

    # --- Random inputs ------------------------------------------------------
    B = 1
    NUM_SAMPLES = 256_000          # 16 s at 16 kHz
    MAX_TOK = 64                   # short sequence for a fast smoke test
    VOCAB = model.tokenizer.vocab_size

    waveform = torch.randn(B, NUM_SAMPLES)
    # Dummy token sequence: <sos> followed by random tokens, padded
    sos = model.tokenizer.special["<sos>"]
    pad = model.tokenizer.special["<pad>"]
    tgt_ids = torch.randint(4, VOCAB, (B, MAX_TOK))
    tgt_ids[:, 0] = sos

    S = MAX_TOK
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(S)
    tgt_padding_mask = tgt_ids == pad

    # --- Forward pass -------------------------------------------------------
    print("\n  Running forward pass ...")
    with torch.no_grad():
        logits = model(waveform, tgt_ids, tgt_mask, tgt_padding_mask)

    expected_logits = (B, MAX_TOK, VOCAB)
    assert logits.shape == expected_logits, (
        f"Logits shape mismatch: got {tuple(logits.shape)}, expected {expected_logits}"
    )
    print(f"  forward() logits shape: {tuple(logits.shape)}  ✓")

    # --- Transcribe ---------------------------------------------------------
    print("\n  Running transcribe() ...")
    generated = model.transcribe(waveform, max_len=32, temperature=0.0)

    assert generated.shape[0] == B, f"Batch size mismatch: {generated.shape[0]}"
    assert generated[0, 0].item() == sos, (
        f"First token should be <sos>={sos}, got {generated[0, 0].item()}"
    )
    print(f"  transcribe() output shape: {tuple(generated.shape)}  ✓")
    print(f"  First token: {generated[0, 0].item()} (expected <sos>={sos})  ✓")
    print(f"  Token sequence: {generated[0].tolist()}")

    print("\nSmoke test passed.")
