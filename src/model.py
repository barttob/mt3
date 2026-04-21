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


def compute_pitch_context(
    tokens: torch.Tensor, note_on_offset: int
) -> torch.Tensor:
    """Compute the MIDI pitch context at each position of a token sequence.

    Returns a tensor of the same shape as ``tokens`` where each entry is the
    MIDI pitch of the most recent ``note_on`` token seen up to and including
    that position, or 128 if no ``note_on`` has been seen yet.  Used to
    supply ``pitch_ids`` to :class:`~src.decoder.PitchAwareDecoderLayer`.

    Args:
        tokens: Integer token IDs of shape (B, S).
        note_on_offset: Token ID of ``note_on(0)`` in the vocabulary.

    Returns:
        Tensor of shape (B, S) with dtype ``torch.long``.
        Values 0–127 are MIDI pitches; 128 means no pitch context.
    """
    is_note_on = (tokens >= note_on_offset) & (tokens < note_on_offset + 128)
    pitch_ids = torch.where(
        is_note_on,
        tokens - note_on_offset,
        torch.full_like(tokens, 128),
    )
    # Forward-fill: carry the last pitch context forward at each position
    for s in range(1, pitch_ids.shape[1]):
        pitch_ids[:, s] = torch.where(
            pitch_ids[:, s] == 128,
            pitch_ids[:, s - 1],
            pitch_ids[:, s],
        )
    return pitch_ids


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

        pitch_ids = None
        if self.decoder.use_pitch_aware_attention:
            pitch_ids = compute_pitch_context(
                tgt_tokens, self.tokenizer.note_on_offset
            )

        logits = self.decoder(
            tgt_tokens, enc_out, pitch_ids, tgt_mask, tgt_padding_mask
        )                                                           # (B, S, vocab_size)
        return logits

    @torch.no_grad()
    def transcribe(
        self,
        waveform: torch.Tensor,
        max_len: int = 1024,
        temperature: float = 0.0,
        prompt_tokens: Optional[torch.Tensor | list[int]] = None,
        return_confidences: bool = False,
        beam_size: int = 1,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive decoding: greedy, temperature sampling, or beam search.

        Encodes the waveform once, then generates tokens one at a time until
        every sequence in the batch has produced ``<eos>`` or ``max_len`` is
        reached.

        Args:
            waveform: Raw audio of shape (B, num_samples).
            max_len: Maximum number of tokens to generate (including the
                initial ``<sos>``).
            temperature: Sampling temperature. ``0.0`` (default) uses greedy
                argmax; positive values sample from the softmax distribution.
                Ignored when ``beam_size > 1``.
            prompt_tokens: Optional decoder prompt. When provided this should
                already contain ``<sos>`` and any tie-section tokens.
            return_confidences: If ``True``, also return a float tensor of
                shape (B, L) containing the max softmax probability at each
                generated step (prompt positions are filled with ``1.0``).
            beam_size: Number of beams for beam search. ``1`` (default) uses
                greedy / temperature decoding. Values > 1 enable beam search
                and ignore ``temperature``.

        Returns:
            generated: Token ID tensor of shape (B, L) where L ≤ max_len.
                Each row starts with ``<sos>``.
            confidences (only when ``return_confidences=True``): Float tensor
                of shape (B, L) with per-step max softmax probability.
                Prompt positions are set to ``1.0``.
        """
        device = waveform.device
        amp_dtype = (
            torch.bfloat16
            if device.type == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            spec = self.frontend(waveform)                          # (B, n_mels, T)
            enc_out = self.encoder(spec)                            # (B, T, d_model)

        B = enc_out.size(0)
        sos_id = self.tokenizer.special["<sos>"]
        eos_id = self.tokenizer.special["<eos>"]

        # Build prompt tensor (B, P)
        if prompt_tokens is None:
            prompt = torch.full((B, 1), sos_id, dtype=torch.long, device=device)
        else:
            prompt = torch.as_tensor(prompt_tokens, dtype=torch.long, device=device)
            if prompt.ndim == 1:
                prompt = prompt.unsqueeze(0).expand(B, -1).clone()
            elif prompt.ndim != 2 or prompt.size(0) != B:
                raise ValueError(
                    f"prompt_tokens must have shape (P,) or (B, P); got {tuple(prompt.shape)}"
                )
            if prompt.size(1) > max_len:
                raise ValueError(
                    f"Prompt length {prompt.size(1)} exceeds max_len={max_len}."
                )

        if beam_size > 1:
            return self._beam_search(
                enc_out, prompt, max_len, beam_size, device, amp_dtype, return_confidences
            )

        # -----------------------------------------------------------------
        # Greedy / temperature decoding (beam_size == 1)
        # -----------------------------------------------------------------
        generated = prompt
        prompt_len = generated.size(1)

        # Track per-step confidence (max softmax prob) for generated tokens only.
        step_confidences: list[torch.Tensor] = []  # each (B,)

        # Pre-allocate the full causal mask once and slice per step — avoids
        # O(max_len) separate triu() allocations inside the decode loop.
        full_mask = nn.Transformer.generate_square_subsequent_mask(max_len, device=device)

        for _ in range(max_len - generated.size(1)):
            S = generated.size(1)
            tgt_mask = full_mask[:S, :S]

            pitch_ids = None
            if self.decoder.use_pitch_aware_attention:
                pitch_ids = compute_pitch_context(
                    generated, self.tokenizer.note_on_offset
                )

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                logits = self.decoder(generated, enc_out, pitch_ids=pitch_ids, tgt_mask=tgt_mask)
            next_logits = logits[:, -1, :]  # (B, vocab_size)

            probs = torch.softmax(next_logits.float(), dim=-1)  # fp32 for stability
            if temperature <= 0.0:
                next_token = probs.argmax(dim=-1, keepdim=True)  # (B, 1)
            else:
                scaled = torch.softmax(next_logits.float() / temperature, dim=-1)
                next_token = torch.multinomial(scaled, num_samples=1)  # (B, 1)

            step_confidences.append(probs.max(dim=-1).values)   # (B,)
            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == eos_id).all():
                break

        if not return_confidences:
            return generated

        # Prepend 1.0 confidence for the prompt positions.
        prompt_conf = torch.ones(B, prompt_len, device=device)
        if step_confidences:
            gen_conf = torch.stack(step_confidences, dim=1)      # (B, steps)
            confidences = torch.cat([prompt_conf, gen_conf], dim=1)
        else:
            confidences = prompt_conf
        return generated, confidences

    @torch.no_grad()
    def _beam_search(
        self,
        enc_out: torch.Tensor,
        prompt: torch.Tensor,
        max_len: int,
        beam_size: int,
        device: torch.device,
        amp_dtype: torch.dtype,
        return_confidences: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Beam search decoding.

        Args:
            enc_out: Encoder output of shape (B, T, d_model).
            prompt: Initial token sequence (B, P) starting with ``<sos>``.
            max_len: Maximum total sequence length including prompt.
            beam_size: Number of beams (K).
            device: Inference device.
            amp_dtype: dtype for autocast.
            return_confidences: Whether to return per-token confidence scores.

        Returns:
            best_seq: (B, L) highest-scoring token sequence per batch item.
            confidences (optional): (B, L) per-token softmax probability.
        """
        B, P = prompt.shape
        K = beam_size
        T, d = enc_out.size(1), enc_out.size(2)
        eos_id = self.tokenizer.special["<eos>"]
        vocab_size = self.tokenizer.vocab_size

        # Expand encoder output for all beams: (B, T, d) → (B*K, T, d)
        enc_out_exp = (
            enc_out.unsqueeze(1)
                   .expand(-1, K, -1, -1)
                   .reshape(B * K, T, d)
        )

        # All K beams start from the same prompt: (B, P) → (B*K, P)
        beams = prompt.unsqueeze(1).expand(-1, K, -1).reshape(B * K, P).clone()

        # Log-prob scores: (B, K). Only first beam active; rest start at -inf
        # so diversity is forced from the very first generation step.
        beam_scores = torch.full((B, K), -float("inf"), device=device)
        beam_scores[:, 0] = 0.0

        # finished[b, k] = True once beam k of batch b emitted <eos>
        finished = torch.zeros(B, K, dtype=torch.bool, device=device)

        # Confidence history: (B*K, P) — prompt positions all 1.0
        beam_conf = torch.ones(B * K, P, device=device)

        full_mask = nn.Transformer.generate_square_subsequent_mask(max_len, device=device)

        for _ in range(max_len - P):
            S = beams.size(1)
            tgt_mask = full_mask[:S, :S]

            pitch_ids = None
            if self.decoder.use_pitch_aware_attention:
                pitch_ids = compute_pitch_context(beams, self.tokenizer.note_on_offset)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                logits = self.decoder(beams, enc_out_exp, pitch_ids=pitch_ids, tgt_mask=tgt_mask)

            next_logits = logits[:, -1, :].float()               # (B*K, vocab)
            log_probs = torch.log_softmax(next_logits, dim=-1)   # (B*K, vocab)
            log_probs_bkv = log_probs.reshape(B, K, vocab_size)

            # Force finished beams to stay at EOS with zero incremental score
            if finished.any():
                log_probs_bkv[finished] = -float("inf")
                log_probs_bkv[finished, eos_id] = 0.0

            # Candidate scores: beam_scores + next log-prob → (B, K, vocab)
            candidate_scores = beam_scores.unsqueeze(-1) + log_probs_bkv
            flat_scores = candidate_scores.reshape(B, K * vocab_size)

            # Top-K candidates per batch item
            topk_scores, topk_flat = flat_scores.topk(K, dim=-1)  # (B, K)
            prev_beam_idx = topk_flat // vocab_size                 # (B, K)
            next_token_id = topk_flat % vocab_size                  # (B, K)

            # Flat indices into (B*K, ...) — maps (b, k) → b*K + prev_beam_idx[b, k]
            batch_offset = torch.arange(B, device=device).unsqueeze(1) * K
            global_beam_idx = (batch_offset + prev_beam_idx).reshape(-1)  # (B*K,)
            next_token_flat = next_token_id.reshape(-1)                    # (B*K,)

            # Per-token confidence: softmax prob of chosen token under parent beam
            step_conf = log_probs[global_beam_idx, next_token_flat].exp()  # (B*K,)

            # Reorder sequences and conf history to follow parent beams
            beams = beams[global_beam_idx]
            beam_conf = beam_conf[global_beam_idx]

            # Append new token / confidence
            beams = torch.cat([beams, next_token_flat.unsqueeze(1)], dim=1)
            beam_conf = torch.cat([beam_conf, step_conf.unsqueeze(1)], dim=1)

            # Update scores and finished flags
            beam_scores = topk_scores
            finished = finished.reshape(B * K)[global_beam_idx].reshape(B, K)
            finished = finished | (next_token_id == eos_id)

            if finished.all():
                break

        # Select best beam per batch item with mild length normalisation
        length_penalty = beams.size(1) ** 0.6
        best_k = (beam_scores / length_penalty).argmax(dim=-1)  # (B,)

        batch_offset_1d = torch.arange(B, device=device) * K
        global_best = batch_offset_1d + best_k                  # (B,)

        best_seq = beams[global_best]                            # (B, L)

        if not return_confidences:
            return best_seq

        best_conf = beam_conf[global_best]                       # (B, L)
        return best_seq, best_conf


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
        num_programs=tok_cfg.get(
            "num_programs",
            129 if tok_cfg.get("multi_instrument", False) else 128,
        ),
        multi_instrument=tok_cfg.get("multi_instrument", False),
        use_hierarchical_time=tok_cfg.get("use_hierarchical_time", False),
        coarse_step_ms=tok_cfg.get("coarse_step_ms", 64),
        num_coarse=tok_cfg.get("num_coarse", 75),
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
        use_2d_patches=model_cfg.get("use_2d_patches", False),
        patch_f=model_cfg.get("patch_f", 64),
        patch_t=model_cfg.get("patch_t", 8),
        use_rope=model_cfg.get("use_rope", False),
        use_conv_frontend=model_cfg.get("use_conv_frontend", False),
        conv_layers=model_cfg.get("conv_layers", 2),
    )

    decoder = EventDecoder(
        vocab_size=tokenizer.vocab_size,
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_layers=model_cfg["dec_layers"],
        dim_feedforward=model_cfg["d_ff"],
        dropout=model_cfg["dropout"],
        max_seq_len=model_cfg["max_token_len"],
        use_pitch_aware_attention=model_cfg.get("use_pitch_aware_attention", False),
        use_rope=model_cfg.get("use_rope", False),
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
