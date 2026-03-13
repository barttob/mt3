"""Dry-run test for TranscriptionDataset using purely synthetic data.

Creates a temporary directory with two fake tracks (audio + notes .npy files),
instantiates a TranscriptionDataset, draws several samples, and verifies
output shapes and dtypes.  No real dataset or audio files are required.

Run with::

    python scripts/test_dataset_dryrun.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure the project root is on sys.path so `src.*` imports work regardless of
# the working directory the script is launched from.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.dataset import TranscriptionDataset, collate_fn
from src.tokenizer import MidiTokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_notes(n: int = 20) -> np.ndarray:
    """Generate n random note-event dicts as an object array."""
    rng = np.random.default_rng(42)
    notes = []
    for _ in range(n):
        onset = float(rng.uniform(0.0, 14.0))
        duration = float(rng.uniform(0.05, 1.0))
        offset = onset + duration
        pitch = int(rng.integers(21, 109))      # piano range
        velocity = int(rng.integers(30, 127))
        notes.append(
            {
                "onset": onset,
                "offset": offset,
                "pitch": pitch,
                "velocity": velocity,
                "program": 0,
            }
        )
    return np.array(notes, dtype=object)


def _make_fake_audio(sample_rate: int = 16_000, duration_s: float = 16.0) -> np.ndarray:
    """Generate a silent (zero) waveform of the given duration."""
    return np.zeros(int(sample_rate * duration_s), dtype=np.float32)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def run_test() -> None:
    """Run the dry-run dataset test."""
    sample_rate = 16_000
    segment_samples = 256_000   # 16 s
    max_token_len = 1024
    batch_size = 2

    print("=" * 60)
    print("TranscriptionDataset — dry-run test (synthetic data)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)

        # Write two fake tracks
        for track_id in range(3):
            audio = _make_fake_audio(sample_rate)
            notes = _make_fake_notes(n=30)
            np.save(data_dir / f"track{track_id:02d}_audio.npy", audio)
            np.save(data_dir / f"track{track_id:02d}_notes.npy", notes)

        print(f"\nCreated 3 synthetic tracks in {data_dir}")

        # ---- Piano-only tokenizer -----------------------------------------
        tokenizer = MidiTokenizer(multi_instrument=False)
        dataset = TranscriptionDataset(
            data_dir=data_dir,
            tokenizer=tokenizer,
            sample_rate=sample_rate,
            segment_samples=segment_samples,
            max_token_len=max_token_len,
        )

        print(f"\nDataset length: {len(dataset)}  (expected 3)")
        assert len(dataset) == 3, f"Expected 3, got {len(dataset)}"

        # ---- Single sample --------------------------------------------------
        waveform, token_ids = dataset[0]

        print(f"\nSingle sample:")
        print(f"  waveform.shape  = {tuple(waveform.shape)}  (expected ({segment_samples},))")
        print(f"  waveform.dtype  = {waveform.dtype}  (expected torch.float32)")
        print(f"  token_ids.shape = {tuple(token_ids.shape)}  (expected ({max_token_len},))")
        print(f"  token_ids.dtype = {token_ids.dtype}  (expected torch.int64)")

        assert waveform.shape == (segment_samples,), \
            f"Bad waveform shape: {waveform.shape}"
        assert waveform.dtype == torch.float32, \
            f"Bad waveform dtype: {waveform.dtype}"
        assert token_ids.shape == (max_token_len,), \
            f"Bad token_ids shape: {token_ids.shape}"
        assert token_ids.dtype == torch.long, \
            f"Bad token_ids dtype: {token_ids.dtype}"

        # ---- Token structure sanity ----------------------------------------
        tokens_list = token_ids.tolist()
        sos_id = tokenizer.special["<sos>"]
        eos_id = tokenizer.special["<eos>"]
        tie_id = tokenizer.special["<tie>"]
        pad_id = tokenizer.special["<pad>"]

        assert tokens_list[0] == sos_id, \
            f"Expected <sos> at position 0, got {tokens_list[0]}"
        assert eos_id in tokens_list, \
            "<eos> token not found in sequence"
        eos_pos = tokens_list.index(eos_id)
        assert tie_id in tokens_list[:eos_pos], \
            "<tie> not found before <eos>"

        # All tokens after <eos> must be <pad>
        after_eos = tokens_list[eos_pos + 1:]
        assert all(t == pad_id for t in after_eos), \
            f"Non-pad tokens after <eos>: {[t for t in after_eos if t != pad_id]}"

        print("\n  Token structure OK:")
        print(f"    tokens[0]        = {tokenizer.token_type(tokens_list[0])}")
        print(f"    <tie> at index   = {tokens_list.index(tie_id)}")
        print(f"    <eos> at index   = {eos_pos}")
        print(f"    trailing <pad>s  = {len(after_eos)}")

        # All token IDs must be in vocabulary range
        assert all(0 <= t < tokenizer.vocab_size for t in tokens_list), \
            "Token out of vocabulary range detected"
        print(f"    all token IDs in [0, {tokenizer.vocab_size})  ✓")

        # ---- DataLoader batch -----------------------------------------------
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=False,
        )
        batch_waveforms, batch_tokens = next(iter(loader))

        print(f"\nDataLoader batch (batch_size={batch_size}):")
        print(f"  batch_waveforms.shape = {tuple(batch_waveforms.shape)}")
        print(f"  batch_tokens.shape    = {tuple(batch_tokens.shape)}")

        assert batch_waveforms.shape == (batch_size, segment_samples), \
            f"Bad batch waveform shape: {batch_waveforms.shape}"
        assert batch_tokens.shape == (batch_size, max_token_len), \
            f"Bad batch token shape: {batch_tokens.shape}"

        # ---- Multi-instrument mode -----------------------------------------
        print("\nMulti-instrument tokenizer test …")
        tok_multi = MidiTokenizer(multi_instrument=True)

        # Create a track with non-zero program numbers
        notes_multi = np.array(
            [
                {"onset": 0.5, "offset": 1.0, "pitch": 60, "velocity": 80, "program": 40},
                {"onset": 1.0, "offset": 1.5, "pitch": 48, "velocity": 70, "program": 0},
            ],
            dtype=object,
        )
        np.save(data_dir / "multi_audio.npy", _make_fake_audio())
        np.save(data_dir / "multi_notes.npy", notes_multi)

        ds_multi = TranscriptionDataset(
            data_dir=data_dir,
            tokenizer=tok_multi,
            sample_rate=sample_rate,
            segment_samples=segment_samples,
            max_token_len=max_token_len,
        )
        # The dataset now has 4 files; just check it loads without error
        assert len(ds_multi) == 4
        w, t = ds_multi[3]
        assert w.shape == (segment_samples,)
        assert t.shape == (max_token_len,)
        print(f"  Multi-instrument dataset length: {len(ds_multi)}  ✓")

    print("\n" + "=" * 60)
    print("All assertions passed — TranscriptionDataset dry-run OK.")
    print("=" * 60)


if __name__ == "__main__":
    run_test()
