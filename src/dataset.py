"""PyTorch Dataset and DataLoader utilities for MT3-style transcription.

Loads preprocessed audio (.npy) and note (.npy) files produced by the
preprocess_maestro.py / preprocess_slakh.py scripts.  Each ``__getitem__``
call draws a random audio segment, tokenises the overlapping note events, and
returns a (waveform, token_ids) pair ready for the training loop.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.augmentation import WaveformAugmenter
from src.tokenizer import MidiTokenizer, NoteEvent, ActiveNote


class TranscriptionDataset(Dataset):
    """Dataset that streams random segments from preprocessed audio/note files.

    Each sample on disk is a pair of ``.npy`` files:

    * ``<stem>_audio.npy`` – 1-D float32 waveform resampled to
      ``sample_rate`` Hz.
    * ``<stem>_notes.npy`` – structured array (or list of dicts) with fields
      ``onset``, ``offset``, ``pitch``, ``velocity``, ``program``.

    At each ``__getitem__`` a contiguous window of ``segment_samples`` samples
    is drawn uniformly at random (or from a fixed deterministic offset when
    ``random_crop=False``), the notes overlapping that window are extracted,
    and the whole sequence is tokenised with ``tokenizer``.

    Args:
        data_dir: Directory containing the ``*_audio.npy`` / ``*_notes.npy``
            file pairs for one split (train / validation / test).
        tokenizer: :class:`~src.tokenizer.MidiTokenizer` instance that
            controls vocabulary and token format.
        sample_rate: Audio sample rate in Hz (must match the ``.npy`` files).
        segments_per_file: Number of random segments to draw from each file
            per epoch.  Higher values improve data utilisation for long
            audio files (default 10).
        augmenter: Optional :class:`~src.augmentation.WaveformAugmenter`
            applied to each waveform segment during training.
        segment_samples: Number of audio samples per training segment.
        max_token_len: Maximum token sequence length (including ``<sos>`` /
            ``<eos>``).  Longer sequences are truncated; shorter ones are
            right-padded with ``<pad>``.
        random_crop: If ``True`` (default) the segment start is sampled
            uniformly at random, giving varied crops across epochs.  Set to
            ``False`` for validation / evaluation to use a deterministic
            centre crop so that val-loss numbers are comparable across runs.
    """

    def __init__(
        self,
        data_dir: str | Path,
        tokenizer: MidiTokenizer,
        sample_rate: int = 16_000,
        segment_samples: int = 256_000,
        max_token_len: int = 1024,
        segments_per_file: int = 10,
        augmenter: WaveformAugmenter | None = None,
        random_crop: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.segment_samples = segment_samples
        self.max_token_len = max_token_len
        self.segments_per_file = segments_per_file
        self.augmenter = augmenter
        self.random_crop = random_crop

        self.samples: list[tuple[Path, Path]] = []
        for audio_path in sorted(self.data_dir.glob("*_audio.npy")):
            stem = audio_path.name.replace("_audio.npy", "")
            notes_path = self.data_dir / f"{stem}_notes.npy"
            if notes_path.exists():
                self.samples.append((audio_path, notes_path))

        if not self.samples:
            raise FileNotFoundError(
                f"No audio/notes pairs found in '{data_dir}'. "
                "Expected files matching *_audio.npy / *_notes.npy."
            )

    def __len__(self) -> int:
        """Total segments per epoch (files × segments_per_file)."""
        return len(self.samples) * self.segments_per_file

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a (waveform, token_ids) pair for the given file index.

        Args:
            idx: Index into ``self.samples``.

        Returns:
            Tuple of:
            * ``waveform`` – float32 tensor of shape ``(segment_samples,)``.
            * ``token_ids`` – int64 tensor of shape ``(max_token_len,)``.
        """
        file_idx = idx % len(self.samples)
        audio_path, notes_path = self.samples[file_idx]

        audio: np.ndarray = np.load(audio_path)
        notes_raw = np.load(notes_path, allow_pickle=True)

        # ---- Segment window --------------------------------------------------
        max_start = max(0, len(audio) - self.segment_samples)
        if self.random_crop:
            start_sample = random.randint(0, max_start)
        else:
            start_sample = max_start // 2
        end_sample = start_sample + self.segment_samples
        segment = audio[start_sample:end_sample]

        # Zero-pad if the file is shorter than one full segment
        if len(segment) < self.segment_samples:
            segment = np.pad(segment, (0, self.segment_samples - len(segment)))

        start_s = start_sample / self.sample_rate
        end_s = end_sample / self.sample_rate

        # ---- Extract overlapping notes ------------------------------------
        # notes_raw may be an object array of dicts (allow_pickle=True)
        note_tuples: list[NoteEvent] = []
        prev_active: list[ActiveNote] = []

        num_programs = self.tokenizer.num_programs
        for note in notes_raw:
            onset = float(note["onset"])
            offset = float(note["offset"])
            pitch = int(note["pitch"])
            velocity = int(note["velocity"])
            program = int(note["program"])

            if program >= num_programs:
                continue

            # Note overlaps with segment if it sounds during [start_s, end_s)
            if onset < end_s and offset > start_s:
                note_tuples.append((onset, offset, pitch, velocity, program))

            # Previously active: started before segment, still sounding at start
            if onset < start_s and offset > start_s:
                prev_active.append((pitch, velocity, program))

        # ---- Tokenise -------------------------------------------------------
        tokens: list[int] = self.tokenizer.notes_to_tokens(
            note_tuples, start_s, end_s, prev_active
        )

        # ---- Truncate / pad to max_token_len --------------------------------
        pad_id = self.tokenizer.special["<pad>"]
        eos_id = self.tokenizer.special["<eos>"]

        if len(tokens) > self.max_token_len:
            tokens = tokens[: self.max_token_len - 1] + [eos_id]
        else:
            tokens = tokens + [pad_id] * (self.max_token_len - len(tokens))

        waveform = torch.tensor(segment, dtype=torch.float32)
        if self.augmenter is not None:
            waveform = self.augmenter(waveform)
        token_ids = torch.tensor(tokens, dtype=torch.long)

        return waveform, token_ids


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack a list of (waveform, token_ids) pairs into batch tensors.

    Args:
        batch: List of ``(waveform, token_ids)`` tuples as returned by
            :meth:`TranscriptionDataset.__getitem__`.

    Returns:
        Tuple of:
        * ``waveforms`` – float32 tensor of shape ``(B, segment_samples)``.
        * ``tokens`` – int64 tensor of shape ``(B, max_token_len)``.
    """
    waveforms = torch.stack([item[0] for item in batch])
    tokens = torch.stack([item[1] for item in batch])
    return waveforms, tokens
