"""Preprocess MAESTRO v3.0.0 for MT3 training.

Reads the CSV metadata, resamples each WAV file to 16 kHz mono, extracts
note events from the paired MIDI file, and saves one ``*_audio.npy`` and one
``*_notes.npy`` file per track under ``<output>/<split>/``.

Usage::

    python scripts/preprocess_maestro.py \\
        --input  data/maestro/maestro-v3.0.0 \\
        --output data/processed/maestro

The ``--workers`` flag controls the number of parallel librosa load/resample
processes (default: 4).  Pass ``--dry-run`` to process only the first 5 files
of each split without writing to disk.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pretty_midi


# ---------------------------------------------------------------------------
# Per-track worker
# ---------------------------------------------------------------------------

def _process_track(args: dict[str, Any]) -> str:
    """Resample audio and extract notes for one MAESTRO track.

    Args:
        args: Dictionary with keys ``audio_path``, ``midi_path``,
            ``out_audio``, ``out_notes``, ``sample_rate``, ``dry_run``.

    Returns:
        Status string (for progress reporting).
    """
    audio_path: Path = args["audio_path"]
    midi_path: Path = args["midi_path"]
    out_audio: Path = args["out_audio"]
    out_notes: Path = args["out_notes"]
    sample_rate: int = args["sample_rate"]
    dry_run: bool = args["dry_run"]

    # Resample audio to 16 kHz mono
    waveform, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True)

    # Extract notes from MIDI
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    notes: list[dict] = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        prog = instrument.program  # 0 for piano in MAESTRO
        for note in instrument.notes:
            notes.append(
                {
                    "onset": note.start,
                    "offset": note.end,
                    "pitch": note.pitch,
                    "velocity": note.velocity,
                    "program": prog,
                }
            )
    notes.sort(key=lambda n: (n["onset"], n["pitch"]))

    if not dry_run:
        np.save(str(out_audio), waveform)
        np.save(str(out_notes), np.array(notes, dtype=object))

    return f"OK  {audio_path.name}  ({len(notes)} notes)"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for MAESTRO preprocessing."""
    parser = argparse.ArgumentParser(
        description="Preprocess MAESTRO v3.0.0 for MT3 training."
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="DIR",
        help="Root of the MAESTRO v3.0.0 dataset (contains maestro-v3.0.0.csv).",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="DIR",
        help="Output directory for processed .npy files.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16_000,
        metavar="HZ",
        help="Target sample rate in Hz (default: 16000).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="Number of parallel worker processes (default: 4).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only the first 5 tracks per split; do not write files.",
    )
    args = parser.parse_args()

    maestro_root = Path(args.input)
    output_root = Path(args.output)
    metadata_path = maestro_root / "maestro-v3.0.0.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")

    # Read CSV and group by split
    rows_by_split: dict[str, list[dict]] = {}
    with open(metadata_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"]
            rows_by_split.setdefault(split, []).append(row)

    all_jobs: list[dict] = []
    for split, rows in rows_by_split.items():
        split_out = output_root / split
        if not args.dry_run:
            split_out.mkdir(parents=True, exist_ok=True)

        if args.dry_run:
            rows = rows[:5]

        for row in rows:
            audio_path = maestro_root / row["audio_filename"]
            midi_path = maestro_root / row["midi_filename"]
            stem = Path(row["audio_filename"]).stem
            out_audio = split_out / f"{stem}_audio.npy"
            out_notes = split_out / f"{stem}_notes.npy"

            # Skip if already processed
            if not args.dry_run and out_audio.exists() and out_notes.exists():
                continue

            all_jobs.append(
                {
                    "audio_path": audio_path,
                    "midi_path": midi_path,
                    "out_audio": out_audio,
                    "out_notes": out_notes,
                    "sample_rate": args.sample_rate,
                    "dry_run": args.dry_run,
                }
            )

    print(f"Processing {len(all_jobs)} tracks with {args.workers} workers …")
    if args.dry_run:
        print("(dry-run mode: no files will be written)")

    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            for i, status in enumerate(
                pool.imap_unordered(_process_track, all_jobs), start=1
            ):
                print(f"  [{i}/{len(all_jobs)}] {status}")
    else:
        for i, job in enumerate(all_jobs, start=1):
            status = _process_track(job)
            print(f"  [{i}/{len(all_jobs)}] {status}")

    print("MAESTRO preprocessing complete.")


if __name__ == "__main__":
    main()
