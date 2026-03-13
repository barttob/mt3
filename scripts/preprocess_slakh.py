"""Preprocess Slakh2100-FLAC-Redux for MT3 multi-instrument training.

Walks each track directory, reads ``metadata.yaml`` for instrument/program
mapping, resamples ``mix.flac`` to 16 kHz mono, collects per-stem MIDI notes
(with their MIDI program numbers), and saves ``*_audio.npy`` / ``*_notes.npy``
pairs under ``<output>/<split>/``.

Usage::

    python scripts/preprocess_slakh.py \\
        --input  data/slakh2100_flac_redux \\
        --output data/processed/slakh

Pass ``--dry-run`` to process only the first 5 tracks of each split without
writing to disk.  Use ``--workers`` to parallelise (default: 4).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pretty_midi
import soundfile as sf
import yaml


# ---------------------------------------------------------------------------
# Per-track worker
# ---------------------------------------------------------------------------

def _process_track(args: dict[str, Any]) -> str:
    """Resample mixture audio and extract multi-instrument notes for one track.

    Args:
        args: Dictionary with keys ``track_dir``, ``out_audio``,
            ``out_notes``, ``sample_rate``, ``dry_run``.

    Returns:
        Status string (for progress reporting).
    """
    track_dir: Path = args["track_dir"]
    out_audio: Path = args["out_audio"]
    out_notes: Path = args["out_notes"]
    sample_rate: int = args["sample_rate"]
    dry_run: bool = args["dry_run"]

    mix_path = track_dir / "mix.flac"
    meta_path = track_dir / "metadata.yaml"

    if not mix_path.exists():
        return f"SKIP {track_dir.name}  (no mix.flac)"

    # Load and resample mixture
    audio, sr = sf.read(str(mix_path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # stereo → mono
    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    audio = audio.astype(np.float32)

    # Load stem metadata
    if not meta_path.exists():
        return f"SKIP {track_dir.name}  (no metadata.yaml)"
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata: dict = yaml.safe_load(f)

    notes: list[dict] = []
    midi_dir = track_dir / "MIDI"

    for stem_id, stem_info in metadata.get("stems", {}).items():
        if not stem_info.get("audio_rendered", False):
            continue

        midi_path = midi_dir / f"{stem_id}.mid"
        if not midi_path.exists():
            continue

        program: int = stem_info.get("program_num", 0)
        is_drum: bool = stem_info.get("is_drum", False)
        # Drums are mapped to program 128 (outside standard 0-127 range)
        effective_program = 128 if is_drum else program

        midi = pretty_midi.PrettyMIDI(str(midi_path))
        for instrument in midi.instruments:
            for note in instrument.notes:
                notes.append(
                    {
                        "onset": note.start,
                        "offset": note.end,
                        "pitch": note.pitch,
                        "velocity": note.velocity,
                        "program": effective_program,
                    }
                )

    notes.sort(key=lambda n: (n["onset"], n["pitch"]))

    if not dry_run:
        np.save(str(out_audio), audio)
        np.save(str(out_notes), np.array(notes, dtype=object))

    return f"OK  {track_dir.name}  ({len(notes)} notes)"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for Slakh2100 preprocessing."""
    parser = argparse.ArgumentParser(
        description="Preprocess Slakh2100-FLAC-Redux for MT3 training."
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="DIR",
        help="Root of the Slakh2100-FLAC-Redux dataset (contains train/, validation/, test/).",
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

    slakh_root = Path(args.input)
    output_root = Path(args.output)

    all_jobs: list[dict] = []
    for split in ("train", "validation", "test"):
        split_in = slakh_root / split
        if not split_in.exists():
            continue

        split_out = output_root / split
        if not args.dry_run:
            split_out.mkdir(parents=True, exist_ok=True)

        track_dirs = sorted(split_in.iterdir())
        if args.dry_run:
            track_dirs = track_dirs[:5]

        for track_dir in track_dirs:
            if not track_dir.is_dir():
                continue

            out_audio = split_out / f"{track_dir.name}_audio.npy"
            out_notes = split_out / f"{track_dir.name}_notes.npy"

            # Skip already-processed tracks
            if not args.dry_run and out_audio.exists() and out_notes.exists():
                continue

            all_jobs.append(
                {
                    "track_dir": track_dir,
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

    print("Slakh preprocessing complete.")


if __name__ == "__main__":
    main()
