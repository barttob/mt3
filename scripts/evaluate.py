"""Evaluation script: run sliding-window transcription on a dataset split
and report onset / onset+offset F1 scores (+ per-program for multi-instrument).

Usage
-----
    python scripts/evaluate.py \\
        --checkpoint checkpoints/best.pt \\
        --config configs/maestro_piano.yaml \\
        --split validation

    # Multi-instrument with per-program breakdown
    python scripts/evaluate.py \\
        --checkpoint checkpoints/slakh_step_200000.pt \\
        --config configs/slakh_multi.yaml \\
        --split test \\
        --per-program

    # Quick dry-run (synthetic data, no real audio needed)
    python scripts/evaluate.py \\
        --checkpoint checkpoints/best.pt \\
        --config configs/maestro_piano.yaml \\
        --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_notes_from_npy(notes_path: Path) -> list[dict]:
    """Load a ``*_notes.npy`` file produced by a preprocessing script.

    Args:
        notes_path: Path to a numpy structured array or object array of note
            dicts.

    Returns:
        List of note dicts with keys ``onset``, ``offset``, ``pitch``,
        ``velocity``, ``program``.
    """
    raw = np.load(notes_path, allow_pickle=True)
    if raw.dtype == object:
        # Object array of dicts (from preprocess_*.py)
        return list(raw)
    # Structured array
    notes = []
    for row in raw:
        notes.append({
            "onset": float(row["onset"]),
            "offset": float(row["offset"]),
            "pitch": int(row["pitch"]),
            "velocity": int(row["velocity"]),
            "program": int(row["program"]),
        })
    return notes


def _collect_samples(data_dir: Path) -> list[tuple[Path, Path]]:
    """Find all (audio_npy, notes_npy) pairs in ``data_dir``.

    Args:
        data_dir: Directory containing ``*_audio.npy`` / ``*_notes.npy``
            file pairs.

    Returns:
        Sorted list of (audio_path, notes_path) tuples.
    """
    pairs = []
    for audio_file in sorted(data_dir.glob("*_audio.npy")):
        stem = audio_file.name.replace("_audio.npy", "")
        notes_file = data_dir / f"{stem}_notes.npy"
        if notes_file.exists():
            pairs.append((audio_file, notes_file))
    return pairs


def _print_metrics(
    label: str,
    metrics: dict[str, float],
    indent: int = 0,
) -> None:
    """Pretty-print a metric dict.

    Args:
        label: A header label for this block of metrics.
        metrics: Dict with keys like ``onset_F1``, ``onset_offset_F1``, etc.
        indent: Number of leading spaces.
    """
    pad = " " * indent
    print(f"{pad}{label}")
    for key in ("onset_P", "onset_R", "onset_F1",
                "onset_offset_P", "onset_offset_R", "onset_offset_F1"):
        if key in metrics:
            print(f"{pad}  {key:<22s}: {metrics[key] * 100:.2f} %")


def _resolve_window_sizes(
    config: dict,
    segment_seconds: float | None,
    hop_seconds: float | None,
) -> tuple[int, int]:
    """Resolve evaluation window sizes from config defaults and CLI overrides."""
    data_cfg = config.get("data", {})
    audio_cfg = config.get("audio", {})
    sample_rate = int(data_cfg.get("sample_rate", 16000))

    if segment_seconds is not None:
        segment_samples = int(segment_seconds * sample_rate)
    else:
        n_frames = data_cfg.get("n_frames")
        hop_length = audio_cfg.get("hop_length")
        if n_frames is not None and hop_length is not None:
            segment_samples = int(n_frames) * int(hop_length)
        else:
            segment_samples = int(data_cfg.get("segment_samples", 256_000))

    if hop_seconds is not None:
        hop_samples = int(hop_seconds * sample_rate)
    else:
        hop_samples = max(1, segment_samples // 2)

    return segment_samples, hop_samples


def _mean_metric_dicts(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    """Average a list of metric dictionaries key-wise."""
    if not metrics_list:
        return {}

    keys = metrics_list[0].keys()
    return {
        key: float(sum(metrics[key] for metrics in metrics_list) / len(metrics_list))
        for key in keys
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    model,
    data_dir: Path,
    sample_rate: int = 16000,
    segment_samples: int = 256_000,
    hop_samples: int = 128_000,
    max_len: int = 1024,
    temperature: float = 0.0,
    per_program: bool = False,
    device: torch.device | None = None,
    max_files: int | None = None,
) -> dict:
    """Run evaluation on all files in ``data_dir``.

    For each audio file the corresponding ground-truth notes are loaded,
    sliding-window inference is run, and the decoded notes are compared with
    the reference using :func:`~src.metrics.evaluate_transcription`.

    Args:
        model: Loaded :class:`~src.model.MT3Model`.
        data_dir: Directory with ``*_audio.npy`` / ``*_notes.npy`` pairs.
        sample_rate: Audio sample rate (Hz).
        segment_samples: Segment length for sliding-window inference.
        hop_samples: Hop size between segments.
        max_len: Max decoding length per segment.
        temperature: Decoding temperature.
        per_program: Whether to accumulate per-program metrics.
        device: Inference device.
        max_files: If given, evaluate only this many files (useful for quick
            sanity checks).

    Returns:
        Dict with keys ``overall`` (aggregated metrics), optionally
        ``per_program`` (macro-averaged per-program metrics), and
        ``num_files`` / ``num_ref_notes`` / ``num_est_notes``.
    """
    from scripts.transcribe import transcribe_full_audio
    from src.metrics import (
        evaluate_transcription,
        instrument_detection_f1,
        macro_average_metrics,
        per_program_metrics,
    )

    if device is None:
        device = next(model.parameters()).device

    pairs = _collect_samples(data_dir)
    if not pairs:
        raise FileNotFoundError(f"No audio/notes pairs found in {data_dir}")
    if max_files is not None:
        pairs = pairs[:max_files]

    overall_metrics_per_file: list[dict[str, float]] = []
    instrument_metrics_per_file: list[dict[str, float]] = []
    per_program_accumulator: dict[int, list[dict[str, float]]] = {}
    per_program_counts: dict[int, dict[str, int]] = {}
    num_ref_notes = 0
    num_est_notes = 0

    for idx, (audio_path, notes_path) in enumerate(pairs):
        audio = np.load(audio_path)
        ref_notes = _load_notes_from_npy(notes_path)

        est_notes = transcribe_full_audio(
            model,
            audio,
            sample_rate=sample_rate,
            segment_samples=segment_samples,
            hop_samples=hop_samples,
            max_len=max_len,
            temperature=temperature,
            device=device,
        )

        overall_metrics_per_file.append(evaluate_transcription(ref_notes, est_notes))
        num_ref_notes += len(ref_notes)
        num_est_notes += len(est_notes)

        if per_program:
            file_prog_metrics = per_program_metrics(ref_notes, est_notes)
            for prog, metrics in file_prog_metrics.items():
                per_program_accumulator.setdefault(prog, []).append(metrics)
                counts = per_program_counts.setdefault(prog, {"ref": 0, "est": 0})
                counts["ref"] += sum(1 for note in ref_notes if note["program"] == prog)
                counts["est"] += sum(1 for note in est_notes if note["program"] == prog)
            instrument_metrics_per_file.append(
                instrument_detection_f1(ref_notes, est_notes)
            )

        if (idx + 1) % 10 == 0 or (idx + 1) == len(pairs):
            print(f"  [{idx + 1}/{len(pairs)}] {audio_path.stem}")

    # ------------------------------------------------------------------
    # Overall metrics
    # ------------------------------------------------------------------
    overall = _mean_metric_dicts(overall_metrics_per_file)
    result: dict = {
        "overall": overall,
        "num_files": len(pairs),
        "num_ref_notes": num_ref_notes,
        "num_est_notes": num_est_notes,
    }

    # ------------------------------------------------------------------
    # Per-program metrics
    # ------------------------------------------------------------------
    if per_program:
        pp = {
            prog: _mean_metric_dicts(metrics_list)
            for prog, metrics_list in per_program_accumulator.items()
        }
        macro = macro_average_metrics(pp)
        inst_det = _mean_metric_dicts(instrument_metrics_per_file)
        result["per_program"] = pp
        result["per_program_counts"] = per_program_counts
        result["macro_avg"] = macro
        result["instrument_detection"] = inst_det

    return result


# ---------------------------------------------------------------------------
# Dry-run (no real data)
# ---------------------------------------------------------------------------

def _dry_run_evaluate(model, config: dict, device: torch.device) -> None:
    """Run a tiny synthetic evaluation to verify the pipeline end-to-end.

    Generates a 4-second silent audio clip and checks that the model produces
    some output without errors.  No meaningful metrics are expected.

    Args:
        model: Loaded :class:`~src.model.MT3Model`.
        config: Config dictionary.
        device: Inference device.
    """
    from scripts.transcribe import transcribe_full_audio
    from src.metrics import evaluate_transcription

    sample_rate: int = config.get("data", {}).get("sample_rate", 16000)
    # 4 seconds of silence
    audio = np.zeros(4 * sample_rate, dtype=np.float32)
    ref_notes: list[dict] = []

    print("[evaluate] dry-run: transcribing 4 s of silence …")
    est_notes = transcribe_full_audio(
        model,
        audio,
        sample_rate=sample_rate,
        segment_samples=min(4 * sample_rate, _resolve_window_sizes(config, None, None)[0]),
        hop_samples=min(2 * sample_rate, _resolve_window_sizes(config, None, None)[1]),
        max_len=64,
        temperature=0.0,
        device=device,
    )
    print(f"[evaluate] dry-run decoded {len(est_notes)} notes (expected ~0 for silence)")

    metrics = evaluate_transcription(ref_notes, est_notes)
    print("[evaluate] dry-run metrics (all zero expected for silence):")
    _print_metrics("Overall", metrics, indent=2)
    print("[evaluate] dry-run passed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Entry point for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MT3 model on a dataset split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a model checkpoint .pt file.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML config path (same one used for training).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--per-program",
        action="store_true",
        help="Print per-program F1 breakdown (multi-instrument models).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Decoding temperature (0 = greedy argmax).",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=1024,
        help="Maximum decoded token length per segment.",
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=None,
        help="Segment length in seconds for sliding-window inference.",
    )
    parser.add_argument(
        "--hop-seconds",
        type=float,
        default=None,
        help="Hop between consecutive segments in seconds.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Evaluate only this many files (useful for quick checks).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device string (e.g. 'cuda', 'cpu').  Auto-detected if omitted.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip real data; run a quick synthetic end-to-end check.",
    )
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # sys.path
    # ------------------------------------------------------------------
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate] device: {device}")

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    sample_rate: int = config.get("data", {}).get("sample_rate", 16000)
    segment_samples, hop_samples = _resolve_window_sizes(
        config, args.segment_seconds, args.hop_seconds
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    from src.model import build_model

    print(f"[evaluate] building model …")
    model = build_model(config)

    print(f"[evaluate] loading checkpoint {args.checkpoint} …")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # Dry-run
    # ------------------------------------------------------------------
    if args.dry_run:
        _dry_run_evaluate(model, config, device)
        return

    # ------------------------------------------------------------------
    # Resolve data directory
    # ------------------------------------------------------------------
    data_cfg = config.get("data", {})
    split_key = "val_dir" if args.split == "validation" else f"{args.split}_dir"
    data_dir = Path(data_cfg.get(split_key, f"data/processed/{data_cfg.get('dataset', 'maestro')}/{args.split}"))

    if not data_dir.exists():
        print(f"[evaluate] ERROR: data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[evaluate] evaluating split '{args.split}' from {data_dir} …")

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    results = evaluate(
        model,
        data_dir,
        sample_rate=sample_rate,
        segment_samples=segment_samples,
        hop_samples=hop_samples,
        max_len=args.max_len,
        temperature=args.temperature,
        per_program=args.per_program,
        device=device,
        max_files=args.max_files,
    )

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print(f"Results — split: {args.split}  |  files: {results['num_files']}")
    print(f"  ref notes : {results['num_ref_notes']:,}")
    print(f"  est notes : {results['num_est_notes']:,}")
    print("=" * 60)
    _print_metrics("Overall", results["overall"])

    if "macro_avg" in results:
        print()
        _print_metrics("Macro-avg (over programs)", results["macro_avg"])

    if "instrument_detection" in results:
        det = results["instrument_detection"]
        print()
        print("Instrument detection")
        print(f"  {'precision':<22s}: {det['precision'] * 100:.2f} %")
        print(f"  {'recall':<22s}: {det['recall'] * 100:.2f} %")
        print(f"  {'f1':<22s}: {det['f1'] * 100:.2f} %")

    if args.per_program and "per_program" in results:
        print()
        print("Per-program F1 (onset / onset+offset)")
        print(f"  {'program':<10s}  {'onset F1':>10s}  {'on+off F1':>10s}  {'# ref':>6s}  {'# est':>6s}")
        print("  " + "-" * 46)
        pp = results["per_program"]
        counts = results.get("per_program_counts", {})
        for prog in sorted(pp.keys()):
            m = pp[prog]
            prog_counts = counts.get(prog, {"ref": 0, "est": 0})
            print(
                f"  {prog:<10d}  {m['onset_F1'] * 100:>9.2f}%  "
                f"{m['onset_offset_F1'] * 100:>9.2f}%  "
                f"{prog_counts['ref']:>6d}  {prog_counts['est']:>6d}"
            )

    print("=" * 60)


if __name__ == "__main__":
    main()
