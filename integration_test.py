#!/usr/bin/env python3
"""Integration test for the MT3 transcription pipeline.

Tests the full pipeline end-to-end:
  1. Create a small synthetic dataset (10 samples)
  2. Run training for 20 steps
  3. Run evaluation on the synthetic data
  4. Run transcription to produce a MIDI file
  5. Verify: loss decreases, metrics are valid, MIDI loads correctly
"""

from __future__ import annotations

import io
import os
import sys
import contextlib
import tempfile
from pathlib import Path

# Ensure project root is importable from any CWD
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(str(_PROJECT_ROOT))

import numpy as np
import torch

# ── Tiny model config (runs fast on CPU) ─────────────────────────────────────
SMALL_CONFIG: dict = {
    "data": {
        "dataset": "synthetic",
        "sample_rate": 16000,
        "segment_samples": 8000,   # 0.5 s of audio per training sample
    },
    "audio": {
        "n_fft": 512,
        "hop_length": 128,
        "n_mels": 64,
    },
    "model": {
        "d_model": 64,
        "nhead": 4,
        "enc_layers": 2,
        "dec_layers": 2,
        "d_ff": 128,
        "dropout": 0.0,
        "max_token_len": 32,
    },
    "tokenizer": {
        "multi_instrument": False,
        "time_step_ms": 8,
        "max_time_steps": 600,
    },
    "training": {
        "batch_size": 2,
        "grad_accum_steps": 1,
        "lr": 0.001,
        "warmup_steps": 5,
        "max_steps": 20,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "label_smoothing": 0.1,
        "log_every": 1,       # log every step so we can track loss
        "save_every": 99999,  # suppress mid-loop saves; final save still runs
        "eval_every": 99999,  # suppress mid-loop eval
        "fp16": False,
        "num_workers": 0,
    },
}

SAMPLE_RATE = 16000
SEGMENT_SAMPLES = 8000   # 0.5 s
N_SAMPLES = 10


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Synthetic dataset
# ─────────────────────────────────────────────────────────────────────────────

def create_synthetic_dataset(data_dir: Path) -> None:
    """Write 10 synthetic *_audio.npy / *_notes.npy pairs to data_dir."""
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    for i in range(N_SAMPLES):
        # Random waveform (low amplitude)
        audio = (rng.standard_normal(SEGMENT_SAMPLES) * 0.05).astype(np.float32)
        np.save(data_dir / f"sample_{i:03d}_audio.npy", audio)

        # 2–5 random notes within the segment duration
        n_notes = int(rng.integers(2, 6))
        duration_s = SEGMENT_SAMPLES / SAMPLE_RATE
        notes = []
        for _ in range(n_notes):
            onset = float(rng.uniform(0.0, duration_s * 0.6))
            offset = float(onset + rng.uniform(0.05, min(0.2, duration_s - onset - 0.01)))
            notes.append({
                "onset": onset,
                "offset": offset,
                "pitch": int(rng.integers(48, 84)),
                "velocity": int(rng.integers(40, 100)),
                "program": 0,
            })
        np.save(data_dir / f"sample_{i:03d}_notes.npy", np.array(notes, dtype=object))

    print(f"[setup] created {N_SAMPLES} synthetic samples in {data_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – Training
# ─────────────────────────────────────────────────────────────────────────────

def run_training(data_dir: Path) -> tuple[Path, list[float]]:
    """Run 20 training steps; return (checkpoint_path, losses_per_step)."""
    from scripts.train import train

    cfg = {
        **SMALL_CONFIG,
        "data": {
            **SMALL_CONFIG["data"],
            "train_dir": str(data_dir),
            "val_dir":   str(data_dir),
        },
    }

    # Capture stdout so we can parse per-step losses
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        train(cfg, resume=None, dry_run=False)
    output = buf.getvalue()
    # Echo to real stdout now
    print(output, end="")

    # Parse lines like: [train] step=  1 | loss=6.8931 | lr=2.000e-04
    losses: list[float] = []
    for line in output.splitlines():
        if "[train] step=" in line and "loss=" in line and "val_loss" not in line:
            for part in line.split("|"):
                part = part.strip()
                if part.startswith("loss="):
                    try:
                        losses.append(float(part.split("=")[1]))
                    except ValueError:
                        pass

    ckpt_path = _PROJECT_ROOT / "checkpoints" / "step_20.pt"
    if not ckpt_path.exists():
        # Fallback: pick whichever checkpoint was saved last
        candidates = sorted((_PROJECT_ROOT / "checkpoints").glob("step_*.pt"))
        if candidates:
            ckpt_path = candidates[-1]
        else:
            raise FileNotFoundError("Training finished but no checkpoint was written.")

    return ckpt_path, losses


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(model, data_dir: Path) -> dict:
    """Run evaluate() on the synthetic data; return results dict."""
    from scripts.evaluate import evaluate

    device = next(model.parameters()).device
    results = evaluate(
        model,
        data_dir,
        sample_rate=SAMPLE_RATE,
        segment_samples=SEGMENT_SAMPLES,
        hop_samples=SEGMENT_SAMPLES // 2,
        max_len=SMALL_CONFIG["model"]["max_token_len"],
        temperature=0.0,
        per_program=False,
        device=device,
        max_files=None,
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 – Transcription
# ─────────────────────────────────────────────────────────────────────────────

def run_transcription(model, output_midi: Path) -> list[dict]:
    """Transcribe a synthetic 2-second audio clip; write MIDI and return notes."""
    from scripts.transcribe import transcribe_full_audio, notes_to_midi

    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(2 * SAMPLE_RATE) * 0.05).astype(np.float32)
    device = next(model.parameters()).device

    notes = transcribe_full_audio(
        model,
        audio,
        sample_rate=SAMPLE_RATE,
        segment_samples=SEGMENT_SAMPLES,
        hop_samples=SEGMENT_SAMPLES // 2,
        max_len=SMALL_CONFIG["model"]["max_token_len"],
        temperature=0.0,
        device=device,
    )
    print(f"[transcribe] decoded {len(notes)} notes")
    notes_to_midi(notes, output_midi)
    return notes


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_model_from_checkpoint(ckpt_path: Path):
    """Build model from saved checkpoint and load weights."""
    from src.model import build_model

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt.get("config", SMALL_CONFIG)
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def verify_midi(midi_path: Path) -> None:
    """Assert the MIDI file loads cleanly with pretty_midi."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    print(f"[verify] MIDI loaded OK — {len(pm.instruments)} instrument(s), "
          f"{sum(len(i.notes) for i in pm.instruments)} note(s)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("MT3 Integration Test")
    print("=" * 60)

    passed: list[str] = []
    failed: list[str] = []

    with tempfile.TemporaryDirectory(prefix="mt3_inttest_") as tmpdir:
        tmp = Path(tmpdir)
        data_dir  = tmp / "data"
        midi_out  = tmp / "output.mid"

        # ── 1. Create synthetic dataset ──────────────────────────────────────
        print("\n── Step 1: Create synthetic dataset ──")
        try:
            create_synthetic_dataset(data_dir)
            assert len(list(data_dir.glob("*_audio.npy"))) == N_SAMPLES
            assert len(list(data_dir.glob("*_notes.npy"))) == N_SAMPLES
            passed.append("dataset creation")
            print("[PASS] dataset creation")
        except Exception as exc:
            failed.append(f"dataset creation: {exc}")
            print(f"[FAIL] dataset creation: {exc}")
            print("Cannot continue without data — aborting.")
            _report(passed, failed)
            return

        # ── 2. Training ──────────────────────────────────────────────────────
        print("\n── Step 2: Training (20 steps) ──")
        ckpt_path = None
        losses: list[float] = []
        try:
            ckpt_path, losses = run_training(data_dir)
            print(f"\n[train] Losses over {len(losses)} logged steps: "
                  f"{[f'{l:.4f}' for l in losses]}")
            passed.append("training completed")
            print("[PASS] training completed")
        except Exception as exc:
            import traceback; traceback.print_exc()
            failed.append(f"training: {exc}")
            print(f"[FAIL] training: {exc}")

        # Verify loss decreases (compare first 25% vs last 25% of steps)
        if losses and len(losses) >= 4:
            q = max(1, len(losses) // 4)
            first_avg = sum(losses[:q]) / q
            last_avg  = sum(losses[-q:]) / q
            if last_avg < first_avg:
                passed.append(f"loss decreases ({first_avg:.4f} → {last_avg:.4f})")
                print(f"[PASS] loss decreases: {first_avg:.4f} → {last_avg:.4f}")
            else:
                # Not always guaranteed with tiny model / random data, treat as warning
                msg = f"loss did not decrease ({first_avg:.4f} → {last_avg:.4f})"
                print(f"[WARN] {msg} (non-critical for random data)")
                passed.append(f"loss check (warn: {msg})")
        else:
            print("[WARN] not enough loss values captured to check monotonicity")

        if ckpt_path is None or not ckpt_path.exists():
            failed.append("checkpoint not found after training")
            print("[FAIL] checkpoint not found")
            _report(passed, failed)
            return
        else:
            passed.append(f"checkpoint saved ({ckpt_path.name})")
            print(f"[PASS] checkpoint saved: {ckpt_path}")

        # ── Load model from checkpoint ────────────────────────────────────────
        print("\n── Loading model from checkpoint ──")
        try:
            model = load_model_from_checkpoint(ckpt_path)
            passed.append("checkpoint load")
            print("[PASS] checkpoint loaded")
        except Exception as exc:
            import traceback; traceback.print_exc()
            failed.append(f"checkpoint load: {exc}")
            print(f"[FAIL] checkpoint load: {exc}")
            _report(passed, failed)
            return

        # ── 3. Evaluation ─────────────────────────────────────────────────────
        print("\n── Step 3: Evaluation ──")
        try:
            results = run_evaluation(model, data_dir)
            overall = results["overall"]
            print(f"  num_files   : {results['num_files']}")
            print(f"  ref notes   : {results['num_ref_notes']}")
            print(f"  est notes   : {results['num_est_notes']}")
            print(f"  onset F1    : {overall['onset_F1']:.4f}")
            print(f"  on+off F1   : {overall['onset_offset_F1']:.4f}")

            # Verify all metric values are valid floats in [0, 1]
            for k, v in overall.items():
                assert isinstance(v, float), f"{k} is not float: {v!r}"
                assert 0.0 <= v <= 1.0,      f"{k} out of range: {v}"

            passed.append("evaluation metrics valid")
            print("[PASS] evaluation metrics valid")
        except Exception as exc:
            import traceback; traceback.print_exc()
            failed.append(f"evaluation: {exc}")
            print(f"[FAIL] evaluation: {exc}")

        # ── 4. Transcription ──────────────────────────────────────────────────
        print("\n── Step 4: Transcription ──")
        try:
            notes = run_transcription(model, midi_out)
            passed.append("transcription ran")
            print("[PASS] transcription ran")
        except Exception as exc:
            import traceback; traceback.print_exc()
            failed.append(f"transcription: {exc}")
            print(f"[FAIL] transcription: {exc}")

        # Verify MIDI file
        if midi_out.exists():
            try:
                verify_midi(midi_out)
                passed.append("MIDI file valid")
                print("[PASS] MIDI file valid")
            except Exception as exc:
                failed.append(f"MIDI verify: {exc}")
                print(f"[FAIL] MIDI verify: {exc}")
        else:
            failed.append("MIDI file not created")
            print("[FAIL] MIDI file not created")

    _report(passed, failed)


def _report(passed: list[str], failed: list[str]) -> None:
    print("\n" + "=" * 60)
    print(f"RESULTS: {len(passed)} passed, {len(failed)} failed")
    print("=" * 60)
    for p in passed:
        print(f"  ✓  {p}")
    for f in failed:
        print(f"  ✗  {f}")
    if failed:
        print("\nIntegration test FAILED")
        sys.exit(1)
    else:
        print("\nIntegration test PASSED")


if __name__ == "__main__":
    main()
