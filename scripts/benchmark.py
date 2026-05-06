"""Benchmark script for collecting experimental data (experiment_params.md).

Collects:
  - Environment info (Tab. 5.1): Python, PyTorch, CUDA, librosa, torchaudio
  - Parameter count
  - VRAM usage: median of 5 runs on a 20 s clip (torch.cuda.max_memory_allocated)
  - RTF_GPU: mean over --gpu-repeats runs (torch.cuda.Event + cudaSynchronize)
  - RTF_CPU: mean over --cpu-repeats runs (time.perf_counter)
  - Scalability table (Tab. 6.3): inference time / VRAM / RTF at 5, 10, 20, 60, 120 s

Usage
-----
    # Full GPU benchmark (random weights — no checkpoint needed to measure perf)
    python scripts/benchmark.py --config configs/maestro_piano.yaml

    # With a real checkpoint
    python scripts/benchmark.py \\
        --config configs/maestro_piano.yaml \\
        --checkpoint checkpoints/best.pt

    # Save results to JSON
    python scripts/benchmark.py \\
        --config configs/maestro_piano.yaml \\
        --checkpoint checkpoints/best.pt \\
        --output results/benchmark_m0.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Environment info
# ---------------------------------------------------------------------------

def collect_environment() -> dict[str, str]:
    """Collect software/hardware environment metadata.

    Returns:
        Dict with keys matching the rows of Tab. 5.1.
    """
    info: dict[str, str] = {}

    info["python_version"] = platform.python_version()
    info["os"] = f"{platform.system()} {platform.release()} ({platform.version()})"
    info["pytorch_version"] = torch.__version__

    try:
        import torchaudio
        info["torchaudio_version"] = torchaudio.__version__
    except ImportError:
        info["torchaudio_version"] = "not installed"

    try:
        import librosa
        info["librosa_version"] = librosa.__version__
    except ImportError:
        info["librosa_version"] = "not installed"

    if torch.cuda.is_available():
        info["cuda_driver_version"] = torch.version.cuda or "unknown"
        info["gpu_name"] = torch.cuda.get_device_name(0)
        gpu_props = torch.cuda.get_device_properties(0)
        info["gpu_vram_gb"] = f"{gpu_props.total_memory / 1024 ** 3:.1f}"
    else:
        info["cuda_driver_version"] = "N/A (no CUDA)"
        info["gpu_name"] = "N/A"
        info["gpu_vram_gb"] = "N/A"

    try:
        import psutil
        vm = psutil.virtual_memory()
        info["ram_gb"] = f"{vm.total / 1024 ** 3:.1f}"
        cpu_count = psutil.cpu_count(logical=False)
        info["cpu_model"] = platform.processor() or "unknown"
        info["cpu_cores"] = str(cpu_count)
    except ImportError:
        info["ram_gb"] = "psutil not installed"
        info["cpu_model"] = platform.processor() or "unknown"
        info["cpu_cores"] = str(torch.get_num_threads())

    return info


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    """Count total and trainable model parameters.

    Args:
        model: The MT3Model instance.

    Returns:
        Dict with ``total`` and ``trainable`` parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


# ---------------------------------------------------------------------------
# Synthetic audio generator
# ---------------------------------------------------------------------------

def _make_waveform(duration_s: float, sample_rate: int, device: torch.device) -> torch.Tensor:
    """Create a batch-1 random waveform of the given duration.

    Args:
        duration_s: Duration in seconds.
        sample_rate: Sample rate in Hz.
        device: Target device.

    Returns:
        Float32 tensor of shape (1, num_samples).
    """
    num_samples = int(duration_s * sample_rate)
    return torch.randn(1, num_samples, device=device)


# ---------------------------------------------------------------------------
# GPU benchmark helpers
# ---------------------------------------------------------------------------

def _reset_vram_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    return 0.0


def _run_inference_gpu(
    model: torch.nn.Module,
    waveform: torch.Tensor,
    max_len: int,
    sample_rate: int,
) -> tuple[float, float]:
    """Run one inference pass on GPU, return (inference_time_s, peak_vram_mb).

    Uses torch.cuda.Event for accurate GPU-side timing.

    Args:
        model: MT3Model in eval mode.
        waveform: Input tensor on CUDA device.
        max_len: Maximum token sequence length.
        sample_rate: Audio sample rate used to derive segment config.

    Returns:
        Tuple of (inference_time_seconds, peak_vram_mb).
    """
    _reset_vram_peak()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        start_evt.record()
        model.transcribe(waveform, max_len=max_len, temperature=0.0)
        end_evt.record()

    torch.cuda.synchronize()
    elapsed_ms = start_evt.elapsed_time(end_evt)
    return elapsed_ms / 1000.0, _peak_vram_mb()


# ---------------------------------------------------------------------------
# CPU benchmark helpers
# ---------------------------------------------------------------------------

def _run_inference_cpu(
    model: torch.nn.Module,
    waveform: torch.Tensor,
    max_len: int,
) -> float:
    """Run one inference pass on CPU, return elapsed time in seconds.

    Args:
        model: MT3Model in eval mode on CPU.
        waveform: Input waveform on CPU.
        max_len: Maximum token sequence length.

    Returns:
        Elapsed wall-clock time in seconds.
    """
    with torch.no_grad():
        t0 = time.perf_counter()
        model.transcribe(waveform, max_len=max_len, temperature=0.0)
        return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# RTF / VRAM at single duration
# ---------------------------------------------------------------------------

def benchmark_single_duration(
    model: torch.nn.Module,
    duration_s: float,
    sample_rate: int,
    max_len: int,
    device: torch.device,
    gpu_repeats: int = 10,
    vram_repeats: int = 5,
    cpu_repeats: int = 0,
    warmup: int = 2,
) -> dict:
    """Measure inference time, VRAM, and RTF for one audio duration.

    Args:
        model: MT3Model in eval mode.
        duration_s: Audio duration in seconds.
        sample_rate: Audio sample rate in Hz.
        max_len: Max decoding length.
        device: Inference device.
        gpu_repeats: Number of GPU timing runs to average over.
        vram_repeats: Number of VRAM samples (reports median).
        cpu_repeats: Number of CPU runs (0 = skip CPU).
        warmup: Warmup runs before measurement (GPU only).

    Returns:
        Dict with keys: duration_s, infer_time_s (mean), vram_mb (median),
        rtf_gpu, rtf_cpu (or None).
    """
    waveform = _make_waveform(duration_s, sample_rate, device)

    result: dict = {"duration_s": duration_s}

    # ------------------------------------------------------------------
    # GPU measurement
    # ------------------------------------------------------------------
    if device.type == "cuda":
        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                model.transcribe(waveform, max_len=max_len, temperature=0.0)
            torch.cuda.synchronize()

        # VRAM: median of vram_repeats
        vram_samples = []
        for _ in range(vram_repeats):
            _, v = _run_inference_gpu(model, waveform, max_len, sample_rate)
            vram_samples.append(v)
        result["vram_mb"] = float(np.median(vram_samples))

        # Timing: mean of gpu_repeats
        times = []
        for _ in range(gpu_repeats):
            t, _ = _run_inference_gpu(model, waveform, max_len, sample_rate)
            times.append(t)
        result["infer_time_s"] = float(np.mean(times))
        result["infer_time_std_s"] = float(np.std(times))
        result["rtf_gpu"] = result["infer_time_s"] / duration_s
    else:
        result["vram_mb"] = None
        result["infer_time_s"] = None
        result["rtf_gpu"] = None

    # ------------------------------------------------------------------
    # CPU measurement (optional; model must be on CPU)
    # ------------------------------------------------------------------
    if cpu_repeats > 0:
        cpu_waveform = waveform.cpu()
        cpu_model = model.cpu()
        cpu_times = []
        for _ in range(cpu_repeats):
            t = _run_inference_cpu(cpu_model, cpu_waveform, max_len)
            cpu_times.append(t)
        # Move model back to original device if we moved it
        if device.type != "cpu":
            model.to(device)
        result["rtf_cpu"] = float(np.mean(cpu_times)) / duration_s
        result["rtf_cpu_std"] = float(np.std(cpu_times)) / duration_s
    else:
        result["rtf_cpu"] = None

    return result


# ---------------------------------------------------------------------------
# Scalability table
# ---------------------------------------------------------------------------

def benchmark_scalability(
    model: torch.nn.Module,
    sample_rate: int,
    max_len: int,
    device: torch.device,
    durations: list[float] | None = None,
    gpu_repeats: int = 10,
    vram_repeats: int = 5,
    cpu_repeats: int = 0,
) -> list[dict]:
    """Measure scalability across multiple audio durations.

    Args:
        model: MT3Model in eval mode.
        sample_rate: Audio sample rate in Hz.
        max_len: Max decoding length.
        device: Inference device.
        durations: List of durations in seconds (default: [5, 10, 20, 60, 120]).
        gpu_repeats: GPU timing repetitions per duration.
        vram_repeats: VRAM sampling repetitions per duration.
        cpu_repeats: CPU timing repetitions per duration (0 = skip).

    Returns:
        List of result dicts (one per duration) as returned by
        :func:`benchmark_single_duration`.
    """
    if durations is None:
        durations = [5, 10, 20, 60, 120]

    rows = []
    for dur in durations:
        print(f"    duration={dur:>4.0f}s …", end="", flush=True)
        row = benchmark_single_duration(
            model,
            duration_s=dur,
            sample_rate=sample_rate,
            max_len=max_len,
            device=device,
            gpu_repeats=gpu_repeats,
            vram_repeats=vram_repeats,
            cpu_repeats=cpu_repeats,
        )
        rows.append(row)
        if row["rtf_gpu"] is not None:
            print(f"  RTF_GPU={row['rtf_gpu']:.3f}  VRAM={row['vram_mb']:.0f} MB", flush=True)
        elif row["rtf_cpu"] is not None:
            print(f"  RTF_CPU={row['rtf_cpu']:.3f}", flush=True)
        else:
            print("  done", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _print_env(env: dict[str, str]) -> None:
    print("\n=== Tab. 5.1 — Environment ===")
    rows = [
        ("CPU model",          env.get("cpu_model", "?")),
        ("CPU cores (phys.)",  env.get("cpu_cores", "?")),
        ("RAM [GB]",           env.get("ram_gb", "?")),
        ("OS",                 env.get("os", "?")),
        ("GPU",                env.get("gpu_name", "?")),
        ("GPU VRAM [GB]",      env.get("gpu_vram_gb", "?")),
        ("CUDA driver",        env.get("cuda_driver_version", "?")),
        ("PyTorch",            env.get("pytorch_version", "?")),
        ("Python",             env.get("python_version", "?")),
        ("librosa",            env.get("librosa_version", "?")),
        ("torchaudio",         env.get("torchaudio_version", "?")),
    ]
    for label, val in rows:
        print(f"  {label:<28s}: {val}")


def _print_params(params: dict[str, int]) -> None:
    print("\n=== Parameter count ===")
    print(f"  Total      : {params['total']:,}  ({params['total'] / 1e6:.2f} M)")
    print(f"  Trainable  : {params['trainable']:,}  ({params['trainable'] / 1e6:.2f} M)")


def _print_perf(row: dict, label: str = "20 s clip") -> None:
    print(f"\n=== Tab. 6.2 — Performance ({label}) ===")
    if row["vram_mb"] is not None:
        print(f"  VRAM [MB]  : {row['vram_mb']:.1f}")
    if row["rtf_gpu"] is not None:
        print(f"  RTF_GPU    : {row['rtf_gpu']:.4f}  (infer={row['infer_time_s']:.3f} s)")
    if row["rtf_cpu"] is not None:
        print(f"  RTF_CPU    : {row['rtf_cpu']:.4f}")


def _print_scalability(rows: list[dict]) -> None:
    print("\n=== Tab. 6.3 — Scalability ===")
    header = f"  {'Audio':>6s}  {'Infer [s]':>10s}  {'VRAM [MB]':>10s}  {'RTF_GPU':>10s}  {'RTF_CPU':>10s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        infer = f"{r['infer_time_s']:.3f}" if r["infer_time_s"] is not None else "    —"
        vram  = f"{r['vram_mb']:.0f}"      if r["vram_mb"]      is not None else "    —"
        rtfg  = f"{r['rtf_gpu']:.4f}"     if r["rtf_gpu"]      is not None else "    —"
        rtfc  = f"{r['rtf_cpu']:.4f}"     if r["rtf_cpu"]      is not None else "    —"
        print(f"  {r['duration_s']:>5.0f}s  {infer:>10s}  {vram:>10s}  {rtfg:>10s}  {rtfc:>10s}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Entry point for the benchmark script."""
    parser = argparse.ArgumentParser(
        description="Collect experimental data for experiment_params.md.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML config (same one used for training).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint .pt file. Random weights are used when omitted "
             "(fine for performance measurements, not for quality metrics).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device string (e.g. 'cuda', 'cpu'). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write full results to this JSON file.",
    )
    parser.add_argument(
        "--gpu-repeats",
        type=int,
        default=10,
        help="Number of GPU timing runs (averaged).",
    )
    parser.add_argument(
        "--vram-repeats",
        type=int,
        default=5,
        help="Number of VRAM measurements (median reported).",
    )
    parser.add_argument(
        "--cpu-repeats",
        type=int,
        default=0,
        help="Number of CPU timing runs. 0 = skip CPU measurement. "
             "Warning: slow for long audio.",
    )
    parser.add_argument(
        "--ref-duration",
        type=float,
        default=20.0,
        help="Reference audio duration in seconds for Tab. 6.2 measurements.",
    )
    parser.add_argument(
        "--scalability-durations",
        type=float,
        nargs="+",
        default=[5, 10, 20, 60, 120],
        help="Audio durations (seconds) for the scalability table (Tab. 6.3).",
    )
    parser.add_argument(
        "--skip-scalability",
        action="store_true",
        help="Skip the scalability sweep (useful for a quick first run).",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=1024,
        help="Max decoder token length passed to model.transcribe().",
    )
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # sys.path
    # ------------------------------------------------------------------
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------
    print("[benchmark] collecting environment info …")
    env = collect_environment()
    _print_env(env)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[benchmark] device: {device}")

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    sample_rate: int = config.get("data", {}).get("sample_rate", 16000)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    from src.model import build_model

    print("[benchmark] building model …")
    model = build_model(config)

    if args.checkpoint is not None:
        print(f"[benchmark] loading checkpoint {args.checkpoint} …")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        state = ckpt.get("model", ckpt)
        if any(k.startswith("_orig_mod.") for k in state):
            state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
        model.load_state_dict(state)
    else:
        print("[benchmark] no checkpoint provided — using random weights "
              "(suitable for performance measurements only)")

    model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    params = count_parameters(model)
    _print_params(params)

    # ------------------------------------------------------------------
    # Tab. 6.2 reference clip
    # ------------------------------------------------------------------
    print(f"\n[benchmark] measuring performance on {args.ref_duration:.0f} s clip …")
    ref_row = benchmark_single_duration(
        model,
        duration_s=args.ref_duration,
        sample_rate=sample_rate,
        max_len=args.max_len,
        device=device,
        gpu_repeats=args.gpu_repeats,
        vram_repeats=args.vram_repeats,
        cpu_repeats=args.cpu_repeats,
    )
    _print_perf(ref_row, label=f"{args.ref_duration:.0f} s clip")

    # ------------------------------------------------------------------
    # Tab. 6.3 scalability
    # ------------------------------------------------------------------
    scalability_rows: list[dict] = []
    if not args.skip_scalability:
        print(f"\n[benchmark] scalability sweep: {args.scalability_durations} …")
        scalability_rows = benchmark_scalability(
            model,
            sample_rate=sample_rate,
            max_len=args.max_len,
            device=device,
            durations=args.scalability_durations,
            gpu_repeats=args.gpu_repeats,
            vram_repeats=args.vram_repeats,
            cpu_repeats=args.cpu_repeats,
        )
        _print_scalability(scalability_rows)

    # ------------------------------------------------------------------
    # JSON output
    # ------------------------------------------------------------------
    results = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "device": str(device),
        "environment": env,
        "parameters": params,
        "ref_performance": ref_row,
        "scalability": scalability_rows,
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\n[benchmark] results saved → {args.output}")

    print("\n[benchmark] done.")


if __name__ == "__main__":
    main()
