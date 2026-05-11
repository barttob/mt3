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

    python scripts/benchmark.py \
        --config checkpoints/slakh_full-done-100k/colab_slakh_multi.yaml \
        --checkpoint checkpoints/slakh_full-done-100k/best.pt

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
) -> tuple[float, float, int]:
    """Run one inference pass on GPU, return (inference_time_s, peak_vram_mb, n_tokens).

    Uses torch.cuda.Event for accurate GPU-side timing.

    Args:
        model: MT3Model in eval mode.
        waveform: Input tensor on CUDA device.
        max_len: Maximum token sequence length.
        sample_rate: Audio sample rate used to derive segment config.

    Returns:
        Tuple of (inference_time_seconds, peak_vram_mb, generated_token_count).
    """
    _reset_vram_peak()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        start_evt.record()
        out = model.transcribe(waveform, max_len=max_len, temperature=0.0)
        end_evt.record()

    torch.cuda.synchronize()
    elapsed_ms = start_evt.elapsed_time(end_evt)
    n_tokens = int(out.shape[1]) - 1  # exclude SOS seed token
    return elapsed_ms / 1000.0, _peak_vram_mb(), n_tokens


# ---------------------------------------------------------------------------
# CPU benchmark helpers
# ---------------------------------------------------------------------------

def _run_inference_cpu(
    model: torch.nn.Module,
    waveform: torch.Tensor,
    max_len: int,
) -> tuple[float, int]:
    """Run one inference pass on CPU, return (elapsed_s, n_tokens).

    Args:
        model: MT3Model in eval mode on CPU.
        waveform: Input waveform on CPU.
        max_len: Maximum token sequence length.

    Returns:
        Tuple of (elapsed_wall_clock_seconds, generated_token_count).
    """
    with torch.no_grad():
        t0 = time.perf_counter()
        out = model.transcribe(waveform, max_len=max_len, temperature=0.0)
        elapsed = time.perf_counter() - t0
    n_tokens = int(out.shape[1]) - 1
    return elapsed, n_tokens


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
            _, v, _ = _run_inference_gpu(model, waveform, max_len, sample_rate)
            vram_samples.append(v)
        result["vram_mb"] = float(np.median(vram_samples))

        # Timing + token count: mean of gpu_repeats
        times = []
        token_counts = []
        for _ in range(gpu_repeats):
            t, _, n = _run_inference_gpu(model, waveform, max_len, sample_rate)
            times.append(t)
            token_counts.append(n)
        result["infer_time_s"] = float(np.mean(times))
        result["infer_time_std_s"] = float(np.std(times))
        result["rtf_gpu"] = result["infer_time_s"] / duration_s
        result["tokens_mean"] = float(np.mean(token_counts))
        result["tokens_per_s_gpu"] = result["tokens_mean"] / result["infer_time_s"] if result["infer_time_s"] > 0 else 0.0
    else:
        result["vram_mb"] = None
        result["infer_time_s"] = None
        result["rtf_gpu"] = None
        result["tokens_mean"] = None
        result["tokens_per_s_gpu"] = None

    # ------------------------------------------------------------------
    # CPU measurement (optional; model must be on CPU)
    # ------------------------------------------------------------------
    if cpu_repeats > 0:
        cpu_waveform = waveform.cpu()
        cpu_model = model.cpu()
        cpu_times = []
        cpu_token_counts = []
        for _ in range(cpu_repeats):
            t, n = _run_inference_cpu(cpu_model, cpu_waveform, max_len)
            cpu_times.append(t)
            cpu_token_counts.append(n)
        # Move model back to original device if we moved it
        if device.type != "cpu":
            model.to(device)
        mean_cpu_time = float(np.mean(cpu_times))
        result["rtf_cpu"] = mean_cpu_time / duration_s
        result["rtf_cpu_std"] = float(np.std(cpu_times)) / duration_s
        result["tokens_per_s_cpu"] = float(np.mean(cpu_token_counts)) / mean_cpu_time if mean_cpu_time > 0 else 0.0
    else:
        result["rtf_cpu"] = None
        result["tokens_per_s_cpu"] = None

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
        print(f"  VRAM [MB]     : {row['vram_mb']:.1f}")
    if row["rtf_gpu"] is not None:
        print(f"  RTF_GPU       : {row['rtf_gpu']:.4f}  (infer={row['infer_time_s']:.3f} s)")
    if row.get("tokens_per_s_gpu") is not None:
        print(f"  Tokens/s GPU  : {row['tokens_per_s_gpu']:.1f}  (mean {row['tokens_mean']:.0f} tokens)")
    if row["rtf_cpu"] is not None:
        print(f"  RTF_CPU       : {row['rtf_cpu']:.4f}")
    if row.get("tokens_per_s_cpu") is not None:
        print(f"  Tokens/s CPU  : {row['tokens_per_s_cpu']:.1f}")


def _print_scalability(rows: list[dict]) -> None:
    print("\n=== Tab. 6.3 — Scalability ===")
    header = f"  {'Audio':>6s}  {'Infer [s]':>10s}  {'VRAM [MB]':>10s}  {'RTF_GPU':>10s}  {'RTF_CPU':>10s}  {'Tok/s GPU':>10s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        infer = f"{r['infer_time_s']:.3f}"      if r["infer_time_s"]      is not None else "         —"
        vram  = f"{r['vram_mb']:.0f}"            if r["vram_mb"]           is not None else "         —"
        rtfg  = f"{r['rtf_gpu']:.4f}"           if r["rtf_gpu"]           is not None else "         —"
        rtfc  = f"{r['rtf_cpu']:.4f}"           if r["rtf_cpu"]           is not None else "         —"
        toks  = f"{r['tokens_per_s_gpu']:.1f}"  if r.get("tokens_per_s_gpu") is not None else "         —"
        print(f"  {r['duration_s']:>5.0f}s  {infer:>10s}  {vram:>10s}  {rtfg:>10s}  {rtfc:>10s}  {toks:>10s}")


# ---------------------------------------------------------------------------
# Per-layer timing (Sec. 8 — dominant layer analysis)
# ---------------------------------------------------------------------------

def profile_layer_timing(
    model: torch.nn.Module,
    waveform: torch.Tensor,
    max_len: int,
    device: torch.device,
    warmup: int = 2,
    repeats: int = 5,
) -> dict[str, float]:
    """Time each major submodule (frontend, encoder, decoder) via forward hooks.

    The decoder accumulates time across all autoregressive steps.  Uses
    synchronize-bracketed wall-clock timing so results are accurate on both
    CPU and CUDA.

    Args:
        model: MT3Model in eval mode.
        waveform: Input waveform on the target device.
        max_len: Max decoding length.
        device: Inference device.
        warmup: Warmup passes before measurement.
        repeats: Number of timed passes to average.

    Returns:
        Dict mapping submodule name -> mean elapsed seconds.
    """
    submodules: dict[str, torch.nn.Module] = {
        name: getattr(model, name)
        for name in ("frontend", "encoder", "decoder")
        if hasattr(model, name)
    }

    def _sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(warmup):
        with torch.no_grad():
            model.transcribe(waveform, max_len=max_len, temperature=0.0)
        _sync()

    all_runs: dict[str, list[float]] = {n: [] for n in submodules}

    for _ in range(repeats):
        run_times: dict[str, float] = {n: 0.0 for n in submodules}
        handles = []

        for mod_name, mod in submodules.items():
            def _make_hooks(name: str):
                def pre_hook(module: torch.nn.Module, inp: tuple) -> None:
                    _sync()
                    module._layer_t0 = time.perf_counter()  # type: ignore[attr-defined]

                def post_hook(module: torch.nn.Module, inp: tuple, out: object) -> None:
                    _sync()
                    run_times[name] += time.perf_counter() - module._layer_t0  # type: ignore[attr-defined]

                return pre_hook, post_hook

            pre, post = _make_hooks(mod_name)
            handles.append(mod.register_forward_pre_hook(pre))
            handles.append(mod.register_forward_hook(post))

        with torch.no_grad():
            model.transcribe(waveform, max_len=max_len, temperature=0.0)
        _sync()

        for h in handles:
            h.remove()
        for name, t in run_times.items():
            all_runs[name].append(t)

    return {name: float(np.mean(ts)) for name, ts in all_runs.items()}


def _print_quality_metrics(results: dict) -> None:
    overall = results["overall"]
    print("\n=== Tab. 6.1 — Quality metrics ===")
    print(f"  Files : {results['num_files']}  |  "
          f"ref notes : {results['num_ref_notes']:,}  |  "
          f"est notes : {results['num_est_notes']:,}")
    print()
    print(f"  {'Metric':<20s}  {'Precision':>10s}  {'Recall':>10s}  {'F1':>10s}")
    print("  " + "-" * 56)
    rows = [
        ("Onset F1",        "onset_P",        "onset_R",        "onset_F1"),
        ("Note F1 (+off.)", "onset_offset_P", "onset_offset_R", "onset_offset_F1"),
        ("Frame F1",        "frame_P",         "frame_R",         "frame_F1"),
    ]
    for label, p_key, r_key, f1_key in rows:
        p   = overall.get(p_key)
        r   = overall.get(r_key)
        f1  = overall.get(f1_key)
        if f1 is not None:
            ps  = f"{p * 100:.2f} %" if p  is not None else "—"
            rs  = f"{r * 100:.2f} %" if r  is not None else "—"
            f1s = f"{f1 * 100:.2f} %"
            print(f"  {label:<20s}  {ps:>10s}  {rs:>10s}  {f1s:>10s}")


def _print_layer_timing(timings: dict[str, float]) -> None:
    total = sum(timings.values())
    print("\n=== Sec. 8 — Per-module timing (dominant layer) ===")
    print(f"  {'Module':<12s}  {'Time [s]':>10s}  {'%':>6s}")
    print("  " + "-" * 32)
    for name, t in sorted(timings.items(), key=lambda x: -x[1]):
        pct = 100.0 * t / total if total > 0 else 0.0
        print(f"  {name:<12s}  {t:>10.4f}  {pct:>5.1f}%")
    print(f"  {'total':<12s}  {total:>10.4f}")


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
        "--skip-layer-profile",
        action="store_true",
        help="Skip per-module timing (Sec. 8 dominant-layer analysis).",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=1024,
        help="Max decoder token length passed to model.transcribe().",
    )
    parser.add_argument(
        "--eval-data-dir",
        type=Path,
        default=None,
        help="Directory with *_audio.npy / *_notes.npy pairs for Tab. 6.1 quality "
             "metrics (onset F1, note F1, frame F1). Falls back to val_dir from "
             "config when omitted. Skip entirely if neither is available.",
    )
    parser.add_argument(
        "--max-eval-files",
        type=int,
        default=None,
        help="Limit quality evaluation to this many files (default: all).",
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
    # Tab. 6.1 — Quality metrics (requires preprocessed evaluation data)
    # ------------------------------------------------------------------
    from scripts.evaluate import evaluate as _evaluate_quality
    from scripts.evaluate import _resolve_window_sizes

    quality_results: dict = {}

    # Resolve eval data dir: explicit flag > config val_dir
    eval_data_dir: Path | None = args.eval_data_dir
    if eval_data_dir is None:
        data_cfg = config.get("data", {})
        val_dir_str = data_cfg.get("val_dir")
        if val_dir_str:
            candidate = Path(val_dir_str)
            if candidate.exists():
                eval_data_dir = candidate

    if eval_data_dir is not None and eval_data_dir.exists():
        segment_samples, hop_samples = _resolve_window_sizes(config, None, None)
        n_files_str = str(args.max_eval_files) if args.max_eval_files else "all"
        print(f"\n[benchmark] quality evaluation on {eval_data_dir} ({n_files_str} files) …")
        quality_results = _evaluate_quality(
            model,
            eval_data_dir,
            sample_rate=sample_rate,
            segment_samples=segment_samples,
            hop_samples=hop_samples,
            max_len=args.max_len,
            device=device,
            max_files=args.max_eval_files,
            beam_size=3,
            confidence_threshold=0.4,  # same default as in evaluate.py
        )
        _print_quality_metrics(quality_results)
    else:
        print(
            "\n[benchmark] NOTE: quality metrics skipped — no eval data found. "
            "Pass --eval-data-dir <path> or set val_dir in config."
        )

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
    # Sec. 8 — per-module timing (dominant layer)
    # ------------------------------------------------------------------
    layer_timings: dict[str, float] = {}
    if not args.skip_layer_profile:
        ref_waveform = _make_waveform(args.ref_duration, sample_rate, device)
        print(f"\n[benchmark] per-module timing on {args.ref_duration:.0f} s clip …")
        layer_timings = profile_layer_timing(
            model,
            ref_waveform,
            max_len=args.max_len,
            device=device,
        )
        _print_layer_timing(layer_timings)

    if args.cpu_repeats == 0:
        print(
            "\n[benchmark] NOTE: RTF_CPU not measured (--cpu-repeats=0). "
            "Tab. 6.2 and Tab. 6.3 require it — re-run with --cpu-repeats 10."
        )

    # ------------------------------------------------------------------
    # JSON output
    # ------------------------------------------------------------------
    results = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "device": str(device),
        "environment": env,
        "parameters": params,
        "quality": quality_results,
        "ref_performance": ref_row,
        "scalability": scalability_rows,
        "layer_timing": layer_timings,
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\n[benchmark] results saved → {args.output}")

    print("\n[benchmark] done.")


if __name__ == "__main__":
    main()
