"""Training entry point for the MT3 music transcription model.

Usage::

    # Normal training
    python scripts/train.py --config configs/maestro_piano.yaml

    # Resume from checkpoint
    python scripts/train.py --config configs/maestro_piano.yaml \\
        --resume checkpoints/step_10000.pt

    # Dry-run (2 steps, synthetic data, no disk I/O)
    python scripts/train.py --config configs/maestro_piano.yaml --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow ``python scripts/train.py`` from the project root without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.dataset import TranscriptionDataset, collate_fn
from src.model import MT3Model, build_model
from src.tokenizer import MidiTokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(path: str | Path) -> dict:
    """Load a YAML config file and return as a nested dict."""
    with open(path) as fh:
        return yaml.safe_load(fh)


def _make_lr_lambda(warmup_steps: int):
    """Return a LambdaLR schedule function: linear warm-up then inv-sqrt decay."""
    def lr_lambda(step: int) -> float:
        step = max(step, 1)
        if step < warmup_steps:
            return step / warmup_steps
        return (warmup_steps / step) ** 0.5
    return lr_lambda


def _build_optimizer_and_scheduler(
    model: MT3Model,
    train_cfg: dict,
) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.LambdaLR]:
    """Create AdamW optimizer and LambdaLR scheduler from training config."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        _make_lr_lambda(train_cfg.get("warmup_steps", 4000)),
    )
    return optimizer, scheduler


def _make_dry_run_loader(
    tokenizer: MidiTokenizer,
    data_cfg: dict,
    train_cfg: dict,
    num_batches: int = 4,
) -> DataLoader:
    """Return a DataLoader that yields random tensors for dry-run testing."""
    B = train_cfg.get("batch_size", 2)
    segment_samples = data_cfg.get("segment_samples", 256_000)
    max_token_len = data_cfg.get("max_token_len") or 1024
    vocab_size = tokenizer.vocab_size

    # Pre-generate a small pool of random batches so the DataLoader just indexes them.
    total_samples = B * num_batches
    waveforms = torch.randn(total_samples, segment_samples)
    # Random token sequences: first token is <sos>
    tokens = torch.randint(4, vocab_size, (total_samples, max_token_len), dtype=torch.long)
    tokens[:, 0] = tokenizer.special["<sos>"]
    # Terminate sequences with <eos> at mid-point so padding mask is non-trivial
    mid = max_token_len // 2
    tokens[:, mid] = tokenizer.special["<eos>"]
    tokens[:, mid + 1:] = tokenizer.special["<pad>"]

    dataset = TensorDataset(waveforms, tokens)
    return DataLoader(dataset, batch_size=B, shuffle=False, drop_last=True)


def _resolve_segment_samples(config: dict) -> int:
    """Resolve the waveform window length used for training/inference."""
    data_cfg = config.get("data", {})
    audio_cfg = config.get("audio", {})

    n_frames = data_cfg.get("n_frames")
    hop_length = audio_cfg.get("hop_length")
    if n_frames is not None and hop_length is not None:
        return int(n_frames) * int(hop_length)
    return int(data_cfg.get("segment_samples", 256_000))


def _compute_val_loss(
    model: MT3Model,
    val_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    use_amp: bool,
    max_batches: int = 50,
) -> float:
    """Compute average cross-entropy loss over at most ``max_batches`` val batches."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for waveform, tokens in val_loader:
            if n_batches >= max_batches:
                break
            waveform = waveform.to(device)
            tokens = tokens.to(device)

            tgt_input = tokens[:, :-1]
            tgt_output = tokens[:, 1:]

            S = tgt_input.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(S, device=device)
            tgt_padding_mask = tgt_input == model.tokenizer.special["<pad>"]

            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(waveform, tgt_input, tgt_mask, tgt_padding_mask)
                loss = criterion(
                    logits.reshape(-1, model.tokenizer.vocab_size),
                    tgt_output.reshape(-1),
                )

            total_loss += loss.item()
            n_batches += 1

    model.train()
    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# TensorBoard writer (optional — graceful fallback if not installed)
# ---------------------------------------------------------------------------

def _try_get_tb_writer(log_dir: str):
    """Return a SummaryWriter or None if tensorboard is unavailable."""
    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore
        return SummaryWriter(log_dir=log_dir)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(config: dict, resume: str | Path | None = None, dry_run: bool = False) -> None:
    """Run the full training loop.

    Args:
        config: Fully loaded configuration dictionary (from YAML).
        resume: Optional path to a checkpoint ``.pt`` file to resume from.
        dry_run: If True, skip real data loading and use synthetic tensors;
            runs exactly 2 optimiser steps then exits.
    """
    # ------------------------------------------------------------------
    # Unpack config sections
    # ------------------------------------------------------------------
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    segment_samples = _resolve_segment_samples(config)

    batch_size: int = train_cfg.get("batch_size", 8)
    grad_accum: int = train_cfg.get("grad_accum_steps", 1)
    max_steps: int = 2 if dry_run else train_cfg.get("max_steps", 500_000)
    grad_clip: float = train_cfg.get("grad_clip", 1.0)
    label_smoothing: float = train_cfg.get("label_smoothing", 0.1)
    log_every: int = 1 if dry_run else train_cfg.get("log_every", 100)
    save_every: int = train_cfg.get("save_every", 10_000)
    eval_every: int = train_cfg.get("eval_every", 10_000)
    use_amp: bool = train_cfg.get("fp16", True) and torch.cuda.is_available()
    num_workers: int = 0 if dry_run else train_cfg.get("num_workers", 4)
    max_token_len: int = model_cfg.get("max_token_len", 1024)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}  amp={use_amp}  dry_run={dry_run}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    # Inject max_token_len from model section into config so build_model sees it
    config["model"].setdefault("max_token_len", max_token_len)
    model: MT3Model = build_model(config).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] parameters={n_params:,}  vocab_size={model.tokenizer.vocab_size}")

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------
    if dry_run:
        dry_data_cfg = dict(data_cfg)
        dry_data_cfg["segment_samples"] = segment_samples
        train_loader = _make_dry_run_loader(model.tokenizer, dry_data_cfg, train_cfg)
        val_loader = _make_dry_run_loader(
            model.tokenizer, dry_data_cfg, train_cfg, num_batches=2
        )
        print("[train] dry-run: using synthetic data loaders")
    else:
        sample_rate: int = data_cfg.get("sample_rate", 16_000)

        train_ds = TranscriptionDataset(
            data_dir=data_cfg["train_dir"],
            tokenizer=model.tokenizer,
            sample_rate=sample_rate,
            segment_samples=segment_samples,
            max_token_len=max_token_len,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=device.type == "cuda",
            drop_last=True,
            persistent_workers=num_workers > 0,
        )

        val_dir = data_cfg.get("val_dir")
        if val_dir:
            val_ds = TranscriptionDataset(
                data_dir=val_dir,
                tokenizer=model.tokenizer,
                sample_rate=sample_rate,
                segment_samples=segment_samples,
                max_token_len=max_token_len,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=device.type == "cuda",
                drop_last=False,
                persistent_workers=num_workers > 0,
            )
        else:
            val_loader = None

        print(f"[train] train samples={len(train_ds)}")
        if val_loader:
            print(f"[train] val samples={len(val_ds)}")

    # ------------------------------------------------------------------
    # Optimizer, scheduler, loss, scaler
    # ------------------------------------------------------------------
    optimizer, scheduler = _build_optimizer_and_scheduler(model, train_cfg)

    criterion = nn.CrossEntropyLoss(
        ignore_index=model.tokenizer.special["<pad>"],
        label_smoothing=label_smoothing,
    )

    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)

    # ------------------------------------------------------------------
    # Checkpointing helpers
    # ------------------------------------------------------------------
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    def save_checkpoint(step: int) -> None:
        path = ckpt_dir / f"step_{step}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "config": config,
            },
            path,
        )
        print(f"[train] checkpoint saved → {path}")

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    global_step = 0
    if resume:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        global_step = ckpt.get("step", 0)
        print(f"[train] resumed from {resume}  (step={global_step})")

    # ------------------------------------------------------------------
    # TensorBoard
    # ------------------------------------------------------------------
    tb_writer = None if dry_run else _try_get_tb_writer("runs/train")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    model.train()
    optimizer.zero_grad()
    accum_loss = 0.0
    accum_steps = 0

    print(f"[train] starting at step {global_step}, target {max_steps} steps")

    epoch = 0
    while global_step < max_steps:
        epoch += 1
        for waveform, tokens in train_loader:
            if global_step >= max_steps:
                break

            waveform = waveform.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)

            # Teacher forcing: decoder input is tokens[:-1], target is tokens[1:]
            tgt_input = tokens[:, :-1]
            tgt_output = tokens[:, 1:]

            S = tgt_input.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(S, device=device)
            tgt_padding_mask = tgt_input == model.tokenizer.special["<pad>"]

            # Forward pass under autocast
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(waveform, tgt_input, tgt_mask, tgt_padding_mask)
                loss = criterion(
                    logits.reshape(-1, model.tokenizer.vocab_size),
                    tgt_output.reshape(-1),
                )
                # Scale loss for gradient accumulation
                loss_scaled = loss / grad_accum

            scaler.scale(loss_scaled).backward()

            accum_loss += loss.item()
            accum_steps += 1

            # Parameter update after accumulating grad_accum micro-batches
            if accum_steps % grad_accum == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                avg_micro_loss = accum_loss / grad_accum
                accum_loss = 0.0

                current_lr = scheduler.get_last_lr()[0]

                # Logging
                if global_step % log_every == 0:
                    print(
                        f"[train] step={global_step:>6d} | "
                        f"loss={avg_micro_loss:.4f} | "
                        f"lr={current_lr:.3e}"
                    )
                    if tb_writer is not None:
                        tb_writer.add_scalar("train/loss", avg_micro_loss, global_step)
                        tb_writer.add_scalar("train/lr", current_lr, global_step)

                # Validation
                if val_loader is not None and global_step % eval_every == 0:
                    val_loss = _compute_val_loss(
                        model, val_loader, criterion, device, use_amp
                    )
                    print(f"[train] step={global_step:>6d} | val_loss={val_loss:.4f}")
                    if tb_writer is not None:
                        tb_writer.add_scalar("val/loss", val_loss, global_step)
                    model.train()

                # Checkpoint
                if not dry_run and global_step % save_every == 0:
                    save_checkpoint(global_step)

                if global_step >= max_steps:
                    break

    # Final checkpoint (skip for dry-run)
    if not dry_run and global_step > 0:
        save_checkpoint(global_step)

    if tb_writer is not None:
        tb_writer.close()

    print(f"[train] done. total steps={global_step}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the MT3 music transcription model."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a YAML config file (e.g. configs/maestro_piano.yaml).",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to a checkpoint .pt file to resume training from.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run 2 optimiser steps with synthetic random data to verify the "
            "loop works without requiring real data on disk."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = _load_config(args.config)
    train(cfg, resume=args.resume, dry_run=args.dry_run)
