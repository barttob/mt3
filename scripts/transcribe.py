"""Inference script: transcribe a single audio file to a MIDI file.

Usage
-----
    python scripts/transcribe.py \\
        --audio input.wav \\
        --output output.mid \\
        --checkpoint checkpoints/best.pt \\
        --config configs/maestro_piano.yaml

    # With temperature sampling and custom max token length
    python scripts/transcribe.py \\
        --audio input.wav \\
        --output output.mid \\
        --checkpoint checkpoints/step_100000.pt \\
        --config configs/maestro_piano.yaml \\
        --temperature 0.05 \\
        --max-len 1024

The script uses a sliding-window approach (segment_samples long, hop_samples
hop) to handle audio of arbitrary length.  Tie notes across segment
boundaries are tracked so that notes spanning a segment boundary are
reconstructed correctly.
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

def _load_audio(audio_path: Path, target_sr: int = 16000) -> np.ndarray:
    """Load and resample audio to mono float32 at ``target_sr`` Hz.

    Args:
        audio_path: Path to the input audio file (WAV, FLAC, MP3, …).
        target_sr: Target sample rate in Hz.

    Returns:
        1-D float32 numpy array of audio samples.
    """
    try:
        import librosa
    except ImportError as exc:
        raise ImportError("librosa is required for audio loading.") from exc

    audio, _ = librosa.load(str(audio_path), sr=target_sr, mono=True)
    return audio


def notes_to_midi(notes: list[dict], output_path: Path) -> None:
    """Convert a list of note dicts to a MIDI file via pretty_midi.

    Args:
        notes: List of note dicts with keys ``onset`` (float), ``offset``
            (float), ``pitch`` (int), ``velocity`` (int), ``program`` (int).
        output_path: Destination ``.mid`` file path.
    """
    try:
        import pretty_midi
    except ImportError as exc:
        raise ImportError("pretty_midi is required for MIDI output.") from exc

    midi = pretty_midi.PrettyMIDI()
    instruments: dict[int, pretty_midi.Instrument] = {}

    for note in notes:
        prog = note["program"]
        if prog not in instruments:
            is_drum = prog == 128
            instruments[prog] = pretty_midi.Instrument(
                program=0 if is_drum else (prog % 128),
                is_drum=is_drum,
                name=f"program_{prog}",
            )

        onset = float(note["onset"])
        offset = float(note["offset"])
        # Ensure minimum note duration of 1 ms to avoid invalid MIDI notes
        if offset <= onset:
            offset = onset + 0.001

        midi_note = pretty_midi.Note(
            velocity=max(1, min(127, int(note["velocity"]))),
            pitch=int(note["pitch"]),
            start=onset,
            end=offset,
        )
        instruments[prog].notes.append(midi_note)

    for inst in instruments.values():
        inst.notes.sort(key=lambda n: n.start)
        midi.instruments.append(inst)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(output_path))


def transcribe_full_audio(
    model,
    audio: np.ndarray,
    sample_rate: int = 16000,
    segment_samples: int = 256_000,
    hop_samples: int = 128_000,
    max_len: int = 1024,
    temperature: float = 0.0,
    device: torch.device | None = None,
) -> list[dict]:
    """Slide a window across ``audio`` and decode MIDI events.

    Handles notes that span segment boundaries using the tie-section
    mechanism from MT3 (notes still open at the end of a segment are tracked
    and passed as ``prev_active`` to the tokenizer for the next segment —
    but at inference time we simply merge the decoded note lists and
    de-duplicate).

    Args:
        model: Loaded :class:`~src.model.MT3Model` in eval mode.
        audio: 1-D float32 audio array at ``sample_rate`` Hz.
        sample_rate: Sample rate of ``audio`` (Hz).
        segment_samples: Number of samples per inference segment.
        hop_samples: Stride between consecutive segments.
        max_len: Maximum token sequence length per segment.
        temperature: Decoding temperature (0 = greedy).
        device: Torch device to run inference on.  Defaults to the model's
            device.

    Returns:
        Sorted, deduplicated list of note dicts.
    """
    from src.metrics import deduplicate_notes

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    total_samples = len(audio)
    if total_samples == 0:
        return []

    num_segments = max(1, (total_samples - segment_samples + hop_samples) // hop_samples)
    all_notes: list[dict] = []

    for i in range(num_segments):
        start = i * hop_samples
        end = start + segment_samples
        segment = audio[start:end]

        # Zero-pad last segment if needed
        if len(segment) < segment_samples:
            segment = np.pad(segment, (0, segment_samples - len(segment)))

        waveform = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).to(device)
        segment_start_s = start / sample_rate

        token_ids = model.transcribe(waveform, max_len=max_len, temperature=temperature)
        tokens = token_ids[0].cpu().tolist()

        segment_notes = model.tokenizer.tokens_to_notes(
            tokens, segment_start_s=segment_start_s
        )
        all_notes.extend(segment_notes)

    all_notes = deduplicate_notes(all_notes)
    return all_notes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Entry point for the transcription script."""
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file to MIDI using a trained MT3 model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--audio",
        type=Path,
        required=True,
        help="Path to input audio file (WAV, FLAC, etc.).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the output MIDI file.",
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
        help="Path to the YAML config used when training the checkpoint.",
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
        help="Maximum token sequence length per segment.",
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=16.0,
        help="Audio segment length in seconds for sliding-window inference.",
    )
    parser.add_argument(
        "--hop-seconds",
        type=float,
        default=8.0,
        help="Hop between consecutive segments in seconds.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device string (e.g. 'cuda', 'cpu').  Auto-detected if omitted.",
    )
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[transcribe] device: {device}")

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    sample_rate: int = config.get("data", {}).get("sample_rate", 16000)
    segment_samples = int(args.segment_seconds * sample_rate)
    hop_samples = int(args.hop_seconds * sample_rate)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    # Add project root to sys.path so ``src`` is importable when the script
    # is run from any working directory.
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.model import build_model  # noqa: E402 — after sys.path fix

    print(f"[transcribe] building model from {args.config} …")
    model = build_model(config)

    print(f"[transcribe] loading checkpoint {args.checkpoint} …")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = ckpt.get("model", ckpt)  # handle both raw state_dict and wrapped ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------
    print(f"[transcribe] loading audio {args.audio} …")
    audio = _load_audio(args.audio, target_sr=sample_rate)
    duration_s = len(audio) / sample_rate
    print(f"[transcribe] audio duration: {duration_s:.1f} s  ({len(audio):,} samples)")

    # ------------------------------------------------------------------
    # Transcribe
    # ------------------------------------------------------------------
    print("[transcribe] running sliding-window inference …")
    notes = transcribe_full_audio(
        model,
        audio,
        sample_rate=sample_rate,
        segment_samples=segment_samples,
        hop_samples=hop_samples,
        max_len=args.max_len,
        temperature=args.temperature,
        device=device,
    )
    print(f"[transcribe] decoded {len(notes)} notes")

    # ------------------------------------------------------------------
    # Save MIDI
    # ------------------------------------------------------------------
    print(f"[transcribe] writing MIDI → {args.output}")
    notes_to_midi(notes, args.output)
    print("[transcribe] done.")


if __name__ == "__main__":
    main()
