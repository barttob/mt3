"""Evaluation metrics for MT3-style music transcription.

Wraps ``mir_eval`` to compute onset, onset+offset, and per-program F1 scores.
Also provides note deduplication for overlapping sliding-window inference.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np

try:
    import librosa
    import mir_eval
    _MIR_EVAL_AVAILABLE = True
except ImportError:
    _MIR_EVAL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
Note = dict  # keys: onset, offset, pitch, velocity, program


# ---------------------------------------------------------------------------
# Note filtering
# ---------------------------------------------------------------------------

def filter_notes(
    notes: list[Note],
    min_duration_s: float = 0.03,
    drum_program: int = 128,
    drum_min_duration_s: float = 0.01,
) -> list[Note]:
    """Remove spuriously short notes produced by the decoder.

    Notes shorter than ``min_duration_s`` are dropped, with a looser threshold
    for drum notes (program == ``drum_program``) because drum hits are
    inherently very short.

    Args:
        notes: List of note dicts (onset, offset, pitch, velocity, program).
        min_duration_s: Minimum duration in seconds for non-drum notes
            (default 30 ms).
        drum_program: MIDI program number used for drum notes (default 128).
        drum_min_duration_s: Minimum duration in seconds for drum notes
            (default 10 ms).

    Returns:
        Filtered list of note dicts, preserving input order.
    """
    kept: list[Note] = []
    for note in notes:
        duration = note["offset"] - note["onset"]
        threshold = (
            drum_min_duration_s
            if note.get("program", 0) == drum_program
            else min_duration_s
        )
        if duration >= threshold:
            kept.append(note)
    return kept


# ---------------------------------------------------------------------------
# Note deduplication
# ---------------------------------------------------------------------------

def deduplicate_notes(notes: list[Note], overlap_s: float = 0.05) -> list[Note]:
    """Remove duplicate notes produced by overlapping sliding-window segments.

    Two notes are considered duplicates if they share the same pitch and
    program and their onsets are within ``overlap_s`` seconds of each other.
    The note with the longer duration is kept.

    Args:
        notes: List of note dicts (onset, offset, pitch, velocity, program).
        overlap_s: Maximum onset difference (seconds) to consider two notes
            duplicates.

    Returns:
        Deduplicated list of note dicts, sorted by onset then pitch.
    """
    if not notes:
        return []

    # Sort by onset for a single forward pass.
    sorted_notes = sorted(
        notes, key=lambda n: (n["onset"], n["pitch"], n["program"])
    )
    drop_indices: set[int] = set()  # indices into sorted_notes to suppress

    # recent[(pitch, program)] = index of the most recent surviving note for
    # that key.  When a new note arrives within overlap_s of the stored one
    # they are duplicates; the shorter is dropped.  When the gap exceeds
    # overlap_s the stored entry is replaced (it can no longer clash).
    recent: dict[tuple[int, int], int] = {}

    for i, note in enumerate(sorted_notes):
        key = (note["pitch"], note["program"])
        onset = note["onset"]

        if key in recent:
            j = recent[key]
            prev = sorted_notes[j]
            if onset - prev["onset"] <= overlap_s:
                # Duplicate pair: keep whichever note has the longer duration.
                dur_cur = note["offset"] - onset
                dur_prev = prev["offset"] - prev["onset"]
                if dur_cur > dur_prev:
                    drop_indices.add(j)
                    recent[key] = i  # current note wins
                else:
                    drop_indices.add(i)
                    # recent[key] stays as j (prev note wins)
                continue  # skip the fallthrough update below

        # New key, or gap exceeded overlap_s — this note becomes the reference.
        recent[key] = i

    kept = [n for i, n in enumerate(sorted_notes) if i not in drop_indices]
    # Already sorted by onset; re-sort to ensure pitch ordering too.
    kept.sort(key=lambda n: (n["onset"], n["pitch"]))
    return kept


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def evaluate_transcription(
    ref_notes: list[Note],
    est_notes: list[Note],
    onset_tolerance: float = 0.05,
    offset_ratio: float = 0.2,
    offset_min_tolerance: float = 0.05,
) -> dict[str, float]:
    """Compute onset and onset+offset transcription F1 using mir_eval.

    Args:
        ref_notes: Ground-truth note list. Each dict must have keys
            ``onset`` (float), ``offset`` (float), ``pitch`` (int),
            ``velocity`` (int).
        est_notes: Estimated note list in the same format.
        onset_tolerance: Allowed onset deviation in seconds (default 50 ms).
        offset_ratio: Offset tolerance as a fraction of note duration
            (default 0.2 = 20 %).
        offset_min_tolerance: Minimum offset tolerance in seconds regardless
            of note duration (default 50 ms).

    Returns:
        Dict with keys:
        ``onset_P``, ``onset_R``, ``onset_F1``,
        ``onset_offset_P``, ``onset_offset_R``, ``onset_offset_F1``.

    Raises:
        ImportError: If ``mir_eval`` or ``librosa`` is not installed.
    """
    if not _MIR_EVAL_AVAILABLE:
        raise ImportError("mir_eval and librosa are required for evaluation metrics.")

    def _to_arrays(
        note_list: list[Note],
    ) -> tuple[np.ndarray, np.ndarray]:
        if not note_list:
            return np.zeros((0, 2)), np.zeros(0)
        intervals = np.array(
            [[n["onset"], max(n["offset"], n["onset"] + 1e-6)] for n in note_list],
            dtype=float,
        )
        pitches = np.array(
            [librosa.midi_to_hz(n["pitch"]) for n in note_list], dtype=float
        )
        return intervals, pitches

    ref_intervals, ref_pitches = _to_arrays(ref_notes)
    est_intervals, est_pitches = _to_arrays(est_notes)

    # Onset-only F1 (offset_ratio=None disables offset matching)
    p_on, r_on, f1_on, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals,
        ref_pitches,
        est_intervals,
        est_pitches,
        onset_tolerance=onset_tolerance,
        offset_ratio=None,
    )

    # Onset + Offset F1
    p_off, r_off, f1_off, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals,
        ref_pitches,
        est_intervals,
        est_pitches,
        onset_tolerance=onset_tolerance,
        offset_ratio=offset_ratio,
        offset_min_tolerance=offset_min_tolerance,
    )

    return {
        "onset_P": float(p_on),
        "onset_R": float(r_on),
        "onset_F1": float(f1_on),
        "onset_offset_P": float(p_off),
        "onset_offset_R": float(r_off),
        "onset_offset_F1": float(f1_off),
    }


# ---------------------------------------------------------------------------
# Per-program metrics
# ---------------------------------------------------------------------------

def per_program_metrics(
    ref_notes: list[Note],
    est_notes: list[Note],
    onset_tolerance: float = 0.05,
    offset_ratio: float = 0.2,
    offset_min_tolerance: float = 0.05,
) -> dict[int, dict[str, float]]:
    """Compute onset and onset+offset F1 separately for each MIDI program.

    Args:
        ref_notes: Ground-truth notes (must include ``program`` key).
        est_notes: Estimated notes (must include ``program`` key).
        onset_tolerance: Onset tolerance in seconds.
        offset_ratio: Offset tolerance as fraction of note duration.
        offset_min_tolerance: Minimum offset tolerance in seconds.

    Returns:
        Dict mapping program number → metric dict (same keys as
        :func:`evaluate_transcription`).  Only programs present in either
        ``ref_notes`` or ``est_notes`` are included.
    """
    programs: set[int] = set()
    for n in ref_notes:
        programs.add(n["program"])
    for n in est_notes:
        programs.add(n["program"])

    results: dict[int, dict[str, float]] = {}
    for prog in sorted(programs):
        ref_prog = [n for n in ref_notes if n["program"] == prog]
        est_prog = [n for n in est_notes if n["program"] == prog]
        results[prog] = evaluate_transcription(
            ref_prog,
            est_prog,
            onset_tolerance=onset_tolerance,
            offset_ratio=offset_ratio,
            offset_min_tolerance=offset_min_tolerance,
        )

    return results


def macro_average_metrics(
    per_prog: dict[int, dict[str, float]],
) -> dict[str, float]:
    """Compute the macro-average (mean over programs) of per-program metrics.

    Args:
        per_prog: Output of :func:`per_program_metrics`.

    Returns:
        Dict with the same keys as :func:`evaluate_transcription`, each value
        being the unweighted mean across all programs.
    """
    if not per_prog:
        return {}

    keys = list(next(iter(per_prog.values())).keys())
    totals: dict[str, float] = defaultdict(float)
    for prog_metrics in per_prog.values():
        for k in keys:
            totals[k] += prog_metrics[k]
    n = len(per_prog)
    return {k: totals[k] / n for k in keys}


# ---------------------------------------------------------------------------
# Instrument detection F1
# ---------------------------------------------------------------------------

def instrument_detection_f1(
    ref_notes: list[Note],
    est_notes: list[Note],
) -> dict[str, float]:
    """Compute instrument-level detection precision, recall, and F1.

    An instrument is considered "detected" if at least one note is assigned
    to its program number.

    Args:
        ref_notes: Ground-truth notes with ``program`` key.
        est_notes: Estimated notes with ``program`` key.

    Returns:
        Dict with keys ``precision``, ``recall``, ``f1``.
    """
    ref_progs = {n["program"] for n in ref_notes}
    est_progs = {n["program"] for n in est_notes}

    tp = len(ref_progs & est_progs)
    fp = len(est_progs - ref_progs)
    fn = len(ref_progs - est_progs)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}
