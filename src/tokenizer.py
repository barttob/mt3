"""MIDI event tokenizer for MT3-style music transcription.

Two vocabulary layouts are supported, selected at construction time via
``use_hierarchical_time``:

Flat layout (default, use_hierarchical_time=False):
    0           <pad>
    1           <sos>
    2           <eos>
    3           <tie>
    4 – 603     time_shift  (600 bins × 8 ms = 0–4.792 s)
    604 – 731   velocity    (128 values, 0–127)
    732 – 859   note_on     (128 MIDI pitches)
    860 – 987   note_off    (128 MIDI pitches)
    988 – 1116  program     (129 program IDs incl. drums; multi-instrument only)

Hierarchical layout (use_hierarchical_time=True, defaults: coarse_step_ms=64, num_coarse=75):
    0 – 3       special tokens (unchanged)
    4 – 78      coarse_time_shift  (75 tokens × 64 ms, covers 0–4.736 s)
    79 – 86     fine_time_shift    (8 tokens × 8 ms, offset within coarse bucket)
    87 – 214    velocity           (128 values)
    215 – 342   note_on            (128 MIDI pitches)
    343 – 470   note_off           (128 MIDI pitches)
    471 – 599   program            (129 IDs; multi-instrument only)

A time event in hierarchical mode is encoded as two consecutive tokens:
    coarse_time_shift(c)  fine_time_shift(f)
representing  c × coarse_step_ms + f × time_step_ms  milliseconds from
segment start. This reduces the time vocabulary from 600 to 83 tokens (86 %)
while preserving 8 ms precision over the same 4.8 s range.
"""

from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
NoteEvent = tuple[float, float, int, int, int]   # onset, offset, pitch, vel, prog
ActiveNote = tuple[int, int, int]                 # pitch, velocity, program


class MidiTokenizer:
    """Converts MIDI note events to/from integer token sequences.

    The token vocabulary is fixed at construction time. Two tokenization modes
    and two instrument modes are supported — see the module docstring for the
    full vocabulary layout.

    Args:
        time_step_ms: Duration of one fine time-shift bin in milliseconds.
        max_time_steps: Number of flat time-shift bins (used only when
            ``use_hierarchical_time=False``).
        num_velocities: Number of MIDI velocity levels (always 128).
        num_pitches: Number of MIDI pitches (always 128).
        num_programs: Number of program IDs. Slakh-style training should use
            129 to cover MIDI programs 0–127 plus the dedicated drum ID 128.
        multi_instrument: Whether to include program tokens in the vocabulary
            and token sequences.
        use_hierarchical_time: If True, encode each timestamp as two
            consecutive tokens (coarse + fine) instead of one flat token.
        coarse_step_ms: Size of one coarse time bucket in milliseconds.
            Must be a multiple of ``time_step_ms``. Ignored when
            ``use_hierarchical_time=False``.
        num_coarse: Number of coarse time buckets. Ignored when
            ``use_hierarchical_time=False``.
    """

    def __init__(
        self,
        time_step_ms: int = 8,
        max_time_steps: int = 600,
        num_velocities: int = 128,
        num_pitches: int = 128,
        num_programs: int = 129,
        multi_instrument: bool = False,
        use_hierarchical_time: bool = False,
        coarse_step_ms: int = 64,
        num_coarse: int = 75,
    ) -> None:
        self.time_step_ms = time_step_ms
        self.max_time_steps = max_time_steps
        self.num_velocities = num_velocities
        self.num_pitches = num_pitches
        self.num_programs = num_programs
        self.multi_instrument = multi_instrument
        self.use_hierarchical_time = use_hierarchical_time
        self.coarse_step_ms = coarse_step_ms
        self.num_coarse = num_coarse
        # Number of fine steps within one coarse bucket
        self.num_fine: int = coarse_step_ms // time_step_ms

        # Special tokens
        self.special: dict[str, int] = {
            "<pad>": 0,
            "<sos>": 1,
            "<eos>": 2,
            "<tie>": 3,
        }

        # Build contiguous vocabulary offsets
        offset = 4

        if use_hierarchical_time:
            self.coarse_time_offset: int = offset
            offset += num_coarse                   # coarse buckets
            self.fine_time_offset: int = offset
            offset += self.num_fine                # fine steps per bucket
            # Alias so external code using time_offset still works for range checks
            self.time_offset: int = self.coarse_time_offset
        else:
            self.time_offset = offset
            offset += max_time_steps
            self.coarse_time_offset = -1           # sentinel: unused
            self.fine_time_offset = -1             # sentinel: unused

        self.velocity_offset: int = offset
        offset += num_velocities

        self.note_on_offset: int = offset
        offset += num_pitches

        self.note_off_offset: int = offset
        offset += num_pitches

        if multi_instrument:
            self.program_offset: int = offset
            offset += num_programs
        else:
            self.program_offset = -1      # sentinel: unused

        self._vocab_size: int = offset

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Total number of tokens in the vocabulary."""
        return self._vocab_size

    def build_tie_prefix(
        self,
        prev_active_notes: Optional[list[ActiveNote]] = None,
    ) -> list[int]:
        """Build a decoder prefix for a segment's tie section."""
        tokens: list[int] = [self.special["<sos>"]]
        if prev_active_notes:
            for pitch, velocity, program in prev_active_notes:
                if self.multi_instrument and not 0 <= program < self.num_programs:
                    raise ValueError(
                        f"Program {program} out of range for tokenizer with "
                        f"{self.num_programs} programs."
                    )
                if self.multi_instrument:
                    tokens.append(self.program_offset + program)
                tokens.append(self.velocity_offset + velocity)
                tokens.append(self.note_on_offset + pitch)
        tokens.append(self.special["<tie>"])
        return tokens

    # ------------------------------------------------------------------
    # Encoding: notes → tokens
    # ------------------------------------------------------------------

    def notes_to_tokens(
        self,
        notes: list[NoteEvent],
        segment_start_s: float,
        segment_end_s: float,
        prev_active_notes: Optional[list[ActiveNote]] = None,
    ) -> list[int]:
        """Convert a list of note events to a token sequence.

        The output follows the MT3 sequence structure::

            <sos> [tie section] <tie> [event section] <eos>

        **Tie section** – notes still sounding from the previous segment,
        encoded as ``[program] velocity note_on`` groups (program only in
        multi-instrument mode).

        **Event section** – time-ordered note-on / note-off events within
        the current segment, encoded as
        ``time_shift [program] velocity note_on`` or
        ``time_shift note_off``.

        At each distinct time step the ``time_shift`` token is emitted once,
        then all events at that time follow. Within a time step the ordering
        is: program (note_on only) → velocity (note_on only) → note_on /
        note_off.

        Args:
            notes: All note events that overlap with or fall inside the
                segment. Each tuple is
                ``(onset_s, offset_s, pitch, velocity, program)``.
            segment_start_s: Absolute start time of the current segment (s).
            segment_end_s: Absolute end time of the current segment (s).
            prev_active_notes: Notes still sounding at the start of this
                segment (from the previous segment). Each tuple is
                ``(pitch, velocity, program)``.

        Returns:
            List of integer token IDs.
        """
        tokens = self.build_tie_prefix(prev_active_notes)

        # ---- Event section ----------------------------------------------
        events: list[tuple[float, str, int, int, int]] = []  # (t_rel, type, pitch, vel, prog)

        for onset, offset, pitch, vel, prog in notes:
            if self.multi_instrument and not 0 <= prog < self.num_programs:
                raise ValueError(
                    f"Program {prog} out of range for tokenizer with "
                    f"{self.num_programs} programs."
                )
            # Note-on: onset falls inside segment
            if segment_start_s <= onset < segment_end_s:
                t_rel = onset - segment_start_s
                events.append((t_rel, "on", pitch, vel, prog))
            # Note-off: offset falls inside segment
            if segment_start_s <= offset < segment_end_s:
                t_rel = offset - segment_start_s
                events.append((t_rel, "off", pitch, 0, prog))

        # Sort: by time, then note-off before note-on, then pitch
        events.sort(key=lambda e: (e[0], 0 if e[1] == "off" else 1, e[2]))

        prev_time_bin: int = -1
        for t_rel, etype, pitch, vel, prog in events:
            time_bin = min(
                int(t_rel * 1000 / self.time_step_ms),
                self.max_time_steps - 1,
            )
            if time_bin != prev_time_bin:
                if self.use_hierarchical_time:
                    delta_ms = time_bin * self.time_step_ms
                    coarse = min(delta_ms // self.coarse_step_ms, self.num_coarse - 1)
                    fine = min(
                        (delta_ms - coarse * self.coarse_step_ms) // self.time_step_ms,
                        self.num_fine - 1,
                    )
                    tokens.append(self.coarse_time_offset + coarse)
                    tokens.append(self.fine_time_offset + fine)
                else:
                    tokens.append(self.time_offset + time_bin)
                prev_time_bin = time_bin

            if etype == "on":
                if self.multi_instrument:
                    tokens.append(self.program_offset + prog)
                tokens.append(self.velocity_offset + vel)
                tokens.append(self.note_on_offset + pitch)
            else:
                tokens.append(self.note_off_offset + pitch)

        tokens.append(self.special["<eos>"])
        return tokens

    # ------------------------------------------------------------------
    # Decoding: tokens → notes
    # ------------------------------------------------------------------

    def tokens_to_notes(
        self,
        tokens: list[int],
        segment_start_s: float = 0.0,
    ) -> list[dict]:
        """Convert a token sequence back to note events.

        Parses the MT3 token structure and reconstructs note events with
        absolute onset/offset times. Notes that are opened (note_on seen)
        but never closed (no note_off before <eos>) are closed at the last
        decoded time position.

        Notes from the **tie section** (between <sos> and <tie>) are
        treated as already-open notes whose onset is ``segment_start_s``;
        their offsets are filled in when the corresponding note_off is
        encountered in the event section.

        Args:
            tokens: Sequence of integer token IDs as produced by
                :meth:`notes_to_tokens`.
            segment_start_s: Absolute start time of the segment in seconds.
                All relative times decoded from ``time_shift`` tokens are
                offset by this value.

        Returns:
            List of note dicts, each with keys:
            ``onset`` (float), ``offset`` (float), ``pitch`` (int),
            ``velocity`` (int), ``program`` (int).
        """
        notes: list[dict] = []
        # (pitch, program) → list of open notes in onset order
        open_notes: dict[tuple[int, int], list[dict]] = {}

        current_time_s: float = segment_start_s
        current_velocity: int = 64   # sensible default
        current_program: int = 0
        pending_coarse: int = 0      # coarse bucket waiting for its fine token

        in_tie_section: bool = True   # between <sos> and <tie>
        sos_seen: bool = False

        i = 0
        while i < len(tokens):
            tok = tokens[i]

            # ---- Special tokens -----------------------------------------
            if tok == self.special["<pad>"]:
                i += 1
                continue

            if tok == self.special["<sos>"]:
                sos_seen = True
                in_tie_section = True
                i += 1
                continue

            if tok == self.special["<eos>"]:
                break

            if tok == self.special["<tie>"]:
                in_tie_section = False
                # After <tie>, reset to segment start for event section
                current_time_s = segment_start_s
                pending_coarse = 0
                i += 1
                continue

            # ---- Hierarchical time tokens (event section only) ----------
            if self.use_hierarchical_time:
                if self.coarse_time_offset <= tok < self.coarse_time_offset + self.num_coarse:
                    if not in_tie_section:
                        pending_coarse = tok - self.coarse_time_offset
                    i += 1
                    continue
                if self.fine_time_offset <= tok < self.fine_time_offset + self.num_fine:
                    if not in_tie_section:
                        fine = tok - self.fine_time_offset
                        total_ms = (
                            pending_coarse * self.coarse_step_ms
                            + fine * self.time_step_ms
                        )
                        current_time_s = segment_start_s + total_ms / 1000.0
                    i += 1
                    continue

            # ---- Flat time_shift (event section only) -------------------
            elif self.time_offset <= tok < self.time_offset + self.max_time_steps:
                if not in_tie_section:
                    time_bin = tok - self.time_offset
                    current_time_s = segment_start_s + time_bin * self.time_step_ms / 1000.0
                i += 1
                continue

            # ---- velocity -----------------------------------------------
            if self.velocity_offset <= tok < self.velocity_offset + self.num_velocities:
                current_velocity = tok - self.velocity_offset
                i += 1
                continue

            # ---- program (multi-instrument only) ------------------------
            if (
                self.multi_instrument
                and self.program_offset <= tok < self.program_offset + self.num_programs
            ):
                current_program = tok - self.program_offset
                i += 1
                continue

            # ---- note_on ------------------------------------------------
            if self.note_on_offset <= tok < self.note_on_offset + self.num_pitches:
                pitch = tok - self.note_on_offset
                onset = segment_start_s if in_tie_section else current_time_s
                key = (pitch, current_program)
                open_notes.setdefault(key, []).append({
                    "onset": onset,
                    "velocity": current_velocity,
                    "program": current_program,
                    "pitch": pitch,
                })
                i += 1
                continue

            # ---- note_off -----------------------------------------------
            if self.note_off_offset <= tok < self.note_off_offset + self.num_pitches:
                pitch = tok - self.note_off_offset
                # Try matching open note; fall back to any open note with this pitch
                key = (pitch, current_program)
                note_bucket = open_notes.get(key)
                if not note_bucket:
                    matching_keys = [
                        k for k, bucket in open_notes.items() if k[0] == pitch and bucket
                    ]
                    if matching_keys:
                        key = min(matching_keys, key=lambda item: open_notes[item][0]["onset"])
                        note_bucket = open_notes[key]
                if note_bucket:
                    note = note_bucket.pop(0)
                    if not note_bucket:
                        open_notes.pop(key, None)
                    note["offset"] = current_time_s
                    notes.append(note)
                i += 1
                continue

            # Unknown token – skip
            i += 1

        # Close any notes that were never given a note_off
        for note_bucket in open_notes.values():
            for note in note_bucket:
                note["offset"] = current_time_s
                notes.append(note)

        notes.sort(key=lambda n: (n["onset"], n["pitch"]))
        return notes

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def token_type(self, token_id: int) -> str:
        """Return a human-readable name for a token ID.

        Args:
            token_id: Integer token ID to look up.

        Returns:
            String such as ``"<sos>"``, ``"time_shift(42)"``,
            ``"note_on(60)"``, etc.

        Raises:
            ValueError: If ``token_id`` is out of vocabulary range.
        """
        if token_id < 0 or token_id >= self._vocab_size:
            raise ValueError(f"Token ID {token_id} out of range [0, {self._vocab_size})")

        for name, val in self.special.items():
            if token_id == val:
                return name

        if self.use_hierarchical_time:
            if self.coarse_time_offset <= token_id < self.coarse_time_offset + self.num_coarse:
                return f"coarse_time_shift({token_id - self.coarse_time_offset})"
            if self.fine_time_offset <= token_id < self.fine_time_offset + self.num_fine:
                return f"fine_time_shift({token_id - self.fine_time_offset})"
        else:
            if self.time_offset <= token_id < self.time_offset + self.max_time_steps:
                return f"time_shift({token_id - self.time_offset})"

        if self.velocity_offset <= token_id < self.velocity_offset + self.num_velocities:
            return f"velocity({token_id - self.velocity_offset})"
        if self.note_on_offset <= token_id < self.note_on_offset + self.num_pitches:
            return f"note_on({token_id - self.note_on_offset})"
        if self.note_off_offset <= token_id < self.note_off_offset + self.num_pitches:
            return f"note_off({token_id - self.note_off_offset})"
        if self.multi_instrument and self.program_offset <= token_id < self.program_offset + self.num_programs:
            return f"program({token_id - self.program_offset})"

        return f"unknown({token_id})"


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math

    def _approx_eq(a: float, b: float, tol: float = 1e-3) -> bool:
        return math.fabs(a - b) <= tol

    def _notes_match(original: list[NoteEvent], decoded: list[dict], tol_s: float = 0.008) -> bool:
        """Check that decoded notes match originals within one time-bin tolerance."""
        if len(original) != len(decoded):
            return False
        # Sort both by onset then pitch for stable comparison
        orig_sorted = sorted(original, key=lambda n: (n[0], n[2]))
        dec_sorted = sorted(decoded, key=lambda n: (n["onset"], n["pitch"]))
        for (onset, offset, pitch, vel, prog), d in zip(orig_sorted, dec_sorted):
            if d["pitch"] != pitch:
                return False
            if d["velocity"] != vel:
                return False
            if d["program"] != prog:
                return False
            if not _approx_eq(d["onset"], onset, tol_s):
                return False
            if not _approx_eq(d["offset"], offset, tol_s):
                return False
        return True

    print("=" * 60)
    print("Test 1: Piano-only round-trip (no ties)")
    print("=" * 60)

    tok_piano = MidiTokenizer(multi_instrument=False)
    print(f"  vocab_size = {tok_piano.vocab_size}  (expected 988)")
    assert tok_piano.vocab_size == 988, f"Got {tok_piano.vocab_size}"

    # Simple two-note sequence: C4 then E4
    notes_piano: list[NoteEvent] = [
        (0.100, 0.500, 60, 80, 0),   # C4
        (0.300, 0.700, 64, 90, 0),   # E4
    ]
    seg_start, seg_end = 0.0, 2.048

    tokens = tok_piano.notes_to_tokens(notes_piano, seg_start, seg_end)
    print(f"  Token sequence ({len(tokens)} tokens):")
    print("   ", [tok_piano.token_type(t) for t in tokens])

    decoded = tok_piano.tokens_to_notes(tokens, seg_start)
    print(f"  Decoded notes: {decoded}")
    assert _notes_match(notes_piano, decoded), f"Round-trip mismatch!\n  orig={notes_piano}\n  dec={decoded}"
    print("  PASSED\n")

    # ------------------------------------------------------------------
    print("=" * 60)
    print("Test 2: Multi-instrument round-trip (piano + strings)")
    print("=" * 60)

    tok_multi = MidiTokenizer(multi_instrument=True)
    print(f"  vocab_size = {tok_multi.vocab_size}  (expected 1117)")
    assert tok_multi.vocab_size == 1117, f"Got {tok_multi.vocab_size}"

    notes_multi: list[NoteEvent] = [
        (0.050, 0.400, 60, 80,  0),   # Piano C4
        (0.050, 0.400, 48, 70, 40),   # Violin G3 (prog 40)
        (0.200, 0.600, 64, 90,  0),   # Piano E4
    ]
    seg_start, seg_end = 0.0, 2.048

    tokens = tok_multi.notes_to_tokens(notes_multi, seg_start, seg_end)
    print(f"  Token sequence ({len(tokens)} tokens):")
    print("   ", [tok_multi.token_type(t) for t in tokens])

    decoded = tok_multi.tokens_to_notes(tokens, seg_start)
    print(f"  Decoded notes: {decoded}")
    assert _notes_match(notes_multi, decoded), (
        f"Round-trip mismatch!\n  orig={notes_multi}\n  dec={decoded}"
    )
    print("  PASSED\n")

    # ------------------------------------------------------------------
    print("=" * 60)
    print("Test 3: Tie section — notes held from previous segment")
    print("=" * 60)

    tok2 = MidiTokenizer(multi_instrument=False)

    # One note spans the segment boundary: onset before seg_start, offset inside
    notes_spanning: list[NoteEvent] = [
        (-0.500, 0.300, 60, 75, 0),   # C4: started before, ends at 0.3 s
        ( 0.100, 0.800, 67, 80, 0),   # G4: starts and ends inside
    ]
    prev_active: list[ActiveNote] = [(60, 75, 0)]   # C4 is held from previous segment

    seg_start, seg_end = 0.0, 2.048
    tokens = tok2.notes_to_tokens(notes_spanning, seg_start, seg_end, prev_active)
    print(f"  Token sequence ({len(tokens)} tokens):")
    print("   ", [tok2.token_type(t) for t in tokens])

    # Tie-section note (C4) has onset clamped to seg_start; its note_off is inside
    decoded = tok2.tokens_to_notes(tokens, seg_start)
    print(f"  Decoded notes: {decoded}")

    # C4: onset should be seg_start (0.0), offset ~0.3
    c4 = next((n for n in decoded if n["pitch"] == 60), None)
    assert c4 is not None, "C4 not found in decoded notes"
    assert _approx_eq(c4["onset"], 0.0), f"C4 onset wrong: {c4['onset']}"
    # 0.3 s → bin 37 → 0.296 s (one-bin quantization error is expected)
    assert _approx_eq(c4["offset"], 0.3, tol=0.008), f"C4 offset wrong: {c4['offset']}"

    # G4: normal note
    g4 = next((n for n in decoded if n["pitch"] == 67), None)
    assert g4 is not None, "G4 not found in decoded notes"
    assert _approx_eq(g4["onset"], 0.1, tol=0.008), f"G4 onset wrong: {g4['onset']}"
    assert _approx_eq(g4["offset"], 0.8, tol=0.008), f"G4 offset wrong: {g4['offset']}"

    print("  PASSED\n")

    # ------------------------------------------------------------------
    print("=" * 60)
    print("Test 4: Repeated-note decoding keeps both notes")
    print("=" * 60)

    repeated_notes: list[NoteEvent] = [
        (0.100, 0.400, 60, 80, 0),
        (0.200, 0.500, 60, 90, 0),
    ]
    tokens = tok_piano.notes_to_tokens(repeated_notes, 0.0, 2.048)
    decoded = tok_piano.tokens_to_notes(tokens, 0.0)
    assert _notes_match(repeated_notes, decoded), (
        f"Repeated-note round-trip mismatch!\n  orig={repeated_notes}\n  dec={decoded}"
    )
    print("  PASSED\n")

    # ------------------------------------------------------------------
    print("=" * 60)
    print("Test 5: token_type helper")
    print("=" * 60)

    t = MidiTokenizer(multi_instrument=True)
    assert t.token_type(0) == "<pad>"
    assert t.token_type(1) == "<sos>"
    assert t.token_type(2) == "<eos>"
    assert t.token_type(3) == "<tie>"
    assert t.token_type(4) == "time_shift(0)"
    assert t.token_type(603) == "time_shift(599)"
    assert t.token_type(604) == "velocity(0)"
    assert t.token_type(731) == "velocity(127)"
    assert t.token_type(732) == "note_on(0)"
    assert t.token_type(859) == "note_on(127)"
    assert t.token_type(860) == "note_off(0)"
    assert t.token_type(987) == "note_off(127)"
    assert t.token_type(988) == "program(0)"
    assert t.token_type(1116) == "program(128)"
    print("  PASSED\n")

    print("All tests passed.")
