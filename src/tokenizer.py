"""MIDI event tokenizer for MT3-style music transcription.

Token vocabulary layout (IDs):
    0           <pad>
    1           <sos>
    2           <eos>
    3           <tie>
    4 – 603     time_shift  (600 bins × 8 ms = 0–4.792 s)
    604 – 731   velocity    (128 values, 0–127)
    732 – 859   note_on     (128 MIDI pitches)
    860 – 987   note_off    (128 MIDI pitches)
    988 – 1115  program     (128 MIDI programs, multi-instrument only)
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

    The token vocabulary is fixed at construction time. Two modes are
    supported:

    * **Piano-only** (``multi_instrument=False``): no program tokens are
      added; all notes are implicitly assumed to be program 0 (piano).
    * **Multi-instrument** (``multi_instrument=True``): 128 program tokens
      are appended to the vocabulary (IDs 988–1115).

    Args:
        time_step_ms: Duration of one time-shift bin in milliseconds.
        max_time_steps: Number of time-shift bins (covers
            ``time_step_ms * max_time_steps`` ms of audio per segment).
        num_velocities: Number of MIDI velocity levels (always 128).
        num_pitches: Number of MIDI pitches (always 128).
        num_programs: Number of MIDI program numbers (always 128).
        multi_instrument: Whether to include program tokens in the
            vocabulary and token sequences.
    """

    def __init__(
        self,
        time_step_ms: int = 8,
        max_time_steps: int = 600,
        num_velocities: int = 128,
        num_pitches: int = 128,
        num_programs: int = 128,
        multi_instrument: bool = False,
    ) -> None:
        self.time_step_ms = time_step_ms
        self.max_time_steps = max_time_steps
        self.num_velocities = num_velocities
        self.num_pitches = num_pitches
        self.num_programs = num_programs
        self.multi_instrument = multi_instrument

        # Special tokens
        self.special: dict[str, int] = {
            "<pad>": 0,
            "<sos>": 1,
            "<eos>": 2,
            "<tie>": 3,
        }

        # Build contiguous vocabulary offsets
        offset = 4
        self.time_offset: int = offset
        offset += max_time_steps          # 4 – 603

        self.velocity_offset: int = offset
        offset += num_velocities          # 604 – 731

        self.note_on_offset: int = offset
        offset += num_pitches             # 732 – 859

        self.note_off_offset: int = offset
        offset += num_pitches             # 860 – 987

        if multi_instrument:
            self.program_offset: int = offset
            offset += num_programs        # 988 – 1115
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
        tokens: list[int] = [self.special["<sos>"]]

        # ---- Tie section ------------------------------------------------
        if prev_active_notes:
            for pitch, velocity, program in prev_active_notes:
                if self.multi_instrument:
                    tokens.append(self.program_offset + program)
                tokens.append(self.velocity_offset + velocity)
                tokens.append(self.note_on_offset + pitch)

        tokens.append(self.special["<tie>"])

        # ---- Event section ----------------------------------------------
        events: list[tuple[float, str, int, int, int]] = []  # (t_rel, type, pitch, vel, prog)

        for onset, offset, pitch, vel, prog in notes:
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
        # (pitch, program) → {"onset": float, "velocity": int, "program": int}
        open_notes: dict[tuple[int, int], dict] = {}

        current_time_s: float = segment_start_s
        current_velocity: int = 64   # sensible default
        current_program: int = 0

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
                i += 1
                continue

            # ---- time_shift (event section only) ------------------------
            if self.time_offset <= tok < self.time_offset + self.max_time_steps:
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
                open_notes[key] = {
                    "onset": onset,
                    "velocity": current_velocity,
                    "program": current_program,
                    "pitch": pitch,
                }
                i += 1
                continue

            # ---- note_off -----------------------------------------------
            if self.note_off_offset <= tok < self.note_off_offset + self.num_pitches:
                pitch = tok - self.note_off_offset
                # Try matching open note; fall back to any open note with this pitch
                key = (pitch, current_program)
                if key not in open_notes:
                    # Scan for any open note with this pitch regardless of program
                    for k in list(open_notes.keys()):
                        if k[0] == pitch:
                            key = k
                            break
                if key in open_notes:
                    note = open_notes.pop(key)
                    note["offset"] = current_time_s
                    notes.append(note)
                i += 1
                continue

            # Unknown token – skip
            i += 1

        # Close any notes that were never given a note_off
        for note in open_notes.values():
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
    print(f"  vocab_size = {tok_multi.vocab_size}  (expected 1116)")
    assert tok_multi.vocab_size == 1116, f"Got {tok_multi.vocab_size}"

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
    print("Test 4: token_type helper")
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
    assert t.token_type(1115) == "program(127)"
    print("  PASSED\n")

    print("All tests passed.")
