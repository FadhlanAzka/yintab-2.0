# -*- coding: utf-8 -*-
"""
tab_midi.py — Generate a MIDI file from CSV detection results (monophonic).

Refinement sesuai info terbaru:
1) CSV TIDAK punya kolom time (tidak ada timestamp)
2) Monophonic: 1 note per step (tidak ada chord)

Asumsi:
- CSV minimal punya kolom "midi"
- Setiap baris = 1 step waktu
- Tempo: notes_per_second (default 2.0) → 1 step = 0.5 detik → 120 BPM
- Jika nilai midi pada baris tidak valid/kosong, default diperlakukan sebagai REST
  (time tetap maju 1 step). Ini penting supaya panjang sequence tidak berubah.

Output:
- MIDI format 0, single-track
- Program default: Acoustic Guitar (nylon) (GM program 24, 0-indexed)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class MidiRenderConfig:
    notes_per_second: float = 2.0
    ticks_per_beat: int = 480
    velocity: int = 90
    program: int = 24  # GM program (0-indexed). 24 ≈ Acoustic Guitar (nylon)


# ===================== MIDI Low-level Helpers =====================

def _varlen(value: int) -> bytes:
    """Encode integer as MIDI variable-length quantity."""
    if value < 0:
        raise ValueError("VLQ value must be >= 0")
    buffer = value & 0x7F
    value >>= 7
    while value:
        buffer <<= 8
        buffer |= 0x80 | (value & 0x7F)
        value >>= 7

    out = bytearray()
    while True:
        out.append(buffer & 0xFF)
        if buffer & 0x80:
            buffer >>= 8
        else:
            break
    return bytes(out)


def _u16be(n: int) -> bytes:
    return int(n).to_bytes(2, "big", signed=False)


def _u32be(n: int) -> bytes:
    return int(n).to_bytes(4, "big", signed=False)


def _tempo_us_per_beat(notes_per_second: float) -> int:
    """
    notes_per_second=2.0 → seconds_per_note=0.5 → bpm=120 → us/beat=500000
    """
    if notes_per_second <= 0:
        raise ValueError("notes_per_second must be > 0")
    seconds_per_note = 1.0 / notes_per_second
    bpm = 60.0 / seconds_per_note
    return int(round(60_000_000 / bpm))


# ===================== CSV Parsing (Monophonic) =====================

def _parse_midi_cell(cell: str) -> Optional[int]:
    """
    Parse one CSV cell into MIDI int (0..127).
    Robustness:
    - trims spaces
    - if cell contains delimiters like ';' or ',' (shouldn't for monophonic),
      takes the first token.
    Returns None if empty/invalid.
    """
    if cell is None:
        return None
    s = str(cell).strip()
    if not s:
        return None

    # If accidentally contains multiple values, pick first token.
    for delim in (";", ",", "|", " "):
        if delim in s:
            s = s.split(delim, 1)[0].strip()
            break

    try:
        v = int(round(float(s)))
    except ValueError:
        return None

    if 0 <= v <= 127:
        return v
    return None


def csv_to_midi_sequence(
    csv_path: Path,
    midi_col: str = "midi",
    rest_policy: str = "keep",  # "keep" = keep step as rest, "skip" = drop the row
) -> List[Optional[int]]:
    """
    Read CSV → monophonic sequence [midi_or_None, ...]
    - Each CSV row = 1 step
    - midi_or_None None means REST at that step

    rest_policy:
    - "keep": invalid/empty midi becomes REST (time advances)
    - "skip": invalid/empty midi row is dropped (time compresses)
    """
    csv_path = Path(csv_path)
    if rest_policy not in ("keep", "skip"):
        raise ValueError("rest_policy must be 'keep' or 'skip'")

    seq: List[Optional[int]] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return seq

        header_lower = [h.strip().lower() for h in header]
        if midi_col.lower() not in header_lower:
            raise ValueError(f"Kolom '{midi_col}' tidak ditemukan di CSV: {csv_path}")
        midi_idx = header_lower.index(midi_col.lower())

        for row in reader:
            if not row or all(c.strip() == "" for c in row):
                if rest_policy == "keep":
                    seq.append(None)
                continue

            cell = row[midi_idx] if midi_idx < len(row) else ""
            midi = _parse_midi_cell(cell)

            if midi is None and rest_policy == "skip":
                continue
            seq.append(midi)

    return seq


# ===================== MIDI Writer (Monophonic) =====================

def write_midi_from_mono_sequence(
    midi_seq: Sequence[Optional[int]],
    out_mid_path: Path,
    config: MidiRenderConfig = MidiRenderConfig(),
) -> Path:
    """
    Write a single-track MIDI (format 0) from monophonic midi sequence.
    - Each step = 1 beat
    - None = REST (no note event, but time advances 1 beat)
    """
    out_mid_path = Path(out_mid_path)
    out_mid_path.parent.mkdir(parents=True, exist_ok=True)

    us_per_beat = _tempo_us_per_beat(config.notes_per_second)
    tpb = int(config.ticks_per_beat)
    step_len_ticks = tpb  # 1 beat per step

    # Build absolute-tick events: (tick, order, bytes)
    # order: ensure note_off before note_on if ever same tick
    events: List[Tuple[int, int, bytes]] = []

    # Tempo meta at time 0
    tempo_bytes = us_per_beat.to_bytes(3, "big", signed=False)
    events.append((0, 0, b"\xFF\x51\x03" + tempo_bytes))

    # Program change at time 0 (channel 0)
    prog = max(0, min(127, int(config.program)))
    events.append((0, 1, bytes([0xC0, prog])))

    vel = max(1, min(127, int(config.velocity)))

    for step, midi in enumerate(midi_seq):
        if midi is None:
            continue
        n = max(0, min(127, int(midi)))
        start = step * step_len_ticks
        end = start + step_len_ticks

        # note_on
        events.append((start, 2, bytes([0x90, n, vel])))
        # note_off
        events.append((end, 1, bytes([0x80, n, 0])))

    events.sort(key=lambda x: (x[0], x[1]))

    # Assemble track with delta-times
    track = bytearray()
    last_tick = 0
    for tick, _, msg in events:
        delta = tick - last_tick
        if delta < 0:
            raise RuntimeError("Event ticks are not non-decreasing.")
        track += _varlen(delta)
        track += msg
        last_tick = tick

    # End of track
    track += _varlen(0) + b"\xFF\x2F\x00"

    # Header chunk (format 0, 1 track)
    header = b"MThd" + _u32be(6) + _u16be(0) + _u16be(1) + _u16be(tpb)
    trk = b"MTrk" + _u32be(len(track)) + bytes(track)

    out_mid_path.write_bytes(header + trk)
    return out_mid_path


# ===================== Convenience API =====================

def csv_notes_to_midi_file(
    csv_path: Path,
    out_mid_path: Path,
    notes_per_second: float = 2.0,
    midi_col: str = "midi",
    rest_policy: str = "keep",
    velocity: int = 90,
    program: int = 24,
    ticks_per_beat: int = 480,
) -> Path:
    """
    One-shot helper:
    CSV → monophonic midi sequence → MIDI file.

    notes_per_second default 2.0 (sesuai permintaan).
    """
    cfg = MidiRenderConfig(
        notes_per_second=notes_per_second,
        ticks_per_beat=ticks_per_beat,
        velocity=velocity,
        program=program,
    )
    seq = csv_to_midi_sequence(csv_path, midi_col=midi_col, rest_policy=rest_policy)
    return write_midi_from_mono_sequence(seq, out_mid_path, config=cfg)
