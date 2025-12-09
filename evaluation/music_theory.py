# music_theory.py
"""
Utility functions for music pitch conversions and normalization.
Part of YIN Evaluation Pipeline.

Functions:
- hz_to_midi_safe
- midi_to_hz_safe
- hz_to_cent_diff
- midi_to_chroma
- normalize_note_name
- note_to_midi
- midi_to_note_sharp        [sharp-only string with '#']
- hz_to_note_sharp          [Hz → sharp-only string with '#']
- normalize_note_to_sharp   [any input → sharp-only with '#']
"""

import numpy as np
import re
import math

# Base pitch-class map (sharp-only)
NOTE_TO_MIDI = {
    "C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
    "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11
}

# Sharp-only pitch classes
SHARP_CLASSES = ["C", "C#", "D", "D#", "E", "F",
                 "F#", "G", "G#", "A", "A#", "B"]


def hz_to_midi_safe(hz: float | np.ndarray) -> np.ndarray:
    """Convert Hz to MIDI pitch, safely handling zeros, NaNs, and negatives."""
    hz = np.asarray(hz, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        midi = 69 + 12 * np.log2(hz / 440.0)
    midi[~np.isfinite(midi)] = np.nan
    midi[midi <= 0] = np.nan
    return midi


def midi_to_hz_safe(midi: float | np.ndarray) -> np.ndarray:
    """Convert MIDI pitch to frequency (Hz)."""
    midi = np.asarray(midi, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        hz = 440.0 * np.power(2.0, (midi - 69) / 12.0)
    hz[~np.isfinite(hz)] = np.nan
    return hz


def hz_to_cent_diff(hz_pred: float, hz_true: float) -> float:
    """Compute cent difference between predicted and true frequencies."""
    if hz_pred <= 0 or hz_true <= 0:
        return np.nan
    return 1200 * math.log2(hz_pred / hz_true)


def midi_to_chroma(midi: float | np.ndarray) -> np.ndarray:
    """Return chroma (MIDI mod 12)."""
    midi = np.asarray(midi, dtype=float)
    chroma = np.mod(np.round(midi), 12)
    chroma[~np.isfinite(chroma)] = np.nan
    return chroma


# ----------------------------
# Normalization of note names
# ----------------------------

def _fix_common_mojibake(s: str) -> str:
    """
    Fix common mojibake and full-width forms:
    - 'â™¯' -> '#', 'â™­' -> 'b'
    - '♯' -> '#', '♭' -> 'b'
    - '＃' (full-width #) -> '#'
    - 'ｂ' (full-width b) -> 'b'
    """
    return (s.replace("â™¯", "#")
             .replace("â™­", "b")
             .replace("♯", "#")
             .replace("♭", "b")
             .replace("＃", "#")
             .replace("ｂ", "b"))


def normalize_note_name(note: str) -> str:
    """
    Normalize note names into a clean ASCII pattern:
    - Fix mojibake/unicode/full-width (#/b) → '#' / 'b'
    - Keep only [A-G][#|b]?[octave], remove spaces/others
    - Uppercase letters (so 'b' → 'B' meaning flat in regex)
    Returns 'N/A' if empty/invalid.
    """
    if not isinstance(note, str):
        return "N/A"
    note = _fix_common_mojibake(note)
    note = note.strip()
    # Allow minus sign for negative octaves
    note = re.sub(r"[^A-Ga-g#b0-9\-]", "", note)
    if not note:
        return "N/A"
    note = note.upper()  # now accidental 'b' becomes 'B'
    return note


def note_to_midi(note: str) -> float:
    """
    Convert a normalized note (e.g., 'C4', 'F#3', 'DB3') to MIDI.
    Assumes normalize_note_name() has been applied.
    Pattern: ^([A-G])([#B]?)(-?\d+)$
      - Group 1: base note A-G
      - Group 2: optional accidental (# for sharp, B for flat)
      - Group 3: octave (can be negative)
    Returns NaN if parsing fails.
    """
    note = normalize_note_name(note)
    match = re.match(r"^([A-G])([#B]?)(-?\d+)$", note)
    if not match:
        return np.nan
    name, accidental, octave = match.groups()
    semitone = NOTE_TO_MIDI.get(name, 0)
    if accidental == "#":
        semitone += 1
    elif accidental == "B":  # flat
        semitone -= 1
    midi = (int(octave) + 1) * 12 + semitone
    return float(midi)


# -------------------------------
# Sharp-only formatting utilities
# -------------------------------

def midi_to_note_sharp(midi: float | int) -> str:
    """
    Return sharp-only note name with octave using '#', e.g., 'F#2'.
    - Rounds MIDI to nearest int.
    - Returns 'N/A' for invalid/non-finite inputs.
    """
    try:
        if midi is None or not np.isfinite(midi):
            return "N/A"
        m = int(round(float(midi)))
    except Exception:
        return "N/A"
    name = SHARP_CLASSES[m % 12]
    octave = (m // 12) - 1
    return f"{name}{octave}"


def hz_to_note_sharp(hz: float) -> str:
    """
    Convert frequency (Hz) to sharp-only note name with '#'.
    Uses hz_to_midi_safe for robust conversion.
    Returns 'N/A' on invalid input.
    """
    try:
        if hz is None or not np.isfinite(hz) or float(hz) <= 0:
            return "N/A"
        midi = hz_to_midi_safe(hz)
        # hz_to_midi_safe returns ndarray; extract scalar
        m = float(midi) if np.ndim(midi) == 0 else float(np.asarray(midi).item())
        return midi_to_note_sharp(m)
    except Exception:
        return "N/A"


def normalize_note_to_sharp(note_str: str) -> str:
    """
    Normalize any note text to sharp-only with ASCII '#'.
    Examples:
      'F♯2' / 'Fâ™¯2' -> 'F#2'
      'Gb2'           -> 'F#2'  (enharmonic)
      'Câ™¯3'         -> 'C#3'
      '＃'/'ｂ' forms are handled as well.
    If parsing fails and it looks like Hz (e.g., '440'), it converts via Hz.
    Returns 'N/A' if cannot parse.
    """
    if not isinstance(note_str, str) or not note_str.strip():
        return "N/A"

    # Fix mojibake/full-width and sanitize
    norm = normalize_note_name(note_str)

    # Try parse as musical note (supports flats/sharps)
    m = note_to_midi(norm)
    if np.isfinite(m):
        return midi_to_note_sharp(m)

    # Fallback: try interpret as Hz
    try:
        hz_val = float(_fix_common_mojibake(note_str).strip())
        return hz_to_note_sharp(hz_val)
    except Exception:
        return "N/A"
