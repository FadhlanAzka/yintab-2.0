# analysis/pitch_stabilizer.py
import numpy as np
import librosa

# >>> Added: sharp-only helpers
from evaluation.music_theory import midi_to_note_sharp, normalize_note_to_sharp

CENT_NOTE_LOCK = 12             # ≤12 cent dianggap note sama
CENT_BOUNDARY_CORRECTION = 18   # ≤18 cent → koreksi balik


def stabilize_pitch(sample_times, sample_f0, cent_lock=CENT_NOTE_LOCK, cent_boundary=CENT_BOUNDARY_CORRECTION, show_debug=False):
    """
    Stabilkan pitch dengan aturan cent.
    Input: sample_times, sample_f0 (Hz)
    Output: stable_notes, stable_f0, stable_midi
    """
    stable_notes, stable_f0, stable_midi = [], [], []
    prev_note, prev_midi = None, None

    for t, f_hz in zip(sample_times, sample_f0):
        if np.isnan(f_hz) or f_hz <= 0:
            stable_notes.append("N/A")
            stable_f0.append(np.nan)
            stable_midi.append(np.nan)
            continue

        midi_exact = librosa.hz_to_midi(f_hz)

        if not np.isfinite(midi_exact):  # filter inf atau NaN
            stable_notes.append("N/A")
            stable_f0.append(np.nan)
            stable_midi.append(np.nan)
            continue

        # >>> Changed: force sharp with ASCII '#'
        note_label = midi_to_note_sharp(midi_exact)

        if prev_note is None:
            stable_notes.append(note_label)
            stable_f0.append(f_hz)
            stable_midi.append(midi_exact)
            prev_note, prev_midi = note_label, midi_exact
            continue

        diff_cents = (midi_exact - prev_midi) * 100.0

        if note_label == prev_note:
            # sama dengan sebelumnya
            stable_notes.append(note_label)
            stable_f0.append(f_hz)
            stable_midi.append(midi_exact)
            prev_midi = midi_exact
        else:
            if abs(diff_cents) <= cent_boundary:
                if show_debug:
                    print(f"[CORRECT] {note_label} -> {prev_note} (Δ {diff_cents:.1f} cent)")
                stable_notes.append(prev_note)
                stable_f0.append(f_hz)
                # update prev_midi dengan sedikit adaptasi
                weight = 0.3
                prev_midi = (1 - weight) * prev_midi + weight * midi_exact
                stable_midi.append(prev_midi)
            else:
                # terima note baru
                stable_notes.append(note_label)
                stable_f0.append(f_hz)
                stable_midi.append(midi_exact)
                prev_note, prev_midi = note_label, midi_exact

    # >>> Final guard: normalize any leftover to sharp with '#'
    stable_notes = [normalize_note_to_sharp(n) if isinstance(n, str) else n for n in stable_notes]

    return (
        np.array(stable_notes, dtype=object),
        np.array(stable_f0),
        np.array(stable_midi),
    )
