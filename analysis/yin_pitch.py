# analysis/yin_pitch.py
import numpy as np
import librosa
from settings import FRAME_LENGTH, HOP_LENGTH, FMIN, FMAX

# >>> Added: sharp-only converter
from evaluation.music_theory import midi_to_note_sharp


def compute_yin(y, sr):
    """
    Compute YIN-based F0 estimation.
    Returns:
        f0: np.ndarray of fundamental frequency (Hz)
        times: np.ndarray of timestamps (sec)
        notes: np.ndarray of sharp-only note labels (e.g., F#3)
    """
    f0 = librosa.yin(
        y=y,
        fmin=FMIN,
        fmax=FMAX,
        sr=sr,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )

    times = librosa.frames_to_time(
        np.arange(len(f0)), sr=sr, hop_length=HOP_LENGTH
    )

    midi = librosa.hz_to_midi(f0)

    # >>> Force sharp-only labels with ASCII '#'
    notes = np.array(
        ["N/A" if np.isnan(m) else midi_to_note_sharp(m) for m in midi],
        dtype=object
    )

    return f0, times, notes
