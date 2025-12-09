import numpy as np
import librosa
from settings import HOP_LENGTH

def compute_beats(y, sr):
    # pastikan finite
    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    if np.max(np.abs(y)) < 1e-6:
        # kalau diam, kembalikan tempo 0 dan tanpa beat
        return 0.0, np.asarray([], dtype=float)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    try:
        tempo = float(np.asarray(tempo).item())
    except Exception:
        tempo = float(np.mean(np.atleast_1d(tempo)))
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP_LENGTH)
    return tempo, beat_times
