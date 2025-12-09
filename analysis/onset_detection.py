import librosa
import numpy as np
from settings import HOP_LENGTH

def compute_onsets(y, sr, hop_length=HOP_LENGTH, min_sep=0.1):
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=hop_length,
        backtrack=True,
        pre_max=20, post_max=20,
        pre_avg=50, post_avg=50,
        delta=0.2
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    # filter minimum jarak antar onset
    filtered = []
    last = -999
    for t in onset_times:
        if t - last >= min_sep:
            filtered.append(t)
            last = t
    return np.array(filtered)
