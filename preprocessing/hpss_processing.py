import numpy as np
import librosa

def apply_hpss(y: np.ndarray) -> np.ndarray:
    """
    HPSS -> pilih komponen harmonic sebagai sinyal untuk pitch tracking.
    """
    y_harm, y_perc = librosa.effects.hpss(y)
    return y_harm