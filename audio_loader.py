import librosa
import numpy as np
from typing import Tuple

def load_audio(path: str, sr: int | None = None, mono: bool = True) -> Tuple[np.ndarray, int]:
    """
    Load audio file with librosa.
    Returns (y, sr)
    """
    y, sr = librosa.load(path, sr=sr, mono=mono)
    return y, sr
