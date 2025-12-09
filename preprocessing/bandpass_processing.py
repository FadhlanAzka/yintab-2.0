# preprocessing/bandpass_processing.py
import numpy as np
from scipy.signal import butter, sosfiltfilt

_EPS = 1e-6

def _safe_norm_freq(edge_hz, sr):
    nyq = 0.5 * sr
    return float(np.clip(edge_hz / nyq, _EPS, 1.0 - _EPS))

def apply_bandpass(y: np.ndarray, sr: int, low_hz: float, high_hz: float) -> np.ndarray:
    """
    Stabil bandpass: SOS + clamp edge supaya tidak nabrak Nyquist.
    """
    if low_hz >= high_hz:
        return y
    low = _safe_norm_freq(low_hz, sr)
    high = _safe_norm_freq(high_hz, sr)
    if low >= high:
        return y
    # order moderat biar sosfiltfilt stabil
    sos = butter(4, [low, high], btype='band', output='sos')
    try:
        y_f = sosfiltfilt(sos, y).astype(y.dtype, copy=False)
    except Exception:
        # fallback aman: jangan bikin NaN—kembalikan input
        return y
    return y_f
