# utils/sanitize.py
import numpy as np

def ensure_finite(y):
    """
    Pastikan audio finite (tanpa NaN/Inf), cast ke float32, dan hilangkan DC offset.
    """
    if y is None:
        return None
    y = np.asarray(y, dtype=np.float32)
    # ganti NaN/Inf -> 0
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    # hilangkan DC offset ringan
    y = y - float(np.mean(y))
    # normalisasi aman (opsional)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0 and not np.isinf(peak) and not np.isnan(peak):
        y = y / peak * 0.9
    return y

def is_silent(y, thresh=1e-6):
    if y is None or y.size == 0:
        return True
    return float(np.max(np.abs(y))) < thresh
