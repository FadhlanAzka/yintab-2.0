# analysis/noise_filter.py
import numpy as np
import librosa

# default percentile
RMS_NOISE_PERCENTILE = 20


def apply_rms_noise_filter(y, sr, f0, hop_length, frame_length=2048, percentile=RMS_NOISE_PERCENTILE):
    """
    Buang frame dengan RMS rendah (noise atau silence) dari f0 array.
    """
    if f0 is None or len(f0) == 0:
        return f0

    rms = librosa.feature.rms(
        y=y, frame_length=frame_length, hop_length=hop_length, center=True
    )[0]
    rms = librosa.util.fix_length(rms, size=len(f0))
    rms_threshold = np.percentile(rms, percentile)

    f0_filtered = f0.copy()
    f0_filtered[rms < rms_threshold] = np.nan

    return f0_filtered