# analysis/block_sampler.py
import numpy as np
import librosa


def block_sample_pitch(times, f0, block_ms=50, sr=44100, hop_length=512):
    """
    Sampling pitch per block waktu (ms). Ambil median pitch di blok tersebut.
    Return: sample_times, sample_f0
    """
    if f0 is None or len(f0) == 0:
        return np.array([]), np.array([])

    duration_sec = times[-1] if len(times) > 0 else 0.0
    block_len_s = block_ms / 1000.0
    sample_block_starts = np.arange(0, duration_sec, block_len_s)

    sample_times, sample_f0 = [], []
    for t0 in sample_block_starts:
        t1 = t0 + block_len_s
        idx = np.where((times >= t0) & (times < t1))[0]
        if len(idx) == 0:
            sample_times.append(t0)
            sample_f0.append(np.nan)
            continue

        block_vals = f0[idx]
        block_vals = block_vals[~np.isnan(block_vals)]
        if len(block_vals) == 0:
            sample_times.append(t0)
            sample_f0.append(np.nan)
            continue

        median_f = np.median(block_vals)
        sample_times.append(t0)
        sample_f0.append(median_f)

    return np.array(sample_times), np.array(sample_f0)
