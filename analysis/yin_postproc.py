# analysis/yin_postproc.py
import numpy as np
import librosa
from scipy.signal import medfilt
from typing import Tuple

# PARAM (sesuaikan)
MEDIAN_KERNEL = 5           # smoothing awal (ganjil)
STFT_N_FFT = 4096
STFT_HOP = 512
ENERGY_BAND_BINS = 4        # band +/- bins around freq bin to sum energy
CONTINUITY_WEIGHT = 0.8     # 0..1, seberapa kuat prior kontinuitas (1 = sangat kuat)
MIN_FREQ = librosa.note_to_hz('E2')  # ~82.4
MAX_HARM = 4                # cek subharmonic sampai f/4

def _hz_to_bin(hz: float, n_fft: int, sr: int) -> int:
    return int(np.round(hz * n_fft / sr))

def _frame_spectral_energy_at_freqs(S_mag: np.ndarray, freq_list_hz, n_fft, sr, band_bins=ENERGY_BAND_BINS):
    """
    S_mag: shape (freq_bins, frames)
    freq_list_hz: list of freqs (length = #candidates)
    returns energy array shape (len(freq_list_hz), frames)
    """
    nbins, nframes = S_mag.shape
    out = np.zeros((len(freq_list_hz), nframes), dtype=float)
    for i, hz in enumerate(freq_list_hz):
        if np.isnan(hz) or hz <= 0:
            continue
        bin_idx = _hz_to_bin(hz, n_fft, sr)
        lo = max(0, bin_idx - band_bins)
        hi = min(nbins - 1, bin_idx + band_bins)
        out[i, :] = S_mag[lo:hi+1, :].sum(axis=0)
    return out

def postprocess_yin(f0_hz: np.ndarray, y: np.ndarray, sr: int,
                    n_fft: int = STFT_N_FFT, hop_length: int = STFT_HOP,
                    median_kernel: int = MEDIAN_KERNEL,
                    max_harm: int = MAX_HARM,
                    continuity_weight: float = CONTINUITY_WEIGHT
                   ) -> np.ndarray:
    """
    Input:
      - f0_hz: output dari librosa.yin (shape (T,))
      - y, sr: original signal (dipakai untuk STFT energy)
    Return:
      - f0_corrected: array same shape with octave-corrections applied
    """
    # 1) smoothing awal (mengurangi spikes)
    if median_kernel is not None and median_kernel >= 3 and median_kernel % 2 == 1:
        f0_s = medfilt(f0_hz, kernel_size=median_kernel)
    else:
        f0_s = f0_hz.copy()

    # 2) compute STFT magnitude (magnitude spectrum)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=True))
    # Align length: yin frames might differ from STFT frames if hop differs.
    # We will map yin-frame index -> stft frame index by time.
    yin_frames = len(f0_s)
    stft_frames = S.shape[1]
    # times
    yin_times = librosa.frames_to_time(np.arange(yin_frames), sr=sr, hop_length=hop_length)
    stft_times = librosa.frames_to_time(np.arange(stft_frames), sr=sr, hop_length=hop_length)
    # map indices: for each yin frame find nearest stft frame
    idx_map = np.searchsorted(stft_times, yin_times)
    idx_map = np.clip(idx_map, 0, stft_frames - 1)

    # 3) build candidates per frame: f0, f0/2, f0/3, ..., down to >= MIN_FREQ
    # We'll compute energy for each candidate across all frames (vectorized).
    # First get maximum number of candidates per frame (max_harm)
    candidate_list = []
    for h in range(1, max_harm + 1):  # h=1 => f0, h=2 => f0/2 (subharmonic)
        candidate_list.append((h))

    # prepare freq arrays per candidate (shape candidates x T)
    T = yin_frames
    candidates_hz = np.full((len(candidate_list), T), np.nan, dtype=float)
    for i, h in enumerate(candidate_list):
        vals = f0_s / float(h)
        # set invalid if below MIN_FREQ or NaN
        vals[~np.isfinite(vals)] = np.nan
        vals[vals < MIN_FREQ] = np.nan
        candidates_hz[i, :] = vals

    # 4) compute spectral energy for each candidate at each frame (using mapped STFT frames)
    # We'll compute energy per stft frame and then pick via idx_map
    # Precompute energy per candidate frequency for all stft frames
    # But efficient: compute energy matrix for candidate freqs (per candidate) across stft frames
    # build list of unique frequencies across candidates (to avoid duplicate bin computes)
    # Simpler approach: for each candidate i compute energy vector for stft frames, then pick entries by idx_map
    cand_energy = np.zeros_like(candidates_hz)
    for i in range(candidates_hz.shape[0]):
        freqs = candidates_hz[i, :]
        # Create array of length stft_frames where energy per stft frame = energy near freq corresponding to that frame's candidate.
        # We'll compute energy per yin-frame by sampling S[:, idx_map[j]] at bin corresponding to freqs[j].
        # Vectorized: compute bin indices for all frames
        bin_idx = np.array([_hz_to_bin(f if np.isfinite(f) else 0.0, n_fft, sr) for f in freqs])
        # clamp
        bin_idx = np.clip(bin_idx, 0, S.shape[0]-1)
        # sum energy in band
        for t in range(T):
            b = bin_idx[t]
            if freqs[t] != freqs[t]:  # NaN
                cand_energy[i, t] = 0.0
                continue
            lo = max(0, b - ENERGY_BAND_BINS)
            hi = min(S.shape[0]-1, b + ENERGY_BAND_BINS)
            stf_idx = idx_map[t]
            cand_energy[i, t] = S[lo:hi+1, stf_idx].sum()

    # 5) pick candidate per frame with energy + continuity prior
    f0_corrected = np.full(T, np.nan, dtype=float)
    prev = np.nan
    for t in range(T):
        energies = cand_energy[:, t]

        # Guard: kalau semua kandidat NaN atau energi nol
        if (not np.isfinite(candidates_hz[:, t]).any()) or np.all(energies == 0):
            f0_corrected[t] = np.nan
            prev = np.nan
            continue

        # continuity
        if np.isfinite(prev):
            continuity_scores = []
            for i in range(len(candidate_list)):
                hz = candidates_hz[i, t]
                if not np.isfinite(hz):
                    continuity_scores.append(np.nan)
                else:
                    continuity_scores.append(-abs(np.log2(hz / prev)))
            continuity_scores = np.array(continuity_scores, dtype=float)

            if np.isfinite(continuity_scores).any():
                cs = (continuity_scores - np.nanmin(continuity_scores)) / (
                    np.nanmax(continuity_scores) - np.nanmin(continuity_scores) + 1e-12
                )
            else:
                cs = np.zeros_like(energies)

            final_score = (1 - continuity_weight) * (
                energies / (energies.max() + 1e-12)
            ) + continuity_weight * cs
        else:
            final_score = energies / (energies.max() + 1e-12)

        # Guard: kalau masih semua NaN
        if not np.isfinite(final_score).any():
            f0_corrected[t] = np.nan
            prev = np.nan
            continue

        best_idx = int(np.nanargmax(final_score))
        chosen_hz = candidates_hz[best_idx, t]
        f0_corrected[t] = chosen_hz
        prev = f0_corrected[t]

    # 6) final median smoothing (to remove remaining flicker)
    try:
        f0_final = medfilt(f0_corrected, kernel_size=median_kernel)
    except Exception:
        f0_final = f0_corrected

    return f0_final