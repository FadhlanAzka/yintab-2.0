# -*- coding: utf-8 -*-
"""
app.py — LSTM Mapper Inference

Single WAV → Pipeline #14 (YIN) → LSTM mapper
→ CSV (hz, note, midi, string, fret, token_idx) + ASCII Tab + Visualization PNG.
"""

from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

# Tkinter (file pickers)
import tkinter as tk
from tkinter import filedialog

# Project modules
from audio_loader import load_audio
from preprocessing.hpss_processing import apply_hpss
from analysis.yin_pitch import compute_yin
from analysis.yin_postproc import postprocess_yin
from analysis.noise_filter import apply_rms_noise_filter
from analysis.block_sampler import block_sample_pitch
from analysis.pitch_stabilizer import stabilize_pitch
from analysis.beat_tracking import compute_beats
from analysis.onset_detection import compute_onsets
from utils.sanitize import ensure_finite
from settings import HOP_LENGTH

# --- Music theory helpers (sharp-only) ---
try:
    from evaluation.music_theory import midi_to_note_sharp, normalize_note_to_sharp
except Exception:
    SHARP_NAMES = [
        "C", "C#", "D", "D#", "E", "F",
        "F#", "G", "G#", "A", "A#", "B"
    ]

    def midi_to_note_sharp(midi: float) -> str:
        if not np.isfinite(midi):
            return ""
        m = int(round(midi))
        octave = (m // 12) - 1
        name = SHARP_NAMES[m % 12]
        return f"{name}{octave}"

    def normalize_note_to_sharp(x: str) -> str:
        return x


# =========================
#  Config
# =========================
MIN_RUN_LEN = 2  # minimal panjang run agar dianggap sustain (>= 2 blok 50ms = >= 100ms)


# =========================
#  Helpers (pitch / dataframe)
# =========================
def _build_dataframe_from_stable(
    stable_f0: np.ndarray,
    stable_notes: np.ndarray,
    stable_midi: np.ndarray,
) -> pd.DataFrame:
    """Bangun DataFrame (hz, note, midi) dari hasil stabilisasi (sharp-only)."""
    notes = np.array(
        [normalize_note_to_sharp(n) if isinstance(n, str) else "" for n in stable_notes],
        dtype=object,
    )
    df = pd.DataFrame({
        "hz": stable_f0.astype(float),
        "note": notes,
        "midi": stable_midi.astype(float),
    })
    mask = np.isfinite(df["midi"]) & (df["note"] != "N/A")
    return df[mask].reset_index(drop=True)


def _collapse_to_sustains(
    df: pd.DataFrame,
    min_run: int = MIN_RUN_LEN,
) -> pd.DataFrame:
    """
    RLE pada kolom 'midi' (dibulatkan) untuk menggabungkan sustain dan
    menghapus nada single-blip (run length == 1).

    Output satu baris per run:
      - hz   -> median pada run (stabil)
      - midi -> median dibulatkan (robust)
      - note -> dari midi median (sharp)
    """
    if len(df) == 0:
        return df

    midi_int = pd.to_numeric(df["midi"], errors="coerce").round().astype("Int64")
    boundary = midi_int.ne(midi_int.shift(1))
    run_id = boundary.cumsum()

    df_tmp = df.copy()
    df_tmp["run_id"] = run_id.values

    lens = df_tmp.groupby("run_id", sort=False).size()
    valid_ids = lens[lens >= int(min_run)].index
    if len(valid_ids) == 0:
        return pd.DataFrame(columns=["hz", "note", "midi"])

    df_valid = df_tmp[df_tmp["run_id"].isin(valid_ids)]

    agg = (
        df_valid
        .assign(
            hz_num=pd.to_numeric(df_valid["hz"], errors="coerce"),
            midi_num=pd.to_numeric(df_valid["midi"], errors="coerce"),
        )
        .groupby("run_id", sort=False)
        .agg(
            hz=("hz_num", lambda s: float(np.nanmedian(s))),
            midi=("midi_num", lambda s: float(np.nanmedian(s))),
        )
        .reset_index(drop=True)
    )

    agg["midi"] = agg["midi"].round()
    agg["note"] = agg["midi"].apply(midi_to_note_sharp)
    return agg[["hz", "note", "midi"]]


def _save_basic_visualization(
    out_png: Path,
    audio_path: Path,
    sr: int,
    times: np.ndarray,
    f0_for_plot: np.ndarray,
    notes_for_plot: np.ndarray,
    beat_times: np.ndarray,
    onset_times: np.ndarray,
) -> None:
    """
    Simpan plot sederhana f0 vs waktu + garis beat & onset.
    Dipakai jika modul viz tidak menyediakan save_visualization.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 4), dpi=120)
    ax = plt.gca()

    ax.plot(times, f0_for_plot, linewidth=1.25)
    ax.scatter(times, f0_for_plot, s=8)

    if beat_times is not None and len(beat_times):
        for t in beat_times:
            ax.axvline(t, alpha=0.15)
    if onset_times is not None and len(onset_times):
        for t in onset_times:
            ax.axvline(t, alpha=0.15)

    ax.set_title(f"YINTab Visualization — {audio_path.name}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("F0 (Hz)")
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


# =========================
#  Core Pipeline #14
# =========================
def run_pipeline14(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fixed recipe (HPSS harmonic) + YIN + noise filter + postprocess
    + block-sample + cent-stabilize.

    Return:
        stable_notes, stable_f0, stable_midi, sample_times
    """
    # 1) HPSS (ambil harmonic track); kompatibel (y, sr) atau (y)
    try:
        hpss_out = apply_hpss(y, sr)
    except TypeError:
        hpss_out = apply_hpss(y)
    y_harm = hpss_out[0] if isinstance(hpss_out, (list, tuple)) else hpss_out

    # 2) YIN
    f0_raw, times, _ = compute_yin(y_harm, sr)

    # 3) Noise filter (RMS adaptif) pada kontur f0
    f0_nf = apply_rms_noise_filter(y_harm, sr, f0_raw, hop_length=HOP_LENGTH)

    # 4) Post-process kontur
    f0_corr = postprocess_yin(
        f0_nf,
        y_harm,
        sr,
        n_fft=4096,
        hop_length=HOP_LENGTH,
        median_kernel=5,
        max_harm=4,
        continuity_weight=0.8,
    )

    # 5) Block-sampling tiap 50 ms
    sample_times, sample_f0 = block_sample_pitch(
        times,
        f0_corr,
        block_ms=50,
        sr=sr,
        hop_length=HOP_LENGTH,
    )

    # 6) Stabilization (cent) => notes (sharp), f0 stabil, midi
    stable_notes, stable_f0, stable_midi = stabilize_pitch(sample_times, sample_f0)
    return stable_notes, stable_f0, stable_midi, sample_times


# =========================
#  UI Pickers
# =========================
def pick_wav_file() -> Optional[Path]:
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Pilih file WAV",
        filetypes=[("WAV file", "*.wav")],
    )
    root.destroy()
    return Path(path) if path else None


def pick_output_folder() -> Optional[Path]:
    root = tk.Tk()
    root.withdraw()
    base = filedialog.askdirectory(
        title="Pilih folder output (akan dibuat subfolder nama WAV)",
    )
    root.destroy()
    return Path(base) if base else None


def pick_token_index_csv() -> Optional[Path]:
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Pilih file token index.csv",
        filetypes=[("CSV file", "*.csv"), ("All files", "*.*")],
    )
    root.destroy()
    return Path(path) if path else None


# =========================
#  ASCII Tab Helpers
# =========================
def render_ascii_tab(strings, frets, width_pad: int = 1) -> str:
    """
    Render tablature ASCII dari dua urutan:
      - strings: list/array nomor senar (1..6)
      - frets:   list/array nomor fret (0..24 atau lebih, multi-digit OK)

    width_pad: jumlah "-" pemisah antar kolom (minimal 1).
    """
    string_labels = {1: "e", 2: "B", 3: "G", 4: "D", 5: "A", 6: "E"}
    lines = {s: [] for s in string_labels}
    width_pad = max(1, int(width_pad))

    if len(strings) == 0 or len(frets) == 0:
        return "\n".join(f"{name}||" for _, name in sorted(string_labels.items()))

    if len(strings) != len(frets):
        raise ValueError(
            f"Panjang strings ({len(strings)}) dan frets ({len(frets)}) tidak sama."
        )

    for s_val, f_val in zip(strings, frets):
        # Tambah satu kolom kosong dulu untuk semua senar
        for s in string_labels:
            lines[s].append("-")

        try:
            s_idx = int(s_val)
            fret = int(f_val)
        except Exception:
            s_idx, fret = None, None

        if (
            s_idx in string_labels
            and fret is not None
            and np.isfinite(fret)
            and fret >= 0
        ):
            mark = str(int(fret))
            lines[s_idx][-1] = mark[0]

            if len(mark) > 1:
                for extra_digit in mark[1:]:
                    for ss in string_labels:
                        lines[ss].append("-")
                    lines[s_idx][-1] = extra_digit

        for s in string_labels:
            for _ in range(width_pad):
                lines[s].append("-")

    out_lines = []
    for s in sorted(string_labels.keys()):
        name = string_labels[s]
        out_lines.append(f"{name}|{''.join(lines[s])}|")
    return "\n".join(out_lines)


# =====================
# Device & file pickers
# =====================
def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _pick_lstm_model() -> Optional[Path]:
    """
    Dialog Tkinter untuk memilih file model LSTM
    (.pt / .ts.pt / .jit) hasil trainer.
    """
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Pilih file model LSTM (best.pt / best.ts.pt / best.jit)",
        filetypes=[("PyTorch model", "*.pt *.jit"), ("All files", "*.*")],
    )
    root.destroy()
    return Path(path) if path else None


# ======================
# Token index / model IO
# ======================
def _load_token_index_df(csv_path: Path) -> pd.DataFrame:
    """
    Load token index v2:
      - wajib punya kolom: token_idx, string, fret
    """
    df_tok = pd.read_csv(csv_path)

    if "token_idx" not in df_tok.columns:
        raise ValueError(
            f"token index CSV '{csv_path}' tidak memiliki kolom 'token_idx'."
        )

    if "string" not in df_tok.columns or "fret" not in df_tok.columns:
        raise ValueError(
            f"token index CSV '{csv_path}' harus memiliki kolom 'string' dan 'fret'."
        )

    if "midi" not in df_tok.columns:
        print(
            "[WARN] Kolom 'midi' tidak ditemukan di token index; "
        )

    return df_tok.set_index("token_idx")


def _load_lstm_model(model_path: Path, device: str):
    """
    Load model LSTM untuk inference.

    Prioritas:
    1) TorchScript (.ts.pt / .jit) via torch.jit.load
    2) Full model .pt via torch.load (hasil torch.save(model))

    Catatan:
    - Checkpoint yang berisi dict dengan 'model_state_dict'
      TIDAK didukung di sini.
    """
    path_str = str(model_path)
    lower = path_str.lower()

    if lower.endswith(".ts.pt") or lower.endswith(".jit"):
        model = torch.jit.load(path_str, map_location=device)
        model.eval()
        return model

    obj = torch.load(path_str, map_location=device)

    if hasattr(obj, "forward"):
        model = obj
    elif isinstance(obj, dict) and "model_state_dict" in obj:
        raise RuntimeError(
            "File ini tampaknya checkpoint (dict dengan 'model_state_dict'). "
            "Untuk inference, gunakan file .pt (full model) atau .ts.pt / .jit (TorchScript)."
        )
    else:
        raise RuntimeError(f"Tidak bisa interpret file model: {model_path}")

    model.eval()
    return model


# ===========================
# Utilities
# ===========================
def _build_midi_to_mask(
    token_index_df: pd.DataFrame,
    num_classes: int,
    device: str,
) -> Dict[int, torch.BoolTensor]:
    """
    Dari token_index_df (indexed by token_idx) bangun:
      midi_to_mask[midi] = BoolTensor [C] yang True di token_idx yang boleh.
    """
    if "midi" not in token_index_df.columns:
        return {}

    midi_to_mask: Dict[int, torch.BoolTensor] = {}

    for token_idx, row in token_index_df.iterrows():
        try:
            midi_val = int(row["midi"])
        except Exception:
            continue

        if midi_val not in midi_to_mask:
            mask = torch.zeros(num_classes, dtype=torch.bool, device=device)
            midi_to_mask[midi_val] = mask

        idx = int(token_idx)
        if 0 <= idx < num_classes:
            midi_to_mask[midi_val][idx] = True

    return midi_to_mask


def _apply_pitch_mask_for_sequence(
    logits: torch.Tensor,          # [B, T, C]
    midi_seq: np.ndarray,          # [T] ints
    midi_to_mask: Dict[int, torch.BoolTensor],
) -> torch.Tensor:
    """
    Terapkan pitch mask ke logits satu sequence.
    Untuk tiap timestep t:
      - ambil midi_val = midi_seq[t]
      - jika ada mask untuk midi_val, paksa logits[:, t, token_idx yg tidak allowed] → -1e9.
    """
    if not midi_to_mask:
        return logits

    if logits.dim() == 2:
        logits = logits.unsqueeze(0)  # [1, T, C]

    B, T, C = logits.shape
    if T != len(midi_seq):
        print(
            f"[WARN] Panjang midi_seq ({len(midi_seq)}) "
            f"≠ panjang logits ({T}); pitch mask mungkin tidak konsisten."
        )

    logit_mask_value = -1e9

    for t in range(min(T, len(midi_seq))):
        midi_val = int(midi_seq[t])
        mask_1d = midi_to_mask.get(midi_val, None)
        if mask_1d is None:
            continue

        logits[:, t, ~mask_1d] = logit_mask_value

    return logits


# ==============================
# LSTM Mapping
# ==============================
def _run_lstm_mapping(
    df_notes: pd.DataFrame,
    model,
    token_index_df: pd.DataFrame,
    device: str,
    use_pitch_mask: bool = True,
) -> pd.DataFrame:
    """
    df_notes : DataFrame collapse sustain dengan kolom minimal ['hz', 'note', 'midi'].
    model    : LSTM mapper yang mengembalikan logits [B, T, C] untuk token_idx.
    token_index_df : DataFrame indexed by token_idx, dengan kolom:
                     'string', 'fret', dan idealnya 'midi'.
    device  : 'cpu' / 'cuda'
    use_pitch_mask : jika True, terapkan rule-based pitch mask (jika kolom 'midi' ada).

    Return:
      df_out = df_notes + kolom 'token_idx', 'string', 'fret'.
    """
    if len(df_notes) == 0:
        return df_notes.copy()

    df_notes = df_notes.reset_index(drop=True)
    midi_seq = df_notes["midi"].round().astype(int).to_numpy(dtype="int64")

    with torch.no_grad():
        midi_tensor = torch.from_numpy(midi_seq)[None, :]
        midi_tensor = midi_tensor.to(device)

        logits = model(midi_tensor)  # [1, T, C] atau [T, C]

        if logits.dim() == 2:
            logits = logits.unsqueeze(0)

        B, T, C = logits.shape

        if use_pitch_mask and ("midi" in token_index_df.columns):
            midi_to_mask = _build_midi_to_mask(
                token_index_df=token_index_df,
                num_classes=C,
                device=logits.device,
            )
            if not midi_to_mask:
                print(
                    "[WARN] midi_to_mask kosong; kolom 'midi' mungkin bermasalah. "
                    "Logits dipakai tanpa pitch mask."
                )
            else:
                logits = _apply_pitch_mask_for_sequence(
                    logits=logits,
                    midi_seq=midi_seq,
                    midi_to_mask=midi_to_mask,
                )
        else:
            if use_pitch_mask:
                print(
                    "[WARN] Kolom 'midi' tidak tersedia di token_index_df; "
                )

        preds = logits.argmax(dim=-1)
        preds = preds.squeeze(0).cpu().numpy()

    if len(preds) != len(df_notes):
        raise RuntimeError(
            f"Panjang prediksi ({len(preds)}) tidak sama "
            f"dengan panjang df_notes ({len(df_notes)})."
        )

    df_out = df_notes.copy()
    df_out["token_idx"] = preds.astype(int)

    df_out["string"] = df_out["token_idx"].map(token_index_df["string"])
    df_out["fret"] = df_out["token_idx"].map(token_index_df["fret"])

    return df_out


# ===============================
# End-to-End Single-file Inference
# ===============================
def run_app() -> None:
    """
    Global inference:
      WAV → YIN Pipeline #14 → LSTM → CSV (hz, note, midi, string, fret, token_idx) + ASCII tab + Visualization.
    """
    # 1) Pilih WAV
    wav_path = pick_wav_file()
    if not wav_path:
        print("Dibatalkan: tidak ada WAV dipilih.")
        return

    # 2) Pilih folder output
    base_out = pick_output_folder()
    if not base_out:
        print("Dibatalkan: tidak ada folder output dipilih.")
        return

    out_dir = base_out / wav_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3) Pilih token index
    tok_csv = pick_token_index_csv()
    if not tok_csv:
        print("Dibatalkan: tidak ada token index.csv dipilih.")
        return

    token_index_df = _load_token_index_df(tok_csv)

    # 4) Pilih model LSTM
    model_path = _pick_lstm_model()
    if not model_path:
        print("Dibatalkan: tidak ada model LSTM dipilih.")
        return

    device = _detect_device()
    print(f"[INFO] Device: {device}")
    model = _load_lstm_model(model_path, device=device)
    print(f"[INFO] Model loaded from: {model_path}")

    # 5) Load audio
    print(f"[INFO] Memuat audio: {wav_path}")
    y, sr = load_audio(str(wav_path))
    y = ensure_finite(y)

    # 6) Beats & onsets (untuk visualisasi)
    try:
        tempo, beat_times = compute_beats(y, sr)
    except Exception as e:
        print(f"[WARN] compute_beats gagal: {e}")
        tempo, beat_times = None, np.asarray([])

    try:
        onset_times = compute_onsets(y, sr)
    except Exception as e:
        print(f"[WARN] compute_onsets gagal: {e}")
        onset_times = np.asarray([])

    # 7) Jalankan Pipeline14 (HPSS + YIN + NF + postproc + block-sample + cent-stab)
    print("[INFO] Menjalankan Pipeline #14 (YIN)...")
    stable_notes, stable_f0, stable_midi, sample_times = run_pipeline14(y, sr)

    # 8) Visualisasi (show + save_visualization + fallback basic PNG)
    duration = len(y) / sr
    f0_vis = stable_f0
    notes_vis = np.asarray(
        [normalize_note_to_sharp(n) if isinstance(n, str) else n for n in stable_notes],
        dtype=object,
    )

    try:
        from viz.visualizer import show_visualization, save_visualization
    except Exception:
        from visualizer import show_visualization, save_visualization

    # Player interaktif
    try:
        show_visualization(
            audio_path=str(wav_path),
            duration=duration,
            sr=sr,
            tempo=tempo,
            beat_times=beat_times,
            onset_times=onset_times,
            times=sample_times,
            f0=f0_vis,
            notes=notes_vis,
        )
    except Exception as e:
        print(f"[WARN] show_visualization gagal: {e}")

    # Simpan PNG via save_visualization, dengan fallback ke _save_basic_visualization
    try:
        save_visualization(out_dir)(
            audio_path=str(wav_path),
            duration=duration,
            sr=sr,
            tempo=tempo,
            beat_times=beat_times,
            onset_times=onset_times,
            times=sample_times,
            f0=f0_vis,
            notes=notes_vis,
        )
        print(f"[OK] Visualization disimpan ke folder: {out_dir}")
    except Exception as e:
        out_png = out_dir / f"{wav_path.stem}_viz.png"
        _save_basic_visualization(
            out_png=out_png,
            audio_path=wav_path,
            sr=sr,
            times=sample_times,
            f0_for_plot=f0_vis,
            notes_for_plot=notes_vis,
            beat_times=np.asarray(beat_times) if beat_times is not None else np.asarray([]),
            onset_times=np.asarray(onset_times),
        )
        print(f"[WARN] Modul viz gagal, pakai fallback sederhana. Disimpan: {out_png} ({e})")

    # 9) DataFrame awal (hz, note, midi)
    df0 = _build_dataframe_from_stable(stable_f0, stable_notes, stable_midi)
    if len(df0) == 0:
        print(f"[SKIP] {wav_path.name}: no valid notes")
        out_csv = out_dir / f"{wav_path.stem}_lstm_mapped.csv"
        pd.DataFrame(
            columns=["hz", "note", "midi", "string", "fret", "token_idx"],
        ).to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[OK] Empty CSV disimpan ke: {out_csv}")
        return

    # 10) Collapse to sustains (RLE)
    df = _collapse_to_sustains(df0)
    if len(df) == 0:
        print(f"[SKIP] {wav_path.name}: no sustained notes after RLE")
        out_csv = out_dir / f"{wav_path.stem}_lstm_mapped.csv"
        pd.DataFrame(
            columns=["hz", "note", "midi", "string", "fret", "token_idx"],
        ).to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[OK] Empty CSV disimpan ke: {out_csv}")
        return

    # 11) LSTM mapping
    print("[INFO] Menjalankan LSTM Mapper...")
    df_out = _run_lstm_mapping(
        df_notes=df,
        model=model,
        token_index_df=token_index_df,
        device=device,
        use_pitch_mask=True,  # tetap True; jika token index tidak punya 'midi', otomatis skip
    )

    # 12) Save CSV
    cols = ["hz", "note", "midi", "string", "fret", "token_idx"]
    cols = [c for c in cols if c in df_out.columns]
    out_csv = out_dir / f"{wav_path.stem}_lstm_mapped.csv"
    df_out[cols].to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] CSV LSTM-mapped disimpan: {out_csv}")

    # 13) ASCII tablature dari kolom string & fret
    if {"string", "fret"}.issubset(df_out.columns):
        try:
            strings = df_out["string"].tolist()
            frets = df_out["fret"].tolist()
            tab_txt = render_ascii_tab(strings, frets, width_pad=1)
            out_tab = out_dir / f"{wav_path.stem}_lstm_tab.txt"
            with open(out_tab, "w", encoding="utf-8") as f:
                f.write(tab_txt)
            print(f"[OK] ASCII tablature LSTM disimpan: {out_tab}")
        except Exception as e:
            print(f"[WARN] Gagal membuat ASCII tablature LSTM: {e}")
    else:
        print(
            "[WARN] Kolom 'string' dan/atau 'fret' tidak ditemukan — ASCII tab tidak dibuat."
        )

    print("[DONE] Inference LSTM Mapper selesai.")


def main():
    run_app()


if __name__ == "__main__":
    main()
