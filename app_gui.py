# -*- coding: utf-8 -*-
"""
app_gui.py — Tk GUI wrapper untuk app.py (YIN → LSTM → TAB)

Fungsi:
- Pilih 1 file WAV
- Pilih output folder (base), nanti dibuat subfolder /<nama_wav>
- Pilih token index CSV
- Pilih model LSTM (.pt / .jit)
- Pilih apakah mau menampilkan player interaktif (show_visualization) atau tidak
- Jalankan inference (app.py pipeline) dan tampilkan preview PNG

Jalankan:
  python app_gui.py
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import torch

# Tkinter
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# Optional PIL untuk preview PNG
try:
    from PIL import Image, ImageTk
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

# Project modules (low-level audio & viz)
from audio_loader import load_audio
from analysis.beat_tracking import compute_beats
from analysis.onset_detection import compute_onsets
from utils.sanitize import ensure_finite

# Import fungsi core dari app.py
from app import (
    run_pipeline14,
    _build_dataframe_from_stable,
    _collapse_to_sustains,
    _save_basic_visualization,
    _load_token_index_df,
    _load_lstm_model,
    _run_lstm_mapping,
    render_ascii_tab,
    _detect_device,
    normalize_note_to_sharp,
)

# visualizer (sama seperti di app.py)
try:
    from viz.visualizer import show_visualization, save_visualization
except Exception:
    from visualizer import show_visualization, save_visualization


APP_TITLE = "YINTab LSTM Mapper (GUI untuk app.py)"


# ==============================
# Util OS
# ==============================
def _safe_open_folder(p: Path):
    try:
        if sys.platform.startswith("win"):
            os.startfile(p)  # type: ignore[attr-defined]
        elif sys.platform.startswith("darwin"):
            os.system(f'open "{p}"')
        else:
            os.system(f'xdg-open "{p}"')
    except Exception:
        import webbrowser
        webbrowser.open(str(p))


# ==============================
# 1x Inference helper
# ==============================
def run_inference_once(
    wav_path: Path,
    base_out_dir: Path,
    token_index_csv: Path,
    model_path: Path,
    device_str: str = "auto",
    show_player: bool = True,
) -> Dict[str, Optional[Path]]:
    """
    Jalankan pipeline yang sama seperti run_app() di app.py,
    tetapi dengan path eksplisit (tanpa Tk dialog).

    Parameter:
      show_player: jika True → panggil show_visualization (player interaktif).
                   jika False → skip player, hanya simpan PNG.

    Return dict berisi path penting:
      - "out_dir"
      - "csv"
      - "tab"
      - "viz"
    """
    # Tentukan folder output final = base_out_dir / <nama_wav>
    if not base_out_dir:
        base_out_dir = wav_path.parent
    out_dir = base_out_dir / wav_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Token index
    token_index_df = _load_token_index_df(token_index_csv)

    # Device
    if device_str == "auto":
        device = _detect_device()
    else:
        device = device_str
    print(f"[INFO] Device (GUI): {device}")

    # Model
    model = _load_lstm_model(model_path, device=device)
    print(f"[INFO] Model loaded from: {model_path}")

    # Audio
    print(f"[INFO] Memuat audio: {wav_path}")
    y, sr = load_audio(str(wav_path))
    y = ensure_finite(y)

    # Beat / onset (untuk viz)
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

    # Pipeline14 (YIN) — sama persis dengan app.py
    print("[INFO] Menjalankan Pipeline #14 (YIN)...")
    stable_notes, stable_f0, stable_midi, sample_times = run_pipeline14(y, sr)

    # Visualisasi
    duration = len(y) / sr
    f0_vis = stable_f0
    notes_vis = np.asarray(
        [normalize_note_to_sharp(n) if isinstance(n, str) else n for n in stable_notes],
        dtype=object,
    )

    # Player interaktif: hanya jika show_player == True
    if show_player:
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
            print(f"[WARN] show_visualization gagal (GUI): {e}")
    else:
        print("[INFO] show_player=False → skip show_visualization()")

    # Simpan PNG via save_visualization, fallback ke _save_basic_visualization
    viz_path: Optional[Path] = None
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
        viz_path = out_dir / f"{wav_path.stem}_viz.png"
        print(f"[OK] Visualization disimpan ke folder: {out_dir}")
    except Exception as e:
        viz_path = out_dir / f"{wav_path.stem}_viz.png"
        _save_basic_visualization(
            out_png=viz_path,
            audio_path=wav_path,
            sr=sr,
            times=sample_times,
            f0_for_plot=f0_vis,
            notes_for_plot=notes_vis,
            beat_times=np.asarray(beat_times) if beat_times is not None else np.asarray([]),
            onset_times=np.asarray(onset_times),
        )
        print(f"[WARN] Modul viz gagal, pakai fallback sederhana. Disimpan: {viz_path} ({e})")

    # DataFrame awal
    df0 = _build_dataframe_from_stable(stable_f0, stable_notes, stable_midi)
    if len(df0) == 0:
        print(f"[SKIP] {wav_path.name}: no valid notes")
        out_csv = out_dir / f"{wav_path.stem}_lstm_mapped.csv"
        pd.DataFrame(
            columns=["hz", "note", "midi", "string", "fret", "token_idx"],
        ).to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[OK] Empty CSV disimpan ke: {out_csv}")
        return {
            "out_dir": out_dir,
            "csv": out_csv,
            "tab": None,
            "viz": viz_path,
        }

    # Collapse sustains
    df = _collapse_to_sustains(df0)
    if len(df) == 0:
        print(f"[SKIP] {wav_path.name}: no sustained notes after RLE")
        out_csv = out_dir / f"{wav_path.stem}_lstm_mapped.csv"
        pd.DataFrame(
            columns=["hz", "note", "midi", "string", "fret", "token_idx"],
        ).to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[OK] Empty CSV disimpan ke: {out_csv}")
        return {
            "out_dir": out_dir,
            "csv": out_csv,
            "tab": None,
            "viz": viz_path,
        }

    # LSTM mapping (sama seperti app.py; pitch-mask auto kalau token_index punya 'midi')
    print("[INFO] Menjalankan LSTM Mapper...")
    df_out = _run_lstm_mapping(
        df_notes=df,
        model=model,
        token_index_df=token_index_df,
        device=device,
        use_pitch_mask=True,
    )

    # Simpan CSV
    cols = ["hz", "note", "midi", "string", "fret", "token_idx"]
    cols = [c for c in cols if c in df_out.columns]
    out_csv = out_dir / f"{wav_path.stem}_lstm_mapped.csv"
    df_out[cols].to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] CSV LSTM-mapped disimpan: {out_csv}")

    # ASCII TAB
    out_tab: Optional[Path] = None
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
        print("[WARN] Kolom 'string'/'fret' tidak ada — TAB tidak dibuat.")

    print("[DONE] Inference LSTM Mapper selesai (GUI).")

    return {
        "out_dir": out_dir,
        "csv": out_csv,
        "tab": out_tab,
        "viz": viz_path,
    }


# ==============================
# Tkinter GUI
# ==============================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("900x680")
        self.minsize(860, 620)

        # state
        self.wav_path = tk.StringVar()
        self.out_dir = tk.StringVar()
        self.token_index = tk.StringVar()
        self.model_file = tk.StringVar()
        self.device = tk.StringVar(value="auto")  # auto / cuda / cpu
        self.show_player = tk.BooleanVar(value=True)  # NEW: toggle show_visualization

        self._img_obj = None          # untuk preview PNG (PIL)
        self._last_artifacts: Dict[str, Optional[Path]] = {}
        self._build_ui()

    # --- UI layout ---
    def _build_ui(self):
        pad = {"padx": 8, "pady": 6}

        # === Paths frame ===
        frm_paths = ttk.LabelFrame(self, text="Paths")
        frm_paths.pack(fill="x", **pad)

        def row_path(r, label, var, picker, width=70):
            ttk.Label(frm_paths, text=label, width=18, anchor="w").grid(
                row=r, column=0, sticky="w", **pad
            )
            ttk.Entry(frm_paths, textvariable=var, width=width).grid(
                row=r, column=1, sticky="we", **pad
            )
            ttk.Button(frm_paths, text="Browse", command=picker).grid(
                row=r, column=2, **pad
            )

        frm_paths.columnconfigure(1, weight=1)
        row_path(0, "WAV file", self.wav_path, self._pick_wav)
        row_path(1, "Output folder", self.out_dir, self._pick_outdir)
        row_path(2, "Token index CSV", self.token_index, self._pick_token_index)
        row_path(3, "Model file (.pt/.jit)", self.model_file, self._pick_model)

        # === Options frame ===
        frm_opts = ttk.LabelFrame(self, text="Options")
        frm_opts.pack(fill="x", **pad)

        ttk.Label(frm_opts, text="Device").grid(row=0, column=0, sticky="w", **pad)
        ttk.Combobox(
            frm_opts,
            textvariable=self.device,
            values=["auto", "cuda", "cpu"],
            width=10,
            state="readonly",
        ).grid(row=0, column=1, **pad)

        # NEW: checkbox show_player
        self.chk_show_player = ttk.Checkbutton(
            frm_opts,
            text="Show player (interactive visualization)",
            variable=self.show_player,
        )
        self.chk_show_player.grid(row=0, column=2, sticky="w", **pad)

        # === Actions ===
        frm_act = ttk.Frame(self)
        frm_act.pack(fill="x", **pad)

        self.btn_run = ttk.Button(frm_act, text="Run Inference", command=self._run_threaded)
        self.btn_run.pack(side="left", padx=8)

        ttk.Button(frm_act, text="Open Output Folder", command=self._open_out).pack(
            side="left", padx=8
        )

        self.lbl_status = ttk.Label(frm_act, text="Ready.", anchor="w")
        self.lbl_status.pack(side="right", padx=8)

        # === Preview ===
        frm_prev = ttk.LabelFrame(self, text="Visualization Preview (PNG)")
        frm_prev.pack(fill="both", expand=True, **pad)
        self.canvas = tk.Canvas(frm_prev, bg="#202020")
        self.canvas.pack(fill="both", expand=True)

        self.canvas.bind("<Configure>", lambda event: self._refresh_preview())

    # --- Pickers ---
    def _pick_wav(self):
        f = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if f:
            self.wav_path.set(f)

    def _pick_outdir(self):
        d = filedialog.askdirectory()
        if d:
            self.out_dir.set(d)

    def _pick_token_index(self):
        f = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if f:
            self.token_index.set(f)

    def _pick_model(self):
        f = filedialog.askopenfilename(
            filetypes=[("PyTorch", "*.pt *.jit"), ("All", "*.*")]
        )
        if f:
            self.model_file.set(f)

    # --- Run inference (threaded) ---
    def _run_threaded(self):
        if not self.wav_path.get():
            messagebox.showwarning(APP_TITLE, "Pilih file WAV terlebih dahulu.")
            return
        if not self.token_index.get():
            messagebox.showwarning(APP_TITLE, "Pilih token index CSV terlebih dahulu.")
            return
        if not self.model_file.get():
            messagebox.showwarning(APP_TITLE, "Pilih model file terlebih dahulu.")
            return

        self.btn_run.configure(state="disabled")
        self.lbl_status.configure(text="Running…")
        threading.Thread(target=self._do_run, daemon=True).start()

    def _do_run(self):
        try:
            wav = Path(self.wav_path.get())
            base_out = Path(self.out_dir.get()) if self.out_dir.get() else wav.parent
            tok_csv = Path(self.token_index.get())
            model_path = Path(self.model_file.get())
            device_str = self.device.get() or "auto"
            show_player = bool(self.show_player.get())

            artifacts = run_inference_once(
                wav_path=wav,
                base_out_dir=base_out,
                token_index_csv=tok_csv,
                model_path=model_path,
                device_str=device_str,
                show_player=show_player,   # <-- pakai toggle
            )
            self._last_artifacts = artifacts

            self.lbl_status.configure(text="Done.")
            viz_path = artifacts.get("viz")
            if viz_path:
                self._show_preview(viz_path)

            msg = ["OK!"]
            if artifacts.get("csv"):
                msg.append(f"CSV: {artifacts['csv']}")
            if artifacts.get("tab"):
                msg.append(f"TAB: {artifacts['tab']}")
            if artifacts.get("viz"):
                msg.append(f"VIZ: {artifacts['viz']}")
            messagebox.showinfo(APP_TITLE, "\n".join(msg))
        except Exception as e:
            self.lbl_status.configure(text="Error.")
            messagebox.showerror(APP_TITLE, f"ERROR:\n{e}")
        finally:
            self.btn_run.configure(state="normal")

    # --- Preview ---
    def _show_preview(self, viz_path: Path):
        if not viz_path or not Path(viz_path).exists():
            return

        cw = self.canvas.winfo_width() or 800
        ch = self.canvas.winfo_height() or 400

        self.canvas.delete("all")

        if _HAS_PIL:
            im = Image.open(viz_path)
            im.thumbnail((cw, ch))
            self._img_obj = ImageTk.PhotoImage(im)
            self.canvas.create_image(cw // 2, ch // 2, image=self._img_obj)
        else:
            self.canvas.create_text(
                10,
                10,
                anchor="nw",
                fill="#ffffff",
                text=f"Visualization saved to:\n{viz_path}",
            )

    def _refresh_preview(self):
        """Dipanggil saat canvas di-resize."""
        viz_path = None
        if self._last_artifacts:
            viz_path = self._last_artifacts.get("viz")
        if viz_path and Path(viz_path).exists():
            self._show_preview(viz_path)

    # --- Open output folder ---
    def _open_out(self):
        p: Optional[Path] = None
        if self._last_artifacts and self._last_artifacts.get("out_dir"):
            p = self._last_artifacts["out_dir"]
        elif self.out_dir.get():
            p = Path(self.out_dir.get())
        elif self.wav_path.get():
            p = Path(self.wav_path.get()).parent

        if p:
            _safe_open_folder(p)
        else:
            messagebox.showinfo(APP_TITLE, "Belum ada output folder untuk dibuka.")


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
