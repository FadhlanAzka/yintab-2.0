# -*- coding: utf-8 -*-
"""
app_web.py — Flask web interface for YINTab (YIN → LSTM → TAB)

Versi ini:
- token index CSV, model, dan device di-hardcode
- Form web hanya upload WAV
- MIDI di-generate dari CSV (monophonic, tanpa kolom time)
- NEW: MIDI dirender ke WAV (FluidSynth + SoundFont) agar suara konsisten (web = file explorer)
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    abort,
)

# MIDI generator dari CSV (monophonic)
from tab_midi import csv_notes_to_midi_file

# Low-level project modules
from audio_loader import load_audio
from analysis.beat_tracking import compute_beats
from analysis.onset_detection import compute_onsets
from utils.sanitize import ensure_finite

# Import core helpers dari app.py (pipeline, mapping, dll)
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

# Visualization saver (tanpa player interaktif di web)
try:
    from viz.visualizer import save_visualization
except Exception:
    from visualizer import save_visualization  # type: ignore


# -------------------------------------------------
# Flask app setup & hardcoded config
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
RESULTS_DIR = STATIC_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(
    __name__,
    static_folder=str(STATIC_DIR),
    template_folder=str(BASE_DIR / "templates"),
)

# ==== HARDCODED PATHS & DEVICE ====
TOKEN_INDEX_CSV = BASE_DIR / "tokens" / "token index v2.csv"
MODEL_PATH = BASE_DIR / "models" / "last.jit"
DEFAULT_DEVICE = "auto"  # "auto" / "cpu" / "cuda"

# ==== NEW: SoundFont & fluidsynth config ====
# Letakkan file .sf2 di sini (buat folder assets kalau belum ada)
SOUNDFONT_PATH = BASE_DIR / "assets" / "FluidR3_GM.sf2"
FLUIDSYNTH_BIN = r"C:\Program Files\fluidsynth-v2.5.1-win10-x64-cpp11\bin\fluidsynth.exe"
RENDER_SAMPLE_RATE = 44100


def _midi_to_wav(
    midi_path: Path,
    wav_out_path: Path,
    soundfont_path: Path = SOUNDFONT_PATH,
    sample_rate: int = RENDER_SAMPLE_RATE,
) -> Optional[Path]:
    midi_path = Path(midi_path)
    wav_out_path = Path(wav_out_path)
    soundfont_path = Path(soundfont_path)
    fluidsynth_exe = Path(FLUIDSYNTH_BIN)

    if not midi_path.exists():
        print(f"[WARN] MIDI tidak ditemukan untuk render: {midi_path}")
        return None

    if not soundfont_path.exists():
        print(f"[WARN] SoundFont (.sf2) tidak ditemukan: {soundfont_path}")
        return None

    if not fluidsynth_exe.exists():
        print(f"[WARN] fluidsynth.exe tidak ditemukan: {fluidsynth_exe}")
        return None

    wav_out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(fluidsynth_exe),
        "-q",
        "-ni",
        "-F", str(wav_out_path),
        "-r", str(int(sample_rate)),
        str(soundfont_path),
        str(midi_path),
    ]

    try:
        print(f"[INFO] Rendering MIDI→WAV: {wav_out_path}")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[WARN] Render MIDI→WAV gagal (fluidsynth): {e}")
        return None

    if not wav_out_path.exists() or wav_out_path.stat().st_size == 0:
        print(f"[WARN] Render selesai tapi file WAV tidak valid: {wav_out_path}")
        return None

    print(f"[OK] Render MIDI→WAV sukses: {wav_out_path}")
    return wav_out_path

# -------------------------------------------------
# Core inference function (dipakai route Flask)
# -------------------------------------------------
def run_inference(
    wav_path: Path,
    base_out_dir: Path,
    token_index_csv: Path,
    model_path: Path,
    device_str: str = "auto",
) -> Dict[str, Optional[Path]]:
    """
    Jalankan YIN Pipeline #14 + LSTM mapper untuk satu file WAV.

    Semua output (CSV, TAB, PNG, MIDI, MIDI_WAV) disimpan langsung di base_out_dir,
    TANPA subfolder nama WAV. Ini supaya gampang di-serve via /static.
    """
    out_dir = base_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Token index
    token_index_df = _load_token_index_df(token_index_csv)

    # Device selection
    if device_str == "auto":
        device = _detect_device()
    else:
        device = device_str
        if device == "cuda" and not torch.cuda.is_available():
            print("[WARN] CUDA diminta tetapi tidak tersedia, fallback ke CPU.")
            device = "cpu"
    print(f"[INFO] Device (web): {device}")

    # Load model
    model = _load_lstm_model(model_path, device=device)
    print(f"[INFO] Model loaded from: {model_path}")

    # Load audio
    print(f"[INFO] Memuat audio: {wav_path}")
    y, sr = load_audio(str(wav_path))
    y = ensure_finite(y)

    # Beat / onset untuk visualisasi
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

    # Pipeline 14 (YIN + post-proc)
    print("[INFO] Menjalankan Pipeline #14 (YIN)...")
    stable_notes, stable_f0, stable_midi, sample_times = run_pipeline14(y, sr)

    # Visualisasi (PNG saja)
    duration = len(y) / sr
    f0_vis = stable_f0
    notes_vis = np.asarray(
        [normalize_note_to_sharp(n) if isinstance(n, str) else n for n in stable_notes],
        dtype=object,
    )

    wav_stem = wav_path.stem

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
        viz_path = out_dir / f"{wav_stem}_viz.png"
        print(f"[OK] Visualization disimpan ke folder: {out_dir}")
    except Exception as e:
        viz_path = out_dir / f"{wav_stem}_viz.png"
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

    # DataFrame awal (hz, note, midi)
    df0 = _build_dataframe_from_stable(stable_f0, stable_notes, stable_midi)

    # Prepare default output paths
    out_csv = out_dir / f"{wav_stem}_lstm_mapped.csv"
    out_midi: Optional[Path] = None
    out_midi_wav: Optional[Path] = None

    if len(df0) == 0:
        print(f"[SKIP] {wav_path.name}: no valid notes")
        pd.DataFrame(
            columns=["hz", "note", "midi", "string", "fret", "token_idx"],
        ).to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[OK] Empty CSV disimpan ke: {out_csv}")

        # MIDI dari CSV (monophonic; tanpa time)
        try:
            out_midi = out_dir / f"{wav_stem}_lstm_mapped.mid"
            csv_notes_to_midi_file(
                csv_path=out_csv,
                out_mid_path=out_midi,
                notes_per_second=2.0,
                midi_col="midi",
                rest_policy="keep",
            )
            print(f"[OK] MIDI disimpan: {out_midi}")
        except Exception as e:
            print(f"[WARN] Gagal membuat MIDI dari CSV: {e}")
            out_midi = None

        # NEW: render MIDI -> WAV (konsisten suara)
        if out_midi is not None:
            out_midi_wav = _midi_to_wav(
                midi_path=out_midi,
                wav_out_path=out_dir / f"{wav_stem}_lstm_mapped_render.wav",
            )

        return {
            "out_dir": out_dir,
            "csv": out_csv,
            "tab": None,
            "viz": viz_path,
            "midi": out_midi,
            "midi_wav": out_midi_wav,
        }

    # Collapse sustains
    df = _collapse_to_sustains(df0)
    if len(df) == 0:
        print(f"[SKIP] {wav_path.name}: no sustained notes setelah RLE")
        pd.DataFrame(
            columns=["hz", "note", "midi", "string", "fret", "token_idx"],
        ).to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[OK] Empty CSV disimpan ke: {out_csv}")

        try:
            out_midi = out_dir / f"{wav_stem}_lstm_mapped.mid"
            csv_notes_to_midi_file(
                csv_path=out_csv,
                out_mid_path=out_midi,
                notes_per_second=2.0,
                midi_col="midi",
                rest_policy="keep",
            )
            print(f"[OK] MIDI disimpan: {out_midi}")
        except Exception as e:
            print(f"[WARN] Gagal membuat MIDI dari CSV: {e}")
            out_midi = None

        if out_midi is not None:
            out_midi_wav = _midi_to_wav(
                midi_path=out_midi,
                wav_out_path=out_dir / f"{wav_stem}_lstm_mapped_render.wav",
            )

        return {
            "out_dir": out_dir,
            "csv": out_csv,
            "tab": None,
            "viz": viz_path,
            "midi": out_midi,
            "midi_wav": out_midi_wav,
        }

    # LSTM mapping
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
    df_out[cols].to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] CSV LSTM-mapped disimpan: {out_csv}")

    # MIDI dari CSV
    try:
        out_midi = out_dir / f"{wav_stem}_lstm_mapped.mid"
        csv_notes_to_midi_file(
            csv_path=out_csv,
            out_mid_path=out_midi,
            notes_per_second=2.0,
            midi_col="midi",
            rest_policy="keep",
        )
        print(f"[OK] MIDI disimpan: {out_midi}")
    except Exception as e:
        print(f"[WARN] Gagal membuat MIDI dari CSV: {e}")
        out_midi = None

    # NEW: render MIDI -> WAV (konsisten suara)
    if out_midi is not None:
        out_midi_wav = _midi_to_wav(
            midi_path=out_midi,
            wav_out_path=out_dir / f"{wav_stem}_lstm_mapped_render.wav",
        )

    # ASCII TAB
    out_tab: Optional[Path] = None
    if {"string", "fret"}.issubset(df_out.columns):
        try:
            strings = df_out["string"].tolist()
            frets = df_out["fret"].tolist()
            tab_txt = render_ascii_tab(strings, frets, width_pad=1)
            out_tab = out_dir / f"{wav_stem}_lstm_tab.txt"
            out_tab.write_text(tab_txt, encoding="utf-8")
            print(f"[OK] ASCII tablature LSTM disimpan: {out_tab}")
        except Exception as e:
            print(f"[WARN] Gagal membuat ASCII tablature LSTM: {e}")
    else:
        print("[WARN] Kolom 'string'/'fret' tidak ada — TAB tidak dibuat.")

    print("[DONE] Inference LSTM Mapper selesai (web).")

    return {
        "out_dir": out_dir,
        "csv": out_csv,
        "tab": out_tab,
        "viz": viz_path,
        "midi": out_midi,
        "midi_wav": out_midi_wav,
    }


# -------------------------------------------------
# Helper: run_id & manifest
# -------------------------------------------------
def _new_run_dir() -> Path:
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = RESULTS_DIR / now_str
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_manifest(
    run_dir: Path,
    wav_file: Path,
    artifacts: Dict[str, Optional[Path]],
) -> None:
    manifest = {
        "run_id": run_dir.name,
        "wav": wav_file.name,
        "csv": artifacts.get("csv").name if artifacts.get("csv") else None,
        "tab": artifacts.get("tab").name if artifacts.get("tab") else None,
        "viz": artifacts.get("viz").name if artifacts.get("viz") else None,
        "midi": artifacts.get("midi").name if artifacts.get("midi") else None,
        "midi_wav": artifacts.get("midi_wav").name if artifacts.get("midi_wav") else None,  # NEW
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def _load_manifest(run_id: str) -> Dict[str, Optional[str]]:
    run_dir = RESULTS_DIR / run_id
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found for run_id={run_id}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


# -------------------------------------------------
# Flask routes
# -------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def run_route():
    wav_file = request.files.get("wav_file")
    if not wav_file or wav_file.filename == "":
        abort(400, "WAV file is required")

    if not TOKEN_INDEX_CSV.exists():
        abort(500, f"Token index CSV tidak ditemukan: {TOKEN_INDEX_CSV}")
    if not MODEL_PATH.exists():
        abort(500, f"Model file tidak ditemukan: {MODEL_PATH}")

    # Optional: jangan abort kalau soundfont tidak ada (kita fallback: tetap ada MIDI)
    if not SOUNDFONT_PATH.exists():
        print(f"[WARN] SoundFont tidak ditemukan: {SOUNDFONT_PATH} (render MIDI→WAV akan di-skip)")

    run_dir = _new_run_dir()
    run_id = run_dir.name

    wav_name = Path(wav_file.filename).name
    wav_path = run_dir / wav_name
    wav_file.save(str(wav_path))

    artifacts = run_inference(
        wav_path=wav_path,
        base_out_dir=run_dir,
        token_index_csv=TOKEN_INDEX_CSV,
        model_path=MODEL_PATH,
        device_str=DEFAULT_DEVICE,
    )

    _save_manifest(run_dir, wav_path, artifacts)
    return redirect(url_for("result", run_id=run_id))


@app.route("/result/<run_id>", methods=["GET"])
def result(run_id: str):
    try:
        manifest = _load_manifest(run_id)
    except FileNotFoundError:
        abort(404, f"Run ID {run_id} not found")

    run_dir = RESULTS_DIR / run_id

    wav_url = csv_url = tab_url = viz_url = midi_url = midi_wav_url = None

    if manifest.get("wav"):
        wav_url = url_for("static", filename=f"results/{run_id}/{manifest['wav']}")
    if manifest.get("csv"):
        csv_url = url_for("static", filename=f"results/{run_id}/{manifest['csv']}")
    if manifest.get("tab"):
        tab_url = url_for("static", filename=f"results/{run_id}/{manifest['tab']}")
    if manifest.get("viz"):
        viz_url = url_for("static", filename=f"results/{run_id}/{manifest['viz']}")
    if manifest.get("midi"):
        midi_url = url_for("static", filename=f"results/{run_id}/{manifest['midi']}")
    if manifest.get("midi_wav"):  # NEW
        midi_wav_url = url_for("static", filename=f"results/{run_id}/{manifest['midi_wav']}")

    csv_text = None
    tab_text = None

    try:
        if manifest.get("csv"):
            csv_path = run_dir / manifest["csv"]
            if csv_path.exists():
                csv_text = csv_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[WARN] Gagal membaca CSV untuk preview: {e}")

    try:
        if manifest.get("tab"):
            tab_path = run_dir / manifest["tab"]
            if tab_path.exists():
                tab_text = tab_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[WARN] Gagal membaca TAB untuk preview: {e}")

    return render_template(
        "result.html",
        run_id=run_id,
        wav_url=wav_url,
        csv_url=csv_url,
        tab_url=tab_url,
        viz_url=viz_url,
        midi_url=midi_url,
        midi_wav_url=midi_wav_url,  # NEW
        csv_text=csv_text,
        tab_text=tab_text,
    )


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}


def main():
    app.run(debug=True, host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()
