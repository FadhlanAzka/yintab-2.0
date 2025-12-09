"""
HTDemucs via subprocess (CLI). Pastikan 'demucs' terinstall:
  pip install demucs
  (dan ffmpeg terpasang di PATH)

Fungsi ini akan menjalankan demucs, lalu mengembalikan path folder
hasil separation. User kemudian bisa memilih sendiri stem mana
yang mau dianalisis.
"""

import subprocess
import sys
import warnings
from pathlib import Path

# Suppress warning dari torchaudio terkait torchcodec
warnings.filterwarnings(
    "ignore",
    message=".*torchcodec.*",
    category=UserWarning
)

def run_htdemucs(input_path: str, save_dir: str, model_name: str = "htdemucs") -> str:
    """
    Jalankan HTDemucs separation.
    - input_path: file audio asli
    - save_dir: folder tempat hasil separation disimpan
    - model_name: nama model (default: htdemucs)
    Return: path folder hasil separation (stem_dir) atau "" jika gagal.
    """
    inp = Path(input_path)
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        sys.executable, "-m", "demucs",
        "--name", model_name,
        "-o", str(out_dir),
        str(inp)
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        return ""

    # Cari folder hasil: save_dir/{model_name}/{file_stem}/
    sub = out_dir / model_name / inp.stem
    if not sub.exists():
        return ""

    return str(sub)
