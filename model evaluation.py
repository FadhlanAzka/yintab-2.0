"""
evaluate_lstm_local.py

Evaluasi model LSTM (NaiveTabMapperLSTM) di validation set:
- File-wise split 80/20 (per file), sama seperti saat training
- Menghitung:
    - accuracy (overall / micro)
    - precision / recall / F1 macro & weighted
- Menyimpan:
    - summary.json
    - summary.png (bar plot macro & weighted metrics)
di folder: <RUN_DIR>/eval_metrics

Pemilihan via Tkinter:
- RUN_DIR (folder artifacts run: berisi 'meta' dan 'ckpt')
- training_config.json (otomatis dari RUN_DIR/meta, atau pilih manual jika tidak ada)
- DATASET_DIR (opsional override)
- token index CSV (opsional override)
- MODEL_FILE: dipilih manual (bisa .ckpt, .pt, .pth, .ts.pt, .jit, dll)
"""

import os
import json
import glob
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

import matplotlib.pyplot as plt  # untuk summary.png

# --- Tkinter untuk pemilihan folder/file ---
import tkinter as tk
from tkinter import filedialog


# =============================================================================
# 1. PILIH PATH DENGAN TKINTER
# =============================================================================
def select_paths_with_tkinter():
    """
    Menggunakan Tkinter untuk memilih:
      - RUN_DIR (folder artifacts run: berisi subfolder 'meta' dan 'ckpt')
      - training_config.json (otomatis dari RUN_DIR/meta, atau pilih manual jika tidak ada)
      - Opsional override DATASET_DIR
      - Opsional override token_index_csv_path
      - MODEL_FILE: dipilih manual (bisa .ckpt / .pt / .jit dll)
    Return:
      run_dir, training_config (dict), dataset_dir, token_index_csv_path, model_path
    """
    root = tk.Tk()
    root.withdraw()  # tidak menampilkan jendela utama

    # --- RUN_DIR ---
    print("Pilih RUN_DIR (folder artifacts run yang berisi 'meta' dan 'ckpt')...")
    run_dir = filedialog.askdirectory(title="Pilih RUN_DIR (artifacts run)")
    if not run_dir:
        print("RUN_DIR tidak dipilih. Keluar.")
        raise SystemExit(0)
    run_dir = os.path.abspath(run_dir)
    print(f"RUN_DIR terpilih: {run_dir}\n")

    # --- training_config.json ---
    meta_dir_default = os.path.join(run_dir, "meta")
    config_default_path = os.path.join(meta_dir_default, "training_config.json")

    if os.path.exists(config_default_path):
        config_path = config_default_path
        print(f"training_config.json ditemukan di: {config_path}")
    else:
        print("training_config.json default tidak ditemukan.")
        print("Silakan pilih training_config.json secara manual.")
        config_path = filedialog.askopenfilename(
            title="Pilih training_config.json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not config_path:
            print("training_config.json tidak dipilih. Keluar.")
            raise SystemExit(0)
        config_path = os.path.abspath(config_path)

    with open(config_path, "r") as f:
        training_config = json.load(f)

    print("\n=== training_config yang dimuat ===")
    print(json.dumps(training_config, indent=2))

    # --- DATASET_DIR & token index dari config ---
    dataset_dir = training_config["dataset_dir"]
    token_index_csv_path = training_config["token_index_csv_path"]

    print("\nPath dari config:")
    print(f"  DATASET_DIR (config): {dataset_dir}")
    print(f"  Token index CSV (config): {token_index_csv_path}")

    # Opsional: override DATASET_DIR
    print("\nOpsional: pilih DATASET_DIR lain (Cancel = pakai path dari config)")
    alt_dataset_dir = filedialog.askdirectory(
        title="Opsional: pilih DATASET_DIR lain (Cancel = pakai dari config)"
    )
    if alt_dataset_dir:
        dataset_dir = os.path.abspath(alt_dataset_dir)
        print(f"DATASET_DIR diganti ke: {dataset_dir}")
    else:
        print("DATASET_DIR tetap memakai path dari config.")

    # Opsional: override token index CSV
    print("\nOpsional: pilih token index CSV lain (Cancel = pakai path dari config)")
    alt_token_index = filedialog.askopenfilename(
        title="Opsional: pilih token index CSV lain (Cancel = pakai dari config)",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    if alt_token_index:
        token_index_csv_path = os.path.abspath(alt_token_index)
        print(f"Token index CSV diganti ke: {token_index_csv_path}")
    else:
        print("Token index CSV tetap memakai path dari config.")

    # --- MODEL_FILE: dipilih manual ---
    print("\nPilih file model yang ingin dievaluasi (.ckpt / .pt / .pth / .ts.pt / .jit, dll)...")
    initial_dir = os.path.join(run_dir, "ckpt")
    if not os.path.isdir(initial_dir):
        initial_dir = run_dir

    model_path = filedialog.askopenfilename(
        title="Pilih file model",
        initialdir=initial_dir,
        filetypes=[
            ("Model files", "*.ckpt *.pt *.pth *.ts.pt *.jit"),
            ("All files", "*.*"),
        ],
    )
    if not model_path:
        print("Model file tidak dipilih. Keluar.")
        raise SystemExit(0)
    model_path = os.path.abspath(model_path)
    print(f"Model yang dipilih: {model_path}")

    print("\n=== Ringkasan path terpilih ===")
    print(f"RUN_DIR              : {run_dir}")
    print(f"training_config.json : {config_path}")
    print(f"DATASET_DIR          : {dataset_dir}")
    print(f"Token index CSV      : {token_index_csv_path}")
    print(f"Model file           : {model_path}")
    print("================================\n")

    return run_dir, training_config, dataset_dir, token_index_csv_path, model_path


# =============================================================================
# 2. DATASET
# =============================================================================
class MidiTokenSequenceDataset(Dataset):
    """
    Naive sequence dataset:
      - Input: midi sequence (int) panjang SEQ_LEN
      - Target: token_idx sequence panjang SEQ_LEN
      - Sequence dibentuk sliding window dengan hop = SEQ_HOP
    """

    def __init__(
        self,
        df,
        seq_len=64,
        seq_hop=16,
        midi_col="midi",
        token_col="token_idx",
        file_col=None,
    ):
        self.seq_len = seq_len
        self.seq_hop = seq_hop if seq_hop is not None and seq_hop > 0 else seq_len

        sequences_midi = []
        sequences_token = []

        if file_col is not None and file_col in df.columns:
            groups = df.groupby(file_col)
        else:
            groups = [(None, df)]

        for _, g in groups:
            midi_vals = g[midi_col].to_numpy(dtype="int64")
            token_vals = g[token_col].to_numpy(dtype="int64")

            if len(midi_vals) < seq_len:
                continue

            for start in range(0, len(midi_vals) - seq_len + 1, self.seq_hop):
                end = start + seq_len
                midi_seq = midi_vals[start:end]
                tok_seq = token_vals[start:end]
                sequences_midi.append(midi_seq)
                sequences_token.append(tok_seq)

        if len(sequences_midi) == 0:
            raise ValueError(
                "No sequences were generated. "
                "Check SEQ_LEN, SEQ_HOP, and dataset size."
            )

        self.sequences_midi = np.stack(sequences_midi)
        self.sequences_token = np.stack(sequences_token)

        print(
            f"[Dataset] Created {len(self.sequences_midi)} sequences "
            f"of length {self.seq_len} (hop={self.seq_hop})."
        )

    def __len__(self):
        return len(self.sequences_midi)

    def __getitem__(self, idx):
        midi_seq = torch.from_numpy(self.sequences_midi[idx])  # [T]
        tok_seq = torch.from_numpy(self.sequences_token[idx])  # [T]
        return midi_seq.long(), tok_seq.long()


# =============================================================================
# 3. MODEL
# =============================================================================
class NaiveTabMapperLSTM(nn.Module):
    def __init__(
        self,
        midi_vocab_size,
        num_classes,
        midi_embedding_dim=32,
        hidden_size=128,
        num_layers=2,
        bidirectional=True,
        dropout=0.1,
    ):
        super().__init__()
        self.midi_embed = nn.Embedding(midi_vocab_size, midi_embedding_dim)
        self.lstm = nn.LSTM(
            input_size=midi_embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_out_dim = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, midi_seq):
        x = self.midi_embed(midi_seq)  # [B, T, E]
        out, _ = self.lstm(x)  # [B, T, H*dir]
        logits = self.fc(out)  # [B, T, C]
        return logits


# =============================================================================
# 4. HELPER: LOAD MODEL BERBAGAI FORMAT
# =============================================================================
def load_model_flexible(model_path, device, midi_vocab_size, num_classes,
                        midi_embedding_dim, hidden_size, num_layers,
                        bidirectional, dropout):
    """
    Coba load model dari berbagai format:
      - TorchScript (.jit / .ts.pt)  -> torch.jit.load
      - Checkpoint dict:
          * dengan key "model_state_dict"
          * atau langsung state_dict
      - nn.Module / ScriptModule
    """
    print("\nLoading model from:", model_path)

    # Jika kelihatan TorchScript
    if model_path.endswith((".jit", ".ts.pt")):
        print("Detected TorchScript model (jit).")
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        return model

    # Kalau bukan explicit jit, coba torch.load dulu
    obj = torch.load(model_path, map_location=device)

    # Jika langsung nn.Module / ScriptModule
    if isinstance(obj, (nn.Module, torch.jit.ScriptModule)):
        print("Loaded nn.Module / ScriptModule langsung dari file.")
        model = obj.to(device)
        model.eval()
        return model

    # Kalau dict: bisa checkpoint training atau state_dict mentah
    if isinstance(obj, dict):
        print("Loaded dict checkpoint. Mencari state_dict...")
        if "model_state_dict" in obj:
            state_dict = obj["model_state_dict"]
            print("Menggunakan 'model_state_dict' dari checkpoint.")
        else:
            # Asumsi seluruh dict adalah state_dict
            if all(isinstance(k, str) for k in obj.keys()):
                state_dict = obj
                print("Checkpoint terlihat seperti state_dict mentah.")
            else:
                raise ValueError(
                    "Checkpoint dict tidak memiliki 'model_state_dict' dan tidak terlihat seperti state_dict."
                )

        model = NaiveTabMapperLSTM(
            midi_vocab_size=midi_vocab_size,
            num_classes=num_classes,
            midi_embedding_dim=midi_embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        ).to(device)

        model.load_state_dict(state_dict)
        model.eval()
        print("Berhasil load state_dict ke NaiveTabMapperLSTM.")
        return model

    # Kalau sampai sini, formatnya tidak didukung
    raise ValueError(
        f"Tipe objek yang diload dari {model_path} tidak didukung: {type(obj)}"
    )


# =============================================================================
# 5. MAIN EVAL LOGIC
# =============================================================================
def main():
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 1) Pilih path via Tkinter
    (
        run_dir,
        training_config,
        dataset_dir,
        token_index_csv_path,
        model_path,
    ) = select_paths_with_tkinter()

    # Ambil parameter penting dari config
    num_classes = training_config["num_classes"]
    midi_vocab_size = training_config["midi_vocab_size"]

    hp = training_config["hyperparameters"]
    SEQ_LEN = hp["SEQ_LEN"]
    SEQ_HOP = hp["SEQ_HOP"]
    BATCH_SIZE = hp["BATCH_SIZE"]
    MIDI_EMBED_DIM = hp["MIDI_EMBED_DIM"]
    HIDDEN_SIZE = hp["HIDDEN_SIZE"]
    NUM_LAYERS = hp["NUM_LAYERS"]
    BIDIRECTIONAL = hp["BIDIRECTIONAL"]
    DROPOUT = hp["DROPOUT"]
    SEED = hp["SEED"]

    print("\nLoaded hyperparameters:")
    print(hp)

    # 2) Set seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # 3) Load token index (untuk konsistensi)
    print("\nLoading token index CSV...")
    token_index_df = pd.read_csv(token_index_csv_path)
    print("Token index shape:", token_index_df.shape)

    # 4) Load semua CSV di DATASET_DIR
    print(f"\nScanning CSV di DATASET_DIR: {dataset_dir}")
    csv_paths = sorted(glob.glob(os.path.join(dataset_dir, "*.csv")))
    print(f"Found {len(csv_paths)} CSV files in {dataset_dir}")

    if len(csv_paths) == 0:
        raise ValueError(f"No CSV files found in {dataset_dir}")

    required_cols = {"hz", "note", "midi", "string", "fret", "token_idx"}
    dfs = []

    for p in csv_paths:
        df = pd.read_csv(p)
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"File {p} is missing columns: {missing}")
        base = os.path.splitext(os.path.basename(p))[0]
        df["file"] = base
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    print("Combined dataset shape:", df_all.shape)

    # 5) Split 80/20 per FILE (sama seperti trainer)
    file_col = "file"
    files = df_all[file_col].unique().tolist()
    random.shuffle(files)
    n_train_files = int(0.8 * len(files))
    train_files = files[:n_train_files]
    val_files = files[n_train_files:]

    df_train = df_all[df_all[file_col].isin(train_files)].reset_index(drop=True)
    df_val = df_all[df_all[file_col].isin(val_files)].reset_index(drop=True)

    print(f"\nUsing file-wise split on '{file_col}'")
    print(f"  Total files : {len(files)}")
    print(f"  Train files : {len(train_files)}, Val files: {len(val_files)}")
    print(f"  Train rows  : {len(df_train)}, Val rows : {len(df_val)}")

    # 6) Dataset & DataLoader VAL
    val_dataset = MidiTokenSequenceDataset(
        df_val,
        seq_len=SEQ_LEN,
        seq_hop=SEQ_HOP,
        midi_col="midi",
        token_col="token_idx",
        file_col=file_col,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    print(f"Val batches: {len(val_loader)}")

    # 7) Load model (format fleksibel)
    model = load_model_flexible(
        model_path=model_path,
        device=device,
        midi_vocab_size=midi_vocab_size,
        num_classes=num_classes,
        midi_embedding_dim=MIDI_EMBED_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        bidirectional=BIDIRECTIONAL,
        dropout=DROPOUT,
    )

    # 8) Inference di validation set
    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch_midi, batch_target in val_loader:
            batch_midi = batch_midi.to(device)  # [B, T]
            batch_target = batch_target.to(device)  # [B, T]

            logits = model(batch_midi)  # [B, T, C]
            preds = logits.argmax(dim=-1)  # [B, T]

            all_true.append(batch_target.cpu().numpy())
            all_pred.append(preds.cpu().numpy())

    all_true = np.concatenate(all_true, axis=0).reshape(-1)  # [N_tokens]
    all_pred = np.concatenate(all_pred, axis=0).reshape(-1)  # [N_tokens]

    print("\nTotal tokens (val):", len(all_true))

    # 9) Hitung metrik
    print("\n=== CLASSIFICATION REPORT (per token_idx) ===")
    print(
        classification_report(
            all_true,
            all_pred,
            labels=list(range(num_classes)),
            zero_division=0,
        )
    )

    # Overall / micro
    accuracy = accuracy_score(all_true, all_pred)
    precision_micro = precision_score(
        all_true, all_pred, average="micro", zero_division=0
    )
    recall_micro = recall_score(all_true, all_pred, average="micro", zero_division=0)
    f1_micro = f1_score(all_true, all_pred, average="micro", zero_division=0)

    # Macro & weighted
    macro_f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_true, all_pred, average="weighted", zero_division=0)

    macro_precision = precision_score(
        all_true, all_pred, average="macro", zero_division=0
    )
    weighted_precision = precision_score(
        all_true, all_pred, average="weighted", zero_division=0
    )

    macro_recall = recall_score(all_true, all_pred, average="macro", zero_division=0)
    weighted_recall = recall_score(
        all_true, all_pred, average="weighted", zero_division=0
    )

    print("\n=== SUMMARY METRICS (VAL SET) ===")
    print(f"Accuracy        : {accuracy:.4f}")
    print(f"Micro Precision : {precision_micro:.4f}")
    print(f"Micro Recall    : {recall_micro:.4f}")
    print(f"Micro F1        : {f1_micro:.4f}")
    print(f"Macro Precision : {macro_precision:.4f}")
    print(f"Weighted Precision: {weighted_precision:.4f}")
    print(f"Macro Recall    : {macro_recall:.4f}")
    print(f"Weighted Recall : {weighted_recall:.4f}")
    print(f"Macro F1        : {macro_f1:.4f}")
    print(f"Weighted F1     : {weighted_f1:.4f}")

    # Confusion matrix tetap dihitung untuk info (tidak disimpan)
    cm = confusion_matrix(all_true, all_pred, labels=list(range(num_classes)))
    print("\nConfusion matrix shape:", cm.shape)

    # 10) Simpan summary.json + summary.png
    eval_dir = os.path.join(run_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    summary = {
        "accuracy": float(accuracy),

        "micro_precision": float(precision_micro),
        "micro_recall": float(recall_micro),
        "micro_f1": float(f1_micro),

        "macro_precision": float(macro_precision),        
        "macro_recall": float(macro_recall),    
        "macro_f1": float(macro_f1),

        #weighted
        "precision": float(weighted_precision), 
        "recall": float(weighted_recall),
        "f1": float(weighted_f1),

        "num_classes": int(num_classes),
        "num_tokens_val": int(len(all_true)),
    }

    summary_path = os.path.join(eval_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved summary.json to: {summary_path}")

    # --- Bar plot 1: macro & weighted metrics saja ---
    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]
    values = [summary[m] for m in metric_names]

    x = np.arange(len(metric_names))

    cmap = plt.cm.get_cmap("tab10", len(metric_names))
    colors = [cmap(i) for i in range(len(metric_names))]

    plt.figure(figsize=(10, 5))
    plt.bar(x, values, width=0.6, color=colors)
    plt.xticks(x, metric_names, rotation=0, ha="right")

    # --- Zoom otomatis Y-axis ---
    min_v = min(values)
    max_v = max(values)
    padding = (max_v - min_v) * 0.2 if max_v != min_v else 0.05
    plt.ylim(min_v - padding, max_v + padding)

    # --- Tampilkan angka value di atas bar ---
    for i, v in enumerate(values):
        plt.text(i, v + padding*0.1, f"{v:.4f}", ha="center", fontsize=10)

    # --- Grid horizontal ---
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.ylabel("Score")
    plt.title("Validation Metrics Summary (Macro & Weighted)")
    plt.tight_layout()

    summary_png_path = os.path.join(eval_dir, "summary.png")
    plt.savefig(summary_png_path, dpi=150)
    plt.close()

    print(f"Saved summary.png to: {summary_png_path}")
    print("Selesai.")


if __name__ == "__main__":
    main()
