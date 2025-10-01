#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

# ===== Paths =====
# Mac
#AUDIO_ROOT = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/RAW sound_ML test sound list")
#JSON_PATH  = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Development/2) Image/2D CNN/Abnormal_Breathing/breathing_nonbreathing_intervals.json")
#OUT_DIR    = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Development/2) Image/2D CNN/Abnormal_Breathing/A/Spectrogram/Processed Data")

# Windows
AUDIO_ROOT = Path("D:\\Stethoscope_Project\\Audio shared\\ML test sound list\\RAW sound_ML test sound list")
JSON_PATH  = Path("D:\\Stethoscope_Project\\Development\\2) Image\\2D CNN\\Abnormal_Breathing\\breathing_nonbreathing_intervals.json")
OUT_DIR    = Path("D:\\Stethoscope_Project\\Development\\2) Image\\2D CNN\\Abnormal_Breathing\\A\\Spectrogram\\Processed Data")


OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = OUT_DIR / "manifest.csv"

# ===== Mel-spec params (same style as before) =====
SR_TARGET = 4000
N_FFT     = 1024
HOP       = 128
N_MELS    = 128
FMIN      = 50
FMAX      = 2000
DB_CLIP   = (-80, 0)

# ---- helpers ----
def _norm_key(s: str) -> str:
    return Path(s).stem.strip().replace(" ", "_").lower()

# filename-specific label overrides
OVERRIDE_LABELS = {
    "kp002_wws_1": "Crackle",
    "kp002_wws_2": "Crackle",
}

def build_meta_index(meta_json: dict):
    idx = {}
    for k, v in meta_json.items():
        idx[_norm_key(k)] = v
    return idx

def get_label_from_meta(meta: dict, fname_key: str) -> str:
    # 1) filename overrides (highest priority)
    if fname_key in OVERRIDE_LABELS:
        return OVERRIDE_LABELS[fname_key]
    # 2) JSON diagnosis/label/class
    if isinstance(meta, dict):
        val = (meta.get("diagnosis") or meta.get("label") or meta.get("class") or "").strip()
    else:
        val = ""
    if val == "Brhonchi":  # common typo
        val = "Bronchi"
    return val if val else "Unknown"

def parse_patient_id(stem: str) -> str:
    return stem.strip().replace(" ", "_").split("_")[0]

def load_audio_30s(path: Path, sr_target=SR_TARGET):
    y, sr = sf.read(str(path), always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != sr_target:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=sr_target)
        sr = sr_target
    need = sr * 30
    if len(y) < need:
        y = np.pad(y, (0, need - len(y)))
    else:
        y = y[:need]
    return y.astype(np.float32), sr

def mel_image(y, sr):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
        fmin=FMIN, fmax=FMAX, power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db = np.clip(S_db, DB_CLIP[0], DB_CLIP[1])
    return S_db

def save_png(S_db, out_path: Path):
    fig = plt.figure(figsize=(10, 4), dpi=200)
    ax = plt.axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(S_db, origin="lower", aspect="auto", cmap="viridis",
              vmin=DB_CLIP[0], vmax=DB_CLIP[1])
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def main():
    with open(JSON_PATH, "r") as f:
        meta_json = json.load(f)
    meta_index = build_meta_index(meta_json)

    audio_paths = sorted([p for p in AUDIO_ROOT.glob("**/*")
                          if p.suffix.lower() in (".wav", ".flac", ".m4a", ".mp3")])
    audio_index = { _norm_key(p.name): p for p in audio_paths }

    rows = []
    counts_json = Counter()
    counts_json_with_audio = Counter()
    counts_saved = Counter()
    missing_audio_keys = []

    for key, meta in meta_index.items():
        label = get_label_from_meta(meta, key)
        counts_json[label] += 1

        wav = audio_index.get(key)
        if wav is None:
            missing_audio_keys.append(key)
            continue

        counts_json_with_audio[label] += 1

        if label == "Unknown":
            # Skip conversion for unknown labels but still tracked in counts
            continue

        patient_id = parse_patient_id(wav.stem)

        out_dir = OUT_DIR / label
        out_dir.mkdir(parents=True, exist_ok=True)
        out_img = out_dir / f"{wav.stem}.png"

        try:
            y, sr = load_audio_30s(wav, SR_TARGET)
            img = mel_image(y, sr)
            save_png(img, out_img)
            counts_saved[label] += 1
            rows.append({
                "path": str(out_img),
                "label": label,
                "patient_id": patient_id,
                "orig_file": str(wav),
                "t_start": 0.0,
                "t_end": 30.0,
                "sr": sr,
                "type": "spectrogram",
            })
        except Exception as e:
            print(f"[ERROR] {wav.name}: {e}")

    pd.DataFrame(rows).to_csv(MANIFEST_PATH, index=False)

    total_json = sum(counts_json.values())
    total_with_audio = sum(counts_json_with_audio.values())
    total_saved = sum(counts_saved.values())

    print(f"✅ Saved spectrograms: {total_saved}  |  🗂️ {MANIFEST_PATH}")
    print(f"[VERIFY] JSON total={total_json} | with_audio={total_with_audio} | saved={total_saved}")
    if missing_audio_keys:
        print(f"[MISSING AUDIO] {len(missing_audio_keys)} JSON items had no matching audio file")
        for i, key in enumerate(missing_audio_keys, 1):
            print(f"    {i}. {key}")
        missing_list_path = OUT_DIR / "missing_audio_from_json.txt"
        try:
            with open(missing_list_path, "w", encoding="utf-8") as f:
                for key in missing_audio_keys:
                    f.write(f"{key}\n")
            print(f"[MISSING AUDIO LIST] Saved to {missing_list_path}")
        except Exception as e:
            print(f"[WARN] Could not write missing list: {e}")
    # Per-label comparison
    all_labels = sorted(set(list(counts_json.keys()) + list(counts_json_with_audio.keys()) + list(counts_saved.keys())))
    print("[LABEL COUNTS] label | json | json_with_audio | saved")
    for lbl in all_labels:
        print(f"  - {lbl}: {counts_json.get(lbl,0)} | {counts_json_with_audio.get(lbl,0)} | {counts_saved.get(lbl,0)}")

if __name__ == "__main__":
    main()
