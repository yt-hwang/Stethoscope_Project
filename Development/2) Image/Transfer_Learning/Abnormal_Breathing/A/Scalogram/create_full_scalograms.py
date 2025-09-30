#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
import librosa
import pywt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ===== Paths =====

# Mac
#AUDIO_ROOT = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/RAW sound_ML test sound list")
#JSON_PATH  = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Development/2) Image/Transfer_Learning/Abnormal_Breathing/breathing_nonbreathing_intervals.json")
#OUT_DIR    = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Development/2) Image/Transfer_Learning/Abnormal_Breathing/A/Scalogram/Processed Data")

# Windows
AUDIO_ROOT = Path("D:\\Stethoscope_Project\\Audio shared\\ML test sound list\\RAW sound_ML test sound list")
JSON_PATH  = Path("D:\\Stethoscope_Project\\Development\\2) Image\\Transfer_Learning\\Abnormal_Breathing\\breathing_nonbreathing_intervals.json")
OUT_DIR    = Path("D:\\Stethoscope_Project\\Development\\2) Image\\Transfer_Learning\\Abnormal_Breathing\\A\\Scalogram\\Processed Data")

OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = OUT_DIR / "manifest.csv"

# ===== CWT params =====
SR_TARGET  = 4000
NUM_SCALES = 128
WAVELET    = "morl"
EPS        = 1e-9
PCTL_CLIP  = (1, 99)

def _norm_key(s: str) -> str:
    return Path(s).stem.strip().replace(" ", "_").lower()

OVERRIDE_LABELS = {
    "kp002_wws_1": "Crackle",
    "kp002_wws_2": "Crackle",
}

def build_meta_index(meta_json: dict):
    return { _norm_key(k): v for k, v in meta_json.items() }

def get_label_from_meta(meta: dict, fname_key: str) -> str:
    if fname_key in OVERRIDE_LABELS:
        return OVERRIDE_LABELS[fname_key]
    if isinstance(meta, dict):
        val = (meta.get("diagnosis") or meta.get("label") or meta.get("class") or "").strip()
    else:
        val = ""
    if val == "Brhonchi": val = "Bronchi"
    return val if val else "Unknown"

def parse_patient_id(stem: str) -> str:
    return stem.strip().replace(" ", "_").split("_")[0]

def load_audio_30s(path: Path, sr_target=SR_TARGET):
    y, sr = sf.read(str(path), always_2d=False)
    if y.ndim > 1: y = np.mean(y, axis=1)
    if sr != sr_target:
        y = librosa.resample(y.astype(np.float32), sr, sr_target)
        sr = sr_target
    need = sr * 30
    if len(y) < need: y = np.pad(y, (0, need - len(y)))
    else: y = y[:need]
    return y.astype(np.float32), sr

def scalogram_img(y, sr):
    scales = np.geomspace(1, 256, NUM_SCALES)
    coefs, freqs = pywt.cwt(y, scales, WAVELET, sampling_period=1.0/sr)
    P = np.abs(coefs)**2
    P = np.log(P + EPS)
    lo, hi = np.percentile(P, PCTL_CLIP)
    P = np.clip(P, lo, hi)
    P = (P - lo) / max(hi - lo, 1e-6)
    P = np.flipud(P)
    return P

def save_png(P, out_path: Path):
    fig = plt.figure(figsize=(10, 4), dpi=200)
    ax = plt.axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(P, origin="lower", aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def main():
    with open(JSON_PATH, "r") as f:
        meta_json = json.load(f)
    meta_index = build_meta_index(meta_json)

    audio_paths = sorted([p for p in AUDIO_ROOT.glob("**/*")
                          if p.suffix.lower() in (".wav", ".flac", ".m4a", ".mp3")])

    rows, hit, miss = [], 0, 0
    for wav in audio_paths:
        key = _norm_key(wav.name)
        meta = meta_index.get(key)
        label = get_label_from_meta(meta, key)
        if label == "Unknown": miss += 1
        else: hit += 1

        patient_id = parse_patient_id(wav.stem)

        # >>> change 1: save directly under diagnosis folder (no patient subfolder)
        out_dir = OUT_DIR / label
        out_dir.mkdir(parents=True, exist_ok=True)
        out_img = out_dir / f"{wav.stem}.png"

        try:
            y, sr = load_audio_30s(wav, SR_TARGET)
            P = scalogram_img(y, sr)
            save_png(P, out_img)
            rows.append({
                "path": str(out_img),
                "label": label,
                "patient_id": patient_id,
                "orig_file": str(wav),
                "t_start": 0.0,
                "t_end": 30.0,
                "sr": sr,
                "type": "scalogram",
            })
        except Exception as e:
            print(f"[ERROR] {wav.name}: {e}")

    pd.DataFrame(rows).to_csv(MANIFEST_PATH, index=False)
    print(f"✅ Saved scalograms: {len(rows)}  |  🗂️ {MANIFEST_PATH}")
    print(f"[LABEL MATCH] labeled={hit}  unknown={miss}")

if __name__ == "__main__":
    main()
