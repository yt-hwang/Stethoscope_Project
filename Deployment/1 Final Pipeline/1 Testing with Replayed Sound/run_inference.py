#!/usr/bin/env python3
"""
Inference-only pipeline for one or more WAV files.

What it does:
- Scans an input directory for .wav files
- Resamples to 16 kHz (if needed)
- Segments into 2.0 s windows with 0.5 s hop (same as training)
- Extracts Log-Mel features (64 mels, 64 ms win, 32 ms hop, 50–7900 Hz)
- Per-segment standardization, then temporal mean pooling -> (64,) features
- Loads learned StandardScaler + LR + MLP, averages softmax probs, applies
  thresholds.json if available, predicts per-segment labels
- Aggregates to per-file probabilities and predictions
- Writes two CSVs under result/: segments.csv, files.csv

Usage (PowerShell):
  python Deployment/Test_with_replayed/run_inference.py \
    --input_dir "D:\\Stethoscope_Project\\Deployment\\Test_with_replayed\\input" \
    --model_dir auto

Notes:
- If --model_dir auto, the script will search the following for the latest run
  containing scaler.pkl, model_lr.pkl, model_mlp.pkl:
    D:\\Stethoscope_Project\\Deployment\\Group Split\\model
    D:\\Stethoscope_Project\\Deployment\\model
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import joblib
import numpy as np
import pandas as pd
import soundfile as sf
import librosa


# ========= Defaults (match training) =========
TARGET_SR = 16000
WIN_SEC = 2.0
HOP_SEC = 0.5
N_MELS = 64
WIN_LEN = int(0.064 * TARGET_SR)
HOP_LEN = int(0.032 * TARGET_SR)
FMIN, FMAX = 50, 7900


def discover_latest_model_dir(candidates: List[Path]) -> Optional[Path]:
    """
    Search candidate roots for latest run_* directory that contains
    required model artifacts.
    """
    required = {"scaler.pkl", "model_lr.pkl", "model_mlp.pkl"}
    found: List[Tuple[float, Path]] = []
    for root in candidates:
        if not root.is_dir():
            continue
        for p in root.iterdir():
            if not p.is_dir():
                continue
            if not p.name.startswith("run_"):
                continue
            try:
                names = {q.name for q in p.iterdir() if q.is_file()}
            except Exception:
                continue
            if required.issubset(names):
                mtime = p.stat().st_mtime
                found.append((mtime, p))
    if not found:
        return None
    found.sort(key=lambda t: t[0], reverse=True)
    return found[0][1]


def load_audio_mono16k(path: Path) -> Tuple[np.ndarray, int]:
    x, sr = sf.read(str(path))
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr != TARGET_SR:
        x = librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    return x.astype(np.float32), sr


def segment_signal(x: np.ndarray, sr: int, win_sec: float, hop_sec: float) -> List[Tuple[int, int, float, float]]:
    """
    Return list of (i0, i1, t0, t1) for each window.
    """
    win_n = int(round(win_sec * sr))
    hop_n = int(round(hop_sec * sr))
    out: List[Tuple[int, int, float, float]] = []
    if len(x) <= win_n:
        pad = win_n - len(x)
        i0, i1 = 0, win_n
        out.append((i0, i1, 0.0, win_sec))
        return out
    t = 0
    while t + win_n <= len(x):
        i0 = t
        i1 = t + win_n
        out.append((i0, i1, i0 / sr, i1 / sr))
        t += hop_n
    return out


def compute_logmel(x: np.ndarray, sr: int) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=x.astype(np.float32), sr=sr, n_fft=2048,
        hop_length=HOP_LEN, win_length=WIN_LEN,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0
    )
    logmel = librosa.power_to_db(S, ref=np.max)
    m = float(logmel.mean())
    s = float(logmel.std()) + 1e-6
    logmel = (logmel - m) / s
    return logmel


def to_feature_vector(logmel: np.ndarray) -> np.ndarray:
    """
    Temporal mean pooling -> (64,) vector.
    """
    return logmel.mean(axis=1).astype(np.float32)


def load_models(model_dir: Path):
    scaler = joblib.load(model_dir / "scaler.pkl")
    lr = joblib.load(model_dir / "model_lr.pkl")
    mlp = joblib.load(model_dir / "model_mlp.pkl")
    class_names: Optional[List[str]] = None
    thresholds: Optional[np.ndarray] = None
    thr_path = model_dir / "thresholds.json"
    if thr_path.exists():
        try:
            obj = json.loads(Path(thr_path).read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                cns = obj.get("class_names")
                ths = obj.get("thresholds")
                if isinstance(cns, list):
                    class_names = [str(x) for x in cns]
                if isinstance(ths, list):
                    thresholds = np.array(ths, dtype=np.float32)
        except Exception:
            pass
    if class_names is None:
        try:
            class_names = [str(x) for x in getattr(lr, "classes_", [])]
        except Exception:
            class_names = None
    return scaler, lr, mlp, class_names, thresholds


def infer_probs(feature_matrix: np.ndarray, scaler, lr, mlp) -> np.ndarray:
    Xs = scaler.transform(feature_matrix)
    p1 = lr.predict_proba(Xs)
    p2 = mlp.predict_proba(Xs)
    probs = (p1 + p2) / 2.0
    return probs


def choose_labels(probs: np.ndarray, class_names: Optional[List[str]], thresholds: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if thresholds is not None and thresholds.shape[0] == probs.shape[1]:
        adj = probs - thresholds[None, :]
        idx = np.argmax(adj, axis=1)
    else:
        idx = np.argmax(probs, axis=1)
    if class_names is None or len(class_names) != probs.shape[1]:
        labels = np.array([f"Class {i}" for i in idx], dtype=object)
    else:
        labels = np.array([class_names[i] for i in idx], dtype=object)
    conf = probs[np.arange(probs.shape[0]), idx]
    return labels, conf


def aggregate_file_level(all_probs: np.ndarray, class_names: Optional[List[str]]) -> Tuple[str, float, Dict[str, float]]:
    mean_probs = all_probs.mean(axis=0)
    k = int(mean_probs.argmax())
    label = class_names[k] if class_names and k < len(class_names) else f"Class {k}"
    conf = float(mean_probs[k])
    by_class = { (class_names[i] if class_names and i < len(class_names) else f"Class {i}"): float(mean_probs[i]) for i in range(mean_probs.shape[0]) }
    return label, conf, by_class


def _sanitize_for_col(name: str) -> str:
    """Sanitize class name to safe CSV column suffix."""
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", str(name)).strip("_")
    if not s:
        s = "Class"
    return s


def main():
    parser = argparse.ArgumentParser(description="Inference-only pipeline for WAV files")
    #parser.add_argument("--input_dir", type=str, default=str(Path(r"D:\\Stethoscope_Project\\Audio shared\\1021 replayed")), help="Directory containing WAV files")
    parser.add_argument("--input_dir", type=str, default=str(Path(r"D:\Stethoscope_Project\Audio shared\ML test sound list\RAW sound_ML test sound list")), help="Directory containing WAV files")
    parser.add_argument("--model_dir", type=str, default=str(Path(r"D:\\Stethoscope_Project\\Deployment\\Group Split\\model\\run_20251008_172910")), help="Model directory or 'auto'")
    parser.add_argument("--result_dir", type=str, default=str(Path(r"D:\\Stethoscope_Project\\Deployment\\1 Final Pipeline\\1 Testing with Replayed Sound\\result")), help="Output directory for CSVs")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    result_dir = Path(args.result_dir)
    # Ensure required directories exist
    input_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    model_dir: Optional[Path]
    if args.model_dir.strip().lower() == "auto":
        # Try both locations
        cand = [
            Path(r"D:\\Stethoscope_Project\\Deployment\\Group Split\\model"),
            Path(r"D:\\Stethoscope_Project\\Deployment\\model"),
        ]
        model_dir = discover_latest_model_dir(cand)
        if model_dir is None:
            raise RuntimeError("No model directory found with required artifacts. Please pass --model_dir.")
    else:
        model_dir = Path(args.model_dir)
        if not model_dir.is_dir():
            raise RuntimeError(f"Model dir not found: {model_dir}")

    scaler, lr, mlp, class_names, thresholds = load_models(model_dir)

    wav_files = sorted([p for p in input_dir.glob("*.wav") if p.is_file()])
    if not wav_files:
        print(f"[WARN] No wav files in {input_dir}")
        return

    seg_rows: List[Dict[str, object]] = []
    file_rows: List[Dict[str, object]] = []

    for wav_path in wav_files:
        try:
            x, sr = load_audio_mono16k(wav_path)
        except Exception as e:
            print(f"[WARN] read fail: {wav_path} -> {e}")
            continue

        windows = segment_signal(x, sr, WIN_SEC, HOP_SEC)
        if not windows:
            print(f"[WARN] no windows for {wav_path}")
            continue

        feats: List[np.ndarray] = []
        for (i0, i1, t0, t1) in windows:
            seg = x[i0:i1]
            logmel = compute_logmel(seg, sr)
            feat = to_feature_vector(logmel)
            feats.append(feat)

        F = np.stack(feats, axis=0)  # (num_segments, 64)
        probs = infer_probs(F, scaler, lr, mlp)
        labels, conf = choose_labels(probs, class_names, thresholds)

        # Save per-segment rows
        for j, (i0, i1, t0, t1) in enumerate(windows):
            names_for_cols = (class_names if class_names and len(class_names) == probs.shape[1]
                              else [f"Class {i}" for i in range(probs.shape[1])])
            probs_map = { names_for_cols[i]: float(probs[j, i]) for i in range(probs.shape[1]) }
            row = {
                "file": wav_path.name,
                "segment_index": j,
                "start": round(t0, 2),
                "end": round(t1, 2),
                "pred": str(labels[j]),
                "conf": float(conf[j]),
                "conf_pct": round(float(conf[j]) * 100.0, 2),
                "probs_json": json.dumps(probs_map, ensure_ascii=False),
            }
            # Add per-class percentage columns
            for i, cname in enumerate(names_for_cols):
                col = f"prob_{_sanitize_for_col(cname)}_pct"
                row[col] = round(float(probs[j, i]) * 100.0, 2)
            seg_rows.append(row)

        # Aggregate per-file
        mean_probs = probs.mean(axis=0)
        file_label, file_conf, file_probs_map = aggregate_file_level(probs, class_names)
        file_row = {
            "file": wav_path.name,
            "num_segments": F.shape[0],
            "pred": file_label,
            "conf": file_conf,
            "conf_pct": round(float(file_conf) * 100.0, 2),
            "probs_json": json.dumps(file_probs_map, ensure_ascii=False),
            "model_dir": str(model_dir),
        }
        # Add per-class percentage columns at file level (mean of segment probs)
        names_for_cols_file = (class_names if class_names and len(class_names) == mean_probs.shape[0]
                               else [f"Class {i}" for i in range(mean_probs.shape[0])])
        for i, cname in enumerate(names_for_cols_file):
            col = f"prob_{_sanitize_for_col(cname)}_pct"
            file_row[col] = round(float(mean_probs[i]) * 100.0, 2)
        file_rows.append(file_row)

    # Write CSVs
    seg_df = pd.DataFrame(seg_rows)
    file_df = pd.DataFrame(file_rows)

    seg_csv = result_dir / "segments.csv"
    file_csv = result_dir / "files.csv"
    seg_df.to_csv(seg_csv, index=False, encoding="utf-8")
    file_df.to_csv(file_csv, index=False, encoding="utf-8")

    print(f"[DONE] segments: {len(seg_df)} -> {seg_csv}")
    print(f"[DONE] files   : {len(file_df)} -> {file_csv}")
    print(f"[MODEL] {model_dir}")


if __name__ == "__main__":
    main()


