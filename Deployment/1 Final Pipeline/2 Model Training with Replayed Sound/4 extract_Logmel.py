# -*- coding: utf-8 -*-
# extract_Logmel.py
# 클래스 폴더 트리를 스캔하여 (라벨-순수 + 증강):
# - Log-Mel(64; 64ms/32ms; 50–7900Hz)
# - 시간평균 → (64,)
# - per-sample 표준화
# - (X, y, classes, filenames) 저장

import numpy as np
from pathlib import Path
import librosa
import soundfile as sf

# ====== 절대경로 ======
TRAIN_ROOT = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Deployment/1 Final Pipeline/2 Model Training with Replayed Sound")

SEG_ROOT = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Deployment/1 Final Pipeline/2 Model Training with Replayed Sound/Output/Segments_2s_hop500ms")
AUG_ROOT = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Deployment/1 Final Pipeline/2 Model Training with Replayed Sound/Output/Augmented_Windows_by_Coverage")

SR = 16000
N_MELS = 64
FMIN, FMAX = 50, 7900
WIN_MS, HOP_MS = 64, 32

INPUT_ROOTS = [p for p in [SEG_ROOT, AUG_ROOT] if p.exists()]

def logmel_64(x, sr=SR):
    m = librosa.feature.melspectrogram(
        y=x, sr=sr, n_mels=N_MELS,
        n_fft=int(sr*WIN_MS/1000), hop_length=int(sr*HOP_MS/1000),
        fmin=FMIN, fmax=FMAX, power=2.0
    )
    return librosa.power_to_db(m, ref=np.max)

def per_sample_standardize(v):
    mu = v.mean()
    sd = v.std() + 1e-8
    return (v - mu) / sd

def collect_wavs(root: Path):
    items = []
    if not root.exists():
        return items
    for cls_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        cls = cls_dir.name
        for wav in cls_dir.rglob("*.wav"):
            items.append((cls, wav))
    return items

def resample_if_needed(x, sr):
    if sr == SR:
        return (x if x.ndim == 1 else np.mean(x, axis=1)).astype(np.float32)
    x = (x if x.ndim == 1 else np.mean(x, axis=1)).astype(np.float32)
    return librosa.resample(x, orig_sr=sr, target_sr=SR, res_type="kaiser_best")

def main():
    Xs, ys, classes, filenames = [], [], [], []
    for root in INPUT_ROOTS:
        items = collect_wavs(root)
        print(f"[INFO] scanning: {root}  files={len(items)}")
        for cls, wav in items:
            x, sr = sf.read(wav)
            x = resample_if_needed(x, sr)
            lm = logmel_64(x, SR)                    # (64, T)
            v = lm.mean(axis=1).astype(np.float32)   # (64,)
            v = per_sample_standardize(v)
            Xs.append(v); ys.append(cls); filenames.append(str(wav))
            if cls not in classes:
                classes.append(cls)

    Xs = np.stack(Xs, axis=0) if len(Xs) > 0 else np.zeros((0, 64), dtype=np.float32)
    y_arr = np.array(ys, dtype=object)
    fn_arr = np.array(filenames, dtype=object)
    classes_arr = np.array(classes, dtype=object)

    print(f"[DONE] features: {Xs.shape}  classes: {classes}")
    out_dir = TRAIN_ROOT / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "features_64mel.npz",
                        X=Xs, y=y_arr, classes=classes_arr, filenames=fn_arr)
    print(f"[DONE] saved: {out_dir/'features_64mel.npz'}")

if __name__ == "__main__":
    main()
