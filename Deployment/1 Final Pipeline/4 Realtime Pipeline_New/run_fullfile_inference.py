# -*- coding: utf-8 -*-
# run_fullfile_inference.py
# 목적:
#  - 30초 WAV 한 파일을 2.0s 윈도우 / 0.5s 홉으로 전구간 슬라이딩
#  - 학습과 동일한 전처리(Log-Mel 64, 64ms/32ms, 50~7900Hz, per-sample 표준화, StandardScaler)
#  - LR+MLP 앙상블 softmax 평균 + thresholds.json 적용
#  - 구간별/파일별 결과를 CSV로 저장하여 "값이 안 바뀌는 문제"를 진단
#
# 절대경로(하드코딩): MODEL_DIR, INPUT_WAV, OUT_DIR만 수정
# 다른 코드는 리팩토링 금지 (독립 스크립트로 제공)

import os
import json
import math
import time
import numpy as np
from pathlib import Path
import soundfile as sf
import librosa
from joblib import load as joblib_load
import csv

# ======== [하드코딩: 경로] ========
MODEL_DIR = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Deployment/1 Final Pipeline/2 Model Training with Replayed Sound/model/run_20251102_225046")
INPUT_WAV = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/Test for Realtime Deployment/KP012_WWS_1.wav")
OUT_DIR   = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Deployment/1 Final Pipeline/4 Realtime Pipeline_New/1 fullfile_inference/result")

# ======== [학습 파이프라인과 동일한 파라미터] ========
SR = 16000
WIN = 2.0
HOP = 0.5
N_MELS = 64
FMIN, FMAX = 50, 7900
WIN_MS, HOP_MS = 64, 32

# 학습 스크립트의 표준 클래스 순서(고정)
CANONICAL_CLASSES = ['Crackle', 'Healthy', 'Non-breathing', 'Rhonchi', 'Wheezing']

# ======== 유틸 ========
def resample_if_needed(x, sr):
    if sr == SR:
        return (x if x.ndim == 1 else np.mean(x, axis=1)).astype(np.float32)
    x = (x if x.ndim == 1 else np.mean(x, axis=1)).astype(np.float32)
    return librosa.resample(x, orig_sr=sr, target_sr=SR, res_type="kaiser_best")

def logmel_64(x, sr=SR):
    m = librosa.feature.melspectrogram(
        y=x, sr=sr, n_mels=N_MELS,
        n_fft=int(sr*WIN_MS/1000), hop_length=int(sr*HOP_MS/1000),
        fmin=FMIN, fmax=FMAX, power=2.0
    )
    return librosa.power_to_db(m, ref=np.max)

def per_sample_standardize(v):
    # (64,) 벡터에 대해 표본 단위 표준화
    mu = v.mean()
    sd = v.std() + 1e-8
    return (v - mu) / sd

def load_models(model_dir: Path):
    scaler = joblib_load(model_dir / "scaler.pkl")
    lr     = joblib_load(model_dir / "model_lr.pkl")
    mlp    = joblib_load(model_dir / "model_mlp.pkl")
    with open(model_dir / "thresholds.json", "r", encoding="utf-8") as f:
        th = json.load(f)
    class_names = th.get("class_names", [])
    thresholds  = np.array(th.get("thresholds", []), dtype=float)
    return scaler, lr, mlp, class_names, thresholds

def make_reorder(local_names, canonical):
    """
    모델 내부 클래스 순서(local_names) → UI/표준 순서(canonical)로 맞추기 위한 reindex
    반환: idx 배열 s.t. probs_local[:, local_idx] -> probs_canon[:, canon_k]
    """
    m = []
    for c in canonical:
        try:
            m.append(local_names.index(c))
        except ValueError:
            m.append(None)
    return m

def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

def debug_feature_stats(feats):
    # 값 고정 의심시 확인용: 몇 개 윈도우의 mean/std를 출력
    K = min(5, feats.shape[0])
    for i in range(K):
        mu = float(feats[i].mean()); sd = float(feats[i].std())
        print(f"[DBG] feat[{i}] mean={mu:.6f} std={sd:.6f}")

def segment_indices(n_samples, sr, win_s=2.0, hop_s=0.5):
    n_win = int(win_s * sr)
    n_hop = int(hop_s * sr)
    i = 0
    while i + n_win <= n_samples:
        yield i, i + n_win
        i += n_hop

# ======== 메인 ========
def main():
    t0 = time.time()
    ensure_dirs()

    # 입력 WAV 로드 + 리샘플/모노
    x, sr = sf.read(str(INPUT_WAV))
    x = resample_if_needed(x, sr)
    n = len(x)
    total_sec = n / SR
    print(f"[WAV] path={INPUT_WAV}  sr_in={sr} -> sr={SR}  n={n} ({total_sec:.3f}s)")

    # 모델 로드
    scaler, lr, mlp, local_class_names, thresholds_local = load_models(MODEL_DIR)
    print(f"[MDL] local class_names={local_class_names}")
    print(f"[MDL] thresholds_local={thresholds_local}")

    # 로컬 순서를 CANONICAL 순서로 재정렬할 인덱스 계산
    reorder = make_reorder(local_class_names, CANONICAL_CLASSES)
    if any(idx is None for idx in reorder):
        raise RuntimeError(f"Model class names do not cover canonical set. reorder={reorder} local={local_class_names} canonical={CANONICAL_CLASSES}")
    reorder = np.array(reorder, dtype=int)
    thresholds = thresholds_local[reorder]
    print(f"[MDL] canonical UI classes: {CANONICAL_CLASSES}")
    print(f"[MDL] reorder local->{reorder} -> thresholds(aligned)={thresholds}")

    # 전구간 슬라이딩 → Log-Mel → 시간평균(64,) → per-sample 표준화
    feats = []
    spans = []
    for i0, i1 in segment_indices(n, SR, WIN, HOP):
        seg = x[i0:i1]
        lm = logmel_64(seg, SR)              # (64, T)
        v = lm.mean(axis=1).astype(np.float32)  # (64,)
        v = per_sample_standardize(v)
        feats.append(v)
        spans.append((i0/SR, i1/SR))
    if not feats:
        raise RuntimeError("No segments generated. Check WAV length or parameters.")
    X = np.stack(feats, axis=0)              # (Nwin, 64)
    debug_feature_stats(X)

    # 스케일러 적용
    Xs = scaler.transform(X)

    # LR/MLP softmax → 앙상블 평균
    P_lr  = lr.predict_proba(Xs)
    P_mlp = mlp.predict_proba(Xs)
    # 로컬 → CANONICAL 순서로 재배열
    P = 0.5 * P_lr[:, reorder] + 0.5 * P_mlp[:, reorder]   # (Nwin, K)
    assert P.shape[1] == len(CANONICAL_CLASSES)

    # threshold 보정 점수 및 argmax 예측
    adj = P - thresholds[None, :]    # (Nwin, K)
    y_idx = np.argmax(adj, axis=1)

    # 구간별 CSV 저장
    seg_csv = OUT_DIR / f"{INPUT_WAV.stem}__segments.csv"
    with open(seg_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["t0", "t1"] + [f"prob_{c}" for c in CANONICAL_CLASSES] + ["pred_label"]
        w.writerow(header)
        for (s, e), probs, yi in zip(spans, P, y_idx):
            row = [f"{s:.2f}", f"{e:.2f}"] + [f"{float(p):.6f}" for p in probs] + [CANONICAL_CLASSES[int(yi)]]
            w.writerow(row)

    # 파일 단위 집계: 평균 softmax, 최빈 라벨, 분포
    mean_prob = P.mean(axis=0)
    labels = [CANONICAL_CLASSES[int(i)] for i in y_idx]
    # 라벨 분포
    dist = {c: 0 for c in CANONICAL_CLASSES}
    for lb in labels:
        dist[lb] += 1
    maj_label = max(dist.items(), key=lambda kv: kv[1])[0]

    agg_csv = OUT_DIR / f"{INPUT_WAV.stem}__file_aggregate.csv"
    with open(agg_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "n_windows"] + [f"mean_prob_{c}" for c in CANONICAL_CLASSES] + ["majority_label"] + [f"count_{c}" for c in CANONICAL_CLASSES])
        w.writerow(
            [str(INPUT_WAV), len(spans)]
            + [f"{float(p):.6f}" for p in mean_prob]
            + [maj_label]
            + [dist[c] for c in CANONICAL_CLASSES]
        )

    # 요약 출력
    print(f"[OUT] segments.csv  -> {seg_csv}")
    print(f"[OUT] file_aggregate.csv -> {agg_csv}")
    print(f"[SUM] windows={len(spans)}  majority={maj_label}  mean_prob={ {c: float(p) for c,p in zip(CANONICAL_CLASSES, mean_prob)} }")
    print(f"[OK ] done in {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
