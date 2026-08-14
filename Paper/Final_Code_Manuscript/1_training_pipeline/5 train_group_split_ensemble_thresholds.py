# -*- coding: utf-8 -*-
# train_group_split_ensemble_thresholds.py
# (변경 요약)
#  - 분할을 9:1 (train:val) 한 번만 수행 (GroupShuffleSplit)
#  - test 세트 및 관련 지표/이미지 저장 로직 제거
#  - 나머지 파이프라인/포맷 유지

import re
import json
import time
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve
import joblib

# ====== 절대경로 (변경/축약 없음) ======
TRAIN_ROOT  = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Deployment/1 Final Pipeline/5 Realtime Pipeline_Final")
FEAT_NPZ    = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Deployment/1 Final Pipeline/5 Realtime Pipeline_Final/features/features_64mel.npz")
MODEL_ROOT  = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Deployment/1 Final Pipeline/5 Realtime Pipeline_Final/model")
RESULT_ROOT = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Deployment/1 Final Pipeline/5 Realtime Pipeline_Final/result")

_TSPAN_RE = re.compile(r"_([0-9]+(?:\.[0-9]+)?)-([0-9]+(?:\.[0-9]+)?)$")

CANONICAL_CLASSES = ['Crackle', 'Healthy', 'Non-breathing', 'Rhonchi', 'Wheezing']

def stem_without_time_range(stem: str) -> str:
    m = _TSPAN_RE.search(stem)
    if not m:
        return stem
    return stem[:m.start()]

def filename_to_group(filename: str) -> str:
    p = Path(filename)
    return stem_without_time_range(p.stem)

def normalize_label(s: str) -> str:
    t = s.strip()
    if t.lower().endswith("_window"):
        t = t[:-(len("_window"))]
    t = t.replace(" ", "")
    t_low_flat = t.lower().replace("_", "").replace("-", "")
    key_map = {
        "crackle": "Crackle",
        "healthy": "Healthy",
        "rhonchi": "Rhonchi",
        "wheezing": "Wheezing",
        "nonbreathing": "Non-breathing",
        "nonbreathingwindow": "Non-breathing",
        "non-breathing": "Non-breathing",
        "non_breathing": "Non-breathing",
    }
    if t_low_flat in key_map:
        return key_map[t_low_flat]
    if t in CANONICAL_CLASSES:
        return t
    t_simple = t.replace("_", "-")
    if t_simple.lower() in ["non-breathing", "nonbreathing"]:
        return "Non-breathing"
    return t

def compute_thresholds_ovr(P_va: np.ndarray, y_va_idx: np.ndarray, n_classes: int):
    thresholds = []
    for k in range(n_classes):
        y_true = (y_va_idx == k).astype(int)
        fpr, tpr, thr = roc_curve(y_true, P_va[:, k])
        j = tpr - fpr
        j_best = int(np.argmax(j))
        thresholds.append(float(thr[j_best]))
    return np.array(thresholds, dtype=float)

def main():
    d = np.load(FEAT_NPZ, allow_pickle=True)
    X          = d["X"]                       # (N, 64)
    y_raw      = d["y"].astype(str)           # (N,)
    filenames  = d["filenames"].astype(str)   # (N,)
    classes_in = d["classes"].astype(str).tolist()

    if X.shape[0] == 0:
        raise RuntimeError("No features found. Run extract_Logmel.py first.")

    y = np.array([normalize_label(s) for s in y_raw], dtype=str)
    classes_present = sorted(list(set(y.tolist())), key=lambda c: CANONICAL_CLASSES.index(c) if c in CANONICAL_CLASSES else 999)
    missing = [c for c in CANONICAL_CLASSES if c not in classes_present]
    extra   = [c for c in classes_present if c not in CANONICAL_CLASSES]
    print(f"[CHK] classes_in(npz): {classes_in}")
    print(f"[CHK] classes_present(normalized from y): {classes_present}")
    if missing or extra:
        raise RuntimeError(f"Class mismatch after normalization. missing={missing}, extra={extra}")

    classes = CANONICAL_CLASSES[:]
    print(f"[OK ] normalized class names (canonical order): {classes}")

    groups = np.array([filename_to_group(fn) for fn in filenames])

    # ===== 9:1 (train:val) =====
    gss = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
    tr_idx, va_idx = next(gss.split(X, y, groups))

    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    g_tr, g_va = groups[tr_idx], groups[va_idx]

    cls2idx = {c: i for i, c in enumerate(classes)}
    y_tr_idx = np.array([cls2idx[c] for c in y_tr])
    y_va_idx = np.array([cls2idx[c] for c in y_va])

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_va_s = scaler.transform(X_va)

    lr  = LogisticRegression(max_iter=2000, multi_class="multinomial", random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=200, random_state=42)

    lr.fit(X_tr_s, y_tr_idx)
    mlp.fit(X_tr_s, y_tr_idx)

    # 앙상블 확률(VAL)
    P_va = 0.5 * lr.predict_proba(X_va_s) + 0.5 * mlp.predict_proba(X_va_s)

    # per-class threshold (val, Youden J)
    thresholds = compute_thresholds_ovr(P_va, y_va_idx, n_classes=len(classes))

    # ===== 저장 =====
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = MODEL_ROOT  / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    res_dir = RESULT_ROOT / run_dir.name
    run_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, run_dir / "scaler.pkl")
    joblib.dump(lr,     run_dir / "model_lr.pkl")
    joblib.dump(mlp,    run_dir / "model_mlp.pkl")
    with open(run_dir / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump({"class_names": classes, "thresholds": thresholds.tolist()}, f, ensure_ascii=False, indent=2)

    summary = {
        "run_id": run_dir.name,
        "n_total": int(X.shape[0]),
        "n_train": int(len(tr_idx)),
        "n_val": int(len(va_idx)),
        "classes": classes,
        "split": {"train": 0.90, "val": 0.10, "test": 0.00},
        "groups": {"train_unique": len(set(g_tr.tolist())), "val_unique": len(set(g_va.tolist()))}
    }
    with open(res_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[DONE] classes (canonical, {len(classes)}): {classes}")
    print(f"[DONE] thresholds (val/YoudenJ): {thresholds}")
    print(f"[SAVE] model : {run_dir}")
    print(f"[SAVE] result: {res_dir}")
    print(f"[INFO] counts: train={len(tr_idx)}, val={len(va_idx)}")
    print(f"[INFO] groups : train={len(set(g_tr.tolist()))}, val={len(set(g_va.tolist()))}")

if __name__ == "__main__":
    main()
