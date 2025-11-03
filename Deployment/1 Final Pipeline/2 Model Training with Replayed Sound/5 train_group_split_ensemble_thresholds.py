# -*- coding: utf-8 -*-
# train_group_split_ensemble_thresholds.py
# GroupShuffleSplit(원본 파일 기준 그룹), StandardScaler, LR+MLP 앙상블,
# OVR ROC 기반 per-class threshold(tau) 산출 + test Confusion Matrix/Report 저장
# ※ 클래스명 정규화: _window 제거, Nonbreathing → Non-breathing

import re
import json
import time
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt

# ====== 절대경로 (변경/축약 없음) ======
TRAIN_ROOT  = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Deployment/1 Final Pipeline/2 Model Training with Replayed Sound")
FEAT_NPZ    = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Deployment/1 Final Pipeline/2 Model Training with Replayed Sound/features/features_64mel.npz")
MODEL_ROOT  = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Deployment/1 Final Pipeline/2 Model Training with Replayed Sound/model")
RESULT_ROOT = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Deployment/1 Final Pipeline/2 Model Training with Replayed Sound/result")

# 정규식: 세그먼트 파일명 끝의 _t0-t1 제거용
_TSPAN_RE = re.compile(r"_([0-9]+(?:\.[0-9]+)?)-([0-9]+(?:\.[0-9]+)?)$")

# 프로젝트 표준 클래스(순서 고정)
CANONICAL_CLASSES = ['Crackle', 'Healthy', 'Non-breathing', 'Rhonchi', 'Wheezing']

def stem_without_time_range(stem: str) -> str:
    m = _TSPAN_RE.search(stem)
    if not m:
        return stem
    return stem[:m.start()]  # "_t0-t1" 직전까지

def filename_to_group(filename: str) -> str:
    p = Path(filename)
    return stem_without_time_range(p.stem)

def normalize_label(s: str) -> str:
    """
    - '_window' 접미사 제거
    - 'Nonbreathing' → 'Non-breathing'
    - 대소문자/하이픈/언더스코어 변형을 흡수하여 CANONICAL_CLASSES로 매핑
    """
    t = s.strip()
    # _window 제거(대소문자 무시)
    if t.lower().endswith("_window"):
        t = t[:-(len("_window"))]
    # 공백 제거
    t = t.replace(" ", "")
    # 특이 케이스 정규화
    t_low_flat = t.lower().replace("_", "").replace("-", "")
    key_map = {
        "crackle": "Crackle",
        "healthy": "Healthy",
        "rhonchi": "Rhonchi",
        "wheezing": "Wheezing",
        "nonbreathing": "Non-breathing",
        "nonbreathings": "Non-breathing",  # 오타 방지
        "nonbreathingg": "Non-breathing",
        "nonbreathingwindow": "Non-breathing",
        "nonbreathingnb": "Non-breathing",
        "nonbreathingnonbreathing": "Non-breathing",
        "nonbreathingnonbreath": "Non-breathing",
        "nonbreathingnon-breathing": "Non-breathing",
        "nonbreathingnon_breathing": "Non-breathing",
        "non-breathing": "Non-breathing",
        "non_breathing": "Non-breathing",
    }
    # 일반 케이스 빠르게 처리
    if t_low_flat in key_map:
        return key_map[t_low_flat]
    # 기본 매핑 시도
    for k, v in {
        "crackle": "Crackle",
        "healthy": "Healthy",
        "rhonchi": "Rhonchi",
        "wheezing": "Wheezing",
        "nonbreathing": "Non-breathing",
        "nonbreath": "Non-breathing",
    }.items():
        if t_low_flat == k:
            return v
    # 혹시 이미 표준형이면 그대로
    if t in CANONICAL_CLASSES:
        return t
    # 마지막 안전장치: 대소문자 표준화로 매칭 시도
    t_simple = t.replace("_", "-")
    if t_simple.lower() in ["non-breathing", "nonbreathing"]:
        return "Non-breathing"
    # 매칭 실패 시 원본 반환(추후 assert로 검출)
    return t

def compute_thresholds_ovr(P_va: np.ndarray, y_va_idx: np.ndarray, n_classes: int):
    """Youden J (tpr - fpr) 최댓값으로 클래스별 threshold 선택"""
    thresholds = []
    for k in range(n_classes):
        y_true = (y_va_idx == k).astype(int)
        fpr, tpr, thr = roc_curve(y_true, P_va[:, k])
        j = tpr - fpr
        j_best = int(np.argmax(j))
        thresholds.append(float(thr[j_best]))
    return np.array(thresholds, dtype=float)

def main():
    # ===== 특징 로드 =====
    d = np.load(FEAT_NPZ, allow_pickle=True)
    X          = d["X"]                        # (N, 64)
    y_raw      = d["y"].astype(str)            # (N,)
    filenames  = d["filenames"].astype(str)    # (N,)
    classes_in = d["classes"].astype(str).tolist()

    if X.shape[0] == 0:
        raise RuntimeError("No features found. Run extract_Logmel.py first.")

    # ===== 라벨/클래스 정규화 =====
    y = np.array([normalize_label(s) for s in y_raw], dtype=str)
    classes_present = sorted(list(set(y.tolist())), key=lambda c: CANONICAL_CLASSES.index(c) if c in CANONICAL_CLASSES else 999)

    # 클래스 5개 모두 존재하는지 보증 (없으면 중단)
    missing = [c for c in CANONICAL_CLASSES if c not in classes_present]
    extra   = [c for c in classes_present if c not in CANONICAL_CLASSES]
    print(f"[CHK] classes_in(npz): {classes_in}")
    print(f"[CHK] classes_present(normalized from y): {classes_present}")
    if missing or extra:
        raise RuntimeError(f"Class mismatch after normalization. missing={missing}, extra={extra}")

    classes = CANONICAL_CLASSES[:]  # 표준 순서 고정
    print(f"[OK ] normalized class names (canonical order): {classes}")

    # ===== 그룹 = 원본파일 단위 (세그먼트 파일명에서 _t0-t1 제거) =====
    groups = np.array([filename_to_group(fn) for fn in filenames])

    # ===== 분할: 80% (train+val) / 20% (test) =====
    gss_outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    trval_idx, te_idx = next(gss_outer.split(X, y, groups))
    X_trval, y_trval, g_trval = X[trval_idx], y[trval_idx], groups[trval_idx]

    # ===== 분할: train+val 중 10%를 val로 (전체 비율로 8%) =====
    gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.10/0.80, random_state=42)
    tr_rel, va_rel = next(gss_inner.split(X_trval, y_trval, g_trval))
    tr_idx = trval_idx[tr_rel]
    va_idx = trval_idx[va_rel]

    X_tr, X_va, X_te = X[tr_idx], X[va_idx], X[te_idx]
    y_tr, y_va, y_te = y[tr_idx], y[va_idx], y[te_idx]

    # ===== 인덱스 인코딩 (클래스 순서 = CANONICAL_CLASSES) =====
    cls2idx = {c: i for i, c in enumerate(classes)}
    y_tr_idx = np.array([cls2idx[c] for c in y_tr])
    y_va_idx = np.array([cls2idx[c] for c in y_va])
    y_te_idx = np.array([cls2idx[c] for c in y_te])

    # ===== 스케일러 / 모델 =====
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    lr  = LogisticRegression(max_iter=2000, multi_class="multinomial", random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=200, random_state=42)

    lr.fit(X_tr_s, y_tr_idx)
    mlp.fit(X_tr_s, y_tr_idx)

    # ===== 앙상블 확률 =====
    P_va = 0.5 * lr.predict_proba(X_va_s) + 0.5 * mlp.predict_proba(X_va_s)
    P_te = 0.5 * lr.predict_proba(X_te_s) + 0.5 * mlp.predict_proba(X_te_s)

    # ===== per-class threshold (val 기준, Youden J) =====
    thresholds = compute_thresholds_ovr(P_va, y_va_idx, n_classes=len(classes))

    # ===== run/result 디렉토리 =====
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = MODEL_ROOT  / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    res_dir = RESULT_ROOT / run_dir.name
    run_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    # ===== 아티팩트 저장 =====
    joblib.dump(scaler, run_dir / "scaler.pkl")
    joblib.dump(lr,     run_dir / "model_lr.pkl")
    joblib.dump(mlp,    run_dir / "model_mlp.pkl")
    with open(run_dir / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump({"class_names": classes, "thresholds": thresholds.tolist()}, f, ensure_ascii=False, indent=2)

    # ======= 평가 (test): confusion matrix + report 저장 =======
    # threshold 적용 예측
    adj_te   = P_te - thresholds[None, :]   # (N_test, K)
    y_pred_te = np.argmax(adj_te, axis=1)

    # Confusion Matrix (% by true label)
    cm = confusion_matrix(y_te_idx, y_pred_te, labels=np.arange(len(classes)))
    cm_percent = cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cm_percent = np.nan_to_num(cm_percent) * 100.0

    # 시각화 저장
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_percent, interpolation="nearest", cmap="Blues")
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cm_percent[i, j]
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    color=("white" if val > 50 else "black"), fontsize=9)
    ax.set_title("Confusion Matrix (Percent by True Label) - TEST")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(classes))); ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    (res_dir / "confusion_matrix.png").parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(res_dir / "confusion_matrix.png")
    plt.close()

    # Classification Report 저장
    rep_txt = classification_report(y_te_idx, y_pred_te, target_names=classes, digits=4)
    with open(res_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write("== Classes (canonical) ==\n")
        f.write(", ".join(classes) + "\n\n")
        f.write("== Thresholds (val, Youden J) ==\n")
        for c, t in zip(classes, thresholds):
            f.write(f"{c:>14}: {t:.4f}\n")
        f.write("\n== Classification Report (TEST) ==\n")
        f.write(rep_txt)

    # Summary 저장
    summary = {
        "run_id": run_dir.name,
        "n_total": int(X.shape[0]),
        "n_train": int(len(tr_idx)),
        "n_val": int(len(va_idx)),
        "n_test": int(len(te_idx)),
        "classes": classes,
        "split": {"train": 0.72, "val": 0.08, "test": 0.20},
    }
    with open(res_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[DONE] classes (canonical, {len(classes)}): {classes}")
    print(f"[DONE] thresholds (val/YoudenJ): {thresholds}")
    print(f"[SAVE] model : {run_dir}")
    print(f"[SAVE] result: {res_dir}")
    print(f"[INFO] counts: train={len(tr_idx)}, val={len(va_idx)}, test={len(te_idx)}")

if __name__ == "__main__":
    main()
