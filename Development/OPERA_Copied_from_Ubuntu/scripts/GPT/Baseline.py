#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import pandas as pd
import joblib

# (중요) 헤드리스 환경에서도 절대 멈추지 않도록 Agg 백엔드 고정
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, recall_score, confusion_matrix

# -------------------------------
# 0) 경로/디렉토리
# -------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = '//home//un_wang//my_stethoscope_project'
FEATURES_DIR = os.path.join(PROJECT_DIR, "features")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "Baseline")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
MODELS_DIR = os.path.join(PROJECT_DIR, "models", "Baseline")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

CSV_PATH = os.path.join(FEATURES_DIR, "opera_features.csv")
print(f"[RUN] classifier_on_Opera.py started", flush=True)
print(f"[PATH] PROJECT_DIR={PROJECT_DIR}", flush=True)
print(f"[PATH] CSV_PATH={CSV_PATH}", flush=True)

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"opera_features.csv not found at: {CSV_PATH}")

# -------------------------------
# 1) 데이터 로드 & 정리
# -------------------------------
print("[STEP] Loading CSV...", flush=True)
df = pd.read_csv(CSV_PATH)
print(f"[INFO] df.shape={df.shape}", flush=True)
print(f"[INFO] df.columns={list(df.columns)}", flush=True)

if "label" not in df.columns or "filename" not in df.columns:
    raise ValueError("opera_features.csv에 'label' 또는 'filename' 컬럼이 없습니다.")

# unknown 제거
before = len(df)
df = df[df["label"] != "unknown"].reset_index(drop=True)
print(f"[CLEAN] removed 'unknown': {before} -> {len(df)} rows", flush=True)

# patient_id 추출 (파일명 규칙: 'PATIENT_...wav' 가정)
def _patient_from_fname(x: str) -> str:
    return str(x).split("_")[0]

df["patient_id"] = df["filename"].apply(_patient_from_fname)

drop_cols = ["filename", "label", "extraction_success", "patient_id"]
feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols].values
y = df["label"].values
groups = df["patient_id"].values
filenames = df["filename"].values

class_names = np.sort(np.unique(y))
print(f"[INFO] samples={len(df)}, classes={list(class_names)}, features={len(feature_cols)}", flush=True)
print(df["label"].value_counts().to_string(), flush=True)

# -------------------------------
# 2) 분류기 (베이스라인)
# -------------------------------
classifiers = {
    "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1),
    "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
}

# -------------------------------
# 3) GroupKFold CV (누수 방지)
# -------------------------------
gkf = GroupKFold(n_splits=5)
summary_rows = []

for name, clf in classifiers.items():
    print(f"\n=== {name} (GroupKFold=5) ===", flush=True)
    all_true, all_pred, all_prob = [], [], []
    all_pat, all_file, all_fold = [], [], []

    fold_idx = 0
    for tr_idx, te_idx in gkf.split(X, y, groups):
        fold_idx += 1
        print(f"[FOLD] {name} fold={fold_idx} | train={len(tr_idx)} test={len(te_idx)}", flush=True)

        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        pat_te = groups[te_idx]
        file_te = filenames[te_idx]

        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        pipe.fit(X_tr, y_tr)

        y_hat = pipe.predict(X_te)
        all_true.extend(y_te); all_pred.extend(y_hat)
        all_pat.extend(pat_te); all_file.extend(file_te)
        all_fold.extend([fold_idx]*len(te_idx))

        # 확률(또는 decision function) 저장 시도
        try:
            proba = pipe.predict_proba(X_te)
            idx_map = {c:i for i,c in enumerate(pipe.classes_)}
            reorder = np.array([idx_map[c] for c in class_names])
            all_prob.append(proba[:, reorder])
        except Exception:
            all_prob.append(None)

        # (선택) 폴드별 혼동행렬 저장
        cm = confusion_matrix(y_te, y_hat, labels=class_names)
        fig, ax = plt.subplots(figsize=(5,4))
        im = ax.imshow(cm, interpolation="nearest")
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            xlabel="Predicted",
            ylabel="True",
            title=f"{name} - Confusion Matrix (Fold {fold_idx})"
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        for (i,j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha='center', va='center')
        fig.tight_layout()
        out_png = os.path.join(FIGURES_DIR, f"{name}_cm_fold{fold_idx}.png")
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"[SAVE] {out_png}", flush=True)

    # 누적 리포트
    print("\n--- Aggregate Classification Report ---", flush=True)
    print(classification_report(all_true, all_pred, labels=class_names, zero_division=0), flush=True)
    macro_recall = recall_score(all_true, all_pred, labels=class_names, average="macro", zero_division=0)
    print(f"[MACRO RECALL] {macro_recall:.3f}", flush=True)

    # 예측 로그 저장
    if any(p is not None for p in all_prob):
        try:
            probs = np.vstack([p for p in all_prob if p is not None])
            prob_cols = [f"prob_{c}" for c in class_names]
        except Exception:
            probs = None
            prob_cols = []
    else:
        probs = None
        prob_cols = []

    rows = {"filename": all_file, "patient_id": all_pat, "fold": all_fold, "y_true": all_true, "y_pred": all_pred}
    pred_df = pd.DataFrame(rows)
    if probs is not None and len(pred_df) == len(probs):
        for i, col in enumerate(prob_cols):
            pred_df[col] = probs[:, i]

    pred_path = os.path.join(RESULTS_DIR, f"{name}_cv_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"[SAVE] {pred_path}", flush=True)

    # 요약
    rpt = classification_report(all_true, all_pred, labels=class_names, output_dict=True, zero_division=0)
    summary_rows.append({
        "model": name,
        "accuracy": rpt["accuracy"],
        "precision_macro": rpt["macro avg"]["precision"],
        "recall_macro": rpt["macro avg"]["recall"],
        "f1_macro": rpt["macro avg"]["f1-score"],
    })

    # 전체 데이터로 재학습 저장
    final_pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    final_pipe.fit(X, y)
    model_path = os.path.join(MODELS_DIR, f"{name}.joblib")
    joblib.dump(final_pipe, model_path)
    print(f"[SAVE] {model_path}", flush=True)

# 성능 요약 저장/출력
summary_df = pd.DataFrame(summary_rows).set_index("model")
sum_path = os.path.join(RESULTS_DIR, "classifier_groupcv_results.csv")
summary_df.to_csv(sum_path)
print("\n=== GroupKFold Aggregated Results ===", flush=True)
print(summary_df, flush=True)
print(f"[SAVE] {sum_path}", flush=True)
print("[DONE]", flush=True)
