#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, math, time
import numpy as np
import pandas as pd

# Matplotlib headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from collections import Counter
from typing import Tuple

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, recall_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# -------------------------------
# 0) 경로/디렉토리
# -------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = '//home//un_wang//my_stethoscope_project'
FEATURES_DIR = os.path.join(PROJECT_DIR, "features")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "Sampler")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
MODELS_DIR = os.path.join(PROJECT_DIR, "models", "Sampler")


os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

CSV_PATH = os.path.join(FEATURES_DIR, "opera_features.csv")

print(f"[RUN] step2_sampler_logreg_torch.py", flush=True)
print(f"[PATH] CSV_PATH={CSV_PATH}", flush=True)
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"opera_features.csv not found at: {CSV_PATH}")

# -------------------------------
# 1) 데이터 로드 & 정리
# -------------------------------
df = pd.read_csv(CSV_PATH)
df = df[df["label"] != "unknown"].reset_index(drop=True)

def _patient_from_fname(x: str) -> str:
    return str(x).split("_")[0]

df["patient_id"] = df["filename"].apply(_patient_from_fname)

drop_cols = ["filename", "label", "extraction_success", "patient_id"]
feature_cols = [c for c in df.columns if c not in drop_cols]

X_all = df[feature_cols].values.astype(np.float32)
y_all = df["label"].values
groups_all = df["patient_id"].values
filenames_all = df["filename"].values

class_names = np.sort(np.unique(y_all))
class_to_idx = {c:i for i,c in enumerate(class_names)}
y_idx_all = np.array([class_to_idx[c] for c in y_all], dtype=np.int64)

print(f"[INFO] samples={len(df)}, classes={list(class_names)}, features={len(feature_cols)}")
print(df["label"].value_counts().to_string())


# -------------------------------
# 2) Torch Dataset  (교체)
# -------------------------------
class NumpyDataset(Dataset):
    def __init__(self, X: np.ndarray, y_idx: np.ndarray, filenames=None, patient_ids=None):
        self.X = X
        self.y = y_idx
        self.filenames = filenames  # None 허용
        self.patient_ids = patient_ids  # None 허용
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        fn = self.filenames[i] if self.filenames is not None else ""
        pid = self.patient_ids[i] if self.patient_ids is not None else ""
        return (
            torch.from_numpy(self.X[i]),
            int(self.y[i]),
            fn,
            pid
        )
# -------------------------------
# 3) 모델 (로지스틱 회귀 = 선형층)
# -------------------------------
class TorchLogReg(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)
    def forward(self, x):
        return self.linear(x)  # logits

def train_one_fold(
    X_tr, y_tr, X_te, y_te, filenames_te, patients_te,
    class_weights: torch.Tensor,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
    use_sampler: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    returns: y_true(idx), y_pred(idx), prob (n_test, n_classes)
    """
    n_class = len(class_weights)
    in_dim = X_tr.shape[1]

    # 표준화는 train 통계로만 (누수 방지)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)

    # Dataset
    ds_tr = NumpyDataset(X_tr, y_tr, filenames=None, patient_ids=None)
    ds_te = NumpyDataset(X_te, y_te, filenames_te, patients_te)

    # WeightedRandomSampler (train 전용)
    if use_sampler:
        # 클래스별 빈도로 샘플 가중치 계산: weight = 1/freq(class)
        cls_counts = Counter(y_tr.tolist())
        weights = np.array([1.0 / cls_counts[c] for c in y_tr], dtype=np.float32)

        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),     # train 샘플 수만큼 에폭당 샘플링
            replacement=True
        )

        dl_tr = DataLoader(
            ds_tr,
            batch_size=batch_size,
            sampler=sampler,              # 샘플러 사용 시 shuffle=False 필수
            drop_last=False,
            num_workers=0,                # 워커 포크 이슈 회피
            pin_memory=(device == "cuda")
        )
    else:
        dl_tr = DataLoader(
            ds_tr,
            batch_size=batch_size,
            shuffle=True,                 # 샘플러 미사용 시 셔플
            drop_last=False,
            num_workers=0,
            pin_memory=(device == "cuda")
        )

    # 평가(검증) DataLoader — 절대 샘플링/셔플 X
    dl_te = DataLoader(
        ds_te,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=(device == "cuda")
    )

    # 모델/옵티마/손실
    model = TorchLogReg(in_dim, n_class).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # class-weighted cross entropy (소수 클래스 오류에 더 큰 페널티)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    model.train()
    for ep in range(1, epochs+1):
        epoch_loss = 0.0
        for xb, yb, _, _ in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * xb.size(0)
        if ep % 5 == 0 or ep == 1:
            print(f"[EP {ep:02d}] loss={epoch_loss/len(ds_tr):.4f}", flush=True)

    # 평가
    model.eval()
    all_true, all_pred = [], []
    all_prob = []
    with torch.no_grad():
        for xb, yb, fns, pats in dl_te:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            yhat = probs.argmax(axis=1)
            all_true.append(yb.numpy())
            all_pred.append(yhat)
            all_prob.append(probs)

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    prob = np.vstack(all_prob)
    return y_true, y_pred, prob

# -------------------------------
# 4) GroupKFold CV (환자 누수 방지)
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DEVICE] {device}")

gkf = GroupKFold(n_splits=5)

all_true, all_pred, all_prob = [], [], []
all_file, all_pat, all_fold = [], [], []

n_classes = len(class_names)
# class weight = 1 / freq (전체 데이터 기준) → 과도하면 sqrt/alpha 조정 가능
global_counts = Counter(y_idx_all.tolist())
cw = np.zeros(n_classes, dtype=np.float32)
for c_idx in range(n_classes):
    cw[c_idx] = 1.0 / max(global_counts[c_idx], 1)
# 정규화(선택): 합= n_classes 로 스케일링
cw = cw * (n_classes / cw.sum())
class_weights = torch.tensor(cw, dtype=torch.float32)

fold_id = 0
for tr_idx, te_idx in gkf.split(X_all, y_idx_all, groups_all):
    fold_id += 1
    print(f"\n[FOLD {fold_id}] train={len(tr_idx)} test={len(te_idx)}", flush=True)

    X_tr, y_tr = X_all[tr_idx], y_idx_all[tr_idx]
    X_te, y_te = X_all[te_idx], y_idx_all[te_idx]
    files_te = filenames_all[te_idx]
    pats_te = groups_all[te_idx]

    # 학습 (WeightedRandomSampler + class-weighted CE)
    y_true, y_pred, prob = train_one_fold(
        X_tr, y_tr, X_te, y_te, files_te, pats_te,
        class_weights=class_weights,
        epochs=30, batch_size=64, lr=1e-3,
        device=device, use_sampler=True
    )

    all_true.append(y_true)
    all_pred.append(y_pred)
    all_prob.append(prob)
    all_file.extend(files_te)
    all_pat.extend(pats_te)
    all_fold.extend([fold_id]*len(te_idx))

# -------------------------------
# 5) 누적 성능/로그 저장
# -------------------------------
y_true_all = np.concatenate(all_true, axis=0)
y_pred_all = np.concatenate(all_pred, axis=0)
probs_all = np.vstack(all_prob)

print("\n--- Aggregate Classification Report ---")
print(classification_report(
    [class_names[i] for i in y_true_all],
    [class_names[i] for i in y_pred_all],
    labels=list(class_names),
    zero_division=0
))
macro_recall = recall_score(y_true_all, y_pred_all, average="macro", zero_division=0)
print(f"[MACRO RECALL] {macro_recall:.3f}")

# 혼동행렬 (aggregate)
cm = confusion_matrix(y_true_all, y_pred_all, labels=np.arange(len(class_names)))
fig, ax = plt.subplots(figsize=(5,4))
im = ax.imshow(cm, interpolation="nearest")
ax.figure.colorbar(im, ax=ax)
ax.set(
    xticks=np.arange(len(class_names)),
    yticks=np.arange(len(class_names)),
    xticklabels=list(class_names),
    yticklabels=list(class_names),
    xlabel="Predicted",
    ylabel="True",
    title="TorchLogReg - Confusion Matrix (Aggregate)"
)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
for (i,j), v in np.ndenumerate(cm):
    ax.text(j, i, str(v), ha='center', va='center')
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "TorchLogReg_cm_aggregate.png"), dpi=150)
plt.close(fig)

# 예측 로그 저장
out_rows = {
    "filename": all_file,
    "patient_id": all_pat,
    "fold": all_fold,
    "y_true": [class_names[i] for i in y_true_all],
    "y_pred": [class_names[i] for i in y_pred_all],
}
pred_df = pd.DataFrame(out_rows)
for i, c in enumerate(class_names):
    pred_df[f"prob_{c}"] = probs_all[:, i]
pred_path = os.path.join(RESULTS_DIR, "TorchLogReg_cv_predictions.csv")
pred_df.to_csv(pred_path, index=False)
print(f"[SAVE] {pred_path}")

# 요약 저장
rpt = classification_report(
    [class_names[i] for i in y_true_all],
    [class_names[i] for i in y_pred_all],
    labels=list(class_names),
    output_dict=True,
    zero_division=0
)
summary_df = pd.DataFrame([{
    "model": "TorchLogReg",
    "accuracy": rpt["accuracy"],
    "precision_macro": rpt["macro avg"]["precision"],
    "recall_macro": rpt["macro avg"]["recall"],
    "f1_macro": rpt["macro avg"]["f1-score"],
}]).set_index("model")
sum_path = os.path.join(RESULTS_DIR, "classifier_groupcv_results.csv")
summary_df.to_csv(sum_path)
print("\n=== GroupKFold Aggregated Results ===")
print(summary_df)
print(f"[SAVE] {sum_path}")
print("[DONE]")
