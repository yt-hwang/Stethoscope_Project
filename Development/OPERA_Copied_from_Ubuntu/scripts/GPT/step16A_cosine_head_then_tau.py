#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step16A: Cosine Head + Per-class Tau Search (Macro Recall objective with NB-precision constraint)
Author: ChatGPT
Notes:
- Reads OPERA feature CSV (one row per audio sample with columns: filename, label, extraction_success, and feature columns)
- Uses patient-based GroupKFold to avoid leakage.
- Trains a lightweight CosineClassifier on feature vectors.
- After each fold's training, performs per-class tau (temperature) search to maximize macro recall
  subject to a minimum Non-breathing (NB) precision constraint.
- Logs per-fold metrics and chosen taus to a CSV for easy copy into your master Excel log.
"""

import os
import json
import math
import time
import random
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from collections import Counter

# =========================
# Paths & Config (EDIT HERE)
# =========================
CSV_PATH = r"D:\Stethoscope_Project\Development\OPERA_Copied_from_Ubuntu\features\opera_features.csv"
RESULTS_DIR = r"D:\Stethoscope_Project\Development\OPERA_Copied_from_Ubuntu\scripts\GPT\results_step16A"
EXPERIMENT_TAG = "Step16A_CosHead_tau_NBprec_ge_0.35"
RANDOM_SEED = 42

# Training
EPOCHS = 60
BATCH_SIZE = 128
LR = 3e-4
WD = 1e-4
SCALE_S = 16.0  # try 16.0 or 32.0

# Tau search
TAU_GRID = np.linspace(0.6, 1.8, 13)  # per-class tau grid
NB_PREC_MIN = 0.35                    # NB precision constraint

# ===============
# Utility helpers
# ===============
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def parse_patient_id_from_filename(path_str: str) -> str:
    # Robust patient-id parse: use base name, split by '_' or '-' and take the first token
    base = os.path.basename(path_str).split('.')[0]
    if '_' in base:
        return base.split('_')[0]
    if '-' in base:
        return base.split('-')[0]
    return base

def as_tensor(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)

# ==============
# Data handling
# ==============
class FeatureDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# =====================
# Cosine Classifier head
# =====================
class CosineClassifier(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, s: float = 16.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_classes, in_dim))
        nn.init.xavier_normal_(self.W)
        self.s = s
    def forward(self, x):
        x = F.normalize(x, dim=1)
        W = F.normalize(self.W, dim=1)
        return self.s * x @ W.t()

# ==================
# Metrics & Searching
# ==================
def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> List[float]:
    recs = []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        recs.append(float(tp / (tp + fn + 1e-9)))
    return recs

def precision_of_class(y_true: np.ndarray, y_pred: np.ndarray, c: int) -> float:
    tp = np.sum((y_true == c) & (y_pred == c))
    fp = np.sum((y_true != c) & (y_pred == c))
    return float(tp / (tp + fp + 1e-9))

def search_per_class_tau(probs: np.ndarray, y_true: np.ndarray, nb_idx: int, 
                         num_classes: int, grid: np.ndarray, nb_prec_min: float) -> Tuple[np.ndarray, float, Dict]:
    """
    Coordinate descent over per-class tau to maximize macro recall, with NB precision >= nb_prec_min constraint.
    """
    tau = np.ones(num_classes, dtype=np.float32)
    best_score = -1e9
    best_tau = tau.copy()
    history = []

    def objective(tau_vec):
        q = probs / tau_vec.reshape(1, -1)  # temperature scaling per class
        y_pred = np.argmax(q, axis=1)
        recs = per_class_recall(y_true, y_pred, num_classes)
        mr = float(np.mean(recs))
        nb_prec = precision_of_class(y_true, y_pred, nb_idx)
        if nb_prec < nb_prec_min:
            # strong penalty to enforce constraint
            mr -= 10.0 * (nb_prec_min - nb_prec)
        return mr, recs, nb_prec

    improved = True
    while improved:
        improved = False
        for c in range(num_classes):
            local_best_tau = tau[c]
            local_best_score, _, _ = objective(tau)
            for g in grid:
                trial = tau.copy()
                trial[c] = g
                sc, recs_c, nbp_c = objective(trial)
                if sc > local_best_score + 1e-9:
                    local_best_score = sc
                    local_best_tau = g
            if not math.isclose(local_best_tau, tau[c]):
                tau[c] = local_best_tau
                improved = True
        score, recs, nbp = objective(tau)
        history.append({"score": score, "taus": tau.copy(), "recs": recs, "nb_prec": nbp})
        if score > best_score + 1e-9:
            best_score = score
            best_tau = tau.copy()

    # Final metrics with best_tau
    q = probs / best_tau.reshape(1, -1)
    y_pred = np.argmax(q, axis=1)
    recs = per_class_recall(y_true, y_pred, num_classes)
    mr = float(np.mean(recs))
    nb_prec = precision_of_class(y_true, y_pred, nb_idx)
    return best_tau, mr, {"recs": recs, "nb_prec": nb_prec, "history": history}

# ============
# Train / Eval
# ============
def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    loss_sum, n = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.item()) * xb.size(0)
        n += xb.size(0)
    return loss_sum / max(n, 1)

@torch.no_grad()
def eval_logits(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_y = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        all_logits.append(logits.detach().cpu().numpy())
        all_y.append(yb.numpy())
    logits = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_y, axis=0)
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    return probs, y_true

# =======
# Logging
# =======
@dataclass
class FoldResult:
    fold: int
    s_scale: float
    epochs: int
    macro_recall_raw: float
    macro_recall_tau: float
    nb_precision_tau: float
    per_class_recall_raw: str
    per_class_recall_tau: str
    taus: str
    class_names: str
    n_train: int
    n_val: int

# =====
# Main
# =====
def main():
    set_seed(RANDOM_SEED)
    ensure_dir(RESULTS_DIR)

    df = pd.read_csv(CSV_PATH)
    # Basic filtering
    if 'extraction_success' in df.columns:
        df = df[df['extraction_success'] == True].copy()
    if 'label' not in df.columns or 'filename' not in df.columns:
        raise ValueError("CSV must contain at least 'filename' and 'label' columns.")

    # Class mapping
    class_names = sorted(df['label'].unique().tolist())
    cls_to_idx = {c:i for i,c in enumerate(class_names)}
    y = df['label'].map(cls_to_idx).values

    # Features: drop non-feature cols
    drop_cols = [c for c in ['filename','label','extraction_success'] if c in df.columns]
    X = df.drop(columns=drop_cols).values
    num_samples, in_dim = X.shape

    # Groups for GroupKFold (patient-based)
    groups = df['filename'].apply(parse_patient_id_from_filename).values

    # Standardize features (fit on train fold only)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    C = len(class_names)
    nb_idx = class_names.index('Non-breathing') if 'Non-breathing' in class_names else None
    if nb_idx is None:
        raise ValueError("'Non-breathing' class not found in class names. Please check labels.")

    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    fold_rows = []

    for k, (tr_idx, va_idx) in enumerate(skf.split(X, y, groups), start=1):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]

        scaler = StandardScaler().fit(X_tr)
        X_tr = scaler.transform(X_tr)
        X_va = scaler.transform(X_va)

        tr_ds = FeatureDataset(X_tr, y_tr)
        va_ds = FeatureDataset(X_va, y_va)

        # optional: inverse-frequency sampler to stabilize minority classes
        cnt = Counter(y_tr.tolist())
        weights = np.array([1.0 / cnt[c] for c in y_tr], dtype=np.float32)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

        tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, sampler=sampler, drop_last=False)
        va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

        model = CosineClassifier(in_dim=in_dim, n_classes=C, s=SCALE_S).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        criterion = nn.CrossEntropyLoss()

        best_va_mr = -1.0
        for ep in range(1, EPOCHS+1):
            tr_loss = train_one_epoch(model, tr_loader, device, optimizer, criterion)
            # simple early snapshot of macro recall (raw)
            if ep % 10 == 0 or ep == EPOCHS:
                probs, y_true = eval_logits(model, va_loader, device)
                y_pred_raw = probs.argmax(1)
                recs_raw = per_class_recall(y_true, y_pred_raw, C)
                mr_raw = float(np.mean(recs_raw))
                if mr_raw > best_va_mr:
                    best_va_mr = mr_raw

        # Final eval on this fold
        probs, y_true = eval_logits(model, va_loader, device)
        y_pred_raw = probs.argmax(1)
        recs_raw = per_class_recall(y_true, y_pred_raw, C)
        mr_raw = float(np.mean(recs_raw))

        # Tau search with NB precision constraint
        best_tau, mr_tau, aux = search_per_class_tau(
            probs=probs, y_true=y_true, nb_idx=nb_idx, num_classes=C,
            grid=TAU_GRID, nb_prec_min=NB_PREC_MIN
        )
        recs_tau = aux["recs"]
        nb_prec_tau = aux["nb_prec"]

        row = FoldResult(
            fold=k,
            s_scale=SCALE_S,
            epochs=EPOCHS,
            macro_recall_raw=mr_raw,
            macro_recall_tau=mr_tau,
            nb_precision_tau=nb_prec_tau,
            per_class_recall_raw=json.dumps(recs_raw),
            per_class_recall_tau=json.dumps(recs_tau),
            taus=json.dumps([float(x) for x in best_tau.tolist()]),
            class_names=json.dumps(class_names),
            n_train=len(tr_idx),
            n_val=len(va_idx)
        )
        fold_rows.append(asdict(row))

        # save per-fold detailed outputs
        out_fold_dir = os.path.join(RESULTS_DIR, f"{EXPERIMENT_TAG}_fold{k}")
        ensure_dir(out_fold_dir)
        # save scaler & class names for reproducibility
        with open(os.path.join(out_fold_dir, "class_names.json"), "w", encoding="utf-8") as f:
            json.dump(class_names, f, ensure_ascii=False, indent=2)
        np.save(os.path.join(out_fold_dir, "taus.npy"), best_tau)
        np.save(os.path.join(out_fold_dir, "probs_val.npy"), probs)
        np.save(os.path.join(out_fold_dir, "y_true_val.npy"), y_true)

    # Aggregate CSV
    out_csv = os.path.join(RESULTS_DIR, f"{EXPERIMENT_TAG}_summary.csv")
    pd.DataFrame(fold_rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[DONE] Saved summary CSV to: {out_csv}")
    # Print quick Excel-friendly summary line
    avg_raw = float(np.mean([r['macro_recall_raw'] for r in fold_rows]))
    avg_tau = float(np.mean([r['macro_recall_tau'] for r in fold_rows]))
    print(f"[SUMMARY] MacroRecall(raw)={avg_raw:.3f} | MacroRecall(tau)={avg_tau:.3f} | tag={EXPERIMENT_TAG}")

if __name__ == "__main__":
    main()
