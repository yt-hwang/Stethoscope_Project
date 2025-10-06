#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step16B: LDAM + DRW on OPERA features, with per-class tau search (NB precision constraint)
- Uses StratifiedGroupKFold(5) by patient id parsed from filename
- Optimizer: AdamW
- Loss: LDAM; DRW schedule (first half epochs: plain CE, second half: LDAM with class-balanced weights)
- After training, runs per-class tau search (coordinate descent) to maximize Macro Recall with NB precision >= threshold
- Saves per-fold: train_log.csv, confusion_matrix_tau.png, taus.npy, probs/y_true, class_names.json
"""

import os, json, math, time, random, argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict
from collections import Counter

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------
# Defaults (CLI overridable)
# -----------------------
DEF_CSV_PATH = r"D:\Stethoscope_Project\Development\OPERA_Copied_from_Ubuntu\features\opera_features.csv"
DEF_RESULTS_DIR = r"D:\Stethoscope_Project\Development\OPERA_Copied_from_Ubuntu\scripts\GPT\results_step16B"
DEF_EXPERIMENT_TAG = "Step16B_LDAM_DRW_tau_NBprec_ge_0.35"
DEF_RANDOM_SEED = 42

DEF_EPOCHS = 60
DEF_BATCH_SIZE = 128
DEF_LR = 3e-4
DEF_WD = 1e-4
DEF_RECALL_EVERY = 5

DEF_TAU_MIN = 0.6
DEF_TAU_MAX = 1.8
DEF_TAU_STEPS = 13
DEF_NB_PREC_MIN = 0.35

# -------------
# Util helpers
# -------------
def safe_tag(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in s)

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def parse_patient_id_from_filename(path_str: str) -> str:
    base = os.path.basename(path_str).split('.')[0]
    if '_' in base: return base.split('_')[0]
    if '-' in base: return base.split('-')[0]
    return base

# -------------
# Dataset
# -------------
class FeatureDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -------------
# Model (Linear head; LDAM works on logits)
# -------------
class LinearClassifier(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)
    def forward(self, x):
        return self.fc(x)

# -------------
# Losses & Metrics
# -------------
class LDAMLoss(nn.Module):
    def __init__(self, cls_counts, max_m=0.5, s=30):
        super().__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(np.array(cls_counts, dtype=np.float32)))
        m_list = max_m * (m_list / m_list.max())
        self.m_list = torch.tensor(m_list, dtype=torch.float32)
        self.s = s
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, target, class_weights=None, device=None):
        if device is None:
            device = logits.device
        m = self.m_list.to(device)[target].unsqueeze(1)
        one_hot = torch.zeros_like(logits).scatter_(1, target.view(-1,1), 1.0)
        logits_m = logits - one_hot * m
        logits_m = self.s * logits_m
        if class_weights is not None:
            loss = self.ce(logits_m, target)
            w = class_weights.to(device)[target]
            return (loss * w).mean()
        else:
            return self.ce(logits_m, target).mean()

def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray, C: int) -> List[float]:
    recs = []
    for c in range(C):
        tp = np.sum((y_true==c) & (y_pred==c))
        fn = np.sum((y_true==c) & (y_pred!=c))
        recs.append(float(tp/(tp+fn+1e-9)))
    return recs

def precision_of_class(y_true: np.ndarray, y_pred: np.ndarray, c: int) -> float:
    tp = np.sum((y_true==c) & (y_pred==c))
    fp = np.sum((y_true!=c) & (y_pred==c))
    return float(tp/(tp+fp+1e-9))

def search_per_class_tau(probs: np.ndarray, y_true: np.ndarray, nb_idx: int, 
                         C: int, grid: np.ndarray, nb_prec_min: float) -> Tuple[np.ndarray, float, Dict]:
    tau = np.ones(C, dtype=np.float32)
    best_score, best_tau = -1e9, tau.copy()

    def objective(tau_vec):
        q = probs / tau_vec.reshape(1, -1)
        y_pred = np.argmax(q, axis=1)
        recs = per_class_recall(y_true, y_pred, C)
        mr = float(np.mean(recs))
        nb_prec = precision_of_class(y_true, y_pred, nb_idx)
        if nb_prec < nb_prec_min:
            mr -= 10.0*(nb_prec_min - nb_prec)
        return mr, recs, nb_prec

    improved = True
    while improved:
        improved = False
        for c in range(C):
            base_score, _, _ = objective(tau)
            best_local, best_val = tau[c], base_score
            for g in grid:
                trial = tau.copy(); trial[c]=g
                sc, _, _ = objective(trial)
                if sc > best_val + 1e-9:
                    best_val, best_local = sc, g
            if not math.isclose(best_local, tau[c]):
                tau[c] = best_local
                improved = True

    final_score, recs, nbp = objective(tau)
    return tau, final_score, {"recs": recs, "nb_prec": nbp}

# -------------
# Train / Eval
# -------------
def train_one_epoch(model, loader, device, optimizer, criterion_fn, use_ldam=False, class_weights=None):
    model.train()
    loss_sum, n = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device); yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        if use_ldam:
            loss = criterion_fn(logits, yb, class_weights=class_weights, device=device)
        else:
            if class_weights is not None:
                ce = nn.CrossEntropyLoss(weight=class_weights.to(device))
            else:
                ce = nn.CrossEntropyLoss()
            loss = ce(logits, yb)
        loss.backward(); optimizer.step()
        loss_sum += float(loss.item()) * xb.size(0); n += xb.size(0)
    return loss_sum / max(n,1)

@torch.no_grad()
def eval_logits(model, loader, device):
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

def plot_confusion_matrix_png(cm: np.ndarray, class_names: List[str], out_path: str, title: str):
    fig = plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)

# -------------
# Main
# -------------
def main():
    set_seed(DEF_RANDOM_SEED)
    ensure_dir(DEF_RESULTS_DIR)
    tag = safe_tag(DEF_EXPERIMENT_TAG)

    df = pd.read_csv(DEF_CSV_PATH)
    if 'extraction_success' in df.columns:
        df = df[df['extraction_success']==True].copy()
    class_names = sorted(df['label'].unique().tolist())
    cls_to_idx = {c:i for i,c in enumerate(class_names)}
    y_all = df['label'].map(cls_to_idx).values
    drop_cols = [c for c in ['filename','label','extraction_success'] if c in df.columns]
    X_all = df.drop(columns=drop_cols).values
    groups = df['filename'].apply(parse_patient_id_from_filename).values

    C = len(class_names)
    nb_idx = class_names.index('Non-breathing')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=DEF_RANDOM_SEED)
    tau_grid = np.linspace(DEF_TAU_MIN, DEF_TAU_MAX, DEF_TAU_STEPS)

    rows = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y_all, groups), start=1):
        print(f"\n[Fold {fold}] ===========")
        X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
        X_va, y_va = X_all[va_idx], y_all[va_idx]
        scaler = StandardScaler().fit(X_tr)
        X_tr, X_va = scaler.transform(X_tr), scaler.transform(X_va)
        tr_ds, va_ds = FeatureDataset(X_tr, y_tr), FeatureDataset(X_va, y_va)
        tr_loader = DataLoader(tr_ds, batch_size=DEF_BATCH_SIZE, shuffle=True)
        va_loader = DataLoader(va_ds, batch_size=DEF_BATCH_SIZE, shuffle=False)

        cnt = Counter(y_tr.tolist())
        counts = np.array([cnt.get(i, 1) for i in range(C)], dtype=np.float32)
        inv = 1.0 / counts
        class_weights = torch.tensor(inv / inv.mean(), dtype=torch.float32)

        in_dim = X_tr.shape[1]
        model = LinearClassifier(in_dim, C).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=DEF_LR, weight_decay=DEF_WD)
        ldam = LDAMLoss(cls_counts=counts, max_m=0.5, s=30)

        train_log = []
        for ep in range(1, DEF_EPOCHS+1):
            use_ldam = (ep > DEF_EPOCHS // 2)
            loss = train_one_epoch(model, tr_loader, device, opt, ldam, use_ldam=use_ldam, class_weights=(class_weights if use_ldam else None))
            probs_va, y_true_va = eval_logits(model, va_loader, device)
            y_pred_raw = probs_va.argmax(1)
            recs = per_class_recall(y_true_va, y_pred_raw, C)
            mr = float(np.mean(recs))
            train_log.append({"epoch": ep, "loss": loss, "val_macro_recall_raw": mr, "per_class_recall_raw": json.dumps(recs)})
            if ep % DEF_RECALL_EVERY == 0 or ep == 1:
                print(f"[Fold {fold}][Ep {ep:03d}] loss={loss:.4f}  valMR={mr:.3f}  rec={['%.3f'%r for r in recs]}")

        fold_dir = os.path.join(DEF_RESULTS_DIR, f"{tag}_fold{fold}")
        ensure_dir(fold_dir)
        pd.DataFrame(train_log).to_csv(os.path.join(fold_dir, "train_log.csv"), index=False, encoding="utf-8-sig")
        with open(os.path.join(fold_dir, "class_names.json"), "w", encoding="utf-8") as f:
            json.dump(class_names, f, ensure_ascii=False, indent=2)

        probs, y_true = eval_logits(model, va_loader, device)
        y_pred_raw = probs.argmax(1)
        recs_raw = per_class_recall(y_true, y_pred_raw, C)
        mr_raw = float(np.mean(recs_raw))
        best_tau, mr_tau, aux = search_per_class_tau(probs, y_true, nb_idx, C, tau_grid, DEF_NB_PREC_MIN)
        recs_tau, nb_prec_tau = aux["recs"], aux["nb_prec"]

        q = probs / best_tau.reshape(1, -1)
        y_pred_tau = q.argmax(1)
        cm = confusion_matrix(y_true, y_pred_tau, labels=list(range(C)))
        plot_confusion_matrix_png(cm, class_names, os.path.join(fold_dir, "confusion_matrix_tau.png"), title=f"Confusion Matrix (Fold {fold})")

        np.save(os.path.join(fold_dir, "taus.npy"), best_tau)
        np.save(os.path.join(fold_dir, "probs_val.npy"), probs)
        np.save(os.path.join(fold_dir, "y_true_val.npy"), y_true)
        print(f"[Fold {fold}] rawMR={mr_raw:.3f} -> tauMR={mr_tau:.3f} (NB-prec={nb_prec_tau:.3f})")
        rows.append({"fold": fold, "epochs": DEF_EPOCHS, "macro_recall_raw": mr_raw, "macro_recall_tau": mr_tau, "nb_precision_tau": nb_prec_tau, "per_class_recall_raw": json.dumps(recs_raw), "per_class_recall_tau": json.dumps(recs_tau), "taus": json.dumps([float(x) for x in best_tau.tolist()]), "class_names": json.dumps(class_names), "n_train": len(tr_idx), "n_val": len(va_idx)})

    summary_csv = os.path.join(DEF_RESULTS_DIR, f"{tag}_summary.csv")
    pd.DataFrame(rows).to_csv(summary_csv, index=False, encoding="utf-8-sig")
    avg_raw = float(np.mean([r["macro_recall_raw"] for r in rows]))
    avg_tau = float(np.mean([r["macro_recall_tau"] for r in rows]))
    print(f"[DONE] Saved summary CSV to: {summary_csv}")
    print(f"[SUMMARY] MacroRecall(raw)={avg_raw:.3f} | MacroRecall(tau)={avg_tau:.3f} | tag={tag}")

if __name__ == "__main__":
    main()
