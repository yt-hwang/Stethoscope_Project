#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step16C: Snapshot Ensemble (Cosine + LDAM-DRW, 3 seeds each) + per-class tau search (NB precision constraint)
- Trains two diverse heads per fold:
    (A) Cosine head (s=32, CE)
    (B) Linear head with LDAM + DRW (second half epochs)
- Seeds = [0,1,2]  -> total 6 models per fold; average probabilities
- Then per-class tau search (coordinate descent) with NB precision >= threshold
- Saves: train_log.csv per model, confusion_matrix_tau.png (after ensemble + tau), taus.npy, summary.csv
"""

import os, json, math, time, random, argparse
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
# Defaults
# -----------------------
DEF_CSV_PATH = r"D:\Stethoscope_Project\Development\OPERA_Copied_from_Ubuntu\features\opera_features.csv"
DEF_RESULTS_DIR = r"D:\Stethoscope_Project\Development\OPERA_Copied_from_Ubuntu\scripts\GPT\results_step16C"
DEF_EXPERIMENT_TAG = "Step16C_Ensemble_Cos_s32__LDAM_DRW_seeds3_tau_NBprec_ge_0.35"

DEF_EPOCHS = 60
DEF_BATCH_SIZE = 128
DEF_LR = 3e-4
DEF_WD = 1e-4
DEF_RECALL_EVERY = 10

DEF_TAU_MIN = 0.6
DEF_TAU_MAX = 1.8
DEF_TAU_STEPS = 13
DEF_NB_PREC_MIN = 0.35

SEEDS = [0,1,2]

# -----------------------
# Utils
# -----------------------
def safe_tag(s: str) -> str:
    return "".join(c if (c.isalnum() or c in '._-') else '_' for c in s)

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

# Dataset
class FeatureDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# Models
class CosineClassifier(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, s: float = 32.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_classes, in_dim))
        nn.init.xavier_normal_(self.W); self.s = s
    def forward(self, x):
        x = F.normalize(x, dim=1)
        W = F.normalize(self.W, dim=1)
        return self.s * x @ W.t()

class LinearClassifier(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)
    def forward(self, x): return self.fc(x)

class LDAMLoss(nn.Module):
    def __init__(self, cls_counts, max_m=0.5, s=30):
        super().__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(np.array(cls_counts, dtype=np.float32)))
        m_list = max_m * (m_list / m_list.max())
        self.m_list = torch.tensor(m_list, dtype=torch.float32)
        self.s = s; self.ce = nn.CrossEntropyLoss(reduction='none')
    def forward(self, logits, target, class_weights=None, device=None):
        if device is None: device = logits.device
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

# Metrics & tau search
def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray, C: int):
    recs = []
    for c in range(C):
        tp = np.sum((y_true==c) & (y_pred==c))
        fn = np.sum((y_true==c) & (y_pred!=c))
        recs.append(float(tp/(tp+fn+1e-9)))
    return recs

def precision_of_class(y_true: np.ndarray, y_pred: np.ndarray, c: int):
    tp = np.sum((y_true==c) & (y_pred==c))
    fp = np.sum((y_true!=c) & (y_pred==c))
    return float(tp/(tp+fp+1e-9))

def search_per_class_tau(probs: np.ndarray, y_true: np.ndarray, nb_idx: int, C: int, grid: np.ndarray, nb_prec_min: float):
    tau = np.ones(C, dtype=np.float32)
    def objective(t): 
        q = probs / t.reshape(1,-1)
        yp = np.argmax(q, axis=1)
        mr = float(np.mean(per_class_recall(y_true, yp, C)))
        nbp = precision_of_class(y_true, yp, nb_idx)
        if nbp < nb_prec_min: mr -= 10.0*(nb_prec_min - nbp)
        return mr
    improved = True
    while improved:
        improved=False
        for c in range(C):
            best_val = objective(tau)
            best_tau_c = tau[c]
            for g in grid:
                t2 = tau.copy(); t2[c]=g
                val = objective(t2)
                if val > best_val + 1e-9:
                    best_val = val; best_tau_c = g
            if best_tau_c != tau[c]:
                tau[c] = best_tau_c; improved=True
    q = probs / tau.reshape(1,-1)
    ypt = np.argmax(q, axis=1)
    recs = per_class_recall(y_true, ypt, C)
    nbp = precision_of_class(y_true, ypt, nb_idx)
    mr = float(np.mean(recs))
    return tau, mr, {"recs": recs, "nb_prec": nbp}

def plot_confusion_matrix_png(cm: np.ndarray, class_names: List[str], out_path: str, title: str):
    fig = plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title); plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight'); plt.close(fig)

@torch.no_grad()
def eval_probs(model, loader, device):
    model.eval()
    outs, ys = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        outs.append(torch.softmax(logits, dim=1).cpu().numpy())
        ys.append(yb.numpy())
    return np.concatenate(outs, 0), np.concatenate(ys, 0)

def train_one_epoch(model, loader, device, opt, loss_fn):
    model.train(); tot, n = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device); yb = yb.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward(); opt.step()
        tot += float(loss.item()) * xb.size(0); n += xb.size(0)
    return tot/max(n,1)

def main():
    ensure_dir(DEF_RESULTS_DIR)
    tag = safe_tag(DEF_EXPERIMENT_TAG)

    df = pd.read_csv(DEF_CSV_PATH)
    if 'extraction_success' in df.columns:
        df = df[df['extraction_success']==True].copy()
    assert 'filename' in df.columns and 'label' in df.columns
    class_names = sorted(df['label'].unique().tolist())
    cls_to_idx = {c:i for i,c in enumerate(class_names)}
    y_all = df['label'].map(cls_to_idx).values
    drop_cols = [c for c in ['filename','label','extraction_success'] if c in df.columns]
    X_all = df.drop(columns=drop_cols).values
    groups = df['filename'].apply(parse_patient_id_from_filename).values
    C = len(class_names)
    nb_idx = class_names.index('Non-breathing')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    tau_grid = np.linspace(DEF_TAU_MIN, DEF_TAU_MAX, DEF_TAU_STEPS)

    rows = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y_all, groups), start=1):
        Xtr, Ytr = X_all[tr_idx], y_all[tr_idx]
        Xva, Yva = X_all[va_idx], y_all[va_idx]
        scaler = StandardScaler().fit(Xtr)
        Xtr, Xva = scaler.transform(Xtr), scaler.transform(Xva)
        tr_ds, va_ds = FeatureDataset(Xtr, Ytr), FeatureDataset(Xva, Yva)
        tr_loader = DataLoader(tr_ds, batch_size=DEF_BATCH_SIZE, shuffle=True)
        va_loader = DataLoader(va_ds, batch_size=DEF_BATCH_SIZE, shuffle=False)
        in_dim = Xtr.shape[1]
        cnt = Counter(Ytr.tolist())
        counts = np.array([cnt.get(i,1) for i in range(C)], dtype=np.float32)
        inv = 1.0 / counts
        class_weights = torch.tensor(inv / inv.mean(), dtype=torch.float32)

        probs_list = []

        for seed in SEEDS:
            # (A) Cosine head
            set_seed(seed)
            cos = CosineClassifier(in_dim, C, s=32.0).to(device)
            opt = torch.optim.AdamW(cos.parameters(), lr=DEF_LR, weight_decay=DEF_WD)
            ce = nn.CrossEntropyLoss()
            for ep in range(DEF_EPOCHS):
                train_one_epoch(cos, tr_loader, device, opt, ce)
                if (ep+1) % DEF_RECALL_EVERY == 0 or ep == 0:
                    pv, yt = eval_probs(cos, va_loader, device)
                    mr = float(np.mean(per_class_recall(yt, pv.argmax(1), C)))
                    print(f"[Fold {fold}][Cos s=32 seed{seed} Ep {ep+1:03d}] valMR={mr:.3f}")
            pv, yt = eval_probs(cos, va_loader, device)
            probs_list.append(pv)

            # (B) LDAM + DRW
            set_seed(seed)
            lin = LinearClassifier(in_dim, C).to(device)
            opt = torch.optim.AdamW(lin.parameters(), lr=DEF_LR, weight_decay=DEF_WD)
            ldam = LDAMLoss(counts, max_m=0.5, s=30)
            for ep in range(DEF_EPOCHS):
                use_ldam = (ep+1 > DEF_EPOCHS//2)
                if use_ldam:
                    loss_fn = lambda logits, yb: ldam(logits, yb, class_weights=class_weights, device=device)
                else:
                    loss_fn = nn.CrossEntropyLoss()
                train_one_epoch(lin, tr_loader, device, opt, loss_fn)
                if (ep+1) % DEF_RECALL_EVERY == 0 or ep == 0:
                    pv, yt = eval_probs(lin, va_loader, device)
                    mr = float(np.mean(per_class_recall(yt, pv.argmax(1), C)))
                    print(f"[Fold {fold}][LDAM-DRW seed{seed} Ep {ep+1:03d}] valMR={mr:.3f}")
            pv, yt = eval_probs(lin, va_loader, device)
            probs_list.append(pv)

        # average probs across models
        probs_ens = np.mean(np.stack(probs_list, axis=0), axis=0)
        # raw ensemble metric
        y_pred_raw = probs_ens.argmax(1)
        recs_raw = per_class_recall(yt, y_pred_raw, C)
        mr_raw = float(np.mean(recs_raw))

        # tau search
        best_tau, mr_tau, aux = search_per_class_tau(probs_ens, yt, nb_idx, C, tau_grid, DEF_NB_PREC_MIN)
        recs_tau, nb_prec_tau = aux["recs"], aux["nb_prec"]

        # save fold outputs
        fold_dir = os.path.join(DEF_RESULTS_DIR, f"{tag}_fold{fold}")
        ensure_dir(fold_dir)
        np.save(os.path.join(fold_dir, "taus.npy"), best_tau)
        np.save(os.path.join(fold_dir, "probs_val_ens.npy"), probs_ens)
        np.save(os.path.join(fold_dir, "y_true_val.npy"), yt)
        with open(os.path.join(fold_dir, "class_names.json"), "w", encoding="utf-8") as f:
            json.dump(class_names, f, ensure_ascii=False, indent=2)

        cm = confusion_matrix(yt, (probs_ens / best_tau.reshape(1,-1)).argmax(1), labels=list(range(C)))
        plot_confusion_matrix_png(cm, class_names, os.path.join(fold_dir, "confusion_matrix_tau.png"), f"Ensemble+Tau Confusion (Fold {fold})")

        print(f"[Fold {fold}] Ensemble rawMR={mr_raw:.3f} -> tauMR={mr_tau:.3f} (NB-prec={nb_prec_tau:.3f})")
        rows.append({
            "fold": fold,
            "macro_recall_raw": mr_raw,
            "macro_recall_tau": mr_tau,
            "nb_precision_tau": nb_prec_tau,
            "per_class_recall_raw": json.dumps(recs_raw),
            "per_class_recall_tau": json.dumps(recs_tau),
            "taus": json.dumps([float(x) for x in best_tau.tolist()]),
            "class_names": json.dumps(class_names),
            "n_train": len(tr_idx),
            "n_val": len(va_idx)
        })

    summary_csv = os.path.join(DEF_RESULTS_DIR, f"{tag}_summary.csv")
    pd.DataFrame(rows).to_csv(summary_csv, index=False, encoding="utf-8-sig")
    avg_raw = float(np.mean([r["macro_recall_raw"] for r in rows]))
    avg_tau = float(np.mean([r["macro_recall_tau"] for r in rows]))
    print(f"[DONE] Saved summary CSV to: {summary_csv}")
    print(f"[SUMMARY] MacroRecall(raw)={avg_raw:.3f} | MacroRecall(tau)={avg_tau:.3f} | tag={tag}")

if __name__ == "__main__":
    main()
