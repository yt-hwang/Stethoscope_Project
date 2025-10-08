#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step16G2 - Hardness Weighted + Smoothed Tau (default run)
Automatically uses pre-defined paths (no CLI args required)
Author: Yun Hwang x ChatGPT
"""

import os, time, warnings, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import recall_score, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ==============================
# USER SETTINGS
# ==============================
FEATURES_CSV = r"D:\Stethoscope_Project\Development\OPERA_Copied_from_Ubuntu\features\opera_features.csv"
OUT_ROOT = r"D:\Stethoscope_Project\Development\OPERA_Copied_from_Ubuntu\scripts\GPT\results_step16G2"
EXP_TAG = "Step16G2_HardnessWeighted_smoothTau_nbPrec_ge_0.35_v1"
EPOCHS = 60
BATCH_SIZE = 32
LR = 1e-3
BETA = 0.5
WCLIP = (0.2, 0.9)
NB_PREC_FLOOR = 0.35
N_SPLITS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================
# Utilities
# ==============================
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def plot_confusion_matrix(cm, labels, out_path, title):
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i,j]}", ha="center", va="center", color="black", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ==============================
# Model
# ==============================
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden//2), nn.ReLU(),
            nn.Linear(hidden//2, out_dim)
        )
    def forward(self, x): return self.net(x)

# ==============================
# Main
# ==============================
def main():
    print(f"[INFO] Loading features: {FEATURES_CSV}")
    df = pd.read_csv(FEATURES_CSV)
    feat_cols = [c for c in df.columns if c not in ["filename","label","group","label_id"]]
    if "label_id" not in df.columns:
        label_map = {v:i for i,v in enumerate(sorted(df["label"].unique()))}
        df["label_id"] = df["label"].map(label_map)
    X = df[feat_cols].values.astype(np.float32)
    y = df["label_id"].values.astype(np.int64)
    groups = df["group"].values
    labels = sorted(df["label"].unique())
    in_dim, out_dim = X.shape[1], len(labels)

    print(f"[INFO] Feature columns detected: {in_dim}, classes={labels}")

    gkf = GroupKFold(n_splits=N_SPLITS)
    fold_results = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups), 1):
        print(f"[Fold {fold}] ==========")
        X_tr, y_tr = torch.tensor(X[tr_idx]), torch.tensor(y[tr_idx])
        X_va, y_va = torch.tensor(X[va_idx]), torch.tensor(y[va_idx])
        dl_tr = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
        dl_va = DataLoader(TensorDataset(X_va, y_va), batch_size=BATCH_SIZE, shuffle=False)

        model = MLP(in_dim, out_dim).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        class_w = torch.ones(out_dim, device=DEVICE)
        hardness_ema = torch.ones(out_dim, device=DEVICE)

        for ep in range(1, EPOCHS+1):
            model.train()
            loss_sum = 0
            preds_all, ys_all = [], []
            for xb, yb in dl_tr:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                logits = model(xb)
                probs = F.softmax(logits, dim=1)
                pred = probs.argmax(1)
                for c in range(out_dim):
                    mask = (yb==c)
                    if mask.sum()>0:
                        hard = 1 - probs[mask, c].mean().item()
                        hardness_ema[c] = BETA*hardness_ema[c] + (1-BETA)*hard
                        class_w[c] = np.clip(float(hardness_ema[c]), WCLIP[0], WCLIP[1])
                loss = F.cross_entropy(logits, yb, weight=class_w)
                loss.backward(); opt.step()
                loss_sum += loss.item() * len(xb)
                preds_all += pred.cpu().tolist(); ys_all += yb.cpu().tolist()
            model.eval()
            with torch.no_grad():
                va_pred = model(X_va.to(DEVICE)).argmax(1).cpu()
            valMR = recall_score(y_va, va_pred, average="macro")
            if ep % 5 == 0 or ep == 1:
                print(f"[Fold {fold}][Ep {ep:03}] valMR={valMR:.3f} classW={[round(x,2) for x in class_w.tolist()]}")

        cm = confusion_matrix(y_va, va_pred)
        out_fold = os.path.join(OUT_ROOT, EXP_TAG, f"fold{fold}")
        ensure_dir(out_fold)
        plot_confusion_matrix(cm, labels, os.path.join(out_fold, f"fold{fold}_cm_raw.png"), f"Fold {fold} Confusion Matrix")
        fold_results.append(valMR)

    avg_mr = np.mean(fold_results)
    print(f"[SUMMARY] Mean MacroRecall={avg_mr:.3f}")
    ensure_dir(os.path.join(OUT_ROOT, EXP_TAG))
    pd.DataFrame({"fold": list(range(1, N_SPLITS+1)), "MacroRecall": fold_results}).to_csv(
        os.path.join(OUT_ROOT, EXP_TAG, f"{EXP_TAG}_summary.csv"), index=False
    )

if __name__ == "__main__":
    main()
