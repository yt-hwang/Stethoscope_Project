#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step16A (v2): Cosine Head + Per-class Tau Search with Rich Logging
- Adds per-epoch logging (loss, val macro recall), per-fold train_log CSVs
- Prints per-class recall every K epochs and at the end
- Uses safe experiment tag for Windows paths
"""

import os, json, math, time, random, argparse
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict
from collections import Counter

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# -----------------------
# Default Config (can CLI)
# -----------------------
DEF_CSV_PATH = r"D:\Stethoscope_Project\Development\OPERA_Copied_from_Ubuntu\features\opera_features.csv"
DEF_RESULTS_DIR = r"D:\Stethoscope_Project\Development\OPERA_Copied_from_Ubuntu\scripts\GPT\results_step16A"
DEF_EXPERIMENT_TAG = "Step16A_CosHead_tau_NBprec_ge_0.35"
DEF_RANDOM_SEED = 42

DEF_EPOCHS = 60
DEF_BATCH_SIZE = 128
DEF_LR = 3e-4
DEF_WD = 1e-4
DEF_SCALE_S = 16.0
DEF_LOG_EVERY = 1           # print every epoch
DEF_RECALL_EVERY = 5        # print per-class recall every K epochs

DEF_TAU_MIN = 0.6
DEF_TAU_MAX = 1.8
DEF_TAU_STEPS = 13
DEF_NB_PREC_MIN = 0.35

# -------------
# Util helpers
# -------------
def safe_tag(s: str) -> str:
    # allow alnum, dot, underscore, hyphen; replace others with underscore
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in s)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -------------
# Model
# -------------
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

# -------------
# Metrics
# -------------
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
    history = []
    while improved:
        improved = False
        for c in range(C):
            local_best_tau = tau[c]
            local_best_score, _, _ = objective(tau)
            for g in grid:
                trial = tau.copy(); trial[c]=g
                sc, recs_c, nbp_c = objective(trial)
                if sc > local_best_score + 1e-9:
                    local_best_score, local_best_tau = sc, g
            if not math.isclose(local_best_tau, tau[c]):
                tau[c] = local_best_tau
                improved = True
        sc, recs, nbp = objective(tau)
        history.append({"score": sc, "taus": tau.copy(), "recs": recs, "nb_prec": nbp})
        if sc > best_score + 1e-9:
            best_score, best_tau = sc, tau.copy()

    q = probs / best_tau.reshape(1, -1)
    y_pred = np.argmax(q, axis=1)
    recs = per_class_recall(y_true, y_pred, C)
    mr = float(np.mean(recs))
    nb_prec = precision_of_class(y_true, y_pred, nb_idx)
    return best_tau, mr, {"recs": recs, "nb_prec": nb_prec, "history": history}

# -------------
# Train / Eval
# -------------
def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    loss_sum, n = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device); yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward(); optimizer.step()
        loss_sum += float(loss.item()) * xb.size(0)
        n += xb.size(0)
    return loss_sum / max(n, 1)

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

# -------------
# DataClass
# -------------
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

# -------------
# Main
# -------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEF_CSV_PATH)
    ap.add_argument("--results_dir", default=DEF_RESULTS_DIR)
    ap.add_argument("--tag", default=DEF_EXPERIMENT_TAG)
    ap.add_argument("--seed", type=int, default=DEF_RANDOM_SEED)
    ap.add_argument("--epochs", type=int, default=DEF_EPOCHS)
    ap.add_argument("--batch_size", type=int, default=DEF_BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=DEF_LR)
    ap.add_argument("--wd", type=float, default=DEF_WD)
    ap.add_argument("--scale_s", type=float, default=DEF_SCALE_S)
    ap.add_argument("--log_every", type=int, default=DEF_LOG_EVERY)
    ap.add_argument("--recall_every", type=int, default=DEF_RECALL_EVERY)
    ap.add_argument("--tau_min", type=float, default=DEF_TAU_MIN)
    ap.add_argument("--tau_max", type=float, default=DEF_TAU_MAX)
    ap.add_argument("--tau_steps", type=int, default=DEF_TAU_STEPS)
    ap.add_argument("--nb_prec_min", type=float, default=DEF_NB_PREC_MIN)
    args = ap.parse_args()

    set_seed(args.seed)

    exp_tag_safe = safe_tag(args.tag)
    ensure_dir(args.results_dir)

    # Load CSV
    df = pd.read_csv(args.csv)
    if 'extraction_success' in df.columns:
        df = df[df['extraction_success']==True].copy()
    if 'label' not in df.columns or 'filename' not in df.columns:
        raise ValueError("CSV must contain 'filename' and 'label'.")

    class_names = sorted(df['label'].unique().tolist())
    cls_to_idx = {c:i for i,c in enumerate(class_names)}
    y = df['label'].map(cls_to_idx).values

    drop_cols = [c for c in ['filename','label','extraction_success'] if c in df.columns]
    X = df.drop(columns=drop_cols).values
    n, in_dim = X.shape

    groups = df['filename'].apply(parse_patient_id_from_filename).values

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    C = len(class_names)
    if 'Non-breathing' not in class_names:
        raise ValueError("'Non-breathing' class not found among labels.")
    nb_idx = class_names.index('Non-breathing')

    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    fold_rows = []

    tau_grid = np.linspace(args.tau_min, args.tau_max, args.tau_steps)

    for k, (tr_idx, va_idx) in enumerate(skf.split(X, y, groups), start=1):
        start_fold = time.time()
        print(f"\n[Fold {k}] ========")

        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]

        scaler = StandardScaler().fit(X_tr)
        X_tr = scaler.transform(X_tr); X_va = scaler.transform(X_va)

        tr_ds = FeatureDataset(X_tr, y_tr)
        va_ds = FeatureDataset(X_va, y_va)

        cnt = Counter(y_tr.tolist())
        weights = np.array([1.0/cnt[c] for c in y_tr], dtype=np.float32)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, sampler=sampler)
        va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False)

        model = CosineClassifier(in_dim=in_dim, n_classes=C, s=args.scale_s).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        criterion = nn.CrossEntropyLoss()

        # per-fold log csv
        fold_dir = os.path.join(args.results_dir, f"{exp_tag_safe}_fold{k}")
        ensure_dir(fold_dir)
        log_rows = []

        best_mr_raw, best_ep = -1.0, -1
        for ep in range(1, args.epochs+1):
            t0 = time.time()
            tr_loss = train_one_epoch(model, tr_loader, device, opt, criterion)
            probs_va, y_true_va = eval_logits(model, va_loader, device)
            y_pred_raw = probs_va.argmax(1)
            recs_raw = per_class_recall(y_true_va, y_pred_raw, C)
            mr_raw = float(np.mean(recs_raw))
            took = time.time() - t0

            log_rows.append({
                "epoch": ep, "train_loss": tr_loss, "val_macro_recall_raw": mr_raw,
                "per_class_recall_raw": json.dumps(recs_raw), "secs": took
            })

            if ep % args.log_every == 0:
                msg = f"[Fold {k}][Ep {ep:03d}] loss={tr_loss:.4f}  valMR={mr_raw:.3f}  time={took:.1f}s"
                if ep % args.recall_every == 0:
                    msg += f"  rec={['%.3f'%r for r in recs_raw]}"
                print(msg)

            if mr_raw > best_mr_raw + 1e-6:
                best_mr_raw, best_ep = mr_raw, ep

        # Save fold train log
        pd.DataFrame(log_rows).to_csv(os.path.join(fold_dir, "train_log.csv"), index=False, encoding="utf-8-sig")

        # Final eval + tau search
        probs, y_true = eval_logits(model, va_loader, device)
        y_pred_raw = probs.argmax(1)
        recs_raw = per_class_recall(y_true, y_pred_raw, C)
        mr_raw = float(np.mean(recs_raw))

        best_tau, mr_tau, aux = search_per_class_tau(
            probs=probs, y_true=y_true, nb_idx=nb_idx, C=C, grid=tau_grid, nb_prec_min=args.nb_prec_min
        )
        recs_tau = aux["recs"]; nb_prec_tau = aux["nb_prec"]

        # Confusion matrix (after tau)
        q = probs / best_tau.reshape(1, -1)
        y_pred_tau = q.argmax(1)
        cm = confusion_matrix(y_true, y_pred_tau, labels=list(range(C)))
        np.savetxt(os.path.join(fold_dir, "confusion_matrix_tau.csv"), cm, delimiter=",", fmt="%d")

        # Save arrays
        np.save(os.path.join(fold_dir, "taus.npy"), best_tau)
        np.save(os.path.join(fold_dir, "probs_val.npy"), probs)
        np.save(os.path.join(fold_dir, "y_true_val.npy"), y_true)
        with open(os.path.join(fold_dir, "class_names.json"), "w", encoding="utf-8") as f:
            json.dump(class_names, f, ensure_ascii=False, indent=2)

        fold_time = time.time() - start_fold
        print(f"[Fold {k}] done in {fold_time/60:.1f} min | rawMR={mr_raw:.3f} -> tauMR={mr_tau:.3f} (NB-prec={nb_prec_tau:.3f})")

        fold_rows = []
        # Load prior rows if exist to append
        # (we'll write outside loop; accumulate in memory is also fine)
        # We'll append after loop to a DataFrame.
        if not hasattr(main, "_rows"):
            main._rows = []
        main._rows.append({
            "fold": k,
            "s_scale": args.scale_s,
            "epochs": args.epochs,
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

    # Aggregate summary
    summary_csv = os.path.join(args.results_dir, f"{exp_tag_safe}_summary.csv")
    pd.DataFrame(main._rows).to_csv(summary_csv, index=False, encoding="utf-8-sig")

    avg_raw = float(np.mean([r["macro_recall_raw"] for r in main._rows]))
    avg_tau = float(np.mean([r["macro_recall_tau"] for r in main._rows]))
    print(f"[DONE] Saved summary CSV to: {summary_csv}")
    print(f"[SUMMARY] MacroRecall(raw)={avg_raw:.3f} | MacroRecall(tau)={avg_tau:.3f} | tag={exp_tag_safe}")

if __name__ == "__main__":
    main()
