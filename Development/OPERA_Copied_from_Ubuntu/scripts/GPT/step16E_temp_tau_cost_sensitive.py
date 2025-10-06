#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step16E: Class-wise Temperature + per-class τ (cost-sensitive) on top of LDAM+DRW linear head
- Train: Linear classifier (first half CE → second half LDAM + DRW)
- Calibrate: Coordinate-descent over class-wise Temperature (T_c) and τ_c
    * Objective = weighted macro recall − λ × sum(max(0, r_min_c − recall_c))
    * Constraint: Precision(Non-breathing) ≥ nb_prec_min (soft penalty)
- Saves: train_log.csv, confusion_matrix_tau.png (with counts), temps.npy, taus.npy,
         probs_val.npy, y_true_val.npy, class_names.json, summary.csv
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
DEF_RESULTS_DIR = r"D:\Stethoscope_Project\Development\OPERA_Copied_from_Ubuntu\scripts\GPT\results_step16E"
DEF_EXPERIMENT_TAG = "Step16E_TempTau_costSensitive_nbPrec_ge_0.35"
DEF_RANDOM_SEED = 42

DEF_EPOCHS = 60
DEF_BATCH_SIZE = 128
DEF_LR = 3e-4
DEF_WD = 1e-4
DEF_RECALL_EVERY = 5

# Temperature grid
DEF_T_MIN = 0.5
DEF_T_MAX = 3.0
DEF_T_STEPS = 11
DEF_T_OUTER_ITERS = 3  # alternate optimize T then τ for these rounds

# τ grid
DEF_TAU_MIN = 0.4
DEF_TAU_MAX = 2.2
DEF_TAU_STEPS = 21
DEF_NB_PREC_MIN = 0.35

# Cost-sensitive params
DEF_RECALL_FLOOR = 0.30
DEF_RECALL_FLOOR_NB = 0.40
DEF_FLOOR_PENALTY = 1.5
DEF_CLASS_WEIGHTS = None  # None -> auto inverse-freq normalized

# -----------------------
# Utils
# -----------------------
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

# -----------------------
# Dataset & Model
# -----------------------
class FeatureDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32); self.y = y.astype(np.int64)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

class LinearClassifier(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__(); self.fc = nn.Linear(in_dim, n_classes)
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

# -----------------------
# Metrics & plotting
# -----------------------
def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray, C: int) -> List[float]:
    recs=[]
    for c in range(C):
        tp = np.sum((y_true==c) & (y_pred==c))
        fn = np.sum((y_true==c) & (y_pred!=c))
        recs.append(float(tp/(tp+fn+1e-9)))
    return recs

def precision_of_class(y_true: np.ndarray, y_pred: np.ndarray, c: int) -> float:
    tp = np.sum((y_true==c) & (y_pred==c))
    fp = np.sum((y_true!=c) & (y_pred==c))
    return float(tp/(tp+fp+1e-9))

def plot_confusion_matrix_png(cm: np.ndarray, class_names: List[str], out_path: str, title: str):
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title(title); fig.colorbar(im, ax=ax)
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(ticks); ax.set_yticklabels(class_names)
    # overlay integer counts
    maxv = cm.max() if hasattr(cm, "max") else 0
    thresh = maxv/2 if maxv>0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            ax.text(j, i, str(val), ha="center", va="center",
                    color=("white" if cm[i, j] > thresh else "black"), fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.tight_layout(); fig.savefig(out_path, dpi=180, bbox_inches='tight'); plt.close(fig)

# -----------------------
# Train / Eval
# -----------------------
def train_one_epoch(model, loader, device, optimizer, criterion_fn, use_ldam=False, class_weights=None):
    model.train(); loss_sum=0.0; n=0
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
    model.eval(); outs=[]; ys=[]
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        outs.append(logits.detach().cpu().numpy())
        ys.append(yb.numpy())
    logits = np.concatenate(outs, 0)
    y_true = np.concatenate(ys, 0)
    return logits, y_true

# -----------------------
# Temperature + τ search (cost-sensitive)
# -----------------------
def softmax_with_temp(logits: np.ndarray, temps: np.ndarray) -> np.ndarray:
    # class-wise temperature: divide each column by T_c
    z = logits / temps.reshape(1, -1)
    z = z - z.max(axis=1, keepdims=True)  # stability
    ez = np.exp(z)
    return ez / (ez.sum(axis=1, keepdims=True) + 1e-12)

def search_cost_sensitive_T_tau(
    logits: np.ndarray,
    y_true: np.ndarray,
    nb_idx: int,
    C: int,
    T_grid: np.ndarray,
    tau_grid: np.ndarray,
    nb_prec_min: float,
    class_weights: np.ndarray,
    recall_floors: np.ndarray,
    floor_penalty: float,
    outer_iters: int = 3,
):
    # init
    T = np.ones(C, dtype=np.float32)
    tau = np.ones(C, dtype=np.float32)

    def objective(Tv, tauv):
        probs = softmax_with_temp(logits, Tv)
        q = probs / tauv.reshape(1, -1)
        y_pred = np.argmax(q, axis=1)
        recs = np.array(per_class_recall(y_true, y_pred, C), dtype=np.float32)
        nb_prec = precision_of_class(y_true, y_pred, nb_idx)
        score = float(np.sum(class_weights * recs) / np.sum(class_weights))
        shortfall = np.clip(recall_floors - recs, a_min=0.0, a_max=None)
        score -= float(floor_penalty * shortfall.sum())
        if nb_prec < nb_prec_min:
            score -= 10.0 * (nb_prec_min - nb_prec)
        return score, recs, nb_prec

    for it in range(outer_iters):
        # optimize T (coordinate-wise)
        for c in range(C):
            base, _, _ = objective(T, tau)
            best_val, best_Tc = base, T[c]
            for g in T_grid:
                T2 = T.copy(); T2[c] = g
                sc, _, _ = objective(T2, tau)
                if sc > best_val + 1e-9:
                    best_val, best_Tc = sc, g
            T[c] = best_Tc
        # optimize tau (coordinate-wise)
        for c in range(C):
            base, _, _ = objective(T, tau)
            best_val, best_tc = base, tau[c]
            for g in tau_grid:
                t2 = tau.copy(); t2[c] = g
                sc, _, _ = objective(T, t2)
                if sc > best_val + 1e-9:
                    best_val, best_tc = sc, g
            tau[c] = best_tc

    final_score, recs, nbp = objective(T, tau)
    probs_final = softmax_with_temp(logits, T)
    y_pred_final = (probs_final / tau.reshape(1, -1)).argmax(1)
    return T, tau, final_score, recs.tolist(), float(nbp), y_pred_final

# -----------------------
# Main
# -----------------------
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
    ap.add_argument("--recall_every", type=int, default=DEF_RECALL_EVERY)
    # grids
    ap.add_argument("--t_min", type=float, default=DEF_T_MIN)
    ap.add_argument("--t_max", type=float, default=DEF_T_MAX)
    ap.add_argument("--t_steps", type=int, default=DEF_T_STEPS)
    ap.add_argument("--t_outer_iters", type=int, default=DEF_T_OUTER_ITERS)
    ap.add_argument("--tau_min", type=float, default=DEF_TAU_MIN)
    ap.add_argument("--tau_max", type=float, default=DEF_TAU_MAX)
    ap.add_argument("--tau_steps", type=int, default=DEF_TAU_STEPS)
    ap.add_argument("--nb_prec_min", type=float, default=DEF_NB_PREC_MIN)
    # cost-sensitive params
    ap.add_argument("--class_weights", type=str, default=None,
        help="Comma-separated weights per class; if omitted uses inverse frequency normalized.")
    ap.add_argument("--recall_floor", type=float, default=DEF_RECALL_FLOOR,
        help="Global recall floor unless recall_floor_perclass provided.")
    ap.add_argument("--recall_floor_perclass", type=str, default=None,
        help="Comma-separated recall floors per class (overrides --recall_floor).")
    ap.add_argument("--recall_floor_nb", type=float, default=DEF_RECALL_FLOOR_NB,
        help="Optional floor for Non-breathing; if <=0, ignored.")
    ap.add_argument("--floor_penalty", type=float, default=DEF_FLOOR_PENALTY)
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.results_dir)
    tag = safe_tag(args.tag)

    # Load data
    df = pd.read_csv(args.csv)
    if 'extraction_success' in df.columns:
        df = df[df['extraction_success']==True].copy()
    assert 'label' in df.columns and 'filename' in df.columns
    class_names = sorted(df['label'].unique().tolist())
    cls_to_idx = {c:i for i,c in enumerate(class_names)}
    y_all = df['label'].map(cls_to_idx).values
    drop_cols = [c for c in ['filename','label','extraction_success'] if c in df.columns]
    X_all = df.drop(columns=drop_cols).values
    groups = df['filename'].apply(parse_patient_id_from_filename).values

    C = len(class_names)
    if 'Non-breathing' not in class_names:
        raise ValueError("'Non-breathing' class not found.")
    nb_idx = class_names.index('Non-breathing')

    # floors
    if args.recall_floor_perclass:
        recall_floors = np.array([float(x) for x in args.recall_floor_perclass.split(',')], dtype=np.float32)
        assert len(recall_floors)==C, "--recall_floor_perclass length must equal #classes"
    else:
        recall_floors = np.full(C, float(args.recall_floor), dtype=np.float32)
    if args.recall_floor_nb > 0:
        recall_floors[nb_idx] = max(recall_floors[nb_idx], float(args.recall_floor_nb))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    T_grid = np.linspace(args.t_min, args.t_max, args.t_steps)
    tau_grid = np.linspace(args.tau_min, args.tau_max, args.tau_steps)

    rows = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y_all, groups), start=1):
        print(f"\n[Fold {fold}] ===========")
        X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
        X_va, y_va = X_all[va_idx], y_all[va_idx]

        scaler = StandardScaler().fit(X_tr)
        X_tr, X_va = scaler.transform(X_tr), scaler.transform(X_va)
        tr_ds, va_ds = FeatureDataset(X_tr, y_tr), FeatureDataset(X_va, y_va)
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
        va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False)

        # Class weights for DRW & score
        cnt = Counter(y_tr.tolist())
        counts = np.array([cnt.get(i, 1) for i in range(C)], dtype=np.float32)
        inv = 1.0 / counts
        class_weights_auto = inv / inv.mean()
        if args.class_weights:
            cw = np.array([float(x) for x in args.class_weights.split(',')], dtype=np.float32)
            assert len(cw)==C, "--class_weights length must equal #classes"
            cw_score = cw
        else:
            cw_score = class_weights_auto
        class_weights_torch = torch.tensor(inv / inv.mean(), dtype=torch.float32)  # for DRW half

        # Model & optimizer
        in_dim = X_tr.shape[1]
        model = LinearClassifier(in_dim, C).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        ldam = LDAMLoss(cls_counts=counts, max_m=0.5, s=30)

        # Train
        train_log = []
        for ep in range(1, args.epochs+1):
            use_ldam = (ep > args.epochs // 2)
            loss = train_one_epoch(
                model, tr_loader, device, opt, ldam,
                use_ldam=use_ldam, class_weights=(class_weights_torch if use_ldam else None)
            )
            # quick val raw MR
            logits_va, y_true_va = eval_logits(model, va_loader, device)
            y_pred_raw = softmax_with_temp(logits_va, np.ones(C)).argmax(1)
            recs = per_class_recall(y_true_va, y_pred_raw, C)
            mr = float(np.mean(recs))
            train_log.append({"epoch": ep, "loss": loss, "val_macro_recall_raw": mr, "per_class_recall_raw": json.dumps(recs)})
            if ep % args.recall_every == 0 or ep == 1:
                print(f"[Fold {fold}][Ep {ep:03d}] loss={loss:.4f}  valMR={mr:.3f}  rec={['%.3f'%r for r in recs]}")

        # Save per-fold logs
        fold_dir = os.path.join(args.results_dir, f"{tag}_fold{fold}")
        ensure_dir(fold_dir)
        pd.DataFrame(train_log).to_csv(os.path.join(fold_dir, "train_log.csv"), index=False, encoding="utf-8-sig")
        with open(os.path.join(fold_dir, "class_names.json"), "w", encoding="utf-8") as f:
            json.dump(class_names, f, ensure_ascii=False, indent=2)

        # Eval logits
        logits, y_true = eval_logits(model, va_loader, device)

        # raw
        probs_raw = softmax_with_temp(logits, np.ones(C))
        y_pred_raw = probs_raw.argmax(1)
        recs_raw = per_class_recall(y_true, y_pred_raw, C); mr_raw = float(np.mean(recs_raw))

        # search T and tau
        T, tau, score_tau, recs_tau, nb_prec_tau, y_pred_tau = search_cost_sensitive_T_tau(
            logits=logits, y_true=y_true, nb_idx=nb_idx, C=C,
            T_grid=T_grid, tau_grid=tau_grid, nb_prec_min=args.nb_prec_min,
            class_weights=np.asarray(cw_score, dtype=np.float32),
            recall_floors=recall_floors, floor_penalty=float(args.floor_penalty),
            outer_iters=int(args.t_outer_iters)
        )
        mr_tau = float(np.mean(recs_tau))

        # Confusion matrix with counts
        cm = confusion_matrix(y_true, y_pred_tau, labels=list(range(C)))
        plot_confusion_matrix_png(cm, class_names, os.path.join(fold_dir, "confusion_matrix_tau.png"),
                                  title=f"Temp+Tau Confusion (Fold {fold})")

        # Save arrays
        np.save(os.path.join(fold_dir, "temps.npy"), T)
        np.save(os.path.join(fold_dir, "taus.npy"), tau)
        np.save(os.path.join(fold_dir, "probs_val.npy"), softmax_with_temp(logits, T))
        np.save(os.path.join(fold_dir, "y_true_val.npy"), y_true)

        print(f"[Fold {fold}] rawMR={mr_raw:.3f} -> tauMR={mr_tau:.3f} (WeightedScore={score_tau:.3f}, NB-prec={nb_prec_tau:.3f})")
        rows.append({
            "fold": fold, "epochs": args.epochs,
            "macro_recall_raw": mr_raw, "macro_recall_tau": mr_tau, "nb_precision_tau": nb_prec_tau,
            "weighted_score_tau": score_tau,
            "per_class_recall_raw": json.dumps(recs_raw), "per_class_recall_tau": json.dumps(recs_tau),
            "temps": json.dumps([float(x) for x in T.tolist()]),
            "taus": json.dumps([float(x) for x in tau.tolist()]),
            "class_names": json.dumps(class_names),
            "n_train": len(tr_idx), "n_val": len(va_idx),
            "recall_floors": json.dumps(recall_floors.tolist()),
            "class_weights_score": json.dumps(list(map(float, cw_score)))
        })

    # Summary
    summary_csv = os.path.join(args.results_dir, f"{tag}_summary.csv")
    pd.DataFrame(rows).to_csv(summary_csv, index=False, encoding="utf-8-sig")
    avg_raw = float(np.mean([r["macro_recall_raw"] for r in rows]))
    avg_tau = float(np.mean([r["macro_recall_tau"] for r in rows]))
    print(f"[DONE] Saved summary CSV to: {summary_csv}")
    print(f"[SUMMARY] MacroRecall(raw)={avg_raw:.3f} | MacroRecall(tau)={avg_tau:.3f} | tag={tag}")

if __name__ == "__main__":
    main()
