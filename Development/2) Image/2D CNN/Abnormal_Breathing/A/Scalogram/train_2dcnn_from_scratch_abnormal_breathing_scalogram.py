#!/usr/bin/env python3
# train_2dcnn_from_scratch_abnormal_breathing.py
# - Uses pre-converted full-length (30s) spectrogram/scalogram images.
# - Patient-based split (GroupKFold). Robust to class-missing folds.
# - Saves: best model, preds.csv, metrics.json, logs.csv, curves.png, cm.png

import os, json, math, random
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    average_precision_score
)
from sklearn.preprocessing import label_binarize

# =========================
# CONFIG
# =========================
# === Choose one of the two by editing MANIFEST / OUTDIR ===
# Spectrogram:
# MANIFEST = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Development/2) Image/2D CNN/Abnormal_Breathing/A/Spectrogram/Processed Data/manifest.csv")
# OUTDIR   = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Development/2) Image/2D CNN/Abnormal_Breathing/A/Spectrogram/Result")

# Scalogram:
MANIFEST = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Development/2) Image/2D CNN/Abnormal_Breathing/A/Scalogram/Processed Data/manifest.csv")
OUTDIR   = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Development/2) Image/2D CNN/Abnormal_Breathing/A/Scalogram/Result")

OUTDIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 40
PATIENCE = 8
LR = 3e-4
WEIGHT_DECAY = 1e-4
N_FOLDS = 5
SEED = 42
NUM_WORKERS = 0  # macOS/python3.12 + spawn 이슈 회피

AUG_TRAIN = True   # True면 간단한 기하/밝기 증강 사용
NORM_MEAN = [0.485, 0.456, 0.406]  # Imagenet 통계 (전처리 정규화만 차용)
NORM_STD  = [0.229, 0.224, 0.225]

# =========================
# Utils
# =========================
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def safe_average_precision(y_true, proba, classes_idx):
    """
    Macro-AP with guards for missing positive classes in val set.
    Works for both binary and multi-class.
    Returns float or None.
    """
    try:
        C = proba.shape[1]
        # if binary (C==2) and at least two classes present
        if C == 2:
            if len(np.unique(y_true)) > 1:
                return average_precision_score(y_true, proba[:,1])
            else:
                return None
        # multi-class
        y_bin = label_binarize(y_true, classes=classes_idx)  # (N, C)
        valid = (y_bin.sum(axis=0) > 0)
        if valid.any():
            return average_precision_score(y_bin[:, valid], proba[:, valid], average='macro')
        return None
    except Exception:
        return None

def compute_class_weights(labels, classes):
    counts = pd.Series(labels).value_counts().reindex(range(len(classes)), fill_value=0).values.astype(float)
    inv = counts.sum() / np.maximum(counts, 1e-9)
    inv = inv * (len(inv) / inv.sum())
    return torch.tensor(inv, dtype=torch.float32)

def ensure_val_covers_all_classes(df, groups, y, n_splits, max_tries=50, seed=SEED):
    """
    Try to find a GroupKFold split whose validation set covers all classes.
    Falls back to the last split if impossible.
    """
    gkf = GroupKFold(n_splits=n_splits)
    rng = np.random.RandomState(seed)

    order = np.arange(len(df))
    last = None
    for _ in range(max_tries):
        for tr_idx, va_idx in gkf.split(order, y[order], groups[order]):
            va_labels = set(y[order][va_idx].tolist())
            if len(va_labels) == len(set(y)):
                return order[tr_idx], order[va_idx]
            last = (order[tr_idx], order[va_idx])
        # shuffle order to change group assignment ordering effect
        rng.shuffle(order)
    return last if last is not None else next(iter(gkf.split(df, y, groups)))

# =========================
# Dataset
# =========================
class ImgDS(Dataset):
    def __init__(self, df, label_to_idx, train=True):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        if train and AUG_TRAIN:
            self.tx = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomApply([
                    transforms.RandomAffine(degrees=7, translate=(0.03,0.03), scale=(0.97,1.03))
                ], p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
            ])
        else:
            self.tx = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
            ])

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row.path).convert("RGB")
        x = self.tx(img)
        y = self.label_to_idx[row.label]
        return x, y, row.path

# =========================
# Simple 2D CNN (from scratch)
# =========================
class SmallCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        # stem
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4   = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn5   = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn6   = nn.BatchNorm2d(128)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)  # 112

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)  # 56

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.max_pool2d(x, 2)  # 28

        x = self.head(x)        # (N, C)
        return x

# =========================
# Train/Eval
# =========================
def run_epoch(dl, model, criterion, optimizer, device, train=True):
    model.train(train)
    total, n = 0.0, 0
    ys, ps = [], []

    for x,y,_ in dl:
        x, y = x.to(device), y.to(device)
        if train: optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        if train:
            loss.backward()
            optimizer.step()

        total += loss.item() * x.size(0)
        n     += x.size(0)
        ys.append(y.detach().cpu().numpy())
        ps.append(logits.detach().softmax(1).cpu().numpy())

    y_true = np.concatenate(ys) if ys else np.array([])
    proba  = np.concatenate(ps) if ps else np.array([[]])
    y_pred = proba.argmax(1) if proba.size else np.array([])
    return total / max(n,1), y_true, y_pred, proba

def plot_curves(history, save_path):
    plt.figure(figsize=(10,4))
    plt.plot(history["epoch"], history["train_loss"], label="train_loss")
    plt.plot(history["epoch"], history["val_loss"], label="val_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()

def plot_cm(cm, classes, save_path):
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix"); plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2. if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label'); plt.xlabel('Predicted label')
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

def main():
    set_seed()

    # ===== Load manifest =====
    df = pd.read_csv(MANIFEST)
    assert {"path","label","patient_id"}.issubset(df.columns), \
        "manifest must have columns: path,label,patient_id"
    df = df.dropna(subset=["path","label","patient_id"]).copy()
    df = df[df.path.apply(lambda p: Path(p).exists())].reset_index(drop=True)

    classes = sorted(df["label"].unique().tolist())
    label_to_idx = {c:i for i,c in enumerate(classes)}
    idx_to_label = {i:c for c,i in label_to_idx.items()}
    df["y"] = df["label"].map(label_to_idx)

    print("=== DATA ===")
    print(f"n={len(df)} | classes={classes}")
    print(df["label"].value_counts().to_string())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # ===== K-Fold (patient-based) =====
    groups = df["patient_id"].astype(str).values
    y_all  = df["y"].values

    tr_idx, va_idx = ensure_val_covers_all_classes(df, groups, y_all, n_splits=N_FOLDS, seed=SEED)
    tr_df, va_df = df.iloc[tr_idx].reset_index(drop=True), df.iloc[va_idx].reset_index(drop=True)

    print("\n=== Split (patient-based) ===")
    print(f"Train n={len(tr_df)}  | class dist:\n{tr_df['label'].value_counts().to_string()}")
    print(f"Val   n={len(va_df)}  | class dist:\n{va_df['label'].value_counts().to_string()}")

    # ===== DataLoaders =====
    tds = ImgDS(tr_df, label_to_idx, train=True)
    vds = ImgDS(va_df, label_to_idx, train=False)
    tdl = DataLoader(tds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=(device=="cuda"))
    vdl = DataLoader(vds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(device=="cuda"))

    # ===== Model / Loss / Opt =====
    n_classes = len(classes)
    model = SmallCNN(n_classes).to(device)

    # class weights from training set
    w = compute_class_weights(tr_df["y"].values, classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # ===== Train loop with early stopping =====
    logs = []
    best_f1 = -1.0
    no_imp = 0
    run_tag = timestamp()
    best_path = OUTDIR / f"best_model_{run_tag}.pt"

    for ep in range(1, EPOCHS+1):
        tr_loss, tr_y, tr_pred, tr_proba = run_epoch(tdl, model, criterion, optimizer, device, train=True)
        va_loss, va_y, va_pred, va_proba = run_epoch(vdl, model, criterion, optimizer, device, train=False)

        # metrics
        all_idx = list(range(n_classes))
        tr_f1 = f1_score(tr_y, tr_pred, labels=all_idx, average="macro", zero_division=0) if len(tr_y) else 0.0
        va_f1 = f1_score(va_y, va_pred, labels=all_idx, average="macro", zero_division=0) if len(va_y) else 0.0
        tr_ap = safe_average_precision(tr_y, tr_proba, all_idx) if tr_proba.size else None
        va_ap = safe_average_precision(va_y, va_proba, all_idx) if va_proba.size else None

        print(f"[Ep {ep:02d}] train {tr_loss:.3f}/F1 {tr_f1:.3f}/AP {tr_ap if tr_ap is not None else 'NA'} "
              f"| val {va_loss:.3f}/F1 {va_f1:.3f}/AP {va_ap if va_ap is not None else 'NA'}")

        logs.append({
            "epoch": ep, "train_loss": tr_loss, "val_loss": va_loss,
            "train_f1": tr_f1, "val_f1": va_f1,
            "train_ap": tr_ap if tr_ap is not None else None,
            "val_ap": va_ap if va_ap is not None else None
        })

        improved = va_f1 > best_f1 + 1e-5
        if improved:
            best_f1 = va_f1
            no_imp = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_imp += 1

        if no_imp >= PATIENCE:
            print("Early stop.")
            break

    # ===== Load best and final eval (VAL) =====
    # weights_only=True 지원 안될 수 있어 try/except
    try:
        state = torch.load(best_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(best_path, map_location=device)
    model.load_state_dict(state)

    _, va_y, va_pred, va_proba = run_epoch(vdl, model, criterion, optimizer, device, train=False)

    # Save predictions
    pred_rows = []
    for (_, y, p, path) in zip(range(len(va_df)), va_y, va_pred, va_df["path"].tolist()):
        pred_rows.append({
            "path": path,
            "true_idx": int(y),
            "true_label": idx_to_label[int(y)],
            "pred_idx": int(p),
            "pred_label": idx_to_label[int(p)],
        })
    preds_df = pd.DataFrame(pred_rows)
    preds_csv = OUTDIR / f"preds_{run_tag}.csv"
    preds_df.to_csv(preds_csv, index=False)

    # Metrics + report
    all_idx = list(range(n_classes))
    final_f1 = f1_score(va_y, va_pred, labels=all_idx, average="macro", zero_division=0)
    final_ap = safe_average_precision(va_y, va_proba, all_idx) if va_proba.size else None

    report = classification_report(
        va_y, va_pred,
        labels=all_idx,
        target_names=classes,
        zero_division=0,
        output_dict=True
    )
    cm = confusion_matrix(va_y, va_pred, labels=all_idx)

    metrics = {
        "classes": classes,
        "final_macro_f1": final_f1,
        "final_macro_ap": final_ap,
        "support_per_class": {c: int((np.array(va_y)==i).sum()) for i,c in enumerate(classes)},
        "best_model_path": str(best_path),
        "preds_csv": str(preds_csv),
    }
    with open(OUTDIR / f"metrics_{run_tag}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save logs.csv
    pd.DataFrame(logs).to_csv(OUTDIR / f"logs_{run_tag}.csv", index=False)

    # Plots
    plot_curves(
        {"epoch": [r["epoch"] for r in logs],
         "train_loss": [r["train_loss"] for r in logs],
         "val_loss": [r["val_loss"] for r in logs]},
        OUTDIR / f"curves_{run_tag}.png"
    )
    plot_cm(cm, classes, OUTDIR / f"cm_{run_tag}.png")

    # Human-readable report
    with open(OUTDIR / f"classification_report_{run_tag}.txt", "w") as f:
        f.write(pd.DataFrame(report).transpose().to_string())

    print("\n=== Final (VAL) ===")
    print(f"macro-F1: {final_f1:.3f} | macro-AP: {final_ap if final_ap is not None else 'NA'}")
    print("Confusion matrix:\n", cm)
    print(f"Saved: \n- best model: {best_path}\n- preds: {preds_csv}\n- metrics: {OUTDIR / f'metrics_{run_tag}.json'}")
    print(f"- logs: {OUTDIR / f'logs_{run_tag}.csv'}\n- curves: {OUTDIR / f'curves_{run_tag}.png'}\n- cm: {OUTDIR / f'cm_{run_tag}.png'}")

if __name__ == "__main__":
    main()
