#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D CNN (from scratch) for multiclass abnormal breathing classification
- Uses pre-converted images (30s spectrograms or scalograms) as inputs.
- Patient-based, label-stratified split with retry until every class appears in both sets.
- Handles tiny/imbalanced datasets with class filtering + weighted sampling.
- Saves logs, metrics, confusion matrix, PR curves, and best model.

Tested with: Python 3.12, torch >=2.2, torchvision >=0.17, scikit-learn >=1.3
"""

import os, sys, json, time, random
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from sklearn.metrics import classification_report, confusion_matrix, f1_score, average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedGroupKFold

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------
# User Config
# ------------------------------
# Choose ONE of these roots (already converted 30s images):
#   Spectrogram root:
#IMG_ROOT = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Development/2) Image/2D CNN/Abnormal_Breathing/A/Spectrogram/Processed Data")
IMG_ROOT = Path("D:\\Stethoscope_Project\\Development\\2) Image\\2D CNN\\Abnormal_Breathing\\A\\Spectrogram\\Processed Data")

#   Or Scalogram root:
# IMG_ROOT = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Development/2) Image/2D CNN/Abnormal_Breathing/A/Scalogram/Processed Data")

#OUT_ROOT = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Development/2) Image/2D CNN/Abnormal_Breathing/A/Spectrogram/Result")
OUT_ROOT = Path("D:\\Stethoscope_Project\\Development\\2) Image\\2D CNN\\Abnormal_Breathing\\A\\Spectrogram\\Result")
# If using scalogram, change OUT_ROOT accordingly.

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

# file name pattern to extract patient id; you can customize if needed.
# Example filenames: KP002_WWS_1.png -> patient_id = KP002
def extract_patient_id(fname: str) -> str:
    base = Path(fname).stem
    # patient id is the first token split by underscore
    return base.split("_")[0] if "_" in base else base

# training params
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 60
EARLY_STOP = 10
LR = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
NUM_WORKERS = 0  # macOS + Python 3.12: keep 0 to avoid mp issues

# splitting
TEST_SIZE = 0.2         # 80/20 (patient-based, stratified)
SPLIT_RETRIES = 200     # try up to N random splits to satisfy constraints
MIN_IMAGES_PER_CLASS = 1 # classes with < this count will be dropped (logged)

# data aug
TRAIN_AUG_PROB = 0.5

# ------------------------------
# Utils
# ------------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def list_images(root: Path):
    items = []
    # Expect structure: root/Diagnosis/*.png (we decided to put all files under diagnosis folders)
    for label_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        label = label_dir.name
        for p in label_dir.rglob("*"):
            if p.suffix.lower() in IMG_EXTS:
                pid = extract_patient_id(p.name)
                items.append((str(p), label, pid))
    return pd.DataFrame(items, columns=["path","label","patient_id"])

def print_dist(df, title):
    vc = df["label"].value_counts().sort_index()
    print(f"{title}\n{vc if len(vc)>0 else '(empty)'}\n")

def compute_class_weights(labels, classes):
    counts = Counter(labels)
    weights = []
    total = sum(counts[c] for c in classes)
    for c in classes:
        # inverse frequency
        cw = total / max(1, counts[c])
        weights.append(cw)
    w = torch.tensor(weights, dtype=torch.float32)
    # normalize to mean = 1 for stability
    return w * (len(w) / w.sum())

def build_sampler(df, classes):
    # per-sample weight inverse to class frequency
    counts = df['label'].value_counts().to_dict()
    weights = df['label'].apply(lambda x: 1.0 / counts[x]).values
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

# ------------------------------
# Dataset & Model
# ------------------------------
class ImgDataset(Dataset):
    def __init__(self, df, classes, train=True):
        self.df = df.reset_index(drop=True)
        self.classes = classes
        self.cls2idx = {c:i for i,c in enumerate(classes)}
        if train:
            self.tx = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02)
                ], p=TRAIN_AUG_PROB),
                transforms.RandomApply([
                    transforms.RandomAffine(degrees=5, translate=(0.02,0.02), scale=(0.98,1.02))
                ], p=TRAIN_AUG_PROB),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])
        else:
            self.tx = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row.path).convert("RGB")
        img = self.tx(img)
        y = self.cls2idx[row.label]
        return img, y, row.path

class SmallCNN(nn.Module):
    """
    A compact CNN suitable for small datasets.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(0.10),

            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(0.15),

            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(0.20),

            nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.body(x)
        x = self.head(x)
        return x

# ------------------------------
# Split (Stratified by label, Grouped by patient)
# ------------------------------
def stratified_group_split(df, test_size=0.2, max_tries=200, min_per_class=1, seed=42):
    """
    Try SGK splits until both splits contain at least min_per_class for every class.
    If impossible (too rare classes), return the best we can (logged outside).
    """
    labels = df['label'].values
    groups = df['patient_id'].values
    # We generate pseudo n_splits but will pick one that satisfies constraints
    for attempt in range(max_tries):
        sgk = StratifiedGroupKFold(n_splits=int(1/test_size), shuffle=True, random_state=seed+attempt)
        ok_splits = []
        for tr_idx, va_idx in sgk.split(np.zeros(len(df)), labels, groups):
            tr_df, va_df = df.iloc[tr_idx], df.iloc[va_idx]
            tr_counts = tr_df['label'].value_counts()
            va_counts = va_df['label'].value_counts()
            classes = sorted(df['label'].unique().tolist())
            cond = all(tr_counts.get(c,0) >= min_per_class and va_counts.get(c,0) >= min_per_class for c in classes)
            if cond:
                return tr_df.copy(), va_df.copy()
            ok_splits.append((tr_df.copy(), va_df.copy()))
        # if no split satisfies, continue to new seed
    # fallback: return the first split from last run
    tr_df, va_df = ok_splits[0] if ok_splits else (df.sample(frac=0.8, random_state=seed), df.drop(df.sample(frac=0.8, random_state=seed).index))
    return tr_df.copy(), va_df.copy()

# ------------------------------
# Training / Evaluation
# ------------------------------
def run_epoch(model, dl, device, criterion=None, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    ys, ps, paths = [], [], []
    for x,y,pth in dl:
        x,y = x.to(device), y.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        if train:
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        else:
            loss = F.cross_entropy(logits, y, label_smoothing=LABEL_SMOOTHING)
        total += loss.item()*x.size(0); n += x.size(0)
        ys.append(y.cpu().numpy()); ps.append(logits.detach().softmax(1).cpu().numpy()); paths += list(pth)
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    pred = p.argmax(1)
    macro_f1 = f1_score(y, pred, average="macro", zero_division=0)
    try:
        ap = average_precision_score(y, p, average="macro")
    except Exception:
        ap = None
    return total/n, macro_f1, ap, y, pred, p, paths

def plot_curves_and_cm(out_dir, classes, y_true, prob, y_pred, tag):
    out_dir.mkdir(parents=True, exist_ok=True)
    # PR Curves (one-vs-rest) only for classes present in y_true
    present = sorted(list(set(y_true.tolist())))
    y_onehot = np.zeros((len(y_true), len(classes)), dtype=np.int64)
    for i,lab in enumerate(y_true):
        y_onehot[i, lab] = 1
    plt.figure(figsize=(7,6))
    for c in present:
        prec, rec, _ = precision_recall_curve(y_onehot[:,c], prob[:,c])
        ap = average_precision_score(y_onehot[:,c], prob[:,c])
        plt.plot(rec, prec, label=f"{classes[c]} (AP={ap:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR Curves ({tag})"); plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"curves_{tag}.png", dpi=150)
    plt.close()

    # Confusion Matrix (only present classes to keep axes meaningful)
    cm_labels = present
    cm = confusion_matrix(y_true, y_pred, labels=cm_labels)
    plt.figure(figsize=(6,5))
    im = plt.imshow(cm, interpolation='nearest')
    plt.title(f"Confusion Matrix ({tag})")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_names = [classes[i] for i in cm_labels]
    plt.xticks(range(len(cm_labels)), tick_names, rotation=45, ha='right')
    plt.yticks(range(len(cm_labels)), tick_names)
    thresh = cm.max() if cm.size else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thresh/2 else "black")
    plt.tight_layout()
    plt.savefig(out_dir / f"cm_{tag}.png", dpi=150)
    plt.close()

def main():
    set_seed(42)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Load data table
    df = list_images(IMG_ROOT)

    # Fix known mislabels quickly if still present:
    # (이미 라벨링이 고쳐졌다면 이 블록은 아무 영향 없음)
    df.loc[df['path'].str.contains("KP002_WWS_1"), "label"] = "Crackle"
    df.loc[df['path'].str.contains("KP002_WWS_2"), "label"] = "Crackle"

    print("=== DATA ===")
    print(f"n={len(df)} | classes={sorted(df['label'].unique().tolist())}")
    print(df['label'].value_counts())

    # Drop ultra-rare classes (cannot be split into both train/val)
    cls_counts = df['label'].value_counts()
    rare = cls_counts[cls_counts < MIN_IMAGES_PER_CLASS].index.tolist()
    if len(rare) > 0:
        print(f"\n[Info] Dropping rare classes (count < {MIN_IMAGES_PER_CLASS}): {rare}")
        df = df[~df['label'].isin(rare)].copy()

    # Retry SGK split until every class appears in both sets
    tr_df, va_df = stratified_group_split(df, test_size=TEST_SIZE,
                                          max_tries=SPLIT_RETRIES,
                                          min_per_class=1, seed=42)

    # If still some classes missing in either split, log & drop those classes consistently
    tr_classes = set(tr_df['label'].unique())
    va_classes = set(va_df['label'].unique())
    common = sorted(list(tr_classes & va_classes))
    if len(common) < len(set(df['label'].unique())):
        missing = sorted(list(set(df['label'].unique()) - set(common)))
        print(f"\n[Warn] Some classes couldn't appear in both sets and will be removed: {missing}")
        df = df[df['label'].isin(common)].copy()
        tr_df, va_df = stratified_group_split(df, test_size=TEST_SIZE,
                                              max_tries=SPLIT_RETRIES,
                                              min_per_class=1, seed=123)

    classes = sorted(df['label'].unique().tolist())
    cls2idx = {c:i for i,c in enumerate(classes)}
    tr_df['y'] = tr_df['label'].map(cls2idx)
    va_df['y'] = va_df['label'].map(cls2idx)

    print("\n=== Split (patient-based, stratified) ===")
    print_dist(tr_df, "Train dist:")
    print_dist(va_df, "Val   dist:")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Datasets / Loaders
    ds_tr = ImgDataset(tr_df, classes, train=True)
    ds_va = ImgDataset(va_df, classes, train=False)

    # Weighted sampler on train
    sampler = build_sampler(tr_df, classes)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, sampler=sampler,
                       num_workers=NUM_WORKERS, pin_memory=(device=="cuda"))
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=(device=="cuda"))

    # Model / Loss
    model = SmallCNN(num_classes=len(classes)).to(device)
    class_weights = compute_class_weights(tr_df['label'].tolist(), classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS, eta_min=LR*0.1)

    # Train
    logs = []
    best_f1 = -1.0
    best_path = OUT_ROOT / f"best_model_{time.strftime('%Y-%m-%d_%H-%M-%S')}.pt"
    no_imp = 0

    for ep in range(1, EPOCHS+1):
        tr_loss, tr_f1, tr_ap, *_ = run_epoch(model, dl_tr, device, criterion=criterion, optimizer=optim)
        va_loss, va_f1, va_ap, y_true, y_pred, prob, paths = run_epoch(model, dl_va, device)

        logs.append({
            "epoch": ep, "train_loss": tr_loss, "train_f1": tr_f1, "train_ap": float(tr_ap or 0.0),
            "val_loss": va_loss, "val_f1": va_f1, "val_ap": float(va_ap or 0.0),
            "lr": optim.param_groups[0]["lr"]
        })

        print(f"[Ep {ep:02d}] train {tr_loss:.3f}/F1 {tr_f1:.3f}/AP {tr_ap} | "
              f"val {va_loss:.3f}/F1 {va_f1:.3f}/AP {va_ap}")

        improved = va_f1 > best_f1 + 1e-4
        if improved:
            best_f1 = va_f1
            torch.save(model.state_dict(), best_path)
            no_imp = 0
        else:
            no_imp += 1

        sched.step()

        if no_imp >= EARLY_STOP:
            print("Early stop.")
            break

    # Reload best and final evaluation (VAL)
    model.load_state_dict(torch.load(best_path, map_location=device))
    va_loss, va_f1, va_ap, y_true, y_pred, prob, paths = run_epoch(model, dl_va, device)

    # Save artifacts
    stamp = best_path.stem.split("best_model_")[-1]
    # Metrics
    metrics = {
        "classes": classes,
        "val_loss": va_loss,
        "val_macro_f1": va_f1,
        "val_macro_ap": float(va_ap or 0.0),
        "n_train": len(tr_df),
        "n_val": len(va_df),
        "patient_split": True,
        "stratified_by_label": True,
    }
    with open(OUT_ROOT / f"metrics_{stamp}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Logs
    pd.DataFrame(logs).to_csv(OUT_ROOT / f"logs_{stamp}.csv", index=False)

    # Predictions CSV
    pred_rows = []
    for pth, yt, yp in zip(paths, y_true, y_pred):
        pred_rows.append({"path": pth, "y_true": int(yt), "y_pred": int(yp),
                          "true_label": classes[int(yt)], "pred_label": classes[int(yp)]})
    pd.DataFrame(pred_rows).to_csv(OUT_ROOT / f"preds_{stamp}.csv", index=False)

    # Curves & CM
    plot_curves_and_cm(OUT_ROOT, classes, y_true, prob, y_pred, tag=stamp)

    # Classification report (only for present classes)
    present_idxs = sorted(list(set(y_true.tolist())))
    present_names = [classes[i] for i in present_idxs]
    print("\n=== Final (VAL) ===")
    print(f"macro-F1: {va_f1:.3f} | macro-AP: {va_ap}")
    print("Confusion matrix (present classes only):")
    print(confusion_matrix(y_true, y_pred, labels=present_idxs))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, labels=present_idxs,
                                target_names=present_names, zero_division=0))

    print("\nSaved:")
    print(f"- best model: {best_path}")
    print(f"- preds: {OUT_ROOT / f'preds_{stamp}.csv'}")
    print(f"- metrics: {OUT_ROOT / f'metrics_{stamp}.json'}")
    print(f"- logs: {OUT_ROOT / f'logs_{stamp}.csv'}")
    print(f"- curves: {OUT_ROOT / f'curves_{stamp}.png'}")
    print(f"- cm: {OUT_ROOT / f'cm_{stamp}.png'}")


if __name__ == "__main__":
    main()
