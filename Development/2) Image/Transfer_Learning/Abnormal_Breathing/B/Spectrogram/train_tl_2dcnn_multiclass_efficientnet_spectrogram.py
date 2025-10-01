#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, time, random
from pathlib import Path
from collections import Counter
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models

from sklearn.metrics import classification_report, confusion_matrix, f1_score, average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Paths ----
IMG_ROOT = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Development/2) Image/Transfer_Learning/Abnormal_Breathing/B/Spectrogram/Processed Data_Segment")
OUT_ROOT = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Development/2) Image/Transfer_Learning/Abnormal_Breathing/B/Spectrogram/Result_Segment")

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_LP = 5
EPOCHS_FT = 30
EARLY_STOP = 8
LR_HEAD = 1e-3
LR_BACKBONE = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
NUM_WORKERS = 0
TEST_SIZE = 0.2   # test 비율
VAL_SIZE = 0.25   # train 중 validation 비율
MIN_IMAGES_PER_CLASS = 2
BACKBONE = "convnext_tiny"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
EXCLUDE_LABELS = {"Unknown"}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_root", type=str, default=str(IMG_ROOT))
    ap.add_argument("--out_root", type=str, default=str(OUT_ROOT))
    ap.add_argument("--backbone", type=str, default=BACKBONE, choices=["efficientnet_b0","convnext_tiny"])
    ap.add_argument("--img_size", type=int, default=IMG_SIZE)
    ap.add_argument("--bs", type=int, default=BATCH_SIZE)
    ap.add_argument("--epochs_lp", type=int, default=EPOCHS_LP)
    ap.add_argument("--epochs_ft", type=int, default=EPOCHS_FT)
    ap.add_argument("--early_stop", type=int, default=EARLY_STOP)
    ap.add_argument("--lr_head", type=float, default=LR_HEAD)
    ap.add_argument("--lr_backbone", type=float, default=LR_BACKBONE)
    ap.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    ap.add_argument("--label_smoothing", type=float, default=LABEL_SMOOTHING)
    ap.add_argument("--test_size", type=float, default=TEST_SIZE)
    ap.add_argument("--val_size", type=float, default=VAL_SIZE)
    ap.add_argument("--min_per_class", type=int, default=MIN_IMAGES_PER_CLASS)
    return ap.parse_args()

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def list_images(root: Path):
    items = []
    for label_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        label = label_dir.name
        for p in label_dir.rglob("*"):
            if p.suffix.lower() in IMG_EXTS:
                items.append((str(p), label))
    return pd.DataFrame(items, columns=["path","label"])

def print_dist(df, title):
    vc = df["label"].value_counts().sort_index()
    print(f"{title}\n{vc if len(vc)>0 else '(empty)'}\n")

def compute_class_weights(labels, classes):
    counts = Counter(labels)
    total = sum(counts[c] for c in classes)
    weights = []
    for c in classes:
        cw = total / max(1, counts[c])
        weights.append(cw)
    w = torch.tensor(weights, dtype=torch.float32)
    return w * (len(w) / w.sum())

def build_sampler(df):
    counts = df['label'].value_counts().to_dict()
    weights = df['label'].apply(lambda x: 1.0 / counts[x]).values
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

def simple_split(df, test_size=0.2, val_size=0.25, seed=42):
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df['label'], random_state=seed
    )
    train_df, val_df = train_test_split(
        train_df, test_size=val_size, stratify=train_df['label'], random_state=seed+1
    )
    return train_df, val_df, test_df

class ImgDataset(Dataset):
    def __init__(self, df, classes, img_size=224, train=True):
        self.df = df.reset_index(drop=True)
        self.classes = classes
        self.cls2idx = {c:i for i,c in enumerate(classes)}
        if train:
            self.tx = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomApply([transforms.ColorJitter(0.10,0.10,0.05,0.02)], p=0.5),
                transforms.RandomApply([transforms.RandomAffine(degrees=5, translate=(0.02,0.02), scale=(0.98,1.02))], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])
        else:
            self.tx = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        x = Image.open(r.path).convert("RGB")
        x = self.tx(x)
        y = self.cls2idx[r.label]
        return x, y, r.path

def load_backbone(name: str, num_classes: int):
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_f = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_f, num_classes)
        return m
    elif name == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        in_f = m.classifier[2].in_features
        m.classifier[2] = nn.Linear(in_f, num_classes)
        return m
    else:
        raise ValueError(f"Unknown backbone: {name}")

@torch.no_grad()
def eval_epoch(model, dl, device, label_smoothing=0.0):
    model.eval()
    total, n = 0.0, 0
    ys, ps, paths = [], [], []
    for x,y,pth in dl:
        x,y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
        total += loss.item()*x.size(0); n += x.size(0)
        ys.append(y.cpu().numpy()); ps.append(logits.softmax(1).cpu().numpy())
        paths += list(pth)
    y = np.concatenate(ys); p = np.concatenate(ps)
    pred = p.argmax(1)
    macro_f1 = f1_score(y, pred, average="macro", zero_division=0)
    try: ap = average_precision_score(y, p, average="macro")
    except: ap = None
    return total/n, macro_f1, ap, y, pred, p, paths

def train_epoch(model, dl, device, criterion, optimizer, use_amp=False):
    model.train(True)
    total, n = 0.0, 0
    ys, ps = [], []
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for x,y,_ in dl:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(x); loss = criterion(logits, y)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            logits = model(x); loss = criterion(logits, y)
            loss.backward(); optimizer.step()
        total += loss.item()*x.size(0); n += x.size(0)
        ys.append(y.detach().cpu().numpy()); ps.append(logits.detach().softmax(1).cpu().numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    pred = p.argmax(1)
    macro_f1 = f1_score(y, pred, average="macro", zero_division=0)
    try: ap = average_precision_score(y, p, average="macro")
    except: ap = None
    return total/n, macro_f1, ap

def main():
    args = parse_args()
    set_seed(42)

    IMG_ROOT = Path(args.img_root)
    OUT_ROOT = Path(args.out_root); OUT_ROOT.mkdir(parents=True, exist_ok=True)

    df = list_images(IMG_ROOT)

    # Unknown drop
    if EXCLUDE_LABELS:
        df = df[~df['label'].isin(EXCLUDE_LABELS)].copy()

    # Rare 클래스 제거
    cls_counts = df['label'].value_counts()
    rare = cls_counts[cls_counts < args.min_per_class].index.tolist()
    if len(rare) > 0:
        df = df[~df['label'].isin(rare)].copy()

    # 🚨 segment 단위 stratified split
    tr_df, va_df, te_df = simple_split(df, test_size=args.test_size, val_size=args.val_size, seed=42)

    classes = sorted(df['label'].unique().tolist())
    cls2idx = {c:i for i,c in enumerate(classes)}
    for d in [tr_df, va_df, te_df]:
        d['y'] = d['label'].map(cls2idx)

    print("=== Split (segment-level stratified) ===")
    print_dist(tr_df, "Train dist:")
    print_dist(va_df, "Val   dist:")
    print_dist(te_df, "Test  dist:")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds_tr = ImgDataset(tr_df, classes, img_size=args.img_size, train=True)
    ds_va = ImgDataset(va_df, classes, img_size=args.img_size, train=False)
    ds_te = ImgDataset(te_df, classes, img_size=args.img_size, train=False)

    sampler = build_sampler(tr_df)
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, sampler=sampler,
                       num_workers=NUM_WORKERS, pin_memory=(device=="cuda"))
    dl_va = DataLoader(ds_va, batch_size=args.bs, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=(device=="cuda"))
    dl_te = DataLoader(ds_te, batch_size=args.bs, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=(device=="cuda"))

    model = load_backbone(args.backbone, num_classes=len(classes)).to(device)

    # Linear probe
    for p in model.features.parameters(): p.requires_grad_(False)

    class_weights = compute_class_weights(tr_df['label'].tolist(), classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr_head, weight_decay=args.weight_decay)
    use_amp = (device == "cuda")

    best_f1 = -1.0
    stamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    best_path = OUT_ROOT / f"best_model_{args.backbone}_{stamp}.pt"
    logs = []

    print("\n=== Linear Probe ===")
    for ep in range(1, args.epochs_lp+1):
        tr_loss, tr_f1, tr_ap = train_epoch(model, dl_tr, device, criterion, opt, use_amp=use_amp)
        va_loss, va_f1, va_ap, y_true, y_pred, prob, paths = eval_epoch(model, dl_va, device)
        improved = va_f1 > best_f1 + 1e-4
        if improved:
            best_f1 = va_f1; torch.save(model.state_dict(), best_path)
        print(f"[LP {ep}] val F1 {va_f1:.3f}")

    # Fine-tune
    state = torch.load(best_path, map_location=device)
    model.load_state_dict(state)
    for name,p in model.features.named_parameters():
        if "6" in name or "7" in name: p.requires_grad_(True)

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr_backbone, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs_ft, eta_min=args.lr_backbone*0.1)

    for ep in range(1, args.epochs_ft+1):
        tr_loss, tr_f1, tr_ap = train_epoch(model, dl_tr, device, criterion, opt, use_amp=use_amp)
        va_loss, va_f1, va_ap, y_true, y_pred, prob, paths = eval_epoch(model, dl_va, device)
        improved = va_f1 > best_f1 + 1e-4
        if improved:
            best_f1 = va_f1; torch.save(model.state_dict(), best_path)
        sched.step()
        print(f"[FT {ep}] val F1 {va_f1:.3f}")

    # 최종 Test 평가
    state = torch.load(best_path, map_location=device)
    model.load_state_dict(state)
    te_loss, te_f1, te_ap, y_true, y_pred, prob, paths = eval_epoch(model, dl_te, device)
    print(f"\n=== Final Test ===\nF1={te_f1:.3f}, AP={te_ap}")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=classes))

if __name__ == "__main__":
    main()
