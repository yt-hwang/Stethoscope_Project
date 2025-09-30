#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, time, random
from pathlib import Path
from collections import Counter
import argparse
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models

from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    average_precision_score, precision_recall_curve
)
from sklearn.model_selection import StratifiedGroupKFold

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Paths: choose spectrogram or scalogram ----
# Mac
# IMG_ROOT = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Development/2) Image/Transfer_Learning/Abnormal_Breathing/A/Scalogram/Processed Data")
# OUT_ROOT = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Development/2) Image/Transfer_Learning/Abnormal_Breathing/A/Scalogram/Result")

# Windows
IMG_ROOT = Path("D:\\Stethoscope_Project\\Development\\2) Image\\Transfer_Learning\\Abnormal_Breathing\\A\\Scalogram\\Processed Data")
OUT_ROOT = Path("D:\\Stethoscope_Project\\Development\\2) Image\\Transfer_Learning\\Abnormal_Breathing\\A\\Scalogram\\Result")

# (switch to scalogram by changing the two lines above)

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_LP = 5
EPOCHS_FT = 30
EARLY_STOP = 8
LR_HEAD = 1e-3
LR_BACKBONE = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
NUM_WORKERS = 2
TEST_SIZE = 0.15
SPLIT_RETRIES = 200
MIN_IMAGES_PER_CLASS = 5
BACKBONE = "efficientnet_b0"  # or "convnext_tiny"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
EXCLUDE_LABELS = {"Unknown"}  # drop these labels entirely

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
    ap.add_argument("--min_per_class", type=int, default=MIN_IMAGES_PER_CLASS)
    ap.add_argument("--split_retries", type=int, default=SPLIT_RETRIES)
    ap.add_argument("--imbalance_strategy", type=str, default="loss",
                    choices=["none", "sampler", "loss", "both"],
                    help="불균형 처리 방식 선택: sampler | loss | both | none (기본: loss)")
    ap.add_argument("--save_softmax", action="store_true",
                help="예측 결과 CSV에 클래스별 softmax 확률(%)을 저장")
    ap.add_argument("--topk", type=int, default=3,
                help="Top-k 요약 저장 (기본 3)")
    ap.add_argument("--print_examples", type=int, default=0,
                help="에폭마다 소프트맥스 예시 n개를 콘솔에 출력(0이면 비활성)")

    return ap.parse_args()

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def extract_patient_id(fname: str) -> str:
    base = Path(fname).stem
    return base.split("_")[0] if "_" in base else base

def list_images(root: Path):
    items = []
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

def stratified_group_split(df, test_size=0.2, max_tries=200, min_per_class=1, seed=42):
    labels = df['label'].values
    groups = df['patient_id'].values
    ok_splits = []
    for attempt in range(max_tries):
        n_splits = max(2, int(round(1/test_size)))
        sgk = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed+attempt)
        for tr_idx, va_idx in sgk.split(np.zeros(len(df)), labels, groups):
            tr_df, va_df = df.iloc[tr_idx], df.iloc[va_idx]
            tr_counts = tr_df['label'].value_counts()
            va_counts = va_df['label'].value_counts()
            classes = sorted(df['label'].unique().tolist())
            cond = all(tr_counts.get(c,0) >= min_per_class and va_counts.get(c,0) >= min_per_class for c in classes)
            if cond:
                return tr_df.copy(), va_df.copy()
            ok_splits.append((tr_df.copy(), va_df.copy()))
    return ok_splits[0] if ok_splits else (df.sample(frac=0.8, random_state=seed), df.drop(df.sample(frac=0.8, random_state=seed).index))

class ImgDataset(Dataset):
    def __init__(self, df, classes, img_size=224, train=True):
        self.df = df.reset_index(drop=True)
        self.classes = classes
        self.cls2idx = {c:i for i,c in enumerate(classes)}
        # 의미 보존형 증강: 시간축(가로) 이동만 허용, 수직/회전/스케일 제거
        if train:
            self.tx = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomApply([transforms.ColorJitter(0.10,0.10,0.05,0.02)], p=0.5),
                transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.05,0.0))], p=0.5),
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
        # 간헐적 이미지 로드 실패 대비
        for _ in range(3):
            r = self.df.iloc[i]
            try:
                x = Image.open(r.path).convert("RGB")
                x = self.tx(x)
                y = self.cls2idx[r.label]
                return x, y, r.path
            except (FileNotFoundError, UnidentifiedImageError, OSError):
                i = np.random.randint(0, len(self.df))
        # 그래도 실패하면 마지막 시도에서 예외
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

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

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

    # 멀티클래스 macro-AP를 안정적으로: 원-핫으로 클래스별 AP 계산 후 평균
    classes = p.shape[1]
    y_onehot = np.zeros((len(y), classes), dtype=int)
    for i,lab in enumerate(y): y_onehot[i, lab] = 1
    aps = []
    for c in range(classes):
        try:
            aps.append(average_precision_score(y_onehot[:,c], p[:,c]))
        except Exception:
            pass
    macro_ap = float(np.mean(aps)) if len(aps) > 0 else None

    return total/n, macro_f1, macro_ap, y, pred, p, paths

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

    # macro-AP(학습 모니터링용)
    classes = p.shape[1]
    y_onehot = np.zeros((len(y), classes), dtype=int)
    for i,lab in enumerate(y): y_onehot[i, lab] = 1
    aps = []
    for c in range(classes):
        try:
            aps.append(average_precision_score(y_onehot[:,c], p[:,c]))
        except Exception:
            pass
    macro_ap = float(np.mean(aps)) if len(aps) > 0 else None

    return total/n, macro_f1, macro_ap

def plot_curves_and_cm(out_dir, classes, y_true, prob, y_pred, tag):
    out_dir.mkdir(parents=True, exist_ok=True)
    present = sorted(list(set(y_true.tolist())))

    # PR curves (클래스별)
    y_onehot = np.zeros((len(y_true), len(classes)), dtype=np.int64)
    for i,lab in enumerate(y_true): y_onehot[i, lab] = 1
    plt.figure(figsize=(7,6))
    for c in present:
        prec, rec, _ = precision_recall_curve(y_onehot[:,c], prob[:,c])
        ap = average_precision_score(y_onehot[:,c], prob[:,c])
        plt.plot(rec, prec, label=f"{classes[c]} (AP={ap:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR Curves ({tag})"); plt.legend()
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(out_dir / f"curves_{tag}.png", dpi=150); plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=present)
    plt.figure(figsize=(6,5))
    im = plt.imshow(cm, interpolation='nearest')
    plt.title(f"Confusion Matrix ({tag})"); plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_names = [classes[i] for i in present]
    plt.xticks(range(len(present)), tick_names, rotation=45, ha='right')
    plt.yticks(range(len(present)), tick_names)
    thresh = cm.max() if cm.size else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thresh/2 else "black")
    plt.tight_layout()
    plt.savefig(out_dir / f"cm_{tag}.png", dpi=150); plt.close()

def main():
    args = parse_args()
    set_seed(42)

    IMG_ROOT = Path(args.img_root)
    OUT_ROOT = Path(args.out_root); OUT_ROOT.mkdir(parents=True, exist_ok=True)

    df = list_images(IMG_ROOT)

    # (Optional) mislabel fix (무해)
    df.loc[df['path'].str.contains("KP002_WWS_1"), "label"] = "Crackle"
    df.loc[df['path'].str.contains("KP002_WWS_2"), "label"] = "Crackle"

    # 라벨 제외 (예: Unknown)
    if EXCLUDE_LABELS:
        before = len(df)
        df = df[~df['label'].isin(EXCLUDE_LABELS)].copy()
        dropped = before - len(df)
        print(f"[Filter] Dropped {dropped} items with labels in {sorted(list(EXCLUDE_LABELS))}")

    print("=== DATA (after exclusion) ===")
    print(f"n={len(df)} | classes={sorted(df['label'].unique().tolist())}")
    print(df['label'].value_counts())

    # 희귀 라벨 제거 (클래스당 최소 이미지 수)
    cls_counts = df['label'].value_counts()
    rare = cls_counts[cls_counts < args.min_per_class].index.tolist()
    if len(rare) > 0:
        print(f"\n[Info] Dropping rare classes (count < {args.min_per_class}): {rare}")
        df = df[~df['label'].isin(rare)].copy()

    # 환자 기반 + 라벨 계층 분할 (재시도)
    tr_df, va_df = stratified_group_split(df, test_size=args.test_size,
                                          max_tries=args.split_retries,
                                          min_per_class=1, seed=42)

    # train/val 모두에 존재하는 공통 클래스만 유지
    tr_classes = set(tr_df['label'].unique()); va_classes = set(va_df['label'].unique())
    common = sorted(list(tr_classes & va_classes))
    if len(common) < len(set(df['label'].unique())):
        missing = sorted(list(set(df['label'].unique()) - set(common)))
        print(f"\n[Warn] Some classes didn’t appear in both sets and will be removed: {missing}")
        df = df[df['label'].isin(common)].copy()
        tr_df, va_df = stratified_group_split(df, test_size=args.test_size,
                                              max_tries=args.split_retries,
                                              min_per_class=1, seed=123)

    classes = sorted(df['label'].unique().tolist())
    cls2idx = {c:i for i,c in enumerate(classes)}
    tr_df['y'] = tr_df['label'].map(cls2idx); va_df['y'] = va_df['label'].map(cls2idx)

    print("\n=== Split (patient-based, stratified) ===")
    print_dist(tr_df, "Train dist:")
    print_dist(va_df, "Val   dist:")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ds_tr = ImgDataset(tr_df, classes, img_size=args.img_size, train=True)
    ds_va = ImgDataset(va_df, classes, img_size=args.img_size, train=False)

    # 불균형 처리 전략 선택
    sampler = None
    if args.imbalance_strategy in ["sampler", "both"]:
        sampler = build_sampler(tr_df)
        dl_tr = DataLoader(ds_tr, batch_size=args.bs, sampler=sampler,
                           num_workers=NUM_WORKERS, pin_memory=(device=="cuda"))
    else:
        dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=(device=="cuda"))

    dl_va = DataLoader(ds_va, batch_size=args.bs, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=(device=="cuda"))

    model = load_backbone(args.backbone, num_classes=len(classes)).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nBackbone: {args.backbone} | params: total={total_params:,} trainable(now)={trainable_params:,}")

    # Linear probe: features freeze
    for p in model.features.parameters():
        p.requires_grad_(False)

    # 손실가중치 사용 여부
    if args.imbalance_strategy in ["loss", "both"]:
        class_weights = compute_class_weights(tr_df['label'].tolist(), classes).to(device)
    else:
        class_weights = None

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
        va_loss, va_f1, va_ap, y_true, y_pred, prob, paths = eval_epoch(model, dl_va, device, label_smoothing=args.label_smoothing)
        # (옵션) 소프트맥스 예시 출력
        if args.print_examples > 0 and len(paths) > 0:
            ex = min(args.print_examples, len(paths))
            print("[Softmax examples - LP]")
            for i in range(ex):
                row_probs = prob[i]
                top_idx = int(np.argmax(row_probs))
                top_conf = float(row_probs[top_idx]) * 100
                print(f"  {i+1}) {paths[i]} | pred={classes[top_idx]} ({top_conf:.1f}%) | "
                    f"true={classes[int(y_true[i])]}")
        improved = va_f1 > best_f1 + 1e-4
        if improved:
            best_f1 = va_f1; torch.save(model.state_dict(), best_path)
        logs.append({
            "phase":"LP","epoch":ep,"train_loss":tr_loss,"train_f1":tr_f1,"train_ap":float(tr_ap or 0.0),
            "val_loss":va_loss,"val_f1":va_f1,"val_ap":float(va_ap or 0.0),"lr":opt.param_groups[0]["lr"]
        })
        print(f"[LP {ep:02d}/{args.epochs_lp}] train {tr_loss:.3f}/F1 {tr_f1:.3f}/AP {tr_ap} | "
              f"val {va_loss:.3f}/F1 {va_f1:.3f}/AP {va_ap} {'(*)' if improved else ''}")

    # Fine-tune: 일부 feature unfreeze
    print("\n=== Fine-Tune ===")
    try:
        state = torch.load(best_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(best_path, map_location=device)
    model.load_state_dict(state)

    # FT 대상 계층 보정
    for p in model.features.parameters():
        p.requires_grad_(False)

    if args.backbone == "efficientnet_b0":
        # features의 마지막 두 stage만 풀기(버전 안전: children() 기준)
        feats = list(model.features.children())
        for blk in feats[-2:]:
            for p in blk.parameters():
                p.requires_grad_(True)
    elif args.backbone == "convnext_tiny":
        # ConvNeXt: 마지막 stage만 풀기
        for p in model.features[-1].parameters():
            p.requires_grad_(True)

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr_backbone, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs_ft, eta_min=args.lr_backbone*0.1)

    no_imp = 0
    for ep in range(1, args.epochs_ft+1):
        tr_loss, tr_f1, tr_ap = train_epoch(model, dl_tr, device, criterion, opt, use_amp=use_amp)
        va_loss, va_f1, va_ap, y_true, y_pred, prob, paths = eval_epoch(model, dl_va, device, label_smoothing=args.label_smoothing)
        # (옵션) 소프트맥스 예시 출력
        if args.print_examples > 0 and len(paths) > 0:
            ex = min(args.print_examples, len(paths))
            print("[Softmax examples - FT]")
            for i in range(ex):
                row_probs = prob[i]
                top_idx = int(np.argmax(row_probs))
                top_conf = float(row_probs[top_idx]) * 100
                print(f"  {i+1}) {paths[i]} | pred={classes[top_idx]} ({top_conf:.1f}%) | "
                    f"true={classes[int(y_true[i])]}")
        if improved:
            best_f1 = va_f1; torch.save(model.state_dict(), best_path); no_imp = 0
        else:
            no_imp += 1
        logs.append({
            "phase":"FT","epoch":ep,"train_loss":tr_loss,"train_f1":tr_f1,"train_ap":float(tr_ap or 0.0),
            "val_loss":va_loss,"val_f1":va_f1,"val_ap":float(va_ap or 0.0),"lr":opt.param_groups[0]["lr"]
        })
        print(f"[FT {ep:02d}/{args.epochs_ft}] train {tr_loss:.3f}/F1 {tr_f1:.3f}/AP {tr_ap} | "
              f"val {va_loss:.3f}/F1 {va_f1:.3f}/AP {va_ap} {'(*)' if improved else ''}")
        sched.step()
        if no_imp >= args.early_stop:
            print("Early stop."); break

    # Final eval
    try:
        state = torch.load(best_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(best_path, map_location=device)
    model.load_state_dict(state)

    va_loss, va_f1, va_ap, y_true, y_pred, prob, paths = eval_epoch(model, dl_va, device, label_smoothing=args.label_smoothing)

    pd.DataFrame(logs).to_csv(OUT_ROOT / f"logs_{args.backbone}_{stamp}.csv", index=False)

    metrics = {
        "backbone": args.backbone,
        "classes": classes,
        "n_params_total": int(sum(p.numel() for p in model.parameters())),
        "n_params_trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "val_loss": va_loss,
        "val_macro_f1": va_f1,
        "val_macro_ap": float(va_ap or 0.0),
        "n_train": len(tr_df),
        "n_val": len(va_df),
        "patient_split": True,
        "stratified_by_label": True,
        "excluded_labels": sorted(list(EXCLUDE_LABELS)),
        "imbalance_strategy": args.imbalance_strategy,
        "train_class_counts": tr_df['label'].value_counts().to_dict(),
        "val_class_counts": va_df['label'].value_counts().to_dict(),
    }
    with open(OUT_ROOT / f"metrics_{args.backbone}_{stamp}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # 소프트맥스 확률 및 Top-k 요약 포함 저장
    pred_rows = []
    k = max(1, min(args.topk, len(classes)))
    for i, (pth, yt) in enumerate(zip(paths, y_true)):
        row = {
            "path": pth,
            "y_true": int(yt),
            "true_label": classes[int(yt)],
            "y_pred": int(y_pred[i]),
            "pred_label": classes[int(y_pred[i])],
        }

        # --save_softmax가 켜진 경우: 클래스별 확률(%) 컬럼 추가
        if args.save_softmax:
            for ci, cname in enumerate(classes):
                row[f"prob_{cname}"] = float(prob[i, ci] * 100.0)

            # Top-k 요약 (예: "Crackle 42.3 | Wheeze 31.8 | Rhonchi 20.4")
            sorted_idx = np.argsort(-prob[i])[:k]
            topk_str = " | ".join([f"{classes[j]} {prob[i, j]*100.0:.1f}" for j in sorted_idx])
            row["topk"] = topk_str

        # 항상 저장: top-1 신뢰도(%)
        row["top1_conf"] = float(prob[i, int(y_pred[i])] * 100.0)

        pred_rows.append(row)

    pred_df = pd.DataFrame(pred_rows)
    pred_csv = OUT_ROOT / f"preds_{args.backbone}_{stamp}.csv"
    pred_df.to_csv(pred_csv, index=False)
    print(f"[Saved] preds with softmax: {pred_csv}")


    plot_curves_and_cm(OUT_ROOT, classes, y_true, prob, y_pred, tag=f"{args.backbone}_{stamp}")

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
    print(f"- best model: {OUT_ROOT / f'best_model_{args.backbone}_{stamp}.pt'}")
    print(f"- preds: {OUT_ROOT / f'preds_{args.backbone}_{stamp}.csv'}")
    print(f"- metrics: {OUT_ROOT / f'metrics_{args.backbone}_{stamp}.json'}")
    print(f"- logs: {OUT_ROOT / f'logs_{args.backbone}_{stamp}.csv'}")
    print(f"- curves: {OUT_ROOT / f'curves_{args.backbone}_{stamp}.png'}")
    print(f"- cm: {OUT_ROOT / f'cm_{args.backbone}_{stamp}.png'}")

if __name__ == "__main__":
    main()
