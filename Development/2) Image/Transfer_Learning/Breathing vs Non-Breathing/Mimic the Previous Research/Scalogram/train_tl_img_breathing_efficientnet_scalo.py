#!/usr/bin/env python3
# train_tl_img_breathing_efficientnet_scalo.py
# Scalogram 타일(2s/0.5hop)로 EfficientNet-B0 이진 분류 (GroupKFold-환자 단위)

import random, os
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, average_precision_score, classification_report, confusion_matrix

# ===== 경로 =====
MANIFEST = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Development/9.2) Reference/BreathingTiles_scalo_paper/manifest.csv")
OUTDIR   = MANIFEST.parent / "models_efficientnet_b0_binary_scalo"
OUTDIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["NonBreathing", "Breathing"]
LABELS_IDX = [0,1]

LOSS_TYPE = "focal"  # "focal" or "bce"
EPOCHS_LP = 5
EPOCHS_FT = 15
BS = 32
LR_HEAD = 1e-3
LR_BACKBONE = 1e-4
WEIGHT_DECAY = 1e-4
N_SPLITS = 5
SEED = 42
USE_AMP = True

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed()

def counts_by_class(df): return df.label.value_counts().reindex(CLASSES, fill_value=0).to_dict()

def print_split_stats(name, df):
    vc = counts_by_class(df)
    print(f"{name} -> n={len(df)}, patients={df.patient_id.nunique()} | " +
          "[" + ", ".join(f"{k}: {vc[k]}" for k in CLASSES) + "]")

class ImgDS(Dataset):
    def __init__(self, df, train=True):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.cls_to_idx = {"NonBreathing":0, "Breathing":1}
        if train:
            self.tx = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandomApply([transforms.RandomAffine(degrees=5, translate=(0.02,0.02), scale=(0.98,1.02))], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])
        else:
            self.tx = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        x = Image.open(row.path).convert("RGB")
        x = self.tx(x)
        y = self.cls_to_idx[row.label]
        gid = row.patient_id
        return x, y, gid

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma
    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)
        return ((1-pt)**self.gamma * ce).mean()

@torch.no_grad()
def evaluate(dl, model, device):
    model.eval()
    ys, ps, tot, n = [], [], 0.0, 0
    for x,y,_ in dl:
        x,y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="none")
        tot += loss.mean().item()*x.size(0); n += x.size(0)
        ys.append(y.cpu().numpy()); ps.append(logits.softmax(1).cpu().numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    pred = p.argmax(1)
    f1 = f1_score(y, pred, labels=LABELS_IDX, average="macro", zero_division=0)
    try: ap = average_precision_score(y, p[:,1])
    except: ap = None
    return tot/n, f1, ap, y, pred, p

def train_fold(train_df, val_df, fold=0, device="cuda"):
    tr = ImgDS(train_df, True); va = ImgDS(val_df, False)
    num_workers = 0; pin_mem = (device=="cuda")
    tdl = DataLoader(tr, batch_size=BS, shuffle=True,  num_workers=num_workers, pin_memory=pin_mem)
    vdl = DataLoader(va, batch_size=BS, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)

    # 클래스 가중
    counts = train_df.label.value_counts().reindex(CLASSES, fill_value=0).values.astype(float)
    alpha = counts.sum()/np.maximum(counts,1e-6)
    alpha = alpha * (len(alpha)/alpha.sum())
    alpha = torch.tensor(alpha, dtype=torch.float32, device=device)

    pos_weight = None
    if counts[1] > 0:
        neg, pos = counts[0], counts[1]
        pos_weight = torch.tensor([neg/max(pos,1)], dtype=torch.float32, device=device)

    # 모델
    m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_f = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_f, 2)
    m = m.to(device)

    use_amp = USE_AMP and device=="cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # 손실
    if LOSS_TYPE.lower()=="bce":
        def criterion(logits, target):
            return F.binary_cross_entropy_with_logits(logits[:,1], (target==1).float(), pos_weight=pos_weight)
    else:
        fl = FocalLoss(alpha=alpha, gamma=2.0)
        def criterion(logits, target): return fl(logits, target)

    # --- Stage 1: Linear Probe ---
    for p in m.features.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, m.parameters()), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)

    best_f1 = -1.0
    best_path = OUTDIR / f"fold{fold}_best.pt"

    def run_epoch(dl, train=True):
        m.train(train)
        ys, ps, tot, n = [], [], 0.0, 0
        for x,y,_ in dl:
            x,y = x.to(device), y.to(device)
            if train: opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = m(x); loss = criterion(logits, y)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                logits = m(x); loss = criterion(logits, y)
                if train: loss.backward(); opt.step()
            tot += loss.item()*x.size(0); n += x.size(0)
            ys.append(y.detach().cpu().numpy()); ps.append(logits.detach().softmax(1).cpu().numpy())
        y_true = np.concatenate(ys); p = np.concatenate(ps)
        pred = p.argmax(1)
        f1 = f1_score(y_true, pred, labels=LABELS_IDX, average="macro", zero_division=0)
        try: ap = average_precision_score(y_true, p[:,1])
        except: ap = None
        return tot/n, f1, ap

    for ep in range(1, EPOCHS_LP+1):
        tr_loss, tr_f1, tr_ap = run_epoch(tdl, True)
        va_loss, va_f1, va_ap = run_epoch(vdl, False)
        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save(m.state_dict(), best_path)
        print(f"[Fold {fold} | LP {ep}/{EPOCHS_LP}] "
              f"train {tr_loss:.3f}/F1 {tr_f1:.3f}/AP {tr_ap if tr_ap is not None else 'NA'} | "
              f"val {va_loss:.3f}/F1 {va_f1:.3f}/AP {va_ap if va_ap is not None else 'NA'}")

    # --- Stage 2: Fine-tune ---
    def safe_load(pth, map_location=None):
        try: return torch.load(pth, map_location=map_location, weights_only=True)
        except TypeError: return torch.load(pth, map_location=map_location)

    m.load_state_dict(safe_load(best_path, map_location=device))
    for name,p in m.features.named_parameters():
        if any(k in name for k in ["6","7"]):
            p.requires_grad_(True)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, m.parameters()), lr=LR_BACKBONE, weight_decay=WEIGHT_DECAY)

    patience, no_imp = 6, 0
    for ep in range(1, EPOCHS_FT+1):
        tr_loss, tr_f1, tr_ap = run_epoch(tdl, True)
        va_loss, va_f1, va_ap = run_epoch(vdl, False)
        improved = va_f1 > best_f1 + 1e-4
        if improved:
            best_f1 = va_f1; torch.save(m.state_dict(), best_path); no_imp = 0
        else:
            no_imp += 1
        print(f"[Fold {fold} | FT {ep}/{EPOCHS_FT}] "
              f"train {tr_loss:.3f}/F1 {tr_f1:.3f}/AP {tr_ap if tr_ap is not None else 'NA'} | "
              f"val {va_loss:.3f}/F1 {va_f1:.3f}/AP {va_ap if va_ap is not None else 'NA'} "
              f"{'(*)' if improved else ''}")
        if no_imp >= patience:
            print("Early stop."); break

    # Best 평가
    m.load_state_dict(safe_load(best_path, map_location=device))
    _, _, _, vy, vp, _ = evaluate(vdl, m, device)
    print(classification_report(vy, vp, labels=LABELS_IDX, target_names=CLASSES, zero_division=0))
    print(confusion_matrix(vy, vp, labels=LABELS_IDX))
    return best_f1

def main():
    df = pd.read_csv(MANIFEST)
    assert {"path","label","patient_id"}.issubset(df.columns)
    df = df[df.label.isin(CLASSES)].copy()

    print("=== FULL DATASET STAT ===")
    print_split_stats("All", df)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device} (AMP {'ON' if (USE_AMP and device=='cuda') else 'OFF'})")

    gkf = GroupKFold(n_splits=N_SPLITS)
    bests = []
    for k,(tr,va) in enumerate(gkf.split(df, groups=df.patient_id)):
        tr_df, va_df = df.iloc[tr].copy(), df.iloc[va].copy()
        print(f"\n=== Fold {k} ===")
        print_split_stats("Train", tr_df)
        print_split_stats("Val",   va_df)
        bests.append(train_fold(tr_df, va_df, fold=k, device=device))

    print("\n==== Summary ====")
    print("best macro-F1 per fold:", [f"{b:.3f}" for b in bests])
    print(f"mean={np.mean(bests):.3f}  std={np.std(bests):.3f}")

if __name__ == "__main__":
    main()
