#!/usr/bin/env python3
# train_tl_img_breathing_efficientnet_75_25_spectrogram.py
# 논문식: 샘플(타일) 기준 75/25 stratified split + EfficientNet-B0 전이학습
# 입력 manifest는 멜 스펙트로그램 타일용 manifest.csv

import os, random, io, csv
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, average_precision_score, classification_report, confusion_matrix

# ===== 경로/설정 =====
MANIFEST = Path(
    "/Users/yunhwang/Desktop/Stethoscope_Project/Development/9.2) Reference/BreathingTiles_mel_paper/manifest.csv"
)
OUTDIR   = MANIFEST.parent / "models_efficientnet_b0_binary_75_25"
OUTDIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["NonBreathing", "Breathing"]
LABELS_IDX = [0,1]

LOSS_TYPE     = "focal"  # "focal" or "bce"
EPOCHS_LP     = 5
EPOCHS_FT     = 15
BS            = 32
LR_HEAD       = 1e-3
LR_BACKBONE   = 1e-4
WEIGHT_DECAY  = 1e-4
USE_AMP       = True

N_REPEATS = 5
SEED_BASE = 42

SAVE_LOGS = True
SAVE_REPORT = True
SAVE_SUMMARY = True
SAVE_SPLIT = True
SAVE_PREDICTIONS = True

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def counts_by_class(df: pd.DataFrame):
    return df.label.value_counts().reindex(CLASSES, fill_value=0).to_dict()

def log_print(msg: str, logger):
    print(msg)
    if logger is not None:
        logger.write(msg + "\n")

def print_split_stats(name: str, df: pd.DataFrame, logger=None):
    vc = counts_by_class(df)
    log_print(f"{name} -> n={len(df)} | [" + ", ".join(f"{k}: {vc[k]}" for k in CLASSES) + "]", logger)

class ImgDS(Dataset):
    def __init__(self, df, train=True):
        self.df = df.reset_index(drop=True)
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
        return x, y

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
    for x,y in dl:
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

def train_once(df: pd.DataFrame, seed: int, device: str):
    set_seed(seed)
    logger = open(OUTDIR / f"train_log_seed{seed}.txt", "w") if SAVE_LOGS else None

    # 샘플 기준 75/25 split
    y = df["label"].map({"NonBreathing":0, "Breathing":1}).values
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    tr_idx, te_idx = next(sss.split(np.zeros(len(df)), y))
    tr_df, te_df = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()

    log_print("\n=== 75/25 Split ===", logger)
    print_split_stats("Train(75%)", tr_df, logger)
    print_split_stats("Test (25%)", te_df, logger)

    if SAVE_SPLIT:
        split_df = pd.concat([
            tr_df.assign(split="train", seed=seed),
            te_df.assign(split="test",  seed=seed)
        ], ignore_index=True)
        split_df.to_csv(OUTDIR / f"split_seed{seed}.csv", index=False)

    num_workers = 0
    pin_mem = (device=="cuda")
    tdl = DataLoader(ImgDS(tr_df, True),  batch_size=BS, shuffle=True,  num_workers=num_workers, pin_memory=pin_mem)
    vdl = DataLoader(ImgDS(te_df, False), batch_size=BS, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)

    counts = tr_df.label.value_counts().reindex(CLASSES, fill_value=0).values.astype(float)
    alpha = counts.sum()/np.maximum(counts,1e-6)
    alpha = alpha * (len(alpha)/alpha.sum())
    alpha = torch.tensor(alpha, dtype=torch.float32, device=device)

    pos_weight = None
    if counts[1] > 0:
        neg, pos = counts[0], counts[1]
        pos_weight = torch.tensor([neg/max(pos,1)], dtype=torch.float32, device=device)

    m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_f = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_f, 2)
    m = m.to(device)

    use_amp = USE_AMP and device=="cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    if LOSS_TYPE.lower()=="bce":
        def criterion(logits, target):
            return F.binary_cross_entropy_with_logits(logits[:,1], (target==1).float(), pos_weight=pos_weight)
    else:
        fl = FocalLoss(alpha=alpha, gamma=2.0)
        def criterion(logits, target): return fl(logits, target)

    # Linear-Probe
    for p in m.features.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, m.parameters()), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)

    def run_epoch(dl, train=True):
        m.train(train)
        ys, ps, tot, n = [], [], 0.0, 0
        for x,y in dl:
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

    best_f1 = -1.0
    best_path = OUTDIR / f"best_seed{seed}.pt"

    for ep in range(1, EPOCHS_LP+1):
        tr_loss, tr_f1, tr_ap = run_epoch(tdl, True)
        te_loss, te_f1, te_ap = run_epoch(vdl, False)
        if te_f1 > best_f1:
            best_f1 = te_f1; torch.save(m.state_dict(), best_path)
        log_print(f"[LP {ep}/{EPOCHS_LP}] "
                  f"train {tr_loss:.3f}/F1 {tr_f1:.3f}/AP {tr_ap if tr_ap is not None else 'NA'} | "
                  f"test {te_loss:.3f}/F1 {te_f1:.3f}/AP {te_ap if te_ap is not None else 'NA'}", logger)

    # Fine-Tune
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
        te_loss, te_f1, te_ap = run_epoch(vdl, False)
        improved = te_f1 > best_f1 + 1e-4
        if improved:
            best_f1 = te_f1; torch.save(m.state_dict(), best_path); no_imp = 0
        else:
            no_imp += 1
        log_print(f"[FT {ep}/{EPOCHS_FT}] "
                  f"train {tr_loss:.3f}/F1 {tr_f1:.3f}/AP {tr_ap if tr_ap is not None else 'NA'} | "
                  f"test {te_loss:.3f}/F1 {te_f1:.3f}/AP {te_ap if te_ap is not None else 'NA'} "
                  f"{'(*)' if improved else ''}", logger)
        if no_imp >= patience:
            log_print("Early stop.", logger); break

    # Best 모델로 최종 Test 평가/저장
    m.load_state_dict(safe_load(best_path, map_location=device))
    te_loss, te_f1, te_ap, y_true, y_pred, prob = evaluate(vdl, m, device)

    report_txt = classification_report(y_true, y_pred, labels=LABELS_IDX, target_names=CLASSES, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS_IDX)

    log_print("\n=== FINAL TEST REPORT ===", logger)
    log_print(report_txt, logger)
    log_print(str(cm), logger)

    if SAVE_REPORT:
        with open(OUTDIR / f"final_report_seed{seed}.txt", "w") as f:
            f.write("=== FINAL TEST REPORT ===\n")
            f.write(report_txt + "\n")
            f.write(str(cm) + "\n")

    if SAVE_PREDICTIONS:
        te_paths = te_df["path"].tolist()
        pred_df = pd.DataFrame({
            "path": te_paths,
            "y_true": y_true,
            "y_pred": y_pred,
            "prob_nonbreathing": prob[:,0],
            "prob_breathing": prob[:,1],
            "seed": seed
        })
        pred_df.to_csv(OUTDIR / f"predictions_seed{seed}.csv", index=False)

    if logger is not None:
        logger.close()

    return best_f1

def main():
    df = pd.read_csv(MANIFEST)
    assert {"path","label"}.issubset(df.columns)
    df = df[df.label.isin(CLASSES)].copy()

    print("=== FULL DATASET STAT ===")
    print_split_stats("All", df, logger=None)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device} (AMP {'ON' if (USE_AMP and device=='cuda') else 'OFF'})")

    bests = []
    for rep in range(N_REPEATS):
        seed = SEED_BASE + rep
        print(f"\n=========== REPEAT {rep+1}/{N_REPEATS} (seed={seed}) ===========")
        bests.append(train_once(df, seed, device))

    if SAVE_SUMMARY:
        summary_path = OUTDIR / "summary_results.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["seed", "best_macroF1"])
            for rep, b in enumerate(bests):
                writer.writerow([SEED_BASE+rep, f"{b:.6f}"])
            writer.writerow(["mean", f"{np.mean(bests):.6f}"])
            writer.writerow(["std",  f"{np.std(bests):.6f}"])

    if N_REPEATS > 1:
        print("\n==== Summary over repeats ====")
        print("best macro-F1 per repeat:", [f"{b:.3f}" for b in bests])
        print(f"mean={np.mean(bests):.3f}  std={np.std(bests):.3f}")

if __name__ == "__main__":
    main()
