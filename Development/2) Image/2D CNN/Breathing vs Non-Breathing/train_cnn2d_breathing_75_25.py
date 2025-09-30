#!/usr/bin/env python3
# train_2dcnn_from_tiles.py
# 2D CNN(EfficientNet-B0) 전이학습: 타일(스케일로그램/스펙트로그램) 이미지로 75/25 stratified 학습

import os, random, json
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, average_precision_score, classification_report, confusion_matrix

# ====================== 경로/설정 ======================
# 스케일로그램 manifest
MANIFEST = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Development/9.2) Reference/BreathingTiles_scalo_paper/manifest.csv")
# 스펙트로그램 manifest 로 바꾸려면 위 경로만 교체:
# MANIFEST = Path("/Users/.../BreathingTiles_mel_paper/manifest.csv")

OUTDIR   = MANIFEST.parent / "models_efficientnet_b0_2dcnn_75_25"
OUTDIR.mkdir(parents=True, exist_ok=True)

CLASSES      = ["NonBreathing", "Breathing"]
LABEL_TO_IDX = {c:i for i,c in enumerate(CLASSES)}
IDX_TO_LABEL = {i:c for c,i in LABEL_TO_IDX.items()}

IMG_SIZE     = 224
BS           = 32
EPOCHS_LP    = 5       # Linear Probe
EPOCHS_FT    = 15      # Fine-tune
LR_HEAD      = 1e-3
LR_BACKBONE  = 1e-4
WEIGHT_DECAY = 1e-4
USE_AMP      = True
LOSS_TYPE    = "focal"  # "focal" or "ce"

N_REPEATS    = 5        # 75/25 반복 횟수
SEED_BASE    = 42

# ====================== 유틸 ======================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def counts_str(df):
    vc = df.label.value_counts().reindex(CLASSES, fill_value=0).to_dict()
    return " | [" + ", ".join(f"{k}: {vc[k]}" for k in CLASSES) + "]"

class ImgDS(Dataset):
    def __init__(self, df, train=True):
        self.df = df.reset_index(drop=True)
        if train:
            self.tx = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomApply([transforms.RandomAffine(degrees=5, translate=(0.02,0.02), scale=(0.98,1.02))], p=0.5),
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
        x = Image.open(row.path).convert("RGB")
        x = self.tx(x)
        y = LABEL_TO_IDX[row.label]
        return x, y, row.path

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)
        return ((1-pt)**self.gamma * ce).mean()

@torch.no_grad()
def evaluate(dl, model, device):
    model.eval()
    ys, ps, paths = [], [], []
    tot, n = 0.0, 0
    for x,y,pth in dl:
        x,y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="none")
        tot += loss.mean().item() * x.size(0); n += x.size(0)
        ys.append(y.cpu().numpy())
        ps.append(logits.softmax(1).cpu().numpy())
        paths += list(pth)
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    pred = p.argmax(1)
    f1 = f1_score(y, pred, average="macro", zero_division=0)
    try:
        ap = average_precision_score(y, p[:, LABEL_TO_IDX["Breathing"]])
    except:
        ap = None
    return tot/n, f1, ap, y, pred, p, paths

def save_predictions_csv(paths, y_true, y_pred, prob, out_csv):
    rows = []
    for i, path in enumerate(paths):
        rows.append({
            "path": path,
            "y_true": int(y_true[i]),
            "y_pred": int(y_pred[i]),
            "p_nonbreathing": float(prob[i,0]),
            "p_breathing": float(prob[i,1])
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False)

def safe_load(pth, map_location=None):
    # PyTorch 2.4 권고 사항 대응
    try: return torch.load(pth, map_location=map_location, weights_only=True)
    except TypeError: return torch.load(pth, map_location=map_location)

# ====================== 학습 루프 ======================
def train_once(df, seed, device):
    set_seed(seed)

    # 75/25 stratified split (세그먼트 기준)
    y = df["label"].map(LABEL_TO_IDX).values
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    tr_idx, te_idx = next(sss.split(np.zeros(len(df)), y))
    tr_df, te_df = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()

    print("\n=== 75/25 Split ===")
    print(f"Train(75%) -> n={len(tr_df)}" + counts_str(tr_df))
    print(f"Test (25%) -> n={len(te_df)}" + counts_str(te_df))

    # DataLoader (macOS 멀티프로세싱 이슈 피하려고 num_workers=0)
    num_workers = 0
    pin_mem = (device == "cuda")
    tdl = DataLoader(ImgDS(tr_df, True),  batch_size=BS, shuffle=True,  num_workers=num_workers, pin_memory=pin_mem)
    vdl = DataLoader(ImgDS(te_df, False), batch_size=BS, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)

    # 클래스 가중치 (focal alpha)
    counts = tr_df.label.value_counts().reindex(CLASSES, fill_value=0).values.astype(float)
    alpha = counts.sum() / np.maximum(counts, 1e-6)
    alpha = alpha * (len(alpha) / alpha.sum())
    alpha = torch.tensor(alpha, dtype=torch.float32, device=device)

    # 모델
    m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_f = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_f, len(CLASSES))
    m = m.to(device)

    use_amp = USE_AMP and device=="cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # 손실
    if LOSS_TYPE.lower() == "focal":
        criterion = FocalLoss(alpha=alpha, gamma=2.0)
        def loss_fn(logits, target): return criterion(logits, target)
    else:
        def loss_fn(logits, target): return F.cross_entropy(logits, target)

    def run_epoch(dl, train=True, opt=None):
        m.train(train)
        ys, ps, tot, n = [], [], 0.0, 0
        for x,y,_ in dl:
            x,y = x.to(device), y.to(device)
            if train: opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = m(x); loss = loss_fn(logits, y)
                if train:
                    scaler.scale(loss).backward()
                    scaler.step(opt); scaler.update()
            else:
                logits = m(x); loss = loss_fn(logits, y)
                if train:
                    loss.backward(); opt.step()
            tot += loss.item() * x.size(0); n += x.size(0)
            ys.append(y.detach().cpu().numpy())
            ps.append(logits.detach().softmax(1).cpu().numpy())
        y_true = np.concatenate(ys); p = np.concatenate(ps)
        pred = p.argmax(1)
        f1 = f1_score(y_true, pred, average="macro", zero_division=0)
        try: ap = average_precision_score(y_true, p[:, LABEL_TO_IDX["Breathing"]])
        except: ap = None
        return tot/n, f1, ap

    best_f1 = -1.0
    rep_dir  = OUTDIR / f"seed_{seed}"
    rep_dir.mkdir(parents=True, exist_ok=True)
    best_path = rep_dir / f"best_model_seed{seed}.pt"

    # = Stage 1: Linear Probe =
    for p in m.features.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, m.parameters()), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)

    for ep in range(1, EPOCHS_LP+1):
        tr_loss, tr_f1, tr_ap = run_epoch(tdl, True, opt)
        te_loss, te_f1, te_ap = run_epoch(vdl, False, None)
        if te_f1 > best_f1:
            best_f1 = te_f1; torch.save(m.state_dict(), best_path)
        print(f"[LP {ep}/{EPOCHS_LP}] train {tr_loss:.3f}/F1 {tr_f1:.3f}/AP {tr_ap if tr_ap is not None else 'NA'} | "
              f"test {te_loss:.3f}/F1 {te_f1:.3f}/AP {te_ap if te_ap is not None else 'NA'}")

    # = Stage 2: Fine-tune (마지막 두 블록만 풀기) =
    m.load_state_dict(safe_load(best_path, map_location=device))
    for name, p in m.features.named_parameters():
        if any(k in name for k in ["6","7"]):
            p.requires_grad_(True)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, m.parameters()), lr=LR_BACKBONE, weight_decay=WEIGHT_DECAY)

    patience, no_imp = 6, 0
    history = []
    for ep in range(1, EPOCHS_FT+1):
        tr_loss, tr_f1, tr_ap = run_epoch(tdl, True, opt)
        te_loss, te_f1, te_ap = run_epoch(vdl, False, None)
        improved = te_f1 > best_f1 + 1e-4
        if improved:
            best_f1 = te_f1; torch.save(m.state_dict(), best_path); no_imp = 0
        else:
            no_imp += 1
        print(f"[FT {ep}/{EPOCHS_FT}] train {tr_loss:.3f}/F1 {tr_f1:.3f}/AP {tr_ap if tr_ap is not None else 'NA'} | "
              f"test {te_loss:.3f}/F1 {te_f1:.3f}/AP {te_ap if te_ap is not None else 'NA'} {'(*)' if improved else ''}")
        history.append({
            "epoch": ep, "train_loss": tr_loss, "train_f1": tr_f1, "train_ap": tr_ap,
            "test_loss": te_loss, "test_f1": te_f1, "test_ap": te_ap, "improved": improved
        })
        if no_imp >= patience:
            print("Early stop."); break

    # = Best 모델 로드 후 최종 Test 리포트 & 저장 =
    m.load_state_dict(safe_load(best_path, map_location=device))
    _, _, _, y_true, y_pred, prob, paths = evaluate(vdl, m, device)

    report_txt = classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])

    # 저장물
    (rep_dir / "history.json").write_text(json.dumps(history, indent=2))
    (rep_dir / "report.txt").write_text(report_txt)
    np.savetxt(rep_dir / "confusion_matrix.txt", cm, fmt="%d")
    save_predictions_csv(paths, y_true, y_pred, prob, rep_dir / "test_predictions.csv")

    # 요약 반환 (평균 낼 때 사용)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    ap = average_precision_score(y_true, prob[:, LABEL_TO_IDX["Breathing"]])
    return dict(seed=seed, test_macro_f1=float(macro_f1), test_ap=float(ap))

def main():
    df = pd.read_csv(MANIFEST)
    assert {"path","label"}.issubset(df.columns)
    df = df[df.label.isin(CLASSES)].copy()

    print("=== FULL DATASET STAT ===")
    print(f"All -> n={len(df)}" + counts_str(df))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device} (AMP {'ON' if (USE_AMP and device=='cuda') else 'OFF'})")

    results = []
    for rep in range(N_REPEATS):
        seed = SEED_BASE + rep
        print(f"\n=========== REPEAT {rep+1}/{N_REPEATS} (seed={seed}) ===========")
        res = train_once(df, seed, device)
        results.append(res)

    # 반복 요약 저장
    summ = pd.DataFrame(results)
    summ_path = OUTDIR / "summary_over_repeats.csv"
    summ.to_csv(summ_path, index=False)
    print("\n==== Summary over repeats ====")
    print("best macro-F1 per repeat:", [f"{x['test_macro_f1']:.3f}" for x in results])
    print(f"mean={summ.test_macro_f1.mean():.3f}  std={summ.test_macro_f1.std():.3f}")
    print("Saved:", summ_path)

if __name__ == "__main__":
    main()
