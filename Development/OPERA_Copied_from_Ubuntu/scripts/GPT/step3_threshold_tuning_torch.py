#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, numpy as np, pandas as pd
from collections import Counter
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, recall_score, confusion_matrix

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import random
np.random.seed(42)
random.seed(42)
import torch
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------ paths ------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = '//home//un_wang//my_stethoscope_project'
FEATURES_DIR = os.path.join(PROJECT_DIR, "features")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "Threshold")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
MODELS_DIR = os.path.join(PROJECT_DIR, "models", "Threshold")

os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)

CSV_PATH = os.path.join(FEATURES_DIR, "opera_features.csv")
print(f"[RUN] Step3 Threshold Tuning"); print(f"[PATH] {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
df = df[df["label"] != "unknown"].reset_index(drop=True)

def _pid(x): return str(x).split("_")[0]
df["patient_id"] = df["filename"].apply(_pid)

drop_cols = ["filename", "label", "extraction_success", "patient_id"]
feat_cols = [c for c in df.columns if c not in drop_cols]

X_all = df[feat_cols].values.astype(np.float32)
y_lbl = df["label"].values
groups_all = df["patient_id"].values
files_all = df["filename"].values

class_names = np.sort(np.unique(y_lbl))
c2i = {c:i for i,c in enumerate(class_names)}
y_all = np.array([c2i[c] for c in y_lbl], dtype=np.int64)

print(f"[INFO] n={len(df)}, classes={list(class_names)}, dims={len(feat_cols)}")
print(df["label"].value_counts().to_string())

device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"   # ← 임시 강제

print(f"[DEVICE] {device}")

# ------------------ dataset/model ------------------
class NumpyDataset(Dataset):
    def __init__(self, X, y, filenames=None, pids=None):
        self.X, self.y, self.filenames, self.pids = X, y, filenames, pids
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        fn = self.filenames[i] if self.filenames is not None else ""
        pid = self.pids[i] if self.pids is not None else ""
        return torch.from_numpy(self.X[i]), int(self.y[i]), fn, pid

class TorchLogReg(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__(); self.linear = nn.Linear(in_dim, n_classes)
    def forward(self, x): return self.linear(x)

def build_loader(ds, batch, sampler=None, shuffle=False):
    return DataLoader(
        ds, batch_size=batch, sampler=sampler,
        shuffle=(sampler is None and shuffle),
        drop_last=False, num_workers=0, pin_memory=(device=="cuda")
    )

# ------------------ train helpers ------------------
def fit_return_probs(X_tr, y_tr, X_va, y_va, epochs=30, batch=64, lr=1e-3, use_sampler=True, class_weights=None):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_va = scaler.transform(X_va).astype(np.float32)

    ds_tr = NumpyDataset(X_tr, y_tr)
    ds_va = NumpyDataset(X_va, y_va)

    sampler = None
    if use_sampler:
        cnt = Counter(y_tr.tolist())
        weights = np.array([1.0 / max(cnt[c],1) for c in y_tr], dtype=np.float32)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    dl_tr = build_loader(ds_tr, batch, sampler=sampler, shuffle=True)
    dl_va = build_loader(ds_va, batch, sampler=None, shuffle=False)

    n_class = len(class_names)
    model = TorchLogReg(X_tr.shape[1], n_class).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    if class_weights is None:
        gcnt = Counter(y_all.tolist())
        cw = np.array([1.0/max(gcnt.get(i,1),1) for i in range(n_class)], dtype=np.float32)
        cw = cw * (n_class / cw.sum())
        class_weights = torch.tensor(cw, dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=class_weights.to(device))

    model.train()
    for ep in range(1, epochs+1):
        tot=0.0
        for xb, yb, _, _ in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); logits = model(xb)
            loss = crit(logits, yb); loss.backward(); opt.step()
            tot += loss.item()*xb.size(0)
        if ep==1 or ep%5==0:
            print(f"[EP {ep:02d}] loss={tot/len(ds_tr):.4f}")

    # validation probs
    model.eval(); probs=[]; y_true=[]
    with torch.no_grad():
        for xb, yb, _, _ in dl_va:
            p = torch.softmax(model(xb.to(device)), dim=1).cpu().numpy()
            probs.append(p); y_true.append(yb.numpy())
    return model, scaler, np.vstack(probs), np.concatenate(y_true, axis=0)

def train_full(X_tr, y_tr, epochs=30, batch=64, lr=1e-3):
    sc = StandardScaler()
    Xs = sc.fit_transform(X_tr).astype(np.float32)
    ds = NumpyDataset(Xs, y_tr)
    dl = build_loader(ds, batch, sampler=None, shuffle=True)
    n_class = len(class_names)
    m = TorchLogReg(Xs.shape[1], n_class).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    cnt = Counter(y_tr.tolist())
    cw = np.array([1.0/max(cnt.get(i,1),1) for i in range(n_class)], dtype=np.float32)
    cw = cw * (n_class / cw.sum())
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(device))
    m.train()
    for ep in range(1, epochs+1):
        tot=0.0
        for xb, yb, _, _ in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(m(xb), yb); loss.backward(); opt.step()
            tot += loss.item()*xb.size(0)
        if ep==1 or ep%5==0:
            print(f"[FULL EP {ep:02d}] loss={tot/len(ds):.4f}")
    return m, sc

def apply_thresholds(probs: np.ndarray, tau: np.ndarray) -> np.ndarray:
    return (probs / tau.reshape(1,-1)).argmax(axis=1)

# ------------------ threshold tuning (inner-CV) ------------------
def tune_thresholds_inner(
    X_tr, y_tr, groups_tr,
    grid: List[float],
    epochs=8, batch=64, lr=1e-3,
    max_loops=40,           # ← 반복 상한
    min_gain=1e-4           # ← 개선폭 임계값(매크로 리콜 증가가 이보다 작으면 중단)
    ):
    inner = GroupKFold(n_splits=3)
    P_list, Y_list = [], []

    # 1) inner-CV로 검증 확률 수집
    for i_tr, i_va in inner.split(X_tr, y_tr, groups_tr):
        _, _, P, Y = fit_return_probs(
            X_tr[i_tr], y_tr[i_tr],
            X_tr[i_va], y_tr[i_va],
            epochs=epochs, batch=batch, lr=lr, use_sampler=True
        )
        P_list.append(P); Y_list.append(Y)
    P = np.vstack(P_list)
    Y = np.concatenate(Y_list, axis=0)

    # 2) 좌표 강하 + 수렴 보장
    n_class = len(class_names)
    tau = np.full(n_class, 0.5, dtype=np.float32)

    # 현재 점수
    def macro_rec_with(t):
        yhat = apply_thresholds(P, t)
        return recall_score(Y, yhat, average="macro", zero_division=0)

    best_score = macro_rec_with(tau)
    print(f"[TUNE] init macro recall={best_score:.3f}, tau={tau}", flush=True)

    for loop in range(1, max_loops + 1):
        improved_any = False
        for c in range(n_class):
            best_tc = tau[c]
            best_local = best_score

            for t in grid:
                if abs(t - tau[c]) < 1e-12:
                    continue
                tau_try = tau.copy(); tau_try[c] = t
                sc = macro_rec_with(tau_try)
                if sc > best_local + 1e-12:   # 미세한 진동 방지
                    best_local = sc
                    best_tc = t

            if best_tc != tau[c]:
                tau[c] = best_tc
                best_score = best_local
                improved_any = True

        print(f"[TUNE] loop={loop:02d} macro={best_score:.4f} tau={tau}", flush=True)

        # 개선이 거의 없으면 중단
        if not improved_any:
            print("[TUNE] no coordinate improved → stop", flush=True)
            break
        # 직전 루프 대비 개선폭이 작으면 중단
        if loop >= 2 and (best_score - prev_score) < min_gain:
            print(f"[TUNE] gain < {min_gain} → stop", flush=True)
            break

        prev_score = best_score

    # 최종 리포트
    final_score = macro_rec_with(tau)
    print(f"[TUNE] final macro recall={final_score:.3f}, tau={tau}", flush=True)
    return tau

# ------------------ outer CV ------------------
outer = GroupKFold(n_splits=5)
grid = np.linspace(0.10, 0.90, 17)

ALL_T, ALL_P, ALL_PR = [], [], []
ALL_FILE, ALL_PID, ALL_FOLD = [], [], []

fold = 0
for tr, te in outer.split(X_all, y_all, groups_all):
    fold += 1
    print(f"\n[OUTER {fold}] train={len(tr)} test={len(te)}")

    X_tr, y_tr, g_tr = X_all[tr], y_all[tr], groups_all[tr]
    X_te, y_te = X_all[te], y_all[te]
    f_te, p_te = files_all[te], groups_all[te]

    tau = tune_thresholds_inner(X_tr, y_tr, g_tr, grid, epochs=15, batch=64, lr=1e-3)
    model, scaler = train_full(X_tr, y_tr, epochs=30, batch=64, lr=1e-3)

    X_te_s = scaler.transform(X_te).astype(np.float32)
    ds_te = NumpyDataset(X_te_s, y_te, f_te, p_te)
    dl_te = build_loader(ds_te, batch=128, sampler=None, shuffle=False)

    model.eval(); probs=[]; ytru=[]
    with torch.no_grad():
        for xb, yb, _, _ in dl_te:
            pr = torch.softmax(model(xb.to(device)), dim=1).cpu().numpy()
            probs.append(pr); ytru.append(yb.numpy())
    P = np.vstack(probs); T = np.concatenate(ytru, axis=0)
    yhat = apply_thresholds(P, tau)

    ALL_T.append(T); ALL_P.append(yhat); ALL_PR.append(P)
    ALL_FILE.extend(f_te); ALL_PID.extend(p_te); ALL_FOLD.extend([fold]*len(te))

# ------------------ aggregate report ------------------
T = np.concatenate(ALL_T, axis=0)
PRED = np.concatenate(ALL_P, axis=0)
PROB = np.vstack(ALL_PR)

print("\n--- Aggregate Classification Report ---")
print(classification_report([class_names[i] for i in T],
                            [class_names[i] for i in PRED],
                            labels=list(class_names), zero_division=0))
macro_rec = recall_score(T, PRED, average="macro", zero_division=0)
print(f"[MACRO RECALL] {macro_rec:.3f}")

cm = confusion_matrix(T, PRED, labels=np.arange(len(class_names)))
fig, ax = plt.subplots(figsize=(5,4))
im = ax.imshow(cm, interpolation="nearest"); ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
       xticklabels=list(class_names), yticklabels=list(class_names),
       xlabel="Predicted", ylabel="True", title="TorchLogReg + Threshold (Aggregate)")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
for (i,j), v in np.ndenumerate(cm): ax.text(j, i, str(v), ha='center', va='center')
fig.tight_layout(); fig.savefig(os.path.join(FIGURES_DIR, "TorchLogReg_threshold_cm_aggregate.png"), dpi=150); plt.close(fig)

# save predictions
out = {"filename": ALL_FILE, "patient_id": ALL_PID, "fold": ALL_FOLD,
       "y_true": [class_names[i] for i in T], "y_pred": [class_names[i] for i in PRED]}
pred_df = pd.DataFrame(out)
for i, c in enumerate(class_names): pred_df[f"prob_{c}"] = PROB[:, i]
pred_df.to_csv(os.path.join(RESULTS_DIR, "TorchLogReg_threshold_cv_predictions.csv"), index=False)

# summary
rpt = classification_report([class_names[i] for i in T],
                            [class_names[i] for i in PRED],
                            labels=list(class_names), output_dict=True, zero_division=0)
pd.DataFrame([{
    "model": "TorchLogReg+Threshold",
    "accuracy": rpt["accuracy"],
    "precision_macro": rpt["macro avg"]["precision"],
    "recall_macro": rpt["macro avg"]["recall"],
    "f1_macro": rpt["macro avg"]["f1-score"]
}]).set_index("model").to_csv(os.path.join(RESULTS_DIR, "classifier_groupcv_results.csv"))

print("[DONE]")
