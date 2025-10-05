#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, random, numpy as np, pandas as pd
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, recall_score, confusion_matrix

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ----- seeds -----
np.random.seed(42); random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----- paths -----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = '//home//un_wang//my_stethoscope_project'
FEATURES_DIR = os.path.join(PROJECT_DIR, "features")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "Focal")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)

CSV_PATH = os.path.join(FEATURES_DIR, "opera_features.csv")
print("[RUN] Step4 Focal Loss"); print(f"[PATH] {CSV_PATH}")
df = pd.read_csv(CSV_PATH).query("label != 'unknown'").reset_index(drop=True)

def _pid(x): return str(x).split("_")[0]
df["patient_id"] = df["filename"].apply(_pid)

drop_cols = ["filename","label","extraction_success","patient_id"]
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
print(f"[DEVICE] {device}")

# ----- dataset & model -----
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
        super().__init__(); self.fc = nn.Linear(in_dim, n_classes)
    def forward(self, x): return self.fc(x)

class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__(); self.alpha = alpha; self.gamma = gamma; self.reduction = reduction
    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce)                      # prob of true class
        at = self.alpha[target]                  # per-sample alpha
        loss = at * (1 - pt) ** self.gamma * ce  # focal
        if self.reduction == "mean": return loss.mean()
        if self.reduction == "sum": return loss.sum()
        return loss

def build_loader(ds, batch, sampler=None, shuffle=False):
    return DataLoader(
        ds, batch_size=batch, sampler=sampler,
        shuffle=(sampler is None and shuffle),
        drop_last=False, num_workers=0, pin_memory=(device=="cuda")
    )

def train_one_fold(X_tr, y_tr, X_te, y_te, files_te, pids_te,
                   epochs=30, batch=64, lr=1e-3, gamma=2.0):
    n_class = len(class_names)

    # α = 1/freq (train fold 기준) 정규화
    cnt = Counter(y_tr.tolist())
    alpha = np.array([1.0 / max(cnt.get(i,1),1) for i in range(n_class)], dtype=np.float32)
    alpha = alpha * (n_class / alpha.sum())
    alpha_t = torch.tensor(alpha, dtype=torch.float32, device=device)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)

    ds_tr = NumpyDataset(X_tr, y_tr)
    ds_te = NumpyDataset(X_te, y_te, files_te, pids_te)

    # WeightedRandomSampler (train only)
    weights = np.array([1.0 / max(cnt[c],1) for c in y_tr], dtype=np.float32)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    dl_tr = build_loader(ds_tr, batch, sampler=sampler, shuffle=True)
    dl_te = build_loader(ds_te, batch, sampler=None, shuffle=False)

    model = TorchLogReg(X_tr.shape[1], n_class).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = FocalLoss(alpha=alpha_t, gamma=gamma)

    model.train()
    for ep in range(1, epochs+1):
        tot=0.0
        for xb, yb, _, _ in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = criterion(model(xb), yb)
            loss.backward(); opt.step()
            tot += loss.item()*xb.size(0)
        if ep==1 or ep%5==0:
            print(f"[EP {ep:02d}] loss={tot/len(ds_tr):.4f}")

    # evaluate
    model.eval(); probs=[]; trues=[]; preds=[]
    with torch.no_grad():
        for xb, yb, _, _ in dl_te:
            logits = model(xb.to(device))
            p = torch.softmax(logits, dim=1).cpu().numpy()
            probs.append(p); trues.append(yb.numpy()); preds.append(p.argmax(axis=1))
    return scaler, model, np.concatenate(trues), np.concatenate(preds), np.vstack(probs)

# ----- outer GroupKFold -----
gkf = GroupKFold(n_splits=5)
ALL_T, ALL_P, ALL_PR = [], [], []
ALL_FILE, ALL_PID, ALL_FOLD = [], [], []
fold = 0

for tr, te in gkf.split(X_all, y_all, groups_all):
    fold += 1
    print(f"\n[FOLD {fold}] train={len(tr)} test={len(te)}")
    X_tr, y_tr = X_all[tr], y_all[tr]
    X_te, y_te = X_all[te], y_all[te]
    files_te, pids_te = files_all[te], groups_all[te]

    scaler, model, T, P, PR = train_one_fold(
        X_tr, y_tr, X_te, y_te, files_te, pids_te,
        epochs=30, batch=64, lr=1e-3, gamma=0.5
    )

    ALL_T.append(T); ALL_P.append(P); ALL_PR.append(PR)
    ALL_FILE.extend(files_te); ALL_PID.extend(pids_te); ALL_FOLD.extend([fold]*len(te))

# ----- aggregate -----
T = np.concatenate(ALL_T, axis=0)
P = np.concatenate(ALL_P, axis=0)
PR = np.vstack(ALL_PR)

print("\n--- Aggregate Classification Report ---")
print(classification_report([class_names[i] for i in T],
                            [class_names[i] for i in P],
                            labels=list(class_names), zero_division=0))
macro = recall_score(T, P, average="macro", zero_division=0)
print(f"[MACRO RECALL] {macro:.3f}")

cm = confusion_matrix(T, P, labels=np.arange(len(class_names)))
fig, ax = plt.subplots(figsize=(5,4))
im = ax.imshow(cm, interpolation="nearest"); ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
       xticklabels=list(class_names), yticklabels=list(class_names),
       xlabel="Predicted", ylabel="True",
       title="TorchLogReg + FocalLoss (Aggregate)")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
for (i,j), v in np.ndenumerate(cm): ax.text(j, i, str(v), ha='center', va='center')
fig.tight_layout(); fig.savefig(os.path.join(FIGURES_DIR, "TorchLogReg_focal_cm_aggregate.png"), dpi=150); plt.close(fig)

# save predictions
out = {"filename": ALL_FILE, "patient_id": ALL_PID, "fold": ALL_FOLD,
       "y_true": [class_names[i] for i in T],
       "y_pred": [class_names[i] for i in P]}
pred_df = pd.DataFrame(out)
for i, c in enumerate(class_names): pred_df[f"prob_{c}"] = PR[:, i]
pred_df.to_csv(os.path.join(RESULTS_DIR, "TorchLogReg_focal_cv_predictions.csv"), index=False)

# summary (for quick logging)
rpt = classification_report([class_names[i] for i in T],
                            [class_names[i] for i in P],
                            labels=list(class_names), output_dict=True, zero_division=0)
pd.DataFrame([{
    "model": "TorchLogReg+Focal(gamma=2, alpha~1/freq)",
    "accuracy": rpt["accuracy"],
    "precision_macro": rpt["macro avg"]["precision"],
    "recall_macro": rpt["macro avg"]["recall"],
    "f1_macro": rpt["macro avg"]["f1-score"]
}]).set_index("model").to_csv(os.path.join(RESULTS_DIR, "classifier_groupcv_results.csv"))

print("[DONE]")
