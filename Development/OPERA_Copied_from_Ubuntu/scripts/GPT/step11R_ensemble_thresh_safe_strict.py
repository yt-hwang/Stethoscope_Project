#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, random, numpy as np, pandas as pd
from collections import Counter
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, recall_score, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

PROJECT_DIR = '//home//un_wang//my_stethoscope_project'
FEATURES_DIR = os.path.join(PROJECT_DIR, 'features')
RESULTS_DIR  = os.path.join(PROJECT_DIR, 'results', 'Ensemble_ThreshSafe_STRICT')
FIGURES_DIR  = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)

np.random.seed(42); random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------- Data ----------
CSV_PATH = os.path.join(FEATURES_DIR, 'opera_features.csv')
print('[RUN] Step11R Ensemble + Safe Threshold Tuning (STRICT)')
print(f'[PATH] {CSV_PATH}')
df = pd.read_csv(CSV_PATH).query("label != 'unknown'").reset_index(drop=True)
def _pid(x): return str(x).split('_')[0]
df['patient_id'] = df['filename'].apply(_pid)

drop_cols = ['filename','label','extraction_success','patient_id']
feat_cols = [c for c in df.columns if c not in drop_cols]
X_all = df[feat_cols].values.astype(np.float32)
y_lbl = df['label'].values
groups_all = df['patient_id'].values
files_all = df['filename'].values

class_names = np.sort(np.unique(y_lbl))
c2i = {c:i for i,c in enumerate(class_names)}
y_all = np.array([c2i[c] for c in y_lbl], dtype=np.int64)

IDX = {c:i for i,c in enumerate(class_names)}
IDX_NB = IDX['Non-breathing']
IDX_H  = IDX['Healthy']
IDX_W  = IDX['Wheezing']
IDX_R  = IDX['Rhonchi']
print(f"[INFO] n={len(df)}, classes={list(class_names)}, dims={len(feat_cols)}")
print(df['label'].value_counts().to_string())
print(f'[DEVICE] {device}')

# ---------- Torch datasets/models ----------
class NumpyDataset(Dataset):
    def __init__(self, X, y, filenames=None, pids=None):
        self.X, self.y, self.filenames, self.pids = X, y, filenames, pids
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        fn = self.filenames[i] if self.filenames is not None else ''
        pid = self.pids[i] if self.pids is not None else ''
        return torch.from_numpy(self.X[i]), int(self.y[i]), fn, pid

def build_loader(ds, batch, sampler=None, shuffle=False):
    return DataLoader(ds, batch_size=batch, sampler=sampler,
                      shuffle=(sampler is None and shuffle),
                      drop_last=False, num_workers=0, pin_memory=(device=='cuda'))

class TorchLogReg(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__(); self.fc = nn.Linear(in_dim, n_classes)
    def forward(self, x): return self.fc(x)

class MLP(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=256, p_drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(hidden, n_classes)
        )
    def forward(self, x): return self.net(x)

class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 1.0, reduction: str = 'mean'):
        super().__init__(); self.alpha=alpha; self.gamma=gamma; self.reduction=reduction
    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(logits, target, reduction='none')
        pt = torch.exp(-ce); at = self.alpha[target]
        return (at * (1 - pt) ** self.gamma * ce).mean()

def make_alpha(y, n_class):
    cnt = Counter(y.tolist())
    a = np.array([1.0/max(cnt.get(i,1),1) for i in range(n_class)], dtype=np.float32)
    return (a * (n_class/a.sum())).astype(np.float32)

def train_model(model_type, X_tr, y_tr, X_te, batch=64, epochs=50, lr=1e-3, wd=1e-4):
    n_class = len(class_names)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)

    alpha = torch.tensor(make_alpha(torch.tensor(y_tr), n_class), device=device)
    ds = NumpyDataset(X_tr, y_tr)
    w = np.array([1.0/max(Counter(y_tr.tolist())[c],1) for c in y_tr], dtype=np.float32)
    sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
    dl = build_loader(ds, batch, sampler=sampler, shuffle=True)

    if model_type=='lr':
        model = TorchLogReg(X_tr.shape[1], n_class).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        model = MLP(X_tr.shape[1], n_class, hidden=256, p_drop=0.3).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    crit = FocalLoss(alpha=alpha, gamma=1.0)

    model.train()
    for ep in range(1, epochs+1):
        tot=0.0
        for xb, yb, _, _ in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(model(xb), yb)
            loss.backward(); opt.step(); tot += loss.item()*xb.size(0)
        if ep==1 or ep%5==0:
            print(f"[{model_type.upper()} EP {ep:02d}] loss={tot/len(ds):.4f}")

    def infer(X):
        d = build_loader(NumpyDataset(X, np.zeros(len(X))), batch, sampler=None, shuffle=False)
        model.eval(); P=[]
        with torch.no_grad():
            for xb, _, _, _ in d:
                P.append(torch.softmax(model(xb.to(device)), dim=1).cpu().numpy())
        return np.vstack(P), scaler, model
    return infer, scaler, model

def ensemble_probs(P_lr, P_mlp, w_lr=0.5, w_mlp=0.5):
    return w_lr*P_lr + w_mlp*P_mlp

def apply_thresholds_with_NB_gate(P, tauH, tauW, tauR, nb_gate=0.45):
    # 1) NB hard gate
    y_pred = np.full(P.shape[0], -1, dtype=np.int64)
    nb_mask = P[:, IDX_NB] >= nb_gate
    y_pred[nb_mask] = IDX_NB
    # 2) others with per-class tau
    if (~nb_mask).any():
        Q = P[~nb_mask].copy()
        tau = np.full(Q.shape[1], 0.5, dtype=np.float32)
        tau[IDX_H] = tauH; tau[IDX_W] = tauW; tau[IDX_R] = tauR
        cls = (Q / tau.reshape(1,-1)).argmax(axis=1)
        y_pred[~nb_mask] = cls
    return y_pred

# ---------- Outer CV with STRICT inner-val tuning ----------
gkf = GroupKFold(n_splits=5)
ALL_T, ALL_P, ALL_PR = [], [], []
ALL_FILE, ALL_PID, ALL_FOLD = [], [], []
TAUS = []
fold = 0

for tr, te in gkf.split(X_all, y_all, groups_all):
    fold += 1
    print(f"\n[FOLD {fold}] train={len(tr)} test={len(te)}")
    X_tr, y_tr = X_all[tr], y_all[tr]
    X_te, y_te = X_all[te], y_all[te]
    files_te, pids_te = files_all[te], groups_all[te]

    # inner val split
    idx = np.arange(len(X_tr))
    tr_idx, va_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y_tr)
    X_trn, y_trn = X_tr[tr_idx], y_tr[tr_idx]
    X_val, y_val = X_tr[va_idx], y_tr[va_idx]

    # train two models on trn
    infer_lr, _, _  = train_model('lr',  X_trn, y_trn, X_val, epochs=30, lr=1e-3, wd=0.0)
    infer_mlp, _, _ = train_model('mlp', X_trn, y_trn, X_val, epochs=50, lr=1e-3, wd=1e-4)
    P_lr_val, _, _  = infer_lr(X_val)
    P_mlp_val, _, _ = infer_mlp(X_val)
    P_val = ensemble_probs(P_lr_val, P_mlp_val, 0.5, 0.5)

    # baseline (tau=0.5, no gate)
    base_pred = P_val.argmax(axis=1)
    base_nb = recall_score(y_val, base_pred, labels=[IDX_NB], average=None, zero_division=0)[0]
    # strict guard: 최소 기준 0.45 적용
    nb_guard = max(base_nb, 0.45)

    grid = np.arange(0.30, 0.71, 0.05)
    best = (-1.0, 0.5, 0.5, 0.5)
    for tH in grid:
        for tW in grid:
            for tR in grid:
                yhat = apply_thresholds_with_NB_gate(P_val, tH, tW, tR, nb_gate=0.45)
                macro = recall_score(y_val, yhat, average='macro', zero_division=0)
                nb_rec = recall_score(y_val, yhat, labels=[IDX_NB], average=None, zero_division=0)[0]
                if nb_rec + 1e-9 < nb_guard:
                    continue
                if macro > best[0]:
                    best = (macro, tH, tW, tR)

    print(f"[TUNE] val macro*={best[0]:.3f} with tau_H={best[1]:.2f} tau_W={best[2]:.2f} tau_R={best[3]:.2f} | NB_guard≥{nb_guard:.3f}")
    TAUS.append((best[1], best[2], best[3]))

    # retrain both on FULL train (tr) then test
    infer_lr_full, _, _  = train_model('lr',  X_tr, y_tr, X_te, epochs=30, lr=1e-3, wd=0.0)
    infer_mlp_full, _, _ = train_model('mlp', X_tr, y_tr, X_te, epochs=50, lr=1e-3, wd=1e-4)
    P_lr_te, _, _  = infer_lr_full(X_te)
    P_mlp_te, _, _ = infer_mlp_full(X_te)
    P_te = ensemble_probs(P_lr_te, P_mlp_te, 0.5, 0.5)

    y_pred = apply_thresholds_with_NB_gate(P_te, best[1], best[2], best[3], nb_gate=0.45)

    ALL_T.append(y_te); ALL_P.append(y_pred); ALL_PR.append(P_te)
    ALL_FILE.extend(files_te); ALL_PID.extend(pids_te); ALL_FOLD.extend([fold]*len(te))

# ---------- Aggregate ----------
T = np.concatenate(ALL_T, axis=0)
P = np.concatenate(ALL_P, axis=0)
PR = np.vstack(ALL_PR)

print("\n--- Aggregate Classification Report (Ensemble + STRICT τ) ---")
print(classification_report([class_names[i] for i in T],
                            [class_names[i] for i in P],
                            labels=list(class_names), zero_division=0))
macro = recall_score(T, P, average='macro', zero_division=0)
print(f"[MACRO RECALL] {macro:.3f}")

cm = confusion_matrix(T, P, labels=np.arange(len(class_names)))
fig, ax = plt.subplots(figsize=(5,4))
im = ax.imshow(cm, interpolation='nearest'); ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
       xticklabels=list(class_names), yticklabels=list(class_names),
       xlabel='Predicted', ylabel='True',
       title='Ensemble + STRICT Threshold (Aggregate)')
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
for (i,j), v in np.ndenumerate(cm): ax.text(j, i, str(v), ha='center', va='center')
fig.tight_layout(); fig.savefig(os.path.join(FIGURES_DIR, 'ensemble_threshsafe_strict_cm.png'), dpi=150); plt.close(fig)

# save predictions & summary
pred = {"filename": ALL_FILE, "patient_id": ALL_PID, "fold": ALL_FOLD,
        "y_true": [class_names[i] for i in T], "y_pred": [class_names[i] for i in P]}
pred_df = pd.DataFrame(pred)
for i, c in enumerate(class_names): pred_df[f"prob_{c}"] = PR[:, i]
pred_df.to_csv(os.path.join(RESULTS_DIR, 'cv_predictions.csv'), index=False)

pd.DataFrame(TAUS, columns=['tau_Healthy','tau_Wheezing','tau_Rhonchi']).to_csv(
    os.path.join(RESULTS_DIR, 'taus_per_fold.csv'), index=False)

rpt = classification_report([class_names[i] for i in T],
                            [class_names[i] for i in P],
                            labels=list(class_names), output_dict=True, zero_division=0)
pd.DataFrame([{
    'model': 'Ensemble(LogReg+MLP)+STRICT_Thresh(H,W,R; NB-gate≥0.45, NB-guard≥max(baseline,0.45))',
    'accuracy': rpt['accuracy'],
    'precision_macro': rpt['macro avg']['precision'],
    'recall_macro': rpt['macro avg']['recall'],
    'f1_macro': rpt['macro avg']['f1-score']
}]).set_index('model').to_csv(os.path.join(RESULTS_DIR, 'classifier_groupcv_results.csv'))

print('[DONE]')
