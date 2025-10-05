#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, random, numpy as np, pandas as pd
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, recall_score, confusion_matrix

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ---------- fixed path & seeds ----------
PROJECT_DIR = '//home//un_wang//my_stethoscope_project'
FEATURES_DIR = os.path.join(PROJECT_DIR, 'features')
RESULTS_DIR  = os.path.join(PROJECT_DIR, 'results', 'MLP_Focal_ConsistentMixup_ThreshHW')
FIGURES_DIR  = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)

np.random.seed(42); random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------- data ----------
CSV_PATH = os.path.join(FEATURES_DIR, 'opera_features.csv')
print('[RUN] Step10 = 9R training + Threshold Tuning (H,W)')
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
print(f"[INFO] n={len(df)}, classes={list(class_names)}, dims={len(feat_cols)}")
print(df['label'].value_counts().to_string())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'[DEVICE] {device}')

IDX = {c:i for i,c in enumerate(class_names)}
IDX_HEALTHY  = int(np.where(class_names=='Healthy')[0][0])
IDX_WHEEZING = int(np.where(class_names=='Wheezing')[0][0])
CONSISTENT_TARGETS = set([IDX.get('Healthy', -1), IDX.get('Wheezing', -1), IDX.get('Rhonchi', -1)])

# ---------- dataset ----------
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

# ---------- model & loss ----------
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
        pt = torch.exp(-ce)
        at = self.alpha[target]
        loss = at * (1 - pt) ** self.gamma * ce
        if self.reduction == 'mean': return loss.mean()
        if self.reduction == 'sum': return loss.sum()
        return loss

# ---------- feature-space aug ----------
def gaussian_noise(x, sigma=0.01):
    if sigma <= 0: return x
    return x + torch.randn_like(x) * sigma

def mixup_batch_consistent(x, y, alpha=0.2, targets=None):
    if alpha <= 0: return x, y
    B = x.size(0)
    from collections import defaultdict as dd
    cls2idx = dd(list)
    for i, yi in enumerate(y.tolist()):
        cls2idx[yi].append(i)
    partner = torch.arange(B, device=x.device)
    for cls, idxs in cls2idx.items():
        if (targets is None) or (cls not in targets) or (len(idxs) < 2):
            continue
        perm = torch.randperm(len(idxs), device=x.device)
        idxs_t = torch.tensor(idxs, device=x.device)
        partner[idxs_t] = idxs_t[perm]
    lam = np.random.beta(alpha, alpha)
    x2 = x[partner]
    x_mix = lam * x + (1 - lam) * x2
    return x_mix, y

def probs_to_preds_with_tau(P, tau_H, tau_W):
    tau = np.full(P.shape[1], 0.5, dtype=np.float32)
    tau[IDX_HEALTHY]  = tau_H
    tau[IDX_WHEEZING] = tau_W
    return (P / tau.reshape(1,-1)).argmax(axis=1)

# ---------- train & tune (one fold) ----------
def train_and_tune(X_tr, y_tr, X_te, y_te, files_te, pids_te,
                   epochs=50, batch=64, lr=1e-3, weight_decay=1e-4,
                   mixup_alpha=0.2, noise_sigma=0.01):

    n_class = len(class_names)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)

    idx = np.arange(len(X_tr))
    tr_idx, va_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y_tr)
    X_trn, y_trn = X_tr[tr_idx], y_tr[tr_idx]
    X_val, y_val = X_tr[va_idx], y_tr[va_idx]

    cnt = Counter(y_trn.tolist())
    alpha = np.array([1.0 / max(cnt.get(i,1),1) for i in range(n_class)], dtype=np.float32)
    alpha = alpha * (n_class / alpha.sum())
    alpha_t = torch.tensor(alpha, dtype=torch.float32, device=device)

    wts = np.array([1.0 / max(cnt[c],1) for c in y_trn], dtype=np.float32)
    sampler = WeightedRandomSampler(wts, num_samples=len(wts), replacement=True)

    dl_trn = build_loader(NumpyDataset(X_trn, y_trn), batch, sampler=sampler, shuffle=True)
    dl_val = build_loader(NumpyDataset(X_val, y_val), batch, sampler=None, shuffle=False)

    model = MLP(X_tr.shape[1], n_class, hidden=256, p_drop=0.3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = FocalLoss(alpha=alpha_t, gamma=1.0)

    best_val = float('inf'); patience, wait = 7, 0
    for ep in range(1, epochs+1):
        model.train(); tr_loss=0.0
        for xb, yb, _, _ in dl_trn:
            xb, yb = xb.to(device), yb.to(device)
            xb_aug = gaussian_noise(xb, sigma=noise_sigma)
            # consistent mixup on H/W/R only
            mask = torch.tensor([ (yy.item() in CONSISTENT_TARGETS) for yy in yb ], device=xb.device)
            if mask.any():
                xb2, yb2 = mixup_batch_consistent(xb_aug[mask], yb[mask], alpha=mixup_alpha, targets=CONSISTENT_TARGETS)
                xb_aug = xb_aug.clone(); xb_aug[mask] = xb2
                yb_aug = yb.clone()
            else:
                yb_aug = yb
            opt.zero_grad(); loss = criterion(model(xb_aug), yb_aug)
            loss.backward(); opt.step(); tr_loss += loss.item()*xb.size(0)
        tr_loss /= len(dl_trn.dataset)

        model.eval(); val_loss=0.0
        with torch.no_grad():
            for xb, yb, _, _ in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item()*xb.size(0)
        val_loss /= len(dl_val.dataset)

        if ep==1 or ep%5==0:
            print(f"[EP {ep:02d}] train={tr_loss:.4f} val={val_loss:.4f}")
        if val_loss < best_val - 1e-4:
            best_val = val_loss; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}; wait=0
        else:
            wait += 1
            if wait >= patience:
                print(f"[EARLY STOP] at ep {ep}, best val={best_val:.4f}")
                break

    model.load_state_dict({k:v.to(device) for k,v in best_state.items()})

    # ---- get probs on val to tune tau ----
    model.eval(); P_val=[]; Y_val=[]
    with torch.no_grad():
        for xb, yb, _, _ in dl_val:
            pr = torch.softmax(model(xb.to(device)), dim=1).cpu().numpy()
            P_val.append(pr); Y_val.append(yb.numpy())
    P_val = np.vstack(P_val); Y_val = np.concatenate(Y_val)

    grid = np.linspace(0.10, 0.90, 17)
    best_tau = (0.5, 0.5); best_score = -1
    for tH in grid:
        for tW in grid:
            yhat = probs_to_preds_with_tau(P_val, tH, tW)
            sc = recall_score(Y_val, yhat, average='macro', zero_division=0)
            if sc > best_score:
                best_score = sc; best_tau = (tH, tW)
    print(f"[TUNE] best on val → macro={best_score:.3f}, tau_H={best_tau[0]:.2f}, tau_W={best_tau[1]:.2f}")

    # ---- retrain on full train (trn+val), then test probs ----
    X_full = np.concatenate([X_trn, X_val], axis=0)
    y_full = np.concatenate([y_trn, y_val], axis=0)
    cnt_full = Counter(y_full.tolist())
    alpha_full = np.array([1.0 / max(cnt_full.get(i,1),1) for i in range(n_class)], dtype=np.float32)
    alpha_full = alpha_full * (n_class / alpha_full.sum())
    alpha_full_t = torch.tensor(alpha_full, dtype=torch.float32, device=device)

    ds_full = NumpyDataset(X_full, y_full)
    w_full = np.array([1.0 / max(cnt_full[c],1) for c in y_full], dtype=np.float32)
    sampler_full = WeightedRandomSampler(w_full, num_samples=len(w_full), replacement=True)
    dl_full = build_loader(ds_full, batch, sampler=sampler_full, shuffle=True)

    final = MLP(X_tr.shape[1], n_class, hidden=256, p_drop=0.3).to(device)
    opt = torch.optim.AdamW(final.parameters(), lr=lr, weight_decay=weight_decay)
    crit_final = FocalLoss(alpha=alpha_full_t, gamma=1.0)

    final.train()
    for ep in range(1, 35):
        tot=0.0
        for xb, yb, _, _ in dl_full:
            xb, yb = xb.to(device), yb.to(device)
            xb = gaussian_noise(xb, sigma=noise_sigma)
            mask = torch.tensor([ (yy.item() in CONSISTENT_TARGETS) for yy in yb ], device=xb.device)
            if mask.any():
                xb2, yb2 = mixup_batch_consistent(xb[mask], yb[mask], alpha=mixup_alpha, targets=CONSISTENT_TARGETS)
                xb = xb.clone(); xb[mask] = xb2
            opt.zero_grad(); loss = crit_final(final(xb), yb)
            loss.backward(); opt.step(); tot += loss.item()*xb.size(0)
        if ep==1 or ep%5==0:
            print(f"[FINAL EP {ep:02d}] loss={tot/len(dl_full.dataset):.4f}")

    # test probs
    dl_te = build_loader(NumpyDataset(X_te, y_te, files_te, pids_te), batch, sampler=None, shuffle=False)
    final.eval(); P_te=[]; T_te=[]
    with torch.no_grad():
        for xb, yb, _, _ in dl_te:
            pr = torch.softmax(final(xb.to(device)), dim=1).cpu().numpy()
            P_te.append(pr); T_te.append(yb.numpy())
    P_te = np.vstack(P_te); T_te = np.concatenate(T_te)
    Y_pred = probs_to_preds_with_tau(P_te, best_tau[0], best_tau[1])

    return T_te, Y_pred, P_te, best_tau

# ---------- outer CV ----------
gkf = GroupKFold(n_splits=5)
ALL_T, ALL_P, ALL_PR = [], [], []
TAUS, ALL_FILE, ALL_PID, ALL_FOLD = [], [], [], []
fold = 0
for tr, te in gkf.split(X_all, y_all, groups_all):
    fold += 1
    print(f"\n[FOLD {fold}] train={len(tr)} test={len(te)}")
    X_tr, y_tr = X_all[tr], y_all[tr]
    X_te, y_te = X_all[te], y_all[te]
    files_te, pids_te = files_all[te], groups_all[te]

    T, P, PR, tau_hw = train_and_tune(
        X_tr, y_tr, X_te, y_te, files_te, pids_te,
        epochs=50, batch=64, lr=1e-3, weight_decay=1e-4,
        mixup_alpha=0.2, noise_sigma=0.01
    )
    ALL_T.append(T); ALL_P.append(P); ALL_PR.append(PR); TAUS.append(tau_hw)
    ALL_FILE.extend(files_te); ALL_PID.extend(pids_te); ALL_FOLD.extend([fold]*len(te))

# ---------- aggregate ----------
T = np.concatenate(ALL_T, axis=0)
P = np.concatenate(ALL_P, axis=0)
PR = np.vstack(ALL_PR)

print("\n--- Aggregate Classification Report ---")
print(classification_report([class_names[i] for i in T],
                            [class_names[i] for i in P],
                            labels=list(class_names), zero_division=0))
macro = recall_score(T, P, average='macro', zero_division=0)
print(f"[MACRO RECALL] {macro:.3f}")

cm = confusion_matrix(T, P, labels=np.arange(len(class_names)))
fig, ax = plt.subplots(figsize=(5,4))
im = ax.imshow(cm, interpolation="nearest"); ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
       xticklabels=list(class_names), yticklabels=list(class_names),
       xlabel="Predicted", ylabel="True",
       title="9R + Threshold(H,W) (Aggregate)")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
for (i,j), v in np.ndenumerate(cm): ax.text(j, i, str(v), ha='center', va='center')
fig.tight_layout(); fig.savefig(os.path.join(FIGURES_DIR, "cm_aggregate.png"), dpi=150); plt.close(fig)

# save predictions & taus
out = {"filename": ALL_FILE, "patient_id": ALL_PID, "fold": ALL_FOLD,
       "y_true": [class_names[i] for i in T], "y_pred": [class_names[i] for i in P]}
pred_df = pd.DataFrame(out)
for i, c in enumerate(class_names): pred_df[f"prob_{c}"] = PR[:, i]
pred_df.to_csv(os.path.join(RESULTS_DIR, "cv_predictions.csv"), index=False)
pd.DataFrame(TAUS, columns=["tau_Healthy","tau_Wheezing"]).to_csv(os.path.join(RESULTS_DIR, "taus_per_fold.csv"), index=False)

# quick summary for excel
rpt = classification_report([class_names[i] for i in T],
                            [class_names[i] for i in P],
                            labels=list(class_names), output_dict=True, zero_division=0)
pd.DataFrame([{
    'model': 'MLP(256)+Focal(γ=1, α=1/freq)+ConsistentMixup(H/W/R, α=0.2)+Noise(0.01)+Thresh(H,W)',
    'accuracy': rpt['accuracy'],
    'precision_macro': rpt['macro avg']['precision'],
    'recall_macro': rpt['macro avg']['recall'],
    'f1_macro': rpt['macro avg']['f1-score']
}]).set_index('model').to_csv(os.path.join(RESULTS_DIR, 'classifier_groupcv_results.csv'))

print('[DONE]')
