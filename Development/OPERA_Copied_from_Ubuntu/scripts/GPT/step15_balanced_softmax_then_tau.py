#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Step15 — Balanced Softmax CE(MLP) + Focal(LogReg) → Temp Scaling → Ensemble → Per-class τ

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
RESULTS_DIR  = os.path.join(PROJECT_DIR, 'results', 'Step15_BalSoftmax_PerClassTau')
FIGURES_DIR  = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)

np.random.seed(42); random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'

CSV_PATH = os.path.join(FEATURES_DIR, 'opera_features.csv')
print('[RUN] Step15 Balanced Softmax → per-class τ')
print(f'[PATH] {CSV_PATH}')

df = pd.read_csv(CSV_PATH)
df = df[df['label']!='unknown'].reset_index(drop=True)

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
IDX_NB = IDX['Non-breathing']; IDX_H = IDX['Healthy']; IDX_W = IDX['Wheezing']; IDX_R = IDX['Rhonchi']

print(f"[INFO] n={len(df)}, classes={list(class_names)}, dims={len(feat_cols)}")
print(df['label'].value_counts())
print(f'[DEVICE] {device}')

# ----------------- Torch helpers -----------------
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
                      drop_last=False, num_workers=0,
                      pin_memory=(device=='cuda'))

class TorchLogReg(nn.Module):
    def __init__(self, in_dim, n_classes): super().__init__(); self.fc = nn.Linear(in_dim, n_classes)
    def forward(self, x): return self.fc(x)

class MLP(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=256, p_drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(hidden, n_classes),
        )
    def forward(self, x): return self.net(x)

# --------- Losses ----------
class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 1.0):
        super().__init__(); self.alpha=alpha; self.gamma=gamma
    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(logits, target, reduction='none')
        pt = torch.exp(-ce); at = self.alpha[target]
        return (at * (1 - pt) ** self.gamma * ce).mean()

class BalancedSoftmaxCE(nn.Module):
    """
    Balanced Softmax Cross-Entropy (Ren et al.). 
    CE on logits where denominator is reweighted by class counts.
    Equivalent to CE on (logits + log(prior)) - log Z, but done stably.
    """
    def __init__(self, class_counts: torch.Tensor):
        super().__init__()
        # prior = counts / sum(counts) ; we use log(counts) up to a constant
        cc = class_counts.clone().float().clamp(min=1.0)
        self.log_prior = torch.log(cc).to(device)

    def forward(self, logits, target):
        # logits shape [B,C], add log_prior to numerator & denominator through log-softmax trick
        z = logits + self.log_prior.view(1, -1)
        return nn.functional.cross_entropy(z, target)

def make_invfreq_alpha(y, n_class):
    cnt = Counter(y.tolist())
    a = np.array([1.0/max(cnt.get(i,1),1) for i in range(n_class)], np.float32)
    a = a / (a.sum()/n_class + 1e-12)
    return a

# --------- training wrappers ----------
def train_lr_focal(X_tr, y_tr, batch=64, epochs=30, lr=1e-3):
    n_class = len(class_names)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)

    alpha_np = make_invfreq_alpha(torch.tensor(y_tr), n_class)
    alpha = torch.tensor(alpha_np, device=device)
    crit = FocalLoss(alpha=alpha, gamma=1.0)

    ds = NumpyDataset(X_tr, y_tr)
    w = np.array([1.0/max(Counter(y_tr.tolist())[c],1) for c in y_tr], np.float32)
    sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
    dl = build_loader(ds, batch, sampler=sampler, shuffle=True)

    model = TorchLogReg(X_tr.shape[1], n_class).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for ep in range(1, epochs+1):
        tot=0.0
        for xb, yb, _, _ in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
            tot += loss.item()*xb.size(0)
        if ep==1 or ep%5==0:
            print(f"[LR EP {ep:02d}] loss={tot/len(ds):.4f}")

    def infer_logits(X):
        X = scaler.transform(X).astype(np.float32)
        d = build_loader(NumpyDataset(X, np.zeros(len(X))), batch, sampler=None, shuffle=False)
        model.eval(); L=[]
        with torch.no_grad():
            for xb, _, _, _ in d:
                L.append(model(xb.to(device)).cpu().numpy())
        return np.vstack(L)
    return infer_logits

def train_mlp_balsoftmax(X_tr, y_tr, batch=64, epochs=60, lr=1e-3, wd=1e-4):
    n_class = len(class_names)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)

    # class counts for Balanced Softmax
    cnt = Counter(y_tr.tolist())
    counts = torch.tensor([cnt.get(c,0) for c in range(n_class)], device=device).float().clamp(min=1.0)
    crit = BalancedSoftmaxCE(class_counts=counts)

    ds = NumpyDataset(X_tr, y_tr)
    # 샘플링은 살짝 보정(완전 균등까지는 아님)
    w = np.array([1.0/np.sqrt(max(Counter(y_tr.tolist())[c],1)) for c in y_tr], np.float32)
    sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
    dl = build_loader(ds, batch, sampler=sampler, shuffle=True)

    model = MLP(X_tr.shape[1], n_class, hidden=256, p_drop=0.3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    model.train()
    for ep in range(1, epochs+1):
        tot=0.0
        for xb, yb, _, _ in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
            tot += loss.item()*xb.size(0)
        if ep==1 or ep%5==0:
            print(f"[MLP(BS) EP {ep:02d}] loss={tot/len(ds):.4f}")

    def infer_logits(X):
        X = scaler.transform(X).astype(np.float32)
        d = build_loader(NumpyDataset(X, np.zeros(len(X))), batch, sampler=None, shuffle=False)
        model.eval(); L=[]
        with torch.no_grad():
            for xb, _, _, _ in d:
                L.append(model(xb.to(device)).cpu().numpy())
        return np.vstack(L)
    return infer_logits

# --------- utils ----------
def softmax_np(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z); return e / e.sum(axis=1, keepdims=True)

def temperature_scaling_logits(logits, y_true, max_iter=200, lr=0.01):
    T = 1.0
    for _ in range(max_iter):
        P = softmax_np(logits / T)
        z = logits; zy = z[np.arange(len(y_true)), y_true]; sum_pz = (P * z).sum(axis=1)
        grad = ((sum_pz - zy).mean()) / (T*T + 1e-12)
        T = max(0.5, min(5.0, T - lr*grad))
        if abs(lr*grad) < 1e-6: break
    return float(T)

def apply_T_to_logits(logits, T): return logits / T
def ensemble_probs(P1, P2, w1=0.5, w2=0.5): return w1*P1 + w2*P2
def predict_with_perclass_tau(P, taus): return (P / taus.reshape(1,-1)).argmax(axis=1)

# ----------------- CV loop -----------------
gkf = GroupKFold(n_splits=5)
ALL_T, ALL_P, ALL_PR = [], [], []
ALL_FILE, ALL_PID, ALL_FOLD = [], [], []
TAUS, WEIGHTS, TEMPS = [], [], []

w_cands = [(0.5,0.5),(0.6,0.4),(0.7,0.3)]
tau_vals = np.arange(0.05, 0.61, 0.05)

fold=0
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

    infer_lr   = train_lr_focal(X_trn, y_trn, epochs=30, lr=1e-3)
    infer_mlpB = train_mlp_balsoftmax(X_trn, y_trn, epochs=60, lr=1e-3, wd=1e-4)

    L_lr_val = infer_lr(X_val)
    L_bs_val = infer_mlpB(X_val)
    T_lr  = temperature_scaling_logits(L_lr_val,  y_val, max_iter=200, lr=0.01)
    T_bs  = temperature_scaling_logits(L_bs_val,  y_val, max_iter=200, lr=0.01)
    print(f"[TEMP] T_lr={T_lr:.3f}, T_bsMLP={T_bs:.3f}")

    P_lr_val = softmax_np(L_lr_val / T_lr)
    P_bs_val = softmax_np(L_bs_val / T_bs)

    # base & guard
    base_pred = ensemble_probs(P_lr_val, P_bs_val, 0.5, 0.5).argmax(axis=1)
    base_nb = recall_score(y_val, base_pred, labels=[IDX_NB], average=None, zero_division=0)[0]
    base_macro = recall_score(y_val, base_pred, average='macro', zero_division=0)
    print(f"[VAL BASE] macro={base_macro:.3f} NB_recall={base_nb:.3f}")

    guard_targets = [max(base_nb, 0.35)] + [x/100 for x in range(30, 0, -5)]
    best = (-1.0, (0.5,0.5), np.full(len(class_names),0.5,np.float32)); found=False
    for guard in guard_targets:
        if found: break
        for w1,w2 in w_cands:
            P_val = ensemble_probs(P_lr_val, P_bs_val, w1, w2)
            nb_ok=False; nb_tau_sel=0.5
            for t_nb in tau_vals:
                taus = np.full(len(class_names), 0.5, np.float32); taus[IDX_NB]=t_nb
                y_hat = predict_with_perclass_tau(P_val, taus)
                nb_rec = recall_score(y_val, y_hat, labels=[IDX_NB], average=None, zero_division=0)[0]
                if nb_rec + 1e-9 >= guard: nb_ok=True; nb_tau_sel=t_nb; break
            if not nb_ok: continue
            for tH in tau_vals:
                for tW in tau_vals:
                    for tR in tau_vals:
                        taus = np.full(len(class_names), 0.5, np.float32)
                        taus[IDX_NB]=nb_tau_sel; taus[IDX_H]=tH; taus[IDX_W]=tW; taus[IDX_R]=tR
                        y_hat = predict_with_perclass_tau(P_val, taus)
                        if recall_score(y_val, y_hat, labels=[IDX_NB], average=None, zero_division=0)[0] + 1e-9 < guard:
                            continue
                        macro = recall_score(y_val, y_hat, average='macro', zero_division=0)
                        if macro > best[0]:
                            best = (macro, (w1,w2), taus.copy()); found=True
        if found: print(f"[TUNE] guard≥{guard:.2f} ✓ macro*={best[0]:.3f} w={best[1]} taus={np.round(best[2],2)}")

    (w1,w2), taus = best[1], best[2]

    # retrain on full-train → test
    infer_lr_full   = train_lr_focal(X_tr, y_tr, epochs=30, lr=1e-3)
    infer_mlpB_full = train_mlp_balsoftmax(X_tr, y_tr, epochs=60, lr=1e-3, wd=1e-4)
    L_lr_te  = infer_lr_full(X_te);  L_bs_te = infer_mlpB_full(X_te)
    P_te = ensemble_probs(softmax_np(L_lr_te / T_lr), softmax_np(L_bs_te / T_bs), w1, w2)
    y_pred = predict_with_perclass_tau(P_te, taus)

    ALL_T.append(y_te); ALL_P.append(y_pred); ALL_PR.append(P_te)
    ALL_FILE.extend(files_te); ALL_PID.extend(pids_te); ALL_FOLD.extend([fold]*len(te))
    TAUS.append(taus); WEIGHTS.append((w1,w2)); TEMPS.append((T_lr,T_bs))

# -------- Aggregate --------
T = np.concatenate(ALL_T); P = np.concatenate(ALL_P); PR = np.vstack(ALL_PR)
print("\n--- Aggregate Classification Report (Step15) ---")
print(classification_report([class_names[i] for i in T], [class_names[i] for i in P],
                            labels=list(class_names), zero_division=0))
macro = recall_score(T, P, average='macro', zero_division=0)
print(f"[MACRO RECALL] {macro:.3f}")

cm = confusion_matrix(T, P, labels=np.arange(len(class_names)))
fig, ax = plt.subplots(figsize=(5,4))
im = ax.imshow(cm, interpolation='nearest'); ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
       xticklabels=list(class_names), yticklabels=list(class_names),
       xlabel='Predicted', ylabel='True',
       title='Step15 (BalancedSoftmax + Temp + Per-class τ) — Aggregate')
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
for (i,j), v in np.ndenumerate(cm): ax.text(j, i, str(v), ha='center', va='center')
fig.tight_layout(); fig.savefig(os.path.join(FIGURES_DIR, 'step15_cm.png'), dpi=150); plt.close(fig)

# save artifacts
pd.DataFrame({
    "filename": ALL_FILE, "patient_id": ALL_PID, "fold": ALL_FOLD,
    "y_true": [class_names[i] for i in T],
    "y_pred": [class_names[i] for i in P],
}).to_csv(os.path.join(RESULTS_DIR, 'cv_predictions.csv'), index=False)

pd.DataFrame(np.vstack(TAUS), columns=[f"tau_{c}" for c in class_names]).to_csv(
    os.path.join(RESULTS_DIR, 'taus_per_fold.csv'), index=False)
pd.DataFrame(WEIGHTS, columns=['w_lr','w_mlpBS']).to_csv(
    os.path.join(RESULTS_DIR, 'weights_per_fold.csv'), index=False)
pd.DataFrame(TEMPS, columns=['T_lr','T_mlpBS']).to_csv(
    os.path.join(RESULTS_DIR, 'temperatures_per_fold.csv'), index=False)

rpt = classification_report([class_names[i] for i in T], [class_names[i] for i in P],
                            labels=list(class_names), output_dict=True, zero_division=0)
pd.DataFrame([{
    'model': 'Step15_BalSoftmax_PerClassTau',
    'accuracy': rpt['accuracy'],
    'precision_macro': rpt['macro avg']['precision'],
    'recall_macro': rpt['macro avg']['recall'],
    'f1_macro': rpt['macro avg']['f1-score']
}]).set_index('model').to_csv(os.path.join(RESULTS_DIR, 'classifier_groupcv_results.csv'))

print('[SAVED] results to:', RESULTS_DIR)
print('[DONE]')
