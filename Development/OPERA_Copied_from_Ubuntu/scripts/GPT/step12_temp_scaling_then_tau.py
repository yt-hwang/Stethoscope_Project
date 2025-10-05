#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step12 — Temperature Scaling (per-model scalar T) → Ensemble → Per-class τ 재튜닝
- LogReg, MLP 각각에 대해 검증셋 NLL 최소화로 T_LR, T_MLP 학습 (scalar)
- 보정된 확률로 앙상블(가중치 그리드), per-class τ 탐색
- NB guard는 완화형(시작=max(baseline,0.35) → 0.30→…→0.05), 실패 시 폴백(NB_tau=0.05)
- 나머지 세팅은 11R3F와 동일
"""

import os, random, numpy as np, pandas as pd
from collections import Counter
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, recall_score, confusion_matrix, log_loss

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ---------------- Basic paths ----------------
PROJECT_DIR = '//home//un_wang//my_stethoscope_project'
FEATURES_DIR = os.path.join(PROJECT_DIR, 'features')
RESULTS_DIR  = os.path.join(PROJECT_DIR, 'results', 'Step12_TempScale_PerClassTau')
FIGURES_DIR  = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------- Repro ----------------
np.random.seed(42); random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------- Data ----------------
CSV_PATH = os.path.join(FEATURES_DIR, 'opera_features.csv')
print('[RUN] Step12 Temperature Scaling → per-class τ')
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

# ---------------- Torch helpers ----------------
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

class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 1.0):
        super().__init__(); self.alpha=alpha; self.gamma=gamma
    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(logits, target, reduction='none')
        pt = torch.exp(-ce); at = self.alpha[target]
        return (at * (1 - pt) ** self.gamma * ce).mean()

def make_alpha(y, n_class):
    cnt = Counter(y.tolist())
    a = np.array([1.0/max(cnt.get(i,1),1) for i in range(n_class)], np.float32)
    return (a * (n_class / a.sum())).astype(np.float32)

def train_model(model_type, X_tr, y_tr, batch=64, epochs=50, lr=1e-3, wd=1e-4):
    """Return: infer(X) -> logits, probs (pre-calibration)"""
    n_class = len(class_names)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)

    alpha = torch.tensor(make_alpha(torch.tensor(y_tr), n_class), device=device)
    ds = NumpyDataset(X_tr, y_tr)
    w = np.array([1.0/max(Counter(y_tr.tolist())[c],1) for c in y_tr], np.float32)
    sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
    dl = build_loader(ds, batch, sampler=sampler, shuffle=True)

    if model_type=='lr':
        model = TorchLogReg(X_tr.shape[1], n_class).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
        epochs = min(epochs, 30)
    else:
        model = MLP(X_tr.shape[1], n_class, hidden=256, p_drop=0.3).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    crit = FocalLoss(alpha=alpha, gamma=1.0)
    model.train()
    for ep in range(1, epochs+1):
        tot=0.0
        for xb, yb, _, _ in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
            tot += loss.item()*xb.size(0)
        if ep==1 or ep%5==0:
            print(f"[{model_type.upper()} EP {ep:02d}] loss={tot/len(ds):.4f}")

    def infer_logits_probs(X):
        X = scaler.transform(X).astype(np.float32)
        d = build_loader(NumpyDataset(X, np.zeros(len(X))), batch, sampler=None, shuffle=False)
        model.eval(); L=[]; P=[]
        with torch.no_grad():
            for xb, _, _, _ in d:
                logits = model(xb.to(device))
                L.append(logits.cpu().numpy())
                P.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(L), np.vstack(P)
    return infer_logits_probs

def softmax_np(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def temperature_scaling_logits(logits, y_true, max_iter=100, lr=0.01):
    """
    단일 스칼라 T >= 0.5 제약. 검증 NLL 최소화.
    간단한 GD로 T 업데이트.
    """
    T = 1.0
    for _ in range(max_iter):
        # grad d(NLL)/dT
        P = softmax_np(logits / T)
        # NLL = -log p(y)
        eps = 1e-12
        n = len(y_true)
        # gradient (approx): d/dT = (1/T^2) * sum ( sum_k p_k z_k - z_y )
        z = logits
        zy = z[np.arange(n), y_true]
        sum_pz = (P * z).sum(axis=1)
        grad = ((sum_pz - zy).mean()) / (T*T + 1e-12)
        T_new = T - lr * grad
        T = max(0.5, min(5.0, T_new))  # clip for stability
        # optional early stop by small grad
        if abs(lr*grad) < 1e-6:
            break
    return float(T)

def apply_T(logits, T):
    return softmax_np(logits / T)

def ensemble_probs(P_lr, P_mlp, w_lr=0.5, w_mlp=0.5):
    return w_lr*P_lr + w_mlp*P_mlp

def predict_with_perclass_tau(P, taus):
    Q = P / taus.reshape(1,-1)
    return Q.argmax(axis=1)

# ---------------- CV with temp scaling + per-class τ ----------------
gkf = GroupKFold(n_splits=5)
ALL_T, ALL_P, ALL_PR = [], [], []
ALL_FILE, ALL_PID, ALL_FOLD = [], [], []
TAUS, WEIGHTS, TEMPS = [], [], []

w_cands = [(0.5,0.5),(0.6,0.4),(0.7,0.3),(0.8,0.2)]
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

    # train models
    infer_lr  = train_model('lr',  X_trn, y_trn, epochs=30, lr=1e-3, wd=0.0)
    infer_mlp = train_model('mlp', X_trn, y_trn, epochs=50, lr=1e-3, wd=1e-4)

    # logits/probs on val
    L_lr_val,  P_lr_val  = infer_lr(X_val)
    L_mlp_val, P_mlp_val = infer_mlp(X_val)

    # ------ Temperature scaling (scalar per model) ------
    T_lr  = temperature_scaling_logits(L_lr_val,  y_val, max_iter=200, lr=0.01)
    T_mlp = temperature_scaling_logits(L_mlp_val, y_val, max_iter=200, lr=0.01)
    P_lr_val_T  = apply_T(L_lr_val,  T_lr)
    P_mlp_val_T = apply_T(L_mlp_val, T_mlp)
    print(f"[TEMP] T_lr={T_lr:.3f}, T_mlp={T_mlp:.3f}")

    # baseline after temp
    P_val_blend = ensemble_probs(P_lr_val_T, P_mlp_val_T, 0.5, 0.5)
    base_pred = P_val_blend.argmax(axis=1)
    base_nb = recall_score(y_val, base_pred, labels=[IDX_NB], average=None, zero_division=0)[0]
    base_macro = recall_score(y_val, base_pred, average='macro', zero_division=0)
    print(f"[VAL BASE (after T)] macro={base_macro:.3f} NB_recall={base_nb:.3f}")

    # guard list (완화)
    guard_targets = [max(base_nb, 0.35)] + [x/100 for x in range(30, 0, -5)]
    best = (-1.0, (0.5,0.5), np.full(len(class_names), 0.5, np.float32))
    found = False

    for guard in guard_targets:
        if found: break
        for w_lr, w_mlp in w_cands:
            P_val = ensemble_probs(P_lr_val_T, P_mlp_val_T, w_lr, w_mlp)
            # NB τ 먼저
            nb_ok=False; nb_tau_sel=0.5
            for t_nb in tau_vals:
                taus = np.full(len(class_names), 0.5, np.float32)
                taus[IDX_NB] = t_nb
                y_hat = predict_with_perclass_tau(P_val, taus)
                nb_rec = recall_score(y_val, y_hat, labels=[IDX_NB], average=None, zero_division=0)[0]
                if nb_rec + 1e-9 >= guard:
                    nb_ok=True; nb_tau_sel=t_nb; break
            if not nb_ok: continue

            for tH in tau_vals:
                for tW in tau_vals:
                    for tR in tau_vals:
                        taus = np.full(len(class_names), 0.5, np.float32)
                        taus[IDX_NB]=nb_tau_sel; taus[IDX_H]=tH; taus[IDX_W]=tW; taus[IDX_R]=tR
                        y_hat = predict_with_perclass_tau(P_val, taus)
                        nb_rec = recall_score(y_val, y_hat, labels=[IDX_NB], average=None, zero_division=0)[0]
                        if nb_rec + 1e-9 < guard: continue
                        macro = recall_score(y_val, y_hat, average='macro', zero_division=0)
                        if macro > best[0]:
                            best = (macro, (w_lr,w_mlp), taus.copy()); found=True
        if found:
            print(f"[TUNE] guard≥{guard:.2f} ✓ | macro*={best[0]:.3f} | w={best[1]} | taus={np.round(best[2],2)}")
            break

    # 폴백: guard 불가 시 NB_tau=0.05로 macro 최대
    if not found:
        print("[TUNE] guard search failed → FALLBACK without guard")
        fb_best = (-1.0, (0.5,0.5), None)
        for w_lr, w_mlp in w_cands:
            P_val = ensemble_probs(P_lr_val_T, P_mlp_val_T, w_lr, w_mlp)
            taus = np.full(len(class_names), 0.5, np.float32)
            taus[IDX_NB] = 0.05
            y_hat = predict_with_perclass_tau(P_val, taus)
            macro = recall_score(y_val, y_hat, average='macro', zero_division=0)
            if macro > fb_best[0]:
                fb_best = (macro, (w_lr,w_mlp), taus.copy())
        best = fb_best
        print(f"[FALLBACK] macro*={best[0]:.3f} | w={best[1]} | taus={np.round(best[2],2)}")

    WEIGHTS.append(best[1]); TAUS.append(best[2]); TEMPS.append((T_lr,T_mlp))

    # ----- full train → test (retrain, then apply same T) -----
    infer_lr_full  = train_model('lr',  X_tr, y_tr, epochs=30, lr=1e-3, wd=0.0)
    infer_mlp_full = train_model('mlp', X_tr, y_tr, epochs=50, lr=1e-3, wd=1e-4)
    L_lr_te,  P_lr_te  = infer_lr_full(X_te);  P_lr_te_T  = apply_T(L_lr_te,  T_lr)
    L_mlp_te, P_mlp_te = infer_mlp_full(X_te); P_mlp_te_T = apply_T(L_mlp_te, T_mlp)

    w_lr, w_mlp = best[1]
    P_te = ensemble_probs(P_lr_te_T, P_mlp_te_T, w_lr, w_mlp)
    y_pred = predict_with_perclass_tau(P_te, best[2])

    ALL_T.append(y_te); ALL_P.append(y_pred); ALL_PR.append(P_te)
    ALL_FILE.extend(files_te); ALL_PID.extend(pids_te); ALL_FOLD.extend([fold]*len(te))

# ---------------- Aggregate ----------------
T = np.concatenate(ALL_T); P = np.concatenate(ALL_P); PR = np.vstack(ALL_PR)
print("\n--- Aggregate Classification Report (Step12) ---")
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
       title='Step12 (TempScaling + Per-class τ) — Aggregate')
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
for (i,j), v in np.ndenumerate(cm): ax.text(j, i, str(v), ha='center', va='center')
fig.tight_layout(); fig.savefig(os.path.join(FIGURES_DIR, 'step12_cm.png'), dpi=150); plt.close(fig)

# save artifacts
pred = {"filename": ALL_FILE, "patient_id": ALL_PID, "fold": ALL_FOLD,
        "y_true": [class_names[i] for i in T], "y_pred": [class_names[i] for i in P]}
pd.DataFrame(pred).to_csv(os.path.join(RESULTS_DIR, 'cv_predictions.csv'), index=False)
pd.DataFrame(np.vstack(TAUS), columns=[f"tau_{c}" for c in class_names]).to_csv(
    os.path.join(RESULTS_DIR, 'taus_per_fold.csv'), index=False)
pd.DataFrame(WEIGHTS, columns=['w_lr','w_mlp']).to_csv(
    os.path.join(RESULTS_DIR, 'weights_per_fold.csv'), index=False)
pd.DataFrame(TEMPS, columns=['T_lr','T_mlp']).to_csv(
    os.path.join(RESULTS_DIR, 'temperatures_per_fold.csv'), index=False)

rpt = classification_report([class_names[i] for i in T], [class_names[i] for i in P],
                            labels=list(class_names), output_dict=True, zero_division=0)
pd.DataFrame([{
    'model': 'Step12_TempScale_PerClassTau',
    'accuracy': rpt['accuracy'],
    'precision_macro': rpt['macro avg']['precision'],
    'recall_macro': rpt['macro avg']['recall'],
    'f1_macro': rpt['macro avg']['f1-score']
}]).set_index('model').to_csv(os.path.join(RESULTS_DIR, 'classifier_groupcv_results.csv'))

print('[SAVED] results to:', RESULTS_DIR)
print('[DONE]')
