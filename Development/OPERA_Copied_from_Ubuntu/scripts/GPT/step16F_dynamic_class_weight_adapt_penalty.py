
"""
Step16F (fix): Dynamic Class Weighting + Adaptive Precision Penalty + Temperature & per-class Tau
- FIX: robust feature-column detection (exclude "filename"/strings, allow f0..fN pattern)
"""

import os, re, argparse, numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix, recall_score, precision_score
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

CLASS_NAMES = ["Healthy","Wheezing","Crackle","Rhonchi","Non-breathing"]
NB_INDEX = 4

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def plot_cm(cm, classes, out_path, normalize=False):
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-9)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True label', xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

def ldam_loss_logits(logits, targets, cls_num_list, max_m=0.5, s=30.0, weights=None):
    m = 1.0 / np.sqrt(np.sqrt(cls_num_list))
    m = max_m * (m / m.max())
    m = torch.tensor(m, dtype=torch.float32, device=logits.device)
    one_hot = torch.zeros_like(logits).scatter_(1, targets.view(-1, 1), 1.0)
    logits_m = logits - (m[targets]).view(-1, 1) * one_hot
    return F.cross_entropy(s * logits_m, targets, weight=weights)

def per_class_tau_search(probs, y_true, tau_grid, min_nb_prec=0.35,
                         floor_per_cls=0.30, floor_nb=0.40, floor_lambda=1.5,
                         adaptive=None):
    C = probs.shape[1]
    tau = np.ones(C, np.float32)
    def pen_scale(nb_prec):
        if adaptive is None: return 1.0
        hi, lo = adaptive.get('hi', 0.9), adaptive.get('lo', 0.7)
        mn, mx = adaptive.get('min_scale', 0.5), adaptive.get('max_scale', 2.0)
        if nb_prec >= hi: return mn
        if nb_prec <= lo: return mx
        r = (hi - nb_prec) / (hi - lo + 1e-9)
        return mn + r * (mx - mn)
    for c in range(C):
        best_t, best_val = 1.0, -1e9
        for t in tau_grid:
            sc = probs.copy(); sc[:, c] /= (t + 1e-12)
            y_pred = sc.argmax(1)
            mr = recall_score(y_true, y_pred, average='macro', zero_division=0)
            nbp = precision_score(y_true == NB_INDEX, y_pred == NB_INDEX, zero_division=0)
            recs = [recall_score(y_true == k, y_pred == k, zero_division=0) for k in range(C)]
            penalty = 0.0
            for k, rk in enumerate(recs):
                floor = floor_nb if k == NB_INDEX else floor_per_cls
                if rk < floor: penalty += (floor - rk)
            score = mr - floor_lambda * penalty * pen_scale(nbp)
            if score > best_val:
                best_val = score; best_t = t
        tau[c] = best_t
    sc = probs.copy()
    for c in range(C): sc[:, c] /= (tau[c] + 1e-12)
    y_pred = sc.argmax(1)
    return tau, dict(
        macro_recall=recall_score(y_true, y_pred, average='macro', zero_division=0),
        nb_precision=precision_score(y_true == NB_INDEX, y_pred == NB_INDEX, zero_division=0)
    )

def grid_search_temperature(logP, y_true, T_grid, tau_grid, **kw):
    best = {'score': -1e9}
    for T in T_grid:
        P = np.exp(logP / T); P /= P.sum(1, keepdims=True)
        tau, met = per_class_tau_search(P, y_true, tau_grid, **kw)
        if met['macro_recall'] > best['score']:
            best = {'T': T, 'tau': tau, 'score': met['macro_recall'], 'metrics': met}
    return best

class TabDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

class LinearHead(nn.Module):
    def __init__(self, d, C): super().__init__(); self.fc = nn.Linear(d, C)
    def forward(self, x): return self.fc(x)

def run_fold(X, y, groups, fold_idx, args, out_dir):
    gkf = GroupKFold(5)
    tr_idx, va_idx = list(gkf.split(X, y, groups))[fold_idx]
    Xtr, ytr, Xva, yva = X[tr_idx], y[tr_idx], X[va_idx], y[va_idx]
    C, d = len(CLASS_NAMES), X.shape[1]
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LinearHead(d, C).to(dev)
    cls_counts = np.bincount(ytr, minlength=C).astype(np.float32) + 1e-6
    inv = (1.0 / cls_counts); inv /= inv.mean()
    w = torch.tensor(inv, dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    dl_tr = DataLoader(TabDataset(Xtr, ytr), batch_size=args.bs, shuffle=True)
    dl_va = DataLoader(TabDataset(Xva, yva), batch_size=args.bs, shuffle=False)
    drw_start = args.epochs // 2
    best_va, best_state = -1.0, None
    run_rec = np.zeros(C, np.float32)
    for ep in range(1, args.epochs + 1):
        model.train(); tot = 0.0
        for xb, yb in dl_tr:
            xb, yb = xb.to(dev), yb.to(dev)
            z = model(xb)
            if ep < drw_start: loss = F.cross_entropy(z, yb, weight=w)
            else: loss = ldam_loss_logits(z, yb, cls_num_list=cls_counts, weights=w)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        model.eval(); P = []; Tt = []
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(dev)
                z = model(xb)
                P.append(F.softmax(z, dim=1).cpu().numpy()); Tt.append(yb.numpy())
        P = np.concatenate(P, 0); Tt = np.concatenate(Tt, 0)
        yp = P.argmax(1)
        mr = recall_score(Tt, yp, average='macro', zero_division=0)
        for k in range(C):
            run_rec[k] = 0.7 * run_rec[k] + 0.3 * recall_score(Tt == k, yp == k, zero_division=0)
        if ep % args.dcw_update == 0 and ep >= drw_start:
            adj = (1.0 / (run_rec + 1e-3)); adj /= adj.mean(); w = torch.tensor(inv * adj, dtype=torch.float32, device=dev)
        if mr > best_va: best_va = mr; best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == 1:
            recs = [recall_score(Tt == k, yp == k, zero_division=0) for k in range(C)]
            print(f"[Fold {fold_idx+1}][Ep {ep:03d}] loss={tot/len(dl_tr.dataset):.4f}  valMR={mr:.3f}  rec={['%.3f'%r for r in recs]}")
    model.load_state_dict({k: v for k, v in best_state.items()})
    with torch.no_grad():
        P = []; Tt = []; Z = []
        for xb, yb in dl_va:
            xb = xb.to(dev); z = model(xb); Z.append(z.cpu().numpy())
            P.append(F.softmax(z, dim=1).cpu().numpy()); Tt.append(yb.numpy())
    P = np.concatenate(P, 0); Tt = np.concatenate(Tt, 0); logP = np.log(P + 1e-12)
    best = grid_search_temperature(
        logP, Tt,
        T_grid=np.linspace(0.5, 3.0, 11),
        tau_grid=np.linspace(0.4, 2.2, 21),
        min_nb_prec=args.nb_prec_floor,
        floor_per_cls=args.recall_floor_perclass,
        floor_nb=args.recall_floor_nb,
        floor_lambda=args.floor_lambda,
        adaptive={'hi': 0.9, 'lo': 0.7, 'min_scale': 0.5, 'max_scale': 2.0}
    )
    Tcur, tau = best['T'], best['tau']
    PT = np.exp(logP / Tcur); PT /= PT.sum(1, keepdims=True)
    for c in range(C): PT[:, c] /= (tau[c] + 1e-12)
    yp_tau = PT.argmax(1)
    mr_raw = recall_score(Tt, P.argmax(1), average='macro', zero_division=0)
    mr_tau = recall_score(Tt, yp_tau, average='macro', zero_division=0)
    nbp = precision_score(Tt == NB_INDEX, yp_tau == NB_INDEX, zero_division=0)
    cm_raw = confusion_matrix(Tt, P.argmax(1), labels=list(range(C)))
    cm_tau = confusion_matrix(Tt, yp_tau, labels=list(range(C)))
    plot_cm(cm_raw, CLASS_NAMES, os.path.join(out_dir, f"fold{fold_idx+1}_cm_raw.png"))
    plot_cm(cm_tau, CLASS_NAMES, os.path.join(out_dir, f"fold{fold_idx+1}_cm_tau.png"))
    plot_cm(cm_raw, CLASS_NAMES, os.path.join(out_dir, f"fold{fold_idx+1}_cm_raw_norm.png"), normalize=True)
    plot_cm(cm_tau, CLASS_NAMES, os.path.join(out_dir, f"fold{fold_idx+1}_cm_tau_norm.png"), normalize=True)
    return mr_raw, mr_tau, nbp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", type=str, default=r"D:\Stethoscope_Project\Development\OPERA_Copied_from_Ubuntu\features\opera_features.csv")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dcw_update", type=int, default=5)
    ap.add_argument("--nb_prec_floor", type=float, default=0.35)
    ap.add_argument("--recall_floor_perclass", type=float, default=0.30)
    ap.add_argument("--recall_floor_nb", type=float, default=0.40)
    ap.add_argument("--floor_lambda", type=float, default=1.5)
    ap.add_argument("--exp_tag", type=str, default="Step16F_DCW_adaptTauTemp_nbPrec_ge_0.35_fix")
    args = ap.parse_args()

    base_out = r"D:\Stethoscope_Project\Development\OPERA_Copied_from_Ubuntu\scripts\GPT\results_step16F"
    ensure_dir(base_out)
    out_dir = os.path.join(base_out, args.exp_tag); ensure_dir(out_dir)

    df = pd.read_csv(args.features_csv)

    # ---- robust feature detection ----
    non_feature_cols = {"filename","file","path","filepath","patient_id","label","label_id"}
    feat_cols = []
    for c in df.columns:
        if c in non_feature_cols: continue
        if re.match(r'^f\d+$', c): 
            feat_cols.append(c); continue
        if df[c].dtype != object and pd.api.types.is_numeric_dtype(df[c]):
            feat_cols.append(c)

    if not feat_cols:
        raise RuntimeError("No feature columns detected. Please check opera_features.csv headers.")

    feat_cols = sorted(feat_cols, key=lambda x: (len(x), x))
    print(f"[INFO] Feature columns detected: {len(feat_cols)}  first={feat_cols[0]}  last={feat_cols[-1]}")

    X = df[feat_cols].values.astype(np.float32)
    y = df["label_id"].values.astype(np.int64)
    groups = df["patient_id"].values

    raws, taus, nbps = [], [], []
    for k in range(5):
        print(f"[Fold {k+1}] ==========")
        r, t, n = run_fold(X, y, groups, k, args, out_dir)
        print(f"[Fold {k+1}] rawMR={r:.3f} -> tauMR={t:.3f} (NB-prec={n:.3f})")
        raws.append(r); taus.append(t); nbps.append(n)

    summ = pd.DataFrame({"fold": np.arange(1, 6), "rawMR": raws, "tauMR": taus, "NB_prec": nbps})
    out_csv = os.path.join(out_dir, f"{args.exp_tag}_summary.csv")
    summ.to_csv(out_csv, index=False)
    print(f"[DONE] Saved summary CSV to: {out_csv}")
    print(f"[SUMMARY] MacroRecall(raw)={np.mean(raws):.3f} | MacroRecall(tau)={np.mean(taus):.3f} | tag={args.exp_tag}")

if __name__ == "__main__":
    main()
