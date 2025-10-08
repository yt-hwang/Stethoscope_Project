
import os, re, argparse, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import recall_score, confusion_matrix
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def detect_columns(df):
    LABEL_CANDS = ["label_id","label","diagnosis","diagnosis_id","y"]
    FILE_CANDS  = ["filename","file","filepath","path"]
    label_col = next((c for c in LABEL_CANDS if c in df.columns), None)
    if label_col is None: raise KeyError("No label column found.")
    fname_col = next((c for c in FILE_CANDS if c in df.columns), None)
    feat_cols = [c for c in df.columns if re.fullmatch(r"\d+", str(c))]
    if len(feat_cols)==0:
        feat_cols = [c for c in df.columns if c not in {label_col,fname_col} and pd.api.types.is_numeric_dtype(df[c])]
    feat_cols = sorted(feat_cols, key=lambda x:int(x))
    feat_cols = [c for c in feat_cols if c not in {"extraction_success","patient_id","group","fold","split"}]
    return feat_cols, label_col, fname_col

def map_labels(y):
    order = ["Healthy","Non-breathing","Rhonchi","Wheezing","Crackle"]
    mp = {k:i for i,k in enumerate(order)}
    out = np.array([mp.setdefault(s, len(mp)) for s in y], dtype=np.int64)
    classes = [None]*len(mp)
    for k,v in mp.items(): classes[v]=k
    return out, classes

def groups_from_filenames(series):
    if series is None: return np.arange(len(series))
    return np.array([os.path.basename(str(x)).split("_")[0] for x in series])

def plot_cm(cm, classes, title, path):
    import numpy as _np
    fig = plt.figure(figsize=(4.8,4.8), dpi=160); ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation='nearest'); plt.colorbar(im, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ticks = _np.arange(len(classes))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(classes, rotation=45, ha='right'); ax.set_yticklabels(classes)
    thr = cm.max()/2. if cm.max()>0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,i,str(int(cm[i,j])), ha='center', va='center', color=('white' if cm[i,j]>thr else 'black'))
    ax.set_xlabel("Pred"); ax.set_ylabel("True"); fig.tight_layout(); fig.savefig(path, bbox_inches='tight'); plt.close(fig)

class LinearHead(nn.Module):
    def __init__(self, d, k): super().__init__(); self.fc=nn.Linear(d,k)
    def forward(self,x): return self.fc(x)

def per_class_recall(y_true, y_pred, k):
    recs=[]
    for c in range(k):
        m = (y_true==c)
        recs.append(float((y_pred[m]==c).sum())/max(m.sum(),1))
    return np.array(recs)

def greedy_tau_search(probs, y_true, classes, nb_prec_floor=0.35, grid=np.linspace(0.25,0.75,21), rounds=3):
    k = len(classes)
    tau = np.array([0.5]*k, dtype=np.float32)
    nb_idx = classes.index("Non-breathing") if "Non-breathing" in classes else None

    def eval_with_tau(tau_vec):
        hard = probs.argmax(1).copy()
        for i in range(probs.shape[0]):
            c = hard[i]
            if probs[i,c] < tau_vec[c]:
                passed = np.where(probs[i] >= tau_vec)[0]
                if len(passed)>0: hard[i] = passed[np.argmax(probs[i,passed])]
        mr = recall_score(y_true, hard, average='macro', zero_division=0)
        cm = confusion_matrix(y_true, hard, labels=list(range(k)))
        if nb_idx is not None:
            nb_prec = cm[nb_idx, nb_idx] / max(cm[:, nb_idx].sum(), 1)
            if nb_prec < nb_prec_floor: return -1.0, None, None
        return mr, hard, cm

    best_mr, _, _ = eval_with_tau(tau)
    for _ in range(rounds):
        improved = False
        for c in range(k):
            best_local = best_mr; best_tc = tau[c]
            for t in grid:
                tau_try = tau.copy(); tau_try[c] = t
                mr, _, _ = eval_with_tau(tau_try)
                if mr > best_local:
                    best_local = mr; best_tc = t
            if best_tc != tau[c]:
                tau[c] = best_tc; best_mr = best_local; improved = True
        if not improved: break

    mr, hard, cm = eval_with_tau(tau)
    return tau.tolist(), mr, hard, cm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", type=str, default=r"D:\Stethoscope_Project\Development\OPERA_Copied_from_Ubuntu\features\opera_features.csv")
    ap.add_argument("--out_root", type=str, default=r"D:\Stethoscope_Project\Development\OPERA_Copied_from_Ubuntu\scripts\GPT\results_step16G")
    ap.add_argument("--exp_tag", type=str, default="Step16G_HardnessWeighted_greedyTau_nbPrec_ge_0.35_v1")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--alpha", type=float, default=1.0, help="class weight exponent for (1-recall)^alpha")
    ap.add_argument("--beta", type=float, default=0.5, help="hardness weight smoothing (EWMA factor)")
    args = ap.parse_args()

    df = pd.read_csv(args.features_csv)
    feat_cols, label_col, fname_col = detect_columns(df)
    print(f"[INFO] Feature columns detected: {len(feat_cols)} first={feat_cols[0]} last={feat_cols[-1]}")
    X = df[feat_cols].values.astype(np.float32)
    y, classes = map_labels(df[label_col].astype(str).values)
    groups = groups_from_filenames(df[fname_col].values) if fname_col else np.arange(len(df))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = os.path.join(args.out_root, args.exp_tag); ensure_dir(out_dir)

    gkf = GroupKFold(n_splits=args.folds)
    macro_raw_list, macro_tau_list = [], []
    for fold_idx, (tr, va) in enumerate(gkf.split(X, y, groups), start=1):
        print(f"[Fold {fold_idx}] ==========")
        Xtr, Xva = X[tr], X[va]; ytr, yva = y[tr], y[va]
        tr_ds = TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(ytr).long())
        va_ds = TensorDataset(torch.from_numpy(Xva).float(), torch.from_numpy(yva).long())
        tr_loader = DataLoader(tr_ds, batch_size=args.bs, shuffle=True)
        va_loader = DataLoader(va_ds, batch_size=args.bs, shuffle=False)

        model = LinearHead(X.shape[1], len(classes)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

        # init weights
        class_w = torch.ones(len(classes), device=device)
        # sample-hardness EWMA buffer
        hardness_buf = torch.zeros(len(classes), device=device)

        best_va = -1.0; best_state=None
        for ep in range(1, args.epochs+1):
            model.train()
            for xb, yb in tr_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                # online hardness: 1 - p(true)
                with torch.no_grad():
                    prob = torch.softmax(logits, dim=1)
                    hard = 1.0 - prob.gather(1, yb.view(-1,1)).squeeze(1)  # per-sample hardness
                    # class-wise mean hardness
                    for c in range(len(classes)):
                        m = (yb==c)
                        if m.any():
                            hm = hard[m].mean()
                            hardness_buf[c] = args.beta*hardness_buf[c] + (1-args.beta)*hm

                # combine class weight & hardness weight
                comb_w = class_w * (1.0 + hardness_buf)
                loss = F.cross_entropy(logits, yb, weight=comb_w)

                opt.zero_grad(); loss.backward(); opt.step()

            # update dynamic class weights by train recall
            model.eval()
            with torch.no_grad():
                allp, allt = [], []
                for xb, yb in tr_loader:
                    xb = xb.to(device)
                    pr = model(xb).argmax(1).cpu().numpy()
                    allp.append(pr); allt.append(yb.numpy())
                allp = np.concatenate(allp); allt = np.concatenate(allt)
            recs = per_class_recall(allt, allp, len(classes))
            w = np.clip((1.0 - recs)**args.alpha, 0.2, 3.0)
            class_w = torch.from_numpy(w).float().to(device)

            # val macro recall (raw argmax)
            with torch.no_grad():
                vp, vt = [], []
                for xb, yb in va_loader:
                    xb = xb.to(device)
                    vp.append(model(xb).argmax(1).cpu().numpy()); vt.append(yb.numpy())
                vp = np.concatenate(vp); vt = np.concatenate(vt)
                val_mr = recall_score(vt, vp, average='macro', zero_division=0)
            if ep % 5 == 0 or ep == 1:
                hr = ", ".join([f"{c}:{r:.2f}" for c,r in zip(classes, recs)])
                print(f"[Fold {fold_idx}][Ep {ep:03d}] valMR={val_mr:.3f}  trainRec=[{hr}]  classW={w.round(2)}  hardEWMA={hardness_buf.detach().cpu().round().tolist()}")

            if val_mr > best_va:
                best_va = val_mr; best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

        model.load_state_dict(best_state)
        # get val logits
        model.eval()
        with torch.no_grad():
            logits = []
            for xb,_ in va_loader:
                xb = xb.to(device)
                logits.append(model(xb).cpu().numpy())
            logits = np.vstack(logits)
        probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()

        raw_pred = probs.argmax(1)
        raw_cm = confusion_matrix(yva, raw_pred, labels=list(range(len(classes))))
        raw_mr = recall_score(yva, raw_pred, average='macro', zero_division=0)

        taus, tau_mr, tau_pred, tau_cm = greedy_tau_search(probs, yva, classes, nb_prec_floor=0.35)
        print(f"[Fold {fold_idx}] rawMR={raw_mr:.3f} -> tauMR={tau_mr:.3f}")

        plot_cm(raw_cm, classes, f"Fold{fold_idx} RAW", os.path.join(out_dir, f"fold{fold_idx}_cm_raw.png"))
        plot_cm(tau_cm, classes, f"Fold{fold_idx} TAU", os.path.join(out_dir, f"fold{fold_idx}_cm_tau.png"))

        macro_raw_list.append(raw_mr); macro_tau_list.append(tau_mr)

    macro_raw = float(np.mean(macro_raw_list)); macro_tau = float(np.mean(macro_tau_list))
    sum_path = os.path.join(out_dir, f"{args.exp_tag}_summary.csv")
    pd.DataFrame([{"rawMR": macro_raw, "tauMR": macro_tau, "tag": args.exp_tag}]).to_csv(sum_path, index=False)
    print(f"[DONE] Saved summary CSV to: {sum_path}")
    print(f"[SUMMARY] MacroRecall(raw)={macro_raw:.3f} | MacroRecall(tau)={macro_tau:.3f} | tag={args.exp_tag}")

if __name__ == "__main__":
    main()
