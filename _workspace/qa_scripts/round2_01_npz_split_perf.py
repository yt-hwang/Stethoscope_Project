# round2_01_npz_split_perf.py
# Adversarial re-verification: inspect npz, count patients/recordings/windows,
# reproduce GroupShuffleSplit, check leakage, recompute val performance & thresholds.
import re, json
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_curve, accuracy_score, confusion_matrix, recall_score
import joblib

BASE = Path("/Users/yunhwang/Desktop/Stethoscope_Project")
FEAT = BASE/"Deployment/1 Final Pipeline/5 Realtime Pipeline_Final/features/features_64mel.npz"
MODEL_DIR = BASE/"Deployment/1 Final Pipeline/5 Realtime Pipeline_Final/model/run_20251107_194938"

CANON = ['Crackle','Healthy','Non-breathing','Rhonchi','Wheezing']
_TSPAN_RE = re.compile(r"_([0-9]+(?:\.[0-9]+)?)-([0-9]+(?:\.[0-9]+)?)$")

def stem_without_time(stem):
    m = _TSPAN_RE.search(stem)
    return stem[:m.start()] if m else stem

def fn_to_group(fn):
    return stem_without_time(Path(fn).stem)

def normalize_label(s):
    t = s.strip()
    if t.lower().endswith("_window"): t = t[:-7]
    t = t.replace(" ","")
    flat = t.lower().replace("_","").replace("-","")
    km = {"crackle":"Crackle","healthy":"Healthy","rhonchi":"Rhonchi","wheezing":"Wheezing",
          "nonbreathing":"Non-breathing"}
    return km.get(flat, t)

d = np.load(FEAT, allow_pickle=True)
X = d["X"]; y_raw = d["y"].astype(str); filenames = d["filenames"].astype(str)
classes_in = d["classes"].astype(str).tolist()
y = np.array([normalize_label(s) for s in y_raw])

print("="*70)
print("NPZ INSPECTION")
print("="*70)
print(f"X shape (N windows x dims): {X.shape}")
print(f"classes in npz: {classes_in}")
print(f"normalized class set: {sorted(set(y))}")
from collections import Counter
print(f"class distribution: {dict(Counter(y))}")

groups = np.array([fn_to_group(fn) for fn in filenames])
uniq_groups = sorted(set(groups))
print(f"\nunique groups (stem w/o time range): {len(uniq_groups)}")
print(f"groups: {uniq_groups}")

# Patient extraction: strip session suffix _1/_2, and everything after patient id.
# Group examples look like KP019_WWS_1, KP018_WWS, H001_..., etc.
def group_to_patient(g):
    # patient id = first token before first underscore (e.g. KP019, H001)
    return g.split("_")[0]
patients = sorted(set(group_to_patient(g) for g in uniq_groups))
print(f"\nunique patients (first token before '_'): {len(patients)}")
print(f"patients: {patients}")

# session-suffix analysis: does removing _1/_2 collapse groups?
_SESS_RE = re.compile(r"_[12]$")
def group_to_recording_noSession(g):
    return _SESS_RE.sub("", g)
recs_noSession = sorted(set(group_to_recording_noSession(g) for g in uniq_groups))
print(f"\nunique recordings after removing _1/_2 session suffix: {len(recs_noSession)}")
print(f"  {recs_noSession}")

# src_file count from group definition = recordings
print(f"\n#groups (=recordings/sessions used) = {len(uniq_groups)}")

print("\n"+"="*70)
print("GROUP SPLIT REPRODUCTION (GroupShuffleSplit, test_size=0.10, rs=42)")
print("="*70)
gss = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
tr_idx, va_idx = next(gss.split(X, y, groups))
g_tr = set(groups[tr_idx]); g_va = set(groups[va_idx])
print(f"train windows={len(tr_idx)}, val windows={len(va_idx)}, total={len(X)}")
print(f"train groups ({len(g_tr)}): {sorted(g_tr)}")
print(f"val groups ({len(g_va)}): {sorted(g_va)}")
overlap = g_tr & g_va
print(f"GROUP overlap train&val: {sorted(overlap)}  -> {'LEAK' if overlap else 'NONE'}")
p_tr = set(group_to_patient(g) for g in g_tr)
p_va = set(group_to_patient(g) for g in g_va)
p_overlap = p_tr & p_va
print(f"PATIENT overlap train&val: {sorted(p_overlap)}  -> {'LEAK' if p_overlap else 'NONE'}")

# val class distribution & Rhonchi positives
y_va = y[va_idx]
print(f"\nval class distribution: {dict(Counter(y_va))}")
print(f"val Rhonchi count: {int((y_va=='Rhonchi').sum())}")

print("\n"+"="*70)
print("THRESHOLD REPRODUCTION + VAL PERFORMANCE (deployed pkl direct)")
print("="*70)
scaler = joblib.load(MODEL_DIR/"scaler.pkl")
lr = joblib.load(MODEL_DIR/"model_lr.pkl")
mlp = joblib.load(MODEL_DIR/"model_mlp.pkl")
th_json = json.load(open(MODEL_DIR/"thresholds.json"))
print(f"deployed class_names: {th_json['class_names']}")
print(f"deployed thresholds:  {th_json['thresholds']}")

classes = CANON[:]
cls2idx = {c:i for i,c in enumerate(classes)}
X_va = X[va_idx]; y_va_idx = np.array([cls2idx[c] for c in y_va])
X_va_s = scaler.transform(X_va)
P_va = 0.5*lr.predict_proba(X_va_s) + 0.5*mlp.predict_proba(X_va_s)

# reproduce thresholds
def compute_thr(P, yidx, n):
    out=[]
    for k in range(n):
        yt=(yidx==k).astype(int)
        fpr,tpr,thr = roc_curve(yt, P[:,k])
        out.append(float(thr[int(np.argmax(tpr-fpr))]))
    return out
repro_thr = compute_thr(P_va, y_va_idx, len(classes))
print(f"\nreproduced thresholds: {repro_thr}")
print(f"match deployed?: {[abs(a-b)<1e-6 or (np.isinf(a) and np.isinf(b)) for a,b in zip(repro_thr, th_json['thresholds'])]}")

# argmax accuracy
pred = P_va.argmax(axis=1)
acc = accuracy_score(y_va_idx, pred)
print(f"\nVAL argmax accuracy: {acc*100:.2f}%")
cm = confusion_matrix(y_va_idx, pred, labels=list(range(len(classes))))
print(f"confusion matrix (rows=true {classes}):")
print(cm)
# per-class recall
for k,c in enumerate(classes):
    n_true = int((y_va_idx==k).sum())
    rec = cm[k,k]/n_true if n_true>0 else float('nan')
    print(f"  recall {c}: {rec if not np.isnan(rec) else 'N/A (0 samples)'}  (n_true={n_true})")

# argmax(p - thr) decision — does Rhonchi ever win?
thr_arr = np.array([float(t) for t in th_json['thresholds']])
dec = (P_va - thr_arr).argmax(axis=1)
print(f"\nargmax(p - thr) decision counts: {dict(Counter([classes[i] for i in dec]))}")
print(f"Rhonchi ever selected by argmax(p-thr)? {'YES' if (dec==classes.index('Rhonchi')).any() else 'NO'}")
