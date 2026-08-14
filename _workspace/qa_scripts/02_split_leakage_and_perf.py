import re, json
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve

NPZ = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Paper/Final_Code_Manuscript/1_training_pipeline/features_64mel.npz")
CANONICAL = ['Crackle', 'Healthy', 'Non-breathing', 'Rhonchi', 'Wheezing']
_TSPAN_RE = re.compile(r"_([0-9]+(?:\.[0-9]+)?)-([0-9]+(?:\.[0-9]+)?)$")

def stem_wo_time(stem):
    m = _TSPAN_RE.search(stem); return stem[:m.start()] if m else stem
def fn2group(f): return stem_wo_time(Path(f).stem)
def norm_label(s):
    t=s.strip()
    if t.lower().endswith("_window"): t=t[:-7]
    flat=t.lower().replace("_","").replace("-","")
    m={"crackle":"Crackle","healthy":"Healthy","rhonchi":"Rhonchi","wheezing":"Wheezing","nonbreathing":"Non-breathing"}
    return m.get(flat,t)

d=np.load(NPZ,allow_pickle=True)
X=d["X"]; y_raw=d["y"].astype(str); fn=d["filenames"].astype(str)
y=np.array([norm_label(s) for s in y_raw])
groups=np.array([fn2group(f) for f in fn])

# patient id: strip trailing _1/_2 style channel/session suffix to get true patient
def patient(g):
    return re.sub(r"_\d+$","", g)  # strip trailing _<n>
patients = np.array([patient(g) for g in groups])
print("[GROUP KEY] unique group keys:", len(set(groups.tolist())))
print("[PATIENT ] unique patients (strip trailing _n):", len(set(patients.tolist())))
print("[PATIENT ] set:", sorted(set(patients.tolist())))

# reproduce split
gss=GroupShuffleSplit(n_splits=1,test_size=0.10,random_state=42)
tr,va=next(gss.split(X,y,groups))
g_tr=set(groups[tr].tolist()); g_va=set(groups[va].tolist())
print("\n[SPLIT] n_train=%d n_val=%d"%(len(tr),len(va)))
print("[SPLIT] train groups=%d val groups=%d"%(len(g_tr),len(g_va)))
print("[SPLIT] val groups:", sorted(g_va))
print("[LEAK ] group overlap:", g_tr & g_va)
# patient-level leak
p_tr=set(patients[tr].tolist()); p_va=set(patients[va].tolist())
print("[LEAK ] PATIENT overlap (after stripping _n):", p_tr & p_va)

# train models, recompute val perf
cls2idx={c:i for i,c in enumerate(CANONICAL)}
ytr=np.array([cls2idx[c] for c in y[tr]]); yva=np.array([cls2idx[c] for c in y[va]])
sc=StandardScaler().fit(X[tr]); Xtr=sc.transform(X[tr]); Xva=sc.transform(X[va])
lr=LogisticRegression(max_iter=2000,multi_class="multinomial",random_state=42).fit(Xtr,ytr)
mlp=MLPClassifier(hidden_layer_sizes=(128,),max_iter=200,random_state=42).fit(Xtr,ytr)
Pva=0.5*lr.predict_proba(Xva)+0.5*mlp.predict_proba(Xva)
pred=Pva.argmax(1)
print("\n[PERF] overall val accuracy (argmax):", round(accuracy_score(yva,pred),4))
cm=confusion_matrix(yva,pred,labels=list(range(5)))
print("[PERF] confusion matrix rows=true cols=pred, order",CANONICAL)
print(cm)
for i,c in enumerate(CANONICAL):
    tot=cm[i].sum()
    acc=cm[i,i]/tot if tot>0 else float('nan')
    print(f"   {c:14s} val support={tot:3d} recall={acc:.3f}")

# thresholds via Youden J (reproduce)
print("\n[THRESH] reproduce OVR Youden thresholds:")
for k,c in enumerate(CANONICAL):
    yt=(yva==k).astype(int)
    fpr,tpr,thr=roc_curve(yt,Pva[:,k])
    j=tpr-fpr; jb=int(np.argmax(j))
    print(f"   {c:14s} n_pos={yt.sum():3d} thr={thr[jb]}")
