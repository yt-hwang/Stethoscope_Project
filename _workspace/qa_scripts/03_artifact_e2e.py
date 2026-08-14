import json, re
import numpy as np
from pathlib import Path
import librosa
from joblib import load as joblib_load
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix

MODEL=Path("/Users/yunhwang/Desktop/Stethoscope_Project/Paper/Final_Code_Manuscript/3_model_run_20251107_194938")
NPZ=Path("/Users/yunhwang/Desktop/Stethoscope_Project/Paper/Final_Code_Manuscript/1_training_pipeline/features_64mel.npz")

scaler=joblib_load(MODEL/"scaler.pkl")
lr=joblib_load(MODEL/"model_lr.pkl")
mlp=joblib_load(MODEL/"model_mlp.pkl")
th=json.load(open(MODEL/"thresholds.json"))
print("[ART] scaler n_features_in_:", scaler.n_features_in_)
print("[ART] lr classes_:", lr.classes_, " n_features:", lr.n_features_in_)
print("[ART] mlp classes_:", mlp.classes_, " n_features:", mlp.coefs_[0].shape)
print("[ART] thresholds.json class_names:", th["class_names"])
print("[ART] thresholds:", th["thresholds"])

# QA-4 dummy 2s audio -> logmel -> scaler -> ensemble
SR=16000; N_MELS=64; FMIN,FMAX=50,7900; WIN_MS,HOP_MS=64,32
seg=np.random.randn(int(2.0*SR)).astype(np.float32)*0.01
m=librosa.feature.melspectrogram(y=seg,sr=SR,n_mels=N_MELS,n_fft=int(SR*WIN_MS/1000),hop_length=int(SR*HOP_MS/1000),fmin=FMIN,fmax=FMAX,power=2.0)
lm=librosa.power_to_db(m,ref=np.max); v=lm.mean(axis=1).astype(np.float32)
v=(v-v.mean())/(v.std()+1e-8)
xs=scaler.transform(v.reshape(1,-1))
p=0.5*lr.predict_proba(xs)[0]+0.5*mlp.predict_proba(xs)[0]
print("\n[E2E] logmel dim:", v.shape[0], "== scaler expects", scaler.n_features_in_)
print("[E2E] prob vector len:", len(p), " sum:", round(float(p.sum()),6), " any NaN:", bool(np.isnan(p).any()))

# Verify shipped model == reproduction on the same split (predict on val features)
CANON=['Crackle','Healthy','Non-breathing','Rhonchi','Wheezing']
_TS=re.compile(r"_([0-9]+(?:\.[0-9]+)?)-([0-9]+(?:\.[0-9]+)?)$")
def g(f):
    s=Path(f).stem; m=_TS.search(s); return s[:m.start()] if m else s
def nl(s):
    t=s.strip()
    if t.lower().endswith("_window"): t=t[:-7]
    flat=t.lower().replace("_","").replace("-","")
    mp={"crackle":"Crackle","healthy":"Healthy","rhonchi":"Rhonchi","wheezing":"Wheezing","nonbreathing":"Non-breathing"}
    return mp.get(flat,t)
d=np.load(NPZ,allow_pickle=True); X=d["X"]; y=np.array([nl(s) for s in d["y"].astype(str)]); groups=np.array([g(f) for f in d["filenames"].astype(str)])
gss=GroupShuffleSplit(n_splits=1,test_size=0.10,random_state=42); tr,va=next(gss.split(X,y,groups))
c2i={c:i for i,c in enumerate(th["class_names"])}
yva=np.array([c2i[c] for c in y[va]])
Xva=scaler.transform(X[va])
Pva=0.5*lr.predict_proba(Xva)+0.5*mlp.predict_proba(Xva)
pred=Pva.argmax(1)
print("\n[SHIPPED MODEL on val] accuracy:", round(accuracy_score(yva,pred),4))
cm=confusion_matrix(yva,pred,labels=list(range(5)))
print("[SHIPPED MODEL on val] per-class recall (order",th["class_names"],"):")
for i,c in enumerate(th["class_names"]):
    tot=cm[i].sum(); print(f"   {c:14s} support={tot:3d} recall={(cm[i,i]/tot if tot else float('nan')):.3f}")
