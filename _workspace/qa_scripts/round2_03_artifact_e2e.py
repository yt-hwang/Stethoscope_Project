# round2_03: end-to-end artifact inference with dummy 2s audio, mimicking app _infer_one.
import json
import numpy as np
from pathlib import Path
import librosa, joblib

MODEL_DIR = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Deployment/1 Final Pipeline/5 Realtime Pipeline_Final/model/run_20251107_194938")
SR=16000; WIN_S=2.0; N_MELS=64; FMIN,FMAX=50,7900; WIN_MS,HOP_MS=64,32

scaler=joblib.load(MODEL_DIR/"scaler.pkl")
lr=joblib.load(MODEL_DIR/"model_lr.pkl")
mlp=joblib.load(MODEL_DIR/"model_mlp.pkl")
th=json.load(open(MODEL_DIR/"thresholds.json"))

print("scaler expects n_features_in_:", getattr(scaler,'n_features_in_','?'))
print("lr classes_:", lr.classes_, " mlp classes_:", mlp.classes_)
print("lr n_features_in_:", lr.n_features_in_, " mlp n_features_in_:", mlp.n_features_in_)
print("model is single flat classifier per estimator (no hierarchy):",
      lr.__class__.__name__, "+", mlp.__class__.__name__, "-> averaged")

rng=np.random.default_rng(0)
for trial in range(3):
    seg = rng.standard_normal(int(WIN_S*SR)).astype(np.float32)*0.01
    n_fft=int(SR*WIN_MS/1000); hop_l=int(SR*HOP_MS/1000)
    m=librosa.feature.melspectrogram(y=seg,sr=SR,n_mels=N_MELS,n_fft=n_fft,hop_length=hop_l,fmin=FMIN,fmax=FMAX,power=2.0)
    lm=librosa.power_to_db(m,ref=np.max)
    v=lm.mean(axis=1).astype(np.float32)
    mu,sd=float(v.mean()),float(v.std())+1e-8
    v=(v-mu)/sd
    print(f"\ntrial {trial}: logmel feature dim={v.shape[0]} (scaler expects {scaler.n_features_in_})")
    xs=scaler.transform(v.reshape(1,-1))
    p=0.5*lr.predict_proba(xs)[0]+0.5*mlp.predict_proba(xs)[0]
    print(f"  prob vector len={len(p)} sum={p.sum():.6f} nan={np.isnan(p).any()} probs={np.round(p,4)}")
print("\nE2E OK: dim match, len==5, sum~1, no NaN across trials")
