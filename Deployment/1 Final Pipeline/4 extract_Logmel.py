# extract_logmel.py
# -----------------------------------------------------------------------------
# - Input: D:\Stethoscope_Project\Deployment\data\Segments_2s_hop500ms\metadata_2s_hop500ms.csv
# - Output:
#    D:\Stethoscope_Project\Deployment\features\features_2s_hop500ms.npz
#        (X, y, class_names, ids, sources, patient_ids)
#    D:\Stethoscope_Project\Deployment\features\features_2s_hop500ms.labels.json
#    D:\Stethoscope_Project\Deployment\features\features_2s_hop500ms.index.csv
# -----------------------------------------------------------------------------

from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import soundfile as sf
import librosa

# ==== Path settings ====
DEPLOY_ROOT = Path(r"D:\Stethoscope_Project\Deployment")
META = DEPLOY_ROOT / r"data\Segments_2s_hop500ms\metadata_2s_hop500ms.csv"

FEATURE_DIR = DEPLOY_ROOT / "features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)
OUT = FEATURE_DIR / "features_2s_hop500ms.npz"
LABELS_JSON = FEATURE_DIR / "features_2s_hop500ms.labels.json"
INDEX_CSV = FEATURE_DIR / "features_2s_hop500ms.index.csv"

# ==== Parameters ====
SR = 16000
N_MELS = 64
WIN_LEN = int(0.064 * SR)  # 64 ms
HOP_LEN = int(0.032 * SR)  # 32 ms
FMIN, FMAX = 50, 7900

def extract_patient_id(name: str) -> str:
    """
    Extract patient ID (prefix) from file name.
    Supported patterns:
      - H001.wav, H002_...         -> 'H001', 'H002'
      - KP021_WWS_2.wav            -> 'KP021'
      - WEBSS-006 TP3_...          -> 'WEBSS-006'
    If not found, return '' (empty string)
    """
    base = Path(name).name
    stem = Path(base).stem
    # WEBSS-xxx
    m = re.match(r'^(WEBSS-\d+)', stem, flags=re.IGNORECASE)
    if m: return m.group(1)
    # H###
    m = re.match(r'^(H\d+)', stem, flags=re.IGNORECASE)
    if m: return m.group(1)
    # KP###
    m = re.match(r'^(KP\d+)', stem, flags=re.IGNORECASE)
    if m: return m.group(1)
    # Check first token
    tok = re.split(r'[_\- ]+', stem)[0]
    m = re.match(r'^(WEBSS-\d+|H\d+|KP\d+)$', tok, flags=re.IGNORECASE)
    if m: return m.group(1)
    return ""

df = pd.read_csv(META)
if df.empty:
    raise RuntimeError(f"Metadata is empty: {META}")

class_names = sorted(df["label"].unique())
cls2idx = {c: i for i, c in enumerate(class_names)}

Xs, ys, ids, sources, patient_ids = [], [], [], [], []
for _, row in df.iterrows():
    wav_path = row["wav_path"]
    y_idx = cls2idx[row["label"]]
    try:
        x, sr = sf.read(wav_path)
    except Exception as e:
        print(f"[WARN] read fail: {wav_path} -> {e}")
        continue

    if sr != SR:
        x = librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=SR)

    # log-mel
    S = librosa.feature.melspectrogram(
        y=x.astype(np.float32), sr=SR, n_fft=2048,
        hop_length=HOP_LEN, win_length=WIN_LEN,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0
    )
    logmel = librosa.power_to_db(S, ref=np.max)

    # Sample-wise standardization
    m = logmel.mean(); s = logmel.std() + 1e-6
    logmel = (logmel - m) / s

    Xs.append(logmel.T)  # (time, mel)
    ys.append(y_idx)
    ids.append(row["id"])
    src = row["source_file"]
    sources.append(src)
    pid = extract_patient_id(src)
    if not pid:
        # Safety check: If not found, use source_file stem (even at the file level)
        pid = Path(src).stem
    patient_ids.append(pid)

if not Xs:
    raise RuntimeError("No features extracted. Check input paths and metadata.")

max_T = max(a.shape[0] for a in Xs)
F = Xs[0].shape[1]
X = np.zeros((len(Xs), max_T, F), dtype=np.float32)
for i, a in enumerate(Xs):
    t = min(max_T, a.shape[0])
    X[i, :t, :] = a[:t, :]

y = np.array(ys, dtype=np.int64)
ids = np.array(ids)
sources = np.array(sources)
patient_ids = np.array(patient_ids)

np.savez_compressed(
    OUT, X=X, y=y, class_names=np.array(class_names),
    ids=ids, sources=sources, patient_ids=patient_ids
)
with open(LABELS_JSON, "w", encoding="utf-8") as f:
    json.dump({"class_names": class_names}, f, ensure_ascii=False, indent=2)

pd.DataFrame({
    "id": ids, "source_file": sources, "patient_id": patient_ids, "y": y
}).to_csv(INDEX_CSV, index=False)

print(f"[DONE] X: {X.shape}, y: {y.shape}")
print(f"[OUT ] feat  : {OUT}")
print(f"[OUT ] labels: {LABELS_JSON}")
print(f"[OUT ] index : {INDEX_CSV}")
