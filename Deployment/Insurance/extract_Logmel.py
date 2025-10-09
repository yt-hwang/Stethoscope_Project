# extract_logmel.py
# -----------------------------------------------------------------------------
# 1) 1초 세그먼트 메타를 읽어 Log-Mel(64 mel) 특징을 산출
# 2) 샘플 단위 표준화 후, 길이차 보정을 위해 time축 최대 길이에 제로패딩
# 3) 결과를 D:\Stethoscope_Project\Deployment\features 하위에 저장(.npz + .json)
# -----------------------------------------------------------------------------

from pathlib import Path
import json
import numpy as np
import pandas as pd
import soundfile as sf
import librosa

# ==== 경로 설정 ====
DEPLOY_ROOT = Path(r"D:\Stethoscope_Project\Deployment")
META = DEPLOY_ROOT / r"data\Segments_1s_hop250ms\metadata_1s_hop250ms.csv"

FEATURE_DIR = DEPLOY_ROOT / "features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)
OUT = FEATURE_DIR / "features_1s_hop250ms.npz"
LABELS_JSON = FEATURE_DIR / "features_1s_hop250ms.labels.json"

# ==== 파라미터 ====
SR = 16000
N_MELS = 64
WIN_LEN = int(0.064 * SR)  # 64ms
HOP_LEN = int(0.032 * SR)  # 32ms
FMIN, FMAX = 50, 7900

df = pd.read_csv(META)
class_names = sorted(df["label"].unique())
cls2idx = {c: i for i, c in enumerate(class_names)}

Xs, ys = [], []
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

    # 샘플 단위 표준화
    m = logmel.mean(); s = logmel.std() + 1e-6
    logmel = (logmel - m) / s

    Xs.append(logmel.T)  # (time, mel)
    ys.append(y_idx)

if not Xs:
    raise RuntimeError("No features extracted. Check input paths.")

max_T = max(x.shape[0] for x in Xs)
F = Xs[0].shape[1]
X = np.zeros((len(Xs), max_T, F), dtype=np.float32)
for i, a in enumerate(Xs):
    t = min(max_T, a.shape[0])
    X[i, :t, :] = a[:t, :]
y = np.array(ys, dtype=np.int64)

np.savez_compressed(OUT, X=X, y=y, class_names=np.array(class_names))
with open(LABELS_JSON, "w", encoding="utf-8") as f:
    json.dump({"class_names": class_names}, f, ensure_ascii=False, indent=2)

print(f"[DONE] X shape: {X.shape}, y: {y.shape}")
print(f"[OUT ] feat  : {OUT}")
print(f"[OUT ] labels: {LABELS_JSON}")
