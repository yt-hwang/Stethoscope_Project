import os
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from joblib import load as joblib_load
import matplotlib.pyplot as plt

# =========================================================
# path setting
# !!!!!!!!!!!! 여기만 경로 설정 바꿔주세요 !!!!!!!!!!!!
# =========================================================
AUDIO_PATH = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/RAW sound_ML test sound list/H002.wav")
MODEL_DIR  = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Deployment/1 Final Pipeline/5 Realtime Pipeline_Final/model/run_20251107_194938")
OUTPUT_DIR = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Deployment/1 Final Pipeline/6 Final Deployment to Dr.Oh/")

audio_stem = AUDIO_PATH.stem           # e.g., "H002"
csv_path = OUTPUT_DIR / f"{audio_stem}_softmax_results.csv"

# =========================================================
# model/feature extraction setting
# same as RealtimeModelAdapter
# =========================================================
SR = 16000          # RealtimeModelAdapter.SR
WIN_S = 2.0         # 2s window
HOP_S = 0.5         # 1s hop
N_MELS = 64
FMIN, FMAX = 50, 7900
WIN_MS, HOP_MS = 64, 32

UI_ORDER = ['Healthy', 'Crackle', 'Rhonchi', 'Wheezing', 'Non-breathing']  # internal class names
CSV_CLASSES = ['Healthy', 'Crackle', 'Rhonchi', 'Wheezing', 'Non-breathing']  # names for CSV/graph

def load_models_and_reorder(model_dir: Path):
    """load scaler, LR, MLP, threshold, reorder index same as RealtimeModelAdapter"""
    scaler = joblib_load(model_dir / "scaler.pkl")
    model_lr = joblib_load(model_dir / "model_lr.pkl")
    model_mlp = joblib_load(model_dir / "model_mlp.pkl")

    with open(model_dir / "thresholds.json", "r", encoding="utf-8") as f:
        th = json.load(f)
    local_names = list(th.get("class_names", []))
    thresholds_local = np.array(th.get("thresholds", []), dtype=np.float32)

    reorder = []
    for c in UI_ORDER:
        key = c  # 'Non-breathing' included in UI_ORDER
        try:
            reorder.append(local_names.index(key))
        except ValueError:
            raise RuntimeError(f"class '{c}' not found in model classes {local_names}")

    reorder_local_to_ui = np.array(reorder, dtype=int)
    thresholds_ui = thresholds_local[reorder_local_to_ui]

    print(f"[MDL] local classes={local_names}")
    print(f"[MDL] UI_ORDER     ={UI_ORDER}")
    print(f"[MDL] reorder local->ui = {reorder_local_to_ui.tolist()}")
    print(f"[MDL] thresholds(ui)    = {thresholds_ui.tolist()}")

    return scaler, model_lr, model_mlp, reorder_local_to_ui, thresholds_ui


def extract_features(seg_16k: np.ndarray) -> np.ndarray:
    """
    2s segment → mel spectrogram → log-mel → time mean → normalize
    final feature shape: (64,)
    """
    n_fft = int(SR * WIN_MS / 1000)
    hop_l = int(SR * HOP_MS / 1000)

    m = librosa.feature.melspectrogram(
        y=seg_16k,
        sr=SR,
        n_mels=N_MELS,
        n_fft=n_fft,
        hop_length=hop_l,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    lm = librosa.power_to_db(m, ref=np.max)
    v = lm.mean(axis=1).astype(np.float32)  # (64,)
    mu, sd = float(v.mean()), float(v.std()) + 1e-8
    v = (v - mu) / sd
    return v


def run_offline_inference():
    # -------------------------------
    # 1) load models
    # -------------------------------
    print("[INFO] Loading models...")
    scaler, model_lr, model_mlp, reorder_local_to_ui, _ = load_models_and_reorder(MODEL_DIR)

    # -------------------------------
    # 2) load audio & convert to 16kHz
    # -------------------------------
    print(f"[INFO] Loading audio: {AUDIO_PATH}")
    if not AUDIO_PATH.exists():
        raise FileNotFoundError(f"Audio file not found: {AUDIO_PATH}")

    x, sr = sf.read(str(AUDIO_PATH))
    if x.ndim > 1:
        x = x.mean(axis=1)  # mono
    x = x.astype(np.float32)

    if sr != SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=SR, res_type="kaiser_best")
        print(f"[INFO] Resampled {sr} Hz → {SR} Hz, length={len(x)} samples")
    else:
        print(f"[INFO] Sampling rate already {SR} Hz, length={len(x)} samples")

    # -------------------------------
    # 3) 2s window / 1s hop sliding
    # -------------------------------
    win = int(WIN_S * SR)   # 32000 samples
    hop = int(HOP_S * SR)   # 16000 samples

    n_windows = 0
    probs_ui_list = []
    times = []

    pos = 0
    while True:
        start = pos
        end = start + win
        if end > len(x):
            break  # if remaining length is less than 2s, discard

        seg = x[start:end]  # shape: (win,)

        # extract features
        v = extract_features(seg)
        xs = scaler.transform(v.reshape(1, -1))

        # softmax prediction (LR + MLP average)
        p_lr = model_lr.predict_proba(xs)[0]
        p_mlp = model_mlp.predict_proba(xs)[0]
        p_loc = 0.5 * p_lr + 0.5 * p_mlp

        # reorder by UI_ORDER
        p_ui = p_loc[reorder_local_to_ui].astype(np.float32)  # order: UI_ORDER

        # time axis: center time of window (s)
        center_time = (start + end) / 2.0 / SR

        probs_ui_list.append(p_ui)
        times.append(center_time)
        n_windows += 1

        pos += hop  # 1s hop

    if n_windows == 0:
        print("[WARN] No full 2-second windows in audio.")
        return

    probs_ui_arr = np.stack(probs_ui_list, axis=0)  # (n_windows, 5)
    times = np.array(times)

    print(f"[INFO] Total windows: {n_windows}")
    print("[INFO] Example probs (first window, UI_ORDER):")
    print(UI_ORDER, probs_ui_arr[0])

    # -------------------------------
    # 4) save CSV
    # -------------------------------

    # index mapping for each class in UI_ORDER
    idx_healthy = UI_ORDER.index('Healthy')
    idx_crackle = UI_ORDER.index('Crackle')
    idx_rhonchi = UI_ORDER.index('Rhonchi')
    idx_wheeze  = UI_ORDER.index('Wheezing')
    idx_nb      = UI_ORDER.index('Non-breathing')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_header = ["window_idx", "time_s"] + CSV_CLASSES
    rows = []

    for i in range(n_windows):
        p = probs_ui_arr[i]
        row = [
            i,
            float(times[i]),
            float(p[idx_healthy]),
            float(p[idx_wheeze]),
            float(p[idx_crackle]),
            float(p[idx_rhonchi]),
            float(p[idx_nb]),
        ]
        rows.append(row)

    np.savetxt(
        str(csv_path),
        np.array(rows, dtype=np.float32),
        delimiter=",",
        header=",".join(csv_header),
        comments="",
    )

    print(f"[✓] CSV saved to: {csv_path}")



    # -------------------------------
    # 5) draw line graph
    # -------------------------------
    # times: (n_windows,)
    # probs_ui_arr: (n_windows, 5) in UI_ORDER
    # in graph, draw in CSV_CLASSES order, but different colors for each class
    #   Healthy, Wheezing, Crackle, Bronchi(Rhonchi), Non-breathing

    plt.figure(figsize=(10, 5))
    plt.plot(times, probs_ui_arr[:, idx_healthy], label="Healthy")
    plt.plot(times, probs_ui_arr[:, idx_wheeze],  label="Wheezing")
    plt.plot(times, probs_ui_arr[:, idx_crackle], label="Crackle")
    plt.plot(times, probs_ui_arr[:, idx_rhonchi], label="Bronchi")
    plt.plot(times, probs_ui_arr[:, idx_nb],      label="Non-breathing")

    plt.xlabel("Time (s)")
    plt.ylabel("Probability")
    plt.ylim(0.0, 1.0)
    plt.title("Softmax probabilities (2s window, 1s hop)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_offline_inference()
