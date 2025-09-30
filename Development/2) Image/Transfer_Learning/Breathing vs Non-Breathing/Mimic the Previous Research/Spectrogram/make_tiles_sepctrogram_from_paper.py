#!/usr/bin/env python3
# make_breathing_tiles.py
# 논문식 전처리(HPF + Wavelet denoise) 후 2s 윈도우/0.5s hop 타일 생성
# 라벨: midpoint + 경계 마진 드롭, 멜 스펙트로그램 224x224 저장(오토크롭)

from pathlib import Path
import json, csv
import numpy as np
import librosa, matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import pywt

# ====== 경로/설정 (필요시 수정) ======
JSON_FILE = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/breathing_nonbreathing_intervals.json")
AUDIO_DIR = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/RAW sound_ML test sound list")
OUT_DIR   = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Development/9.2) Reference/BreathingTiles_mel_paper")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST = OUT_DIR / "manifest.csv"

# ====== 타일/라벨 파라미터 ======
SR = 4000
WIN_SEC = 2.0          # 논문 윈도우
HOP_SEC = 0.5          # 75% overlap
MARGIN_SEC = 0.30      # 경계 ±margin 내 midpoint 드롭
START_OFFSET_SEC = 0.0

# ====== 전처리(논문식) ======
# HPF: 3차 Butterworth, fc=4 Hz
HPF_ON = True
HPF_ORDER = 3
HPF_CUTOFF_HZ = 4.0

# Wavelet denoise: soft-threshold
WAV_DENOISE_ON = True
WAV_WAVELET = "db8"
WAV_LEVEL   = 6           # 5~7 권장, 신호/샘플레이트 고려
WAV_METHOD  = "bayes"     # "bayes" | "sure"
WAV_MODE    = "soft"      # "soft" | "hard"

# ====== 멜 스펙트로그램 설정 ======
N_FFT=1024; HOP=256; N_MELS=64
FMAX = int(0.95 * (SR/2))  # Nyquist 이내 (경고 방지)
# y축 AUTO-CROP
AUTO_CROP   = True
CROP_DB_REL = -40.0
MIN_SHOW_BINS = 32
SMOOTH_BINS  = 1
# 표시 스케일
VMIN_PCT=5.0; VMAX_PCT=99.5

# ====== 타일 상한 (중복 억제) ======
MAX_TILES_PER_FILE_PER_CLASS = 60   # 2s로 줄어드는 양 고려, 필요시 조정
GLOBAL_MAX_TILES_PER_CLASS   = None # None=무제한

# ====== JSON pass 필터 ======
PASS_KEYS = ["strict_pass", "passed", "pass", "valid", "clean_ok"]

# ====== 유틸 ======
def load_json(p: Path): return json.loads(p.read_text())

def record_passes(rec: dict) -> bool:
    for k in PASS_KEYS:
        if k in rec:
            return bool(rec[k])
    return len(rec.get("breathing", [])) > 0

def load_audio(p: Path, sr=SR):
    y, _ = librosa.load(str(p), sr=sr, mono=True)
    return y

def patient_id_from_name(name: str):
    # 예: "KP005_WWS_1" -> "KP005"
    return name.split("_")[0]

# ---- 전처리: HPF ----
def butter_highpass(y, sr=SR, cutoff=HPF_CUTOFF_HZ, order=HPF_ORDER):
    nyq = 0.5 * sr
    norm = cutoff / nyq
    b,a = butter(order, norm, btype="highpass")
    return filtfilt(b, a, y).astype(np.float32)

# ---- 전처리: Wavelet denoise ----
def wavelet_denoise(y, wavelet=WAV_WAVELET, level=WAV_LEVEL, method=WAV_METHOD, mode=WAV_MODE):
    # 분해
    coeffs = pywt.wavedec(y, wavelet=wavelet, level=level)
    cA, cDs = coeffs[0], coeffs[1:]

    # 노이즈 추정(최고주파 세부계수의 MAD)
    sigma = np.median(np.abs(cDs[-1])) / 0.6745 + 1e-12
    denoised_coeffs = [cA]

    for i, cD in enumerate(cDs, start=1):
        if method == "bayes":
            # BayesShrink
            var_y = np.var(cD)
            thresh = sigma**2 / (np.sqrt(max(var_y, 1e-12)) + 1e-12)
        elif method == "sure":
            # SUREshrink (간단 구현: universal vs adaptive 중간선)
            n = cD.size
            thr_univ = sigma * np.sqrt(2*np.log(n))
            # 간략 SURE-like: energy-based scale
            thr_sure = np.minimum(thr_univ, np.sqrt(max(np.mean(cD**2) - sigma**2, 0)))
            thresh = float(thr_sure)
        else:
            # fallback: universal threshold
            n = cD.size
            thresh = sigma * np.sqrt(2*np.log(n))

        den_cD = pywt.threshold(cD, value=thresh, mode=mode)
        denoised_coeffs.append(den_cD)

    return pywt.waverec(denoised_coeffs, wavelet=wavelet).astype(np.float32)

# ---- 멜/이미지 ----
def mel_db(seg, sr=SR):
    S = librosa.feature.melspectrogram(
        y=seg, sr=sr, n_fft=N_FFT, hop_length=HOP,
        n_mels=N_MELS, fmax=FMAX, power=2.0
    )
    return librosa.power_to_db(S, ref=np.max)

def auto_crop_rows(S_db: np.ndarray) -> int:
    rows = S_db.shape[0]
    if not AUTO_CROP:
        return rows
    row_max = S_db.max(axis=1)
    if SMOOTH_BINS > 1:
        k = int(SMOOTH_BINS)
        ker = np.ones(k, dtype=float) / k
        row_max = np.convolve(row_max, ker, mode="same")
    gmax = float(row_max.max())
    keep = np.where(row_max >= (gmax + CROP_DB_REL))[0]
    if keep.size > 0:
        last_idx = int(keep.max())
        return max(last_idx + 1, MIN_SHOW_BINS)
    return MIN_SHOW_BINS

def save_img(arr, out_png, vmin, vmax):
    plt.figure(figsize=(4,4), dpi=224/4)  # ~224x224
    plt.imshow(arr, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="plasma")
    plt.axis("off"); plt.tight_layout(pad=0)
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0)
    plt.close()

# ---- 라벨링 보조 ----
def merge_intervals(ints):
    if not ints: return []
    ints = sorted(ints)
    merged = [list(ints[0])]
    for s,e in ints[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s,e])
    return [(s,e) for s,e in merged]

def in_interval(t, intervals):
    for s,e in intervals:
        if s <= t <= e:
            return True
    return False

def dist_to_nearest_boundary(t, intervals):
    if not intervals: return float("inf")
    d = float("inf")
    for s,e in intervals:
        d = min(d, abs(t - s), abs(t - e))
        if s <= t <= e:
            d = min(d, t - s, e - t)
    return d

# ====== 메인 ======
def main():
    meta = load_json(JSON_FILE)
    rows = []
    per_file_counts = {}
    global_counts = {"Breathing":0, "NonBreathing":0}
    dropped_margin = 0

    win_samps = int(WIN_SEC*SR)
    hop_samps = int(HOP_SEC*SR)
    start_off = int(START_OFFSET_SEC*SR)

    for fname, rec in meta.items():
        if not record_passes(rec):
            continue

        wav = AUDIO_DIR / f"{fname}.wav"
        if not wav.exists():
            continue

        # --- 로드 & 전처리 ---
        y = load_audio(wav).astype(np.float32)
        if HPF_ON:
            y = butter_highpass(y, sr=SR, cutoff=HPF_CUTOFF_HZ, order=HPF_ORDER)
        if WAV_DENOISE_ON:
            y = wavelet_denoise(y, wavelet=WAV_WAVELET, level=WAV_LEVEL, method=WAV_METHOD, mode=WAV_MODE)

        pid = patient_id_from_name(fname)
        breathing = merge_intervals(rec.get("breathing", []))

        # --- 윈도우링 ---
        start0 = start_off
        while start0 + win_samps <= len(y):
            end0 = start0 + win_samps
            t_mid = (start0 + end0) / 2 / SR

            # midpoint + 경계 마진
            if dist_to_nearest_boundary(t_mid, breathing) < MARGIN_SEC:
                dropped_margin += 1
                start0 += hop_samps
                continue

            label = "Breathing" if in_interval(t_mid, breathing) else "NonBreathing"

            key = (fname, label)
            per_file_counts.setdefault(key, 0)
            if MAX_TILES_PER_FILE_PER_CLASS is not None and per_file_counts[key] >= MAX_TILES_PER_FILE_PER_CLASS:
                start0 += hop_samps; continue
            if GLOBAL_MAX_TILES_PER_CLASS is not None and global_counts[label] >= GLOBAL_MAX_TILES_PER_CLASS:
                start0 += hop_samps; continue

            seg = y[start0:end0]
            Sdb = mel_db(seg)
            show_rows = auto_crop_rows(Sdb)
            Sshow = Sdb[:show_rows, :]
            vmin = float(np.percentile(Sshow, VMIN_PCT))
            vmax = float(np.percentile(Sshow, VMAX_PCT))

            t0 = start0 / SR
            out_png = OUT_DIR / f"{fname}_{label}_{t0:.2f}s.png"
            save_img(Sshow, out_png, vmin, vmax)

            rows.append([str(out_png), label, pid, fname, t0, t0+WIN_SEC])
            per_file_counts[key] += 1
            global_counts[label] += 1

            start0 += hop_samps

    # 저장
    with MANIFEST.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path","label","patient_id","orig_file","t_start","t_end"])
        w.writerows(rows)

    # 로그
    n_b = sum(1 for r in rows if r[1]=="Breathing")
    n_n = sum(1 for r in rows if r[1]=="NonBreathing")
    print(f"✅ Saved tiles: Breathing={n_b}, NonBreathing={n_n}, Dropped(margin)={dropped_margin}")
    print(f"🗂️ Manifest: {MANIFEST}")

if __name__ == "__main__":
    main()
