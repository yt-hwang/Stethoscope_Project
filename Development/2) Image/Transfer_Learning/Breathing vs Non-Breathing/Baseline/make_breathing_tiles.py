#!/usr/bin/env python3
# make_breathing_tiles.py
# Breathing vs NonBreathing 이진 타일 생성 (midpoint + 경계 마진 드롭)
# ▲ y-axis AUTO_CROP 추가: 상단 빈 대역 제거

from pathlib import Path
import json, csv
import numpy as np
import librosa, matplotlib.pyplot as plt

# ====== 경로/설정 (필요시 수정) ======
JSON_FILE = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/breathing_nonbreathing_intervals.json")
AUDIO_DIR = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/RAW sound_ML test sound list")
OUT_DIR   = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Development/9.2) Reference/BreathingTiles_mel")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST = OUT_DIR / "manifest.csv"

# 오디오/타일
SR = 4000
TILE_SEC = 0.5       # 0.25/0.5/1.0 등 실험 가능
HOP_SEC  = 0.25      # 50% overlap
START_OFFSET_SEC = 0.0

# 멜스펙 파라미터
N_FFT=1024; HOP=256; N_MELS=64; FMAX=4000
# y축 AUTO-CROP 파라미터
AUTO_CROP   = True     # ← 상단 빈 대역 제거
CROP_DB_REL = -40.0    # 전역 최대 대비 -40 dB 이내 행만 유지
MIN_SHOW_BINS = 32     # 최소 보존 행 수
SMOOTH_BINS  = 1       # 1=off, >1일 때 행별 최대치 스무딩

# 표시 스케일
VMIN_PCT=5.0; VMAX_PCT=99.5

# 라벨링 규칙
MARGIN_SEC = 0.2     # 경계 ±margin 내 midpoint면 드롭
MAX_TILES_PER_FILE_PER_CLASS = 30
GLOBAL_MAX_TILES_PER_CLASS   = None

# JSON 패스 필터
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
    return name.split("_")[0]

def mel_db(seg, sr=SR):
    S = librosa.feature.melspectrogram(
        y=seg, sr=sr, n_fft=N_FFT, hop_length=HOP,
        n_mels=N_MELS, fmax=FMAX, power=2.0
    )
    return librosa.power_to_db(S, ref=np.max)  # [mel_bins, frames], dB

def auto_crop_rows(S_db: np.ndarray) -> int:
    """y축(행) 오토크롭: 전역 최대 대비 CROP_DB_REL 이상인 행만 남김(아래쪽부터 keep)."""
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

    tile_samps = int(TILE_SEC*SR)
    hop_samps  = int(HOP_SEC*SR)
    start_off  = int(START_OFFSET_SEC*SR)

    for fname, rec in meta.items():
        if not record_passes(rec):
            continue
        wav = AUDIO_DIR / f"{fname}.wav"
        if not wav.exists():
            continue

        y = load_audio(wav)
        pid = patient_id_from_name(fname)

        breathing = merge_intervals(rec.get("breathing", []))

        start0 = start_off
        while start0 + tile_samps <= len(y):
            end0 = start0 + tile_samps
            t_mid = (start0 + end0) / 2 / SR

            # midpoint + 경계 마진 드롭
            if dist_to_nearest_boundary(t_mid, breathing) < MARGIN_SEC:
                dropped_margin += 1
                start0 += hop_samps
                continue

            label = "Breathing" if in_interval(t_mid, breathing) else "NonBreathing"

            # 상한 체크
            key = (fname, label)
            per_file_counts.setdefault(key, 0)
            if MAX_TILES_PER_FILE_PER_CLASS is not None and per_file_counts[key] >= MAX_TILES_PER_FILE_PER_CLASS:
                start0 += hop_samps; continue
            if GLOBAL_MAX_TILES_PER_CLASS is not None and global_counts[label] >= GLOBAL_MAX_TILES_PER_CLASS:
                start0 += hop_samps; continue

            # 멜 dB + y-오토크롭
            seg = y[start0:end0]
            Sdb = mel_db(seg)
            show_rows = auto_crop_rows(Sdb)
            Sshow = Sdb[:show_rows, :]

            # 크롭 후 vmin/vmax 재산정
            vmin = float(np.percentile(Sshow, VMIN_PCT))
            vmax = float(np.percentile(Sshow, VMAX_PCT))

            t0 = start0 / SR
            out_png = OUT_DIR / f"{fname}_{label}_{t0:.2f}s.png"
            save_img(Sshow, out_png, vmin, vmax)

            rows.append([str(out_png), label, pid, fname, t0, t0+TILE_SEC])
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
