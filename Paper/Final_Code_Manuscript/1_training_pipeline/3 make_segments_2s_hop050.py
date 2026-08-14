# -*- coding: utf-8 -*-
# make_segments_2s_hop050.py
# (변경 요약)
#  - '라벨-순수 세그먼트 재슬라이스' 단계 호출 제거(코드는 남겨두되 main에서 미사용)
#  - 커버리지 기반 2s/0.5s 슬라이딩만 수행
#  - 패딩 제거: 2초 미만 마지막 조각은 저장하지 않음
#  - 커버리지 임계(MIN_BREATH_SEC) 규칙은 기존 그대로 사용

from pathlib import Path
import re
import json
import csv
import numpy as np
import soundfile as sf
import librosa

# ====== 절대경로 (생략/축약 금지) ======
DEPLOY_ROOT = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Deployment/1 Final Pipeline/5 Realtime Pipeline_Final")

# (이전 산출물) 파일명에서 t0-t1 역추정을 위한 소스
RAW_SEGMENTS_SRC = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Deployment/1 Final Pipeline/5 Realtime Pipeline_Final/Output/Segments_from_JSON")

# (최종 출력) 2s/0.5s 윈도우
OUT_DIR = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Deployment/1 Final Pipeline/5 Realtime Pipeline_Final/Output/Segments_2s_hop500ms")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 커버리지 계산에 사용할 원본 WAV 루트
WAV_ROOT = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/RAW sound_ML test sound list")

# (커버리지 라벨링 출력 루트)
AUG_OUT_DIR = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Deployment/1 Final Pipeline/5 Realtime Pipeline_Final/Output/Augmented_Windows_by_Coverage")
AUG_OUT_DIR.mkdir(parents=True, exist_ok=True)

# ====== 파라미터 ======
SR = 16000
WIN = 2.0
HOP = 0.5
MIN_RMS = 5e-4   # 0/None 이면 무음 필터 off

# 커버리지 규칙
MIN_BREATH_SEC = 1.3   # 윈도우 내 호흡 누적 길이 임계값 (x 값)
BREATHING = {"Healthy", "Crackle", "Wheezing", "Rhonchi"}

FOLDER_NAME_BY_LABEL_AUG = {
    "Crackle": "Crackle_window",
    "Healthy": "Healthy_window",
    "Wheezing": "Wheezing_window",
    "Rhonchi": "Rhonchi_window",
    "Non-breathing": "Nonbreathing_window",
}

_TSPAN_RE = re.compile(r"_([0-9]+(?:\.[0-9]+)?)-([0-9]+(?:\.[0-9]+)?)$")

def parse_tspan_from_stem(stem: str):
    m = _TSPAN_RE.search(stem)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))

def resample_if_needed(x, sr):
    if sr == SR:
        return (x if x.ndim == 1 else np.mean(x, axis=1)).astype(np.float32)
    x = (x if x.ndim == 1 else np.mean(x, axis=1)).astype(np.float32)
    return librosa.resample(x, orig_sr=sr, target_sr=SR, res_type="kaiser_best")

def pass_min_rms(x):
    if not MIN_RMS:
        return True
    rms = float(np.sqrt(np.mean(np.square(x))))
    return rms >= MIN_RMS

def save_seg(seg, sr, out_dir: Path, out_name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(out_dir / out_name, seg.astype(np.float32), sr)

def sec_overlap(a0, a1, b0, b1):
    lo = max(a0, b0)
    hi = min(a1, b1)
    return max(0.0, hi - lo)

# ---------- (참조용, 미사용) 라벨-순수 재슬라이스 ----------
def make_label_pure_segments():
    """
    (유지하되 main에서 호출하지 않음)
    RAW_SEGMENTS_SRC/<Label>/*.wav 를 2s/0.5s로 재슬라이스 → OUT_DIR/<Label>/*.wav
    """
    meta = []
    for label_dir in sorted(RAW_SEGMENTS_SRC.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        target_dir = OUT_DIR / label
        target_dir.mkdir(parents=True, exist_ok=True)

        for wav_path in sorted(label_dir.rglob("*.wav")):
            x, sr = sf.read(wav_path)
            x = resample_if_needed(x, sr)
            n_per = int(WIN * SR)
            hop = int(HOP * SR)
            N = len(x)
            for i0 in range(0, max(0, N - n_per + 1), hop):
                i1 = i0 + n_per
                if i1 > N:
                    break  # 패딩 금지
                chunk = x[i0:i1]
                if pass_min_rms(chunk):
                    out_name = f"{wav_path.stem}_{i0/SR:.2f}-{i1/SR:.2f}.wav"
                    save_seg(chunk, SR, target_dir, out_name)
                    meta.append([wav_path.name, f"{i0/SR:.3f}", f"{i1/SR:.3f}", label, str(target_dir / out_name)])

    meta_csv = OUT_DIR / "metadata_2s_hop500ms.csv"
    with open(meta_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["src_file", "t0", "t1", "label", "dest_wav"])
        w.writerows(meta)
    print(f"[DONE] pure segments: {len(meta)}  -> {OUT_DIR}")
    print(f"[DONE] metadata: {meta_csv}")

# ---------- 커버리지 기반 라벨 선택 ----------
def choose_label_by_coverage(t0, t1, intervals):
    by = {}
    for seg in intervals:
        s, e = float(seg["start"]), float(seg["end"])
        lab = str(seg["label"])
        ov = sec_overlap(t0, t1, s, e)
        if ov > 0:
            by[lab] = by.get(lab, 0.0) + ov
    # 호흡 클래스 중 최댓값
    best_lab, best_sec = None, 0.0
    for lab in BREATHING:
        sec = by.get(lab, 0.0)
        if sec > best_sec:
            best_sec, best_lab = sec, lab
    if best_lab is not None and best_sec >= MIN_BREATH_SEC:
        return best_lab, best_sec, by
    return "Non-breathing", by.get("Non-breathing", 0.0), by

def build_intervals_map_from_raw_segments():
    _TSPAN_RE = re.compile(r"_([0-9]+(?:\.[0-9]+)?)-([0-9]+(?:\.[0-9]+)?)$")
    _TAIL_RE  = re.compile(r"_(?:NB|B)\d+$")
    intervals_map = {}
    for label_dir in sorted(RAW_SEGMENTS_SRC.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for wav_path in sorted(label_dir.rglob("*.wav")):
            stem = wav_path.stem
            m_span = _TSPAN_RE.search(stem)
            base_stem = stem[:m_span.start()] if m_span else stem
            base_stem = _TAIL_RE.sub("", base_stem)
            orig_name = base_stem + ".wav"
            if not m_span:
                continue
            t0 = float(m_span.group(1)); t1 = float(m_span.group(2))
            intervals_map.setdefault(orig_name, []).append({"start": t0, "end": t1, "label": label})
    for k in intervals_map.keys():
        intervals_map[k].sort(key=lambda z: (z["start"], z["end"]))
    return intervals_map

def make_wav_name_map_exact(root: Path):
    name_map = {}
    for p in root.rglob("*.wav"):
        name_map[p.name.lower()] = p
    for p in root.rglob("*.WAV"):
        nm = p.name.lower()
        if nm not in name_map:
            name_map[nm] = p
    return name_map

def make_augmented_windows_from_intervals(intervals_map):
    """
    원본 WAV 전구간을 2s/0.5s로 슬라이딩(패딩 없이)하고
    커버리지 규칙으로 라벨링하여 AUG_OUT_DIR/*_window/*.wav 저장
    """
    meta = []
    wav_name_map = make_wav_name_map_exact(WAV_ROOT)
    print(f"[CHK] WAV_ROOT={WAV_ROOT}  found_originals={len(wav_name_map)}")

    matched = 0
    for orig_name, intervals in intervals_map.items():
        p = wav_name_map.get(orig_name.lower(), None)
        if p is None:
            print(f"[WRN] original WAV not found for {orig_name}")
            continue
        matched += 1
        x, sr = sf.read(p)
        x = resample_if_needed(x, sr)
        n = len(x)
        n_per = int(WIN * SR)
        hop = int(HOP * SR)

        # 패딩 제거: 정확히 2초 길이만
        for i0 in range(0, max(0, n - n_per + 1), hop):
            i1 = i0 + n_per
            if i1 > n:
                break
            t0, t1 = i0 / SR, i1 / SR
            chunk = x[i0:i1]
            lab, best_sec, detail = choose_label_by_coverage(t0, t1, intervals)
            folder = FOLDER_NAME_BY_LABEL_AUG[lab]
            out_dir = AUG_OUT_DIR / folder
            out_name = f"{Path(orig_name).stem}_{t0:.2f}-{t1:.2f}.wav"
            if pass_min_rms(chunk):
                save_seg(chunk, SR, out_dir, out_name)
                meta.append([
                    orig_name, f"{t0:.3f}", f"{t1:.3f}", lab,
                    f"{best_sec:.3f}", json.dumps(detail, ensure_ascii=False),
                    str(out_dir / out_name)
                ])

    meta_csv = AUG_OUT_DIR / "metadata_windows.csv"
    with open(meta_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["src_file","t0","t1","label","best_overlap_sec","overlap_detail_json","dest_wav"])
        w.writerows(meta)
    print(f"[INFO] coverage originals matched: {matched}/{len(intervals_map)}")
    print(f"[DONE] augmented windows: {len(meta)} -> {AUG_OUT_DIR}")
    print(f"[DONE] metadata: {meta_csv}")

def main():
    # (중요) 라벨-순수 재슬라이스는 호출하지 않음 → 커버리지 기반만 수행
    intervals_map = build_intervals_map_from_raw_segments()
    make_augmented_windows_from_intervals(intervals_map)

if __name__ == "__main__":
    main()
