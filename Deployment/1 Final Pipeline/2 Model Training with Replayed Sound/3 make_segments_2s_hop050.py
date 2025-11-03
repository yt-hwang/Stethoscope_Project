# -*- coding: utf-8 -*-
# make_segments_2s_hop050.py
# (원래 기능) 라벨-순수 구간에서 2s/0.5s 세그먼트 생성
# (옵션) Segments_from_JSON 파일명(t0-t1)으로 intervals 역추정 →
#        RAW 원본 WAV 전구간 2s/0.5s 슬라이딩 → coverage 규칙(≥0.7s)으로 라벨링
#        (탐색 강화 없음: 지정 루트에서 파일명 완전일치로만 매칭)

from pathlib import Path
import re
import json
import csv
import numpy as np
import soundfile as sf
import librosa

# ====== 절대경로 (생략/축약 금지) ======
DEPLOY_ROOT = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Deployment/1 Final Pipeline/2 Model Training with Replayed Sound")

# 라벨-순수 세그먼트의 원천(네 기존 파이프라인 산출물)
RAW_SEGMENTS_SRC = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Deployment/1 Final Pipeline/2 Model Training with Replayed Sound/Output/Segments_from_JSON")

# (라벨-순수) 출력 루트
OUT_DIR = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Deployment/1 Final Pipeline/2 Model Training with Replayed Sound/Output/Segments_2s_hop500ms")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (옵션) coverage 증강: **2번 스크립트가 쓰는 RAW 원본 WAV 루트 그대로 사용**
WAV_ROOT = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/RAW sound_ML test sound list")

# (옵션) coverage 증강 출력 루트
AUG_OUT_DIR = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Deployment/1 Final Pipeline/2 Model Training with Replayed Sound/Output/Augmented_Windows_by_Coverage")
AUG_OUT_DIR.mkdir(parents=True, exist_ok=True)

# ====== 파라미터 ======
SR = 16000
WIN = 2.0
HOP = 0.5
MIN_RMS = 5e-4   # 0/None 이면 무음 필터 off

# (옵션) coverage 규칙
AUGMENT_WINDOWS_BY_COVERAGE = True
MIN_BREATH_SEC = 1.3
BREATHING = {"Healthy", "Crackle", "Wheezing", "Rhonchi"}

FOLDER_NAME_BY_LABEL_AUG = {
    "Crackle": "Crackle_window",
    "Healthy": "Healthy_window",
    "Wheezing": "Wheezing_window",
    "Rhonchi": "Rhonchi_window",
    "Non-breathing": "Nonbreathing_window",
}

# ---------- 유틸 ----------
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

# ---------- 1) (원래) 라벨-순수 세그먼트 ----------
def make_label_pure_segments():
    """
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
                    break
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

# ---------- 2) (옵션) coverage 증강: Segments_from_JSON에서 intervals 역추정 ----------
def build_intervals_map_from_raw_segments():
    """
    RAW_SEGMENTS_SRC/<Label>/*.wav 의 파일명에 포함된 _t0-t1(초)로
    { '원본파일명.wav': [ {'start':t0,'end':t1,'label':Label}, ...] } 맵을 생성

    규칙:
      - 세그먼트 stem의 맨 끝에 붙는 시간범위(_t0-t1) 제거
      - 그 다음 맨 끝에 붙는 세그먼트 꼬리표만 제거:
          *_B###   또는  *_NB###      (예: KP004_WWS_B001 -> KP004_WWS)
          *_1_B### 등은 *_1 유지 후 _B###만 제거 (예: KP017_WWS_1_B000 -> KP017_WWS_1)
      - 최종적으로 ".wav" 확장자 부여
    """
    _TSPAN_RE = re.compile(r"_([0-9]+(?:\.[0-9]+)?)-([0-9]+(?:\.[0-9]+)?)$")   # _t0-t1
    _TAIL_RE  = re.compile(r"_(?:NB|B)\d+$")                                   # _B### or _NB###

    intervals_map = {}
    for label_dir in sorted(RAW_SEGMENTS_SRC.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for wav_path in sorted(label_dir.rglob("*.wav")):
            stem = wav_path.stem

            # 1) 끝의 _t0-t1 제거
            m_span = _TSPAN_RE.search(stem)
            base_stem = stem[:m_span.start()] if m_span else stem

            # 2) 끝의 세그먼트 꼬리표(_B### / _NB###)만 제거 (다른 접미사 예: _1, _2는 보존)
            base_stem = _TAIL_RE.sub("", base_stem)

            # 3) 원본 파일명 생성
            orig_name = base_stem + ".wav"

            # 4) interval 기록
            if m_span:
                t0 = float(m_span.group(1))
                t1 = float(m_span.group(2))
            else:
                # 안전장치: 시간 구간이 없으면 스킵
                continue

            intervals_map.setdefault(orig_name, []).append({
                "start": t0, "end": t1, "label": label
            })

    # 시작시간 기준 정렬
    for k in intervals_map.keys():
        intervals_map[k].sort(key=lambda z: (z["start"], z["end"]))
    return intervals_map


def make_wav_name_map_exact(root: Path):
    """
    root 하위에서 .wav/.WAV 파일을 스캔하여
    '파일명(대소문자 무시)' → Path 딕셔너리 생성. (파일명 완전일치만 허용)
    """
    name_map = {}
    for p in root.rglob("*.wav"):
        name_map[p.name.lower()] = p
    for p in root.rglob("*.WAV"):
        nm = p.name.lower()
        if nm not in name_map:
            name_map[nm] = p
    return name_map

def choose_label_by_coverage(t0, t1, intervals):
    by = {}
    for seg in intervals:
        s, e = float(seg["start"]), float(seg["end"])
        lab = str(seg["label"])
        ov = sec_overlap(t0, t1, s, e)
        if ov > 0:
            by[lab] = by.get(lab, 0.0) + ov
    best_lab, best_sec = None, 0.0
    for lab in BREATHING:
        sec = by.get(lab, 0.0)
        if sec > best_sec:
            best_sec, best_lab = sec, lab
    if best_lab is not None and best_sec >= MIN_BREATH_SEC:
        return best_lab, best_sec, by
    return "Non-breathing", by.get("Non-breathing", 0.0), by

def make_augmented_windows_from_intervals(intervals_map):
    """
    RAW 원본(WAV_ROOT)에서 파일명 완전일치(대소문자 무시)로만 매칭하여:
      - 전구간 2s/0.5s 슬라이딩
      - coverage 규칙으로 라벨링
      - AUG_OUT_DIR/*_window/*.wav 저장
    """
    meta = []
    wav_name_map = make_wav_name_map_exact(WAV_ROOT)
    print(f"[CHK] WAV_ROOT={WAV_ROOT}  found_originals={len(wav_name_map)}")

    # 샘플 키 5개로 빠른 대조
    example_keys = list(intervals_map.keys())[:5]
    hit = [k for k in example_keys if k.lower() in wav_name_map]
    miss = [k for k in example_keys if k.lower() not in wav_name_map]
    print(f"[DBG] sample expected originals (first5): {example_keys}")
    print(f"[DBG] sample hit: {hit}")
    print(f"[DBG] sample miss: {miss}")

    if len(wav_name_map) == 0:
        print("[ERR] No original WAVs under WAV_ROOT (including subfolders). Check WAV_ROOT path.")
        return

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
        total = n / SR
        t = 0.0
        while t < total:
            t0, t1 = t, t + WIN
            i0 = int(round(t0 * SR))
            i1 = int(round(t1 * SR))
            if i0 >= n:
                break
            chunk = x[i0:min(i1, n)]
            if len(chunk) < int(WIN * SR):
                pad = np.zeros(int(WIN * SR) - len(chunk), dtype=np.float32)
                chunk = np.concatenate([chunk, pad], axis=0)

            lab, best_sec, detail = choose_label_by_coverage(t0, t1, intervals)
            folder = FOLDER_NAME_BY_LABEL_AUG[lab]
            out_dir = AUG_OUT_DIR / folder
            out_dir.mkdir(parents=True, exist_ok=True)
            out_name = f"{Path(orig_name).stem}_{t0:.2f}-{t1:.2f}.wav"

            if pass_min_rms(chunk):
                save_seg(chunk, SR, out_dir, out_name)
                meta.append([
                    orig_name, f"{t0:.3f}", f"{t1:.3f}", lab,
                    f"{best_sec:.3f}", json.dumps(detail, ensure_ascii=False),
                    str(out_dir / out_name)
                ])

            t += HOP

    meta_csv = AUG_OUT_DIR / "metadata_windows.csv"
    with open(meta_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["src_file","t0","t1","label","best_overlap_sec","overlap_detail_json","dest_wav"])
        w.writerows(meta)

    print(f"[INFO] coverage originals matched: {matched}/{len(intervals_map)}")
    print(f"[DONE] augmented windows: {len(meta)} -> {AUG_OUT_DIR}")
    print(f"[DONE] metadata: {meta_csv}")

def main():
    # 1) (원래) 라벨-순수 세그먼트
    make_label_pure_segments()

    # 2) (옵션) coverage 증강
    if AUGMENT_WINDOWS_BY_COVERAGE:
        intervals_map = build_intervals_map_from_raw_segments()
        make_augmented_windows_from_intervals(intervals_map)

if __name__ == "__main__":
    main()
