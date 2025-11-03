# -*- coding: utf-8 -*-
"""
make_synthetic_tests.py
- 다섯 클래스 폴더에서 무작위 구간을 잘라 30초짜리 합성 WAV 생성
- 합성 타임라인의 GT(클래스, 시작/종료초)와 원본 메타정보를 엑셀로 저장

고정 사양:
- 출력 샘플레이트: 16,000 Hz (mono)
- 최종 길이: 정확히 30.0 s
- 클래스 표기: {'Crackle','Healthy','Non-breathing','Rhonchi','Wheezing'}
- 엑셀 시트: "segments" (가독성 좋은 열 구조, 파일별/시간순 정렬)

필요 라이브러리: numpy, pandas, librosa, soundfile, openpyxl
"""

import os
import sys
import random
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa

# ─────────────────────────────────────────────────────────
# [하드코딩] 소스 폴더 (사용자가 지정한 경로 그대로)
# ─────────────────────────────────────────────────────────
SRC_DIRS = {
    "Crackle":      Path("/Users/yunhwang/Desktop/Stethoscope_Project/Deployment/1 Final Pipeline/2 Model Training with Replayed Sound/Output/Augmented_Windows_by_Coverage/Crackle_window"),
    "Healthy":      Path("/Users/yunhwang/Desktop/Stethoscope_Project/Deployment/1 Final Pipeline/2 Model Training with Replayed Sound/Output/Augmented_Windows_by_Coverage/Healthy_window"),
    "Non-breathing":Path("/Users/yunhwang/Desktop/Stethoscope_Project/Deployment/1 Final Pipeline/2 Model Training with Replayed Sound/Output/Augmented_Windows_by_Coverage/Nonbreathing_window"),
    "Rhonchi":      Path("/Users/yunhwang/Desktop/Stethoscope_Project/Deployment/1 Final Pipeline/2 Model Training with Replayed Sound/Output/Augmented_Windows_by_Coverage/Rhonchi_window"),
    "Wheezing":     Path("/Users/yunhwang/Desktop/Stethoscope_Project/Deployment/1 Final Pipeline/2 Model Training with Replayed Sound/Output/Augmented_Windows_by_Coverage/Wheezing_window"),
}

# ─────────────────────────────────────────────────────────
# [하드코딩] 출력 위치
# - 합성한 wav: 이 폴더에 생성
# - gt 엑셀: 이 폴더에 1개 파일로 생성
# ─────────────────────────────────────────────────────────
OUT_DIR = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Deployment/1 Final Pipeline/4 Realtime Pipeline_New/0 Test Data Creation/Output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────
# 합성 파라미터 (하드코딩)
# ─────────────────────────────────────────────────────────
TARGET_SR = 16000          # 모델 파이프라인과 일치
TARGET_LEN_S = 30.0        # 정확히 30초
MIN_SEG_S = 2.0            # 한 번에 붙일 최소 구간 길이
MAX_SEG_S = 8.0            # 한 번에 붙일 최대 구간 길이
N_OUTPUT_FILES = 3         # 30초짜리 파일 몇 개 만들지
RANDOM_SEED = 20251103     # 재현성 확보

# ─────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────
def list_wavs(folder: Path):
    return sorted([p for p in folder.glob("**/*.wav") if p.is_file()])

def load_mono_resample(path: Path, sr: int) -> np.ndarray:
    y, in_sr = sf.read(str(path), always_2d=False)
    # mono
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    # resample
    if in_sr != sr:
        y = librosa.resample(y, orig_sr=in_sr, target_sr=sr, res_type="kaiser_best")
    return y

def normalize_peak(audio: np.ndarray, peak_db=-3.0) -> np.ndarray:
    """최대치를 -3 dBFS 정도로 정규화(클리핑 방지)."""
    if audio.size == 0:
        return audio
    peak = np.max(np.abs(audio))
    if peak < 1e-9:
        return audio
    target_lin = 10.0 ** (peak_db / 20.0)
    return audio * (target_lin / peak)

def write_int16_wav(path: Path, audio: np.ndarray, sr: int):
    # int16로 저장
    audio16 = np.clip(audio, -1.0, 1.0)
    audio16 = (audio16 * 32767.0).astype(np.int16)
    sf.write(str(path), audio16, sr, subtype="PCM_16")

# ─────────────────────────────────────────────────────────
# 메인 합성 로직
# ─────────────────────────────────────────────────────────
def build_one_synthetic(idx: int):
    """
    30초짜리 1개 합성
    반환:
      out_wav: Path
      segments_meta: List[dict]  (타임라인 순서대로)
    """
    rng = random.Random(RANDOM_SEED + idx)

    # 각 클래스별 파일 캐시
    class_files = {c: list_wavs(SRC_DIRS[c]) for c in SRC_DIRS.keys()}
    for c, flist in class_files.items():
        if not flist:
            raise FileNotFoundError(f"[{c}] 폴더에 wav가 없습니다: {SRC_DIRS[c]}")

    target_len = int(TARGET_SR * TARGET_LEN_S)
    cur = np.zeros(0, dtype=np.float32)
    timeline = []  # dicts with out_start_s/out_end_s/label/src_path/src_seg_start_s/src_seg_end_s

    out_t = 0.0
    while cur.shape[0] < target_len:
        # 1) 클래스 선택 (균등확률)
        label = rng.choice(list(SRC_DIRS.keys()))
        # 2) 파일 선택
        cand_files = class_files[label]
        src_path = rng.choice(cand_files)

        # 3) 로드/리샘플
        y = load_mono_resample(src_path, TARGET_SR)
        if y.shape[0] < int(TARGET_SR * MIN_SEG_S):
            # 너무 짧으면 스킵
            continue

        # 4) 세그 길이 선택
        seg_len_s = rng.uniform(MIN_SEG_S, MAX_SEG_S)
        seg_len = int(TARGET_SR * seg_len_s)
        if seg_len > y.shape[0]:
            seg_len = y.shape[0]

        # 5) 시작 위치 선택
        max_start = max(0, y.shape[0] - seg_len)
        start = rng.randint(0, max_start) if max_start > 0 else 0
        end = start + seg_len
        seg = y[start:end].copy()

        # 6) 타겟 길이를 넘길 경우 잘라내기
        remain = target_len - cur.shape[0]
        if seg.shape[0] > remain:
            seg = seg[:remain]
            seg_len = seg.shape[0]

        # 7) 이어붙이기 + 타임라인 기록
        cur = np.concatenate([cur, seg], axis=0)
        out_start_s = out_t
        out_end_s = out_t + seg_len / TARGET_SR
        timeline.append({
            "out_wav": f"synthetic_{idx:02d}.wav",
            "segment_idx": len(timeline) + 1,
            "label": label,  # 'Non-breathing' 표기 포함
            "out_start_s": round(out_start_s, 3),
            "out_end_s": round(out_end_s, 3),
            "src_folder": str(SRC_DIRS[label]),
            "src_file": str(src_path.name),
            "src_seg_start_s": round(start / TARGET_SR, 3),
            "src_seg_end_s": round(end / TARGET_SR, 3),
        })
        out_t = out_end_s

    # 정확히 30초 보장
    if cur.shape[0] > target_len:
        cur = cur[:target_len]

    # 정규화(클리핑 방지)
    cur = normalize_peak(cur, peak_db=-3.0)

    # 저장
    out_wav = OUT_DIR / f"synthetic_{idx:02d}.wav"
    write_int16_wav(out_wav, cur, TARGET_SR)

    return out_wav, timeline

def main():
    print("[SETUP] output dir:", OUT_DIR)
    segments_rows = []

    # 여러 파일 생성
    for i in range(1, N_OUTPUT_FILES + 1):
        out_wav, tl = build_one_synthetic(i)
        print(f"[OK] wrote {out_wav}  ({TARGET_LEN_S:.1f}s)")
        segments_rows.extend(tl)

    # 엑셀 저장 (가독성 좋은 컬럼 순서/정렬)
    df = pd.DataFrame(segments_rows)
    df = df[[
        "out_wav", "segment_idx", "label",
        "out_start_s", "out_end_s",
        "src_folder", "src_file", "src_seg_start_s", "src_seg_end_s",
    ]].sort_values(by=["out_wav", "out_start_s", "segment_idx"])

    xlsx_path = OUT_DIR / "synthetic_ground_truth.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="segments", index=False)

        # (선택) 파일별 타임라인 요약 시트: 보기 편하게 그룹/집계
        summary = (
            df.groupby(["out_wav", "label"], as_index=False)
              .agg(total_duration_s=("out_end_s", lambda s: round(float(s.max() - df.loc[s.index, "out_start_s"].min()), 3)))
        )
        # 위의 집계가 직관적이지 않다면, 단순히 구간 개수만 보여주는 표도 추가
        counts = df.groupby(["out_wav", "label"], as_index=False).size().rename(columns={"size": "num_segments"})
        summary2 = counts

        summary.to_excel(writer, sheet_name="summary_duration_hint", index=False)
        summary2.to_excel(writer, sheet_name="summary_counts", index=False)

    print(f"[OK] wrote GT excel: {xlsx_path}")

if __name__ == "__main__":
    main()
