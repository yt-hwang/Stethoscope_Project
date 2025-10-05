#!/usr/bin/env python3
import os, json, glob
from pathlib import Path
import torchaudio
import torch
import pandas as pd

PROJECT_DIR = '//home//un_wang//my_stethoscope_project'
RAW_DIR = f'{PROJECT_DIR}//data//audio//Raw//RAW sound_ML test sound list'
INTERVAL_JSON = f'{PROJECT_DIR}//data//audio//Raw//breathing_nonbreathing_intervals.json'
SEG_OUT_DIR = f'{PROJECT_DIR}//data//audio//Segments_from_JSON'
META_CSV = f'{PROJECT_DIR}//data//audio//Segments_from_JSON//metadata.csv'

TARGET_SR = 16000

LABELS_ALLOWED = {'Healthy','Crackle','Rhonchi','Wheezing'}  # breathing 진단 라벨
NB_LABEL = 'Non-breathing'

def find_raw_path(basename: str) -> str:
    # 확장자/대소문자 무시하고 RAW_DIR에서 매칭
    pat = os.path.join(RAW_DIR, f'*{basename}*')
    cands = []
    for p in glob.glob(pat):
        if os.path.isdir(p): 
            continue
        name = os.path.splitext(os.path.basename(p))[0].lower()
        if basename.lower() in name:
            cands.append(p)
    # 가장 길이가 가까운 것을 우선 선택
    if not cands:
        return None
    cands.sort(key=lambda x: abs(len(os.path.splitext(os.path.basename(x))[0]) - len(basename)))
    return cands[0]

def load_audio(path):
    wav, sr = torchaudio.load(path)  # [ch, T]
    if wav.shape[0] > 1:  # mono 변환
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav.squeeze(0), TARGET_SR  # [T], sr

def slice_and_save(wav, sr, start_s, end_s, out_wav_path):
    start = int(round(start_s * sr))
    end   = int(round(end_s   * sr))
    seg = wav[max(0,start):min(end, wav.numel())]
    seg = seg.unsqueeze(0)  # [1, T]
    Path(os.path.dirname(out_wav_path)).mkdir(parents=True, exist_ok=True)
    torchaudio.save(out_wav_path, seg, sr)

def main():
    Path(SEG_OUT_DIR).mkdir(parents=True, exist_ok=True)

    with open(INTERVAL_JSON, 'r') as f:
        intervals = json.load(f)

    rows = []
    miss = []

    for key, info in intervals.items():
        diag = info.get('diagnosis', '').strip()
        breathing = info.get('breathing', [])
        non_breathing = info.get('non_breathing', [])
        group = info.get('group', 'UNK')  # 없으면 UNK

        raw_path = find_raw_path(key)
        if raw_path is None or not os.path.exists(raw_path):
            miss.append(key)
            continue

        wav, sr = load_audio(raw_path)

        # breathing → diagnosis 라벨(Healthy/Crackle/Rhonchi/Wheezing 등)
        if diag and len(breathing)>0:
            label = diag if diag in LABELS_ALLOWED or diag=='Healthy' else diag
            # 허용 라벨에 없는 특이 라벨도 그냥 그대로 기록(후처리서 맵핑 가능)
            for idx, (s, e) in enumerate(breathing):
                out_dir = os.path.join(SEG_OUT_DIR, label)
                out_name = f'{key}_B{idx:03d}_{s:.2f}-{e:.2f}.wav'
                out_path = os.path.join(out_dir, out_name)
                slice_and_save(wav, sr, float(s), float(e), out_path)
                rows.append({
                    'id': f'{key}_B{idx:03d}',
                    'wav_path': out_path,
                    'label': label,
                    'group': group,
                    'start': float(s),
                    'end': float(e),
                    'source_file': raw_path
                })

        # non_breathing → "Non-breathing"
        for idx, (s, e) in enumerate(non_breathing):
            out_dir = os.path.join(SEG_OUT_DIR, NB_LABEL)
            out_name = f'{key}_NB{idx:03d}_{s:.2f}-{e:.2f}.wav'
            out_path = os.path.join(out_dir, out_name)
            slice_and_save(wav, sr, float(s), float(e), out_path)
            rows.append({
                'id': f'{key}_NB{idx:03d}',
                'wav_path': out_path,
                'label': NB_LABEL,
                'group': group,
                'start': float(s),
                'end': float(e),
                'source_file': raw_path
            })

    if miss:
        print('[WARN] JSON에는 있는데 RAW 폴더에서 못 찾은 파일 키:', miss)

    df = pd.DataFrame(rows)
    df.to_csv(META_CSV, index=False, encoding='utf-8')
    print(f'[DONE] segments={len(df)} → {META_CSV}')

if __name__ == '__main__':
    main()
