#!/usr/bin/env python3
import os, sys
import pandas as pd
import torch
import torchaudio
from pathlib import Path

PROJECT_DIR = '//home//un_wang//my_stethoscope_project'
META_CSV = f'{PROJECT_DIR}//data//audio//Segments_from_JSON//metadata.csv'
FEATURES_CSV = f'{PROJECT_DIR}//features//opera_features.csv'
TARGET_SR = 16000

def load_opera_or_fail():
    try:
        # 👇 너의 환경에 맞게 수정: 예) from opera import Opera
        from opera import Opera
        model = Opera(device='cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(
            f'OPERA 모듈을 불러오지 못했습니다: {e}\n'
            '→ (대안) step: B안으로 openl3/다른 임베딩 또는 '
            '기존 OPERA 추출 스크립트를 호출하도록 수정하세요.'
        )

def load_audio(path):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav.squeeze(0), TARGET_SR

@torch.no_grad()
def embed(model, wav, sr):
    # 👇 반드시 너의 Opera wrapper에 맞게 수정 (아래는 예시 stub)
    # 예: feats = model.embed(wav.unsqueeze(0), sr)  # [1, 768]
    feats = model.embed(wav.unsqueeze(0), sr)  # 구현체에 맞게
    return feats.squeeze(0).cpu().numpy()

def main():
    os.makedirs(os.path.dirname(FEATURES_CSV), exist_ok=True)
    df = pd.read_csv(META_CSV)
    model = load_opera_or_fail()

    rows = []
    for i, row in df.iterrows():
        wav_path = row['wav_path']
        label    = row['label']
        group    = row.get('group', 'UNK')
        seg_id   = row['id']
        wav, sr = load_audio(wav_path)
        vec = embed(model, wav, sr)  # 768-D 가정
        rows.append({'id': seg_id, 'label': label, 'group': group, **{f'f{k}':v for k,v in enumerate(vec)}})

        if (i+1) % 50 == 0:
            print(f'[EMBED] {i+1}/{len(df)}')

    out = pd.DataFrame(rows)
    out.to_csv(FEATURES_CSV, index=False)
    print(f'[DONE] saved features to {FEATURES_CSV}, n={len(out)}')

if __name__ == '__main__':
    main()
