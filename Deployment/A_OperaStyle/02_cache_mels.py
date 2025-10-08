# 02_cache_mels.py — zero padding only (no skipping)
import torch, torchaudio, torch.nn.functional as F
import torchaudio.transforms as T
import pandas as pd
from pathlib import Path
from config import *

CACHE_MEL.mkdir(parents=True, exist_ok=True)

mel = T.MelSpectrogram(
    sample_rate=SR,
    n_fft=N_FFT,
    win_length=N_FFT,
    hop_length=HOP,
    n_mels=N_MELS,
    f_min=0.0,
    f_max=SR / 2,
    center=True,
    pad_mode="constant",   # reflect → constant padding (zero-fill)
    power=2.0
)
to_db = T.AmplitudeToDB()

def load_16k_mono(path):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav  # [1, T]

def main():
    df = pd.read_csv(ARTI / "dataset.csv")

    for _, r in df.iterrows():
        wid = r["id"]
        wp = r["wav_path_local"]
        out = CACHE_MEL / f"{wid}.pt"
        if out.exists():
            continue

        wav = load_16k_mono(wp)
        Tlen = wav.shape[-1]

        # 너무 짧으면 제로패딩으로 길이를 n_fft 이상 맞춤
        if Tlen < N_FFT:
            wav = F.pad(wav, (0, N_FFT - Tlen))

        # 멜 스펙트로그램 생성 (constant pad 모드)
        M = mel(wav)      # [1, n_mels, T]
        Mdb = to_db(M)
        torch.save(Mdb.cpu(), out)

    print("[DONE] Mel cache created →", CACHE_MEL)

if __name__ == "__main__":
    main()
