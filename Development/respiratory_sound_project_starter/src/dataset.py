import csv, os
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa

from .config import SR, N_FFT, HOP_LEN, N_MELS, FMIN, FMAX
from .audio_io import load_audio
from .features import logmel

LABEL2IDX = {"breathing": 0, "wheezing": 1, "noise": 2}

class RespiratoryDataset(Dataset):
    """CSV columns: path,start_sec,end_sec,label
    Segments are cut from the audio and converted to log-mel features.
    """
    def __init__(self, csv_path: str, max_frames: int = 400):
        self.items = []
        with open(csv_path, newline='') as f:
            r = csv.DictReader(f)
            for row in r:
                self.items.append(row)
        self.max_frames = max_frames

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        y = load_audio(it['path'])
        s = int(float(it['start_sec']) * SR)
        e = int(float(it['end_sec']) * SR)
        seg = y[s:e] if e > s else y
        M = logmel(seg)  # (n_mels, T)
        # truncate/pad time dim
        if M.shape[1] > self.max_frames:
            M = M[:, :self.max_frames]
        if M.shape[1] < self.max_frames:
            pad = self.max_frames - M.shape[1]
            M = np.pad(M, ((0,0),(0,pad)), mode='constant', constant_values=M.min())
        x = torch.tensor(M, dtype=torch.float32).unsqueeze(0)  # (1, n_mels, T)
        ylab = torch.tensor(LABEL2IDX[it['label']], dtype=torch.long)
        return x, ylab
