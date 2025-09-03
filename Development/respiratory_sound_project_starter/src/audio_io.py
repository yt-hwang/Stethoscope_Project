import numpy as np
import soundfile as sf
import librosa

from .config import SR

def load_audio(path: str, sr: int = SR):
    y, native_sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if native_sr != sr:
        y = librosa.resample(y, orig_sr=native_sr, target_sr=sr, res_type="kaiser_best")
    # optional de-mean
    y = y - np.mean(y)
    return y

def pre_emphasis(y: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    return np.append(y[0], y[1:] - coeff * y[:-1])
