import argparse
import numpy as np
import librosa
from .config import SR, N_FFT, HOP_LEN, N_MELS, FMIN, FMAX
from .audio_io import load_audio, pre_emphasis

def stft_mag_db(y: np.ndarray):
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LEN, win_length=int(0.025*SR), window='hann'))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    return S_db

def logmel(y: np.ndarray):
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LEN, win_length=int(0.025*SR), window='hann'))**2
    M = librosa.feature.melspectrogram(S=S, sr=SR, n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
    M_db = librosa.power_to_db(M, ref=np.max)
    return M_db

def mfcc(y: np.ndarray, n_mfcc: int = 20):
    M = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LEN, n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
    M_db = librosa.power_to_db(M, ref=np.max)
    return librosa.feature.mfcc(S=M_db, n_mfcc=n_mfcc)

def wheeze_indicators(S_db: np.ndarray):
    """
    Lightweight wheeze heuristics on a spectrogram in dB:
    - spectral flatness (tonality)
    - spectral centroid
    - band energy ratios in (100–1000 Hz) vs. (1000–2500 Hz)
    Returns a dict of per-frame features.
    """
    flat = librosa.feature.spectral_flatness(S=librosa.db_to_amplitude(S_db))
    cent = librosa.feature.spectral_centroid(S=librosa.db_to_amplitude(S_db), sr=SR)
    # band energies
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    S_lin = librosa.db_to_amplitude(S_db)
    def band_energy(f_lo, f_hi):
        idx = np.where((freqs >= f_lo) & (freqs < f_hi))[0]
        if idx.size == 0:
            return np.zeros(S_lin.shape[1])
        return S_lin[idx].mean(axis=0)
    e_100_1k = band_energy(100, 1000)
    e_1k_2_5k = band_energy(1000, 2500)
    ratio = (e_100_1k + 1e-8) / (e_1k_2_5k + 1e-8)
    return {
        "flatness": flat.squeeze(),
        "centroid": cent.squeeze(),
        "e_ratio_100_1k__1k_2_5k": ratio,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav', type=str, required=True)
    parser.add_argument('--out', type=str, required=False, default=None)
    args = parser.parse_args()

    y = load_audio(args.wav)
    y = pre_emphasis(y)

    M = logmel(y)
    mf = mfcc(y)
    S = stft_mag_db(y)
    inds = wheeze_indicators(S)

    out = {
        "logmel": M.astype(np.float32),
        "mfcc": mf.astype(np.float32),
        "stft_db": S.astype(np.float32),
        "indicators": {k: np.asarray(v, dtype=np.float32) for k, v in inds.items()},
    }
    if args.out:
        np.save(args.out, out, allow_pickle=True)
    else:
        print({k: (v.shape if hasattr(v, 'shape') else len(v)) for k, v in out.items()})

if __name__ == '__main__':
    main()
