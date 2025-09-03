import numpy as np
import librosa
from .config import SR, HOP_LEN, N_FFT

def frame_rms(y: np.ndarray, frame_length: int = int(0.050*SR), hop_length: int = int(0.010*SR)):
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length).squeeze()
    return rms

def smooth(x: np.ndarray, win: int = 7):
    if win <= 1:
        return x
    pad = win//2
    xpad = np.pad(x, (pad, pad), mode='edge')
    c = np.convolve(xpad, np.ones(win)/win, mode='valid')
    return c

def energy_segmentation(y: np.ndarray, k: float = 1.0):
    """Return boolean mask of voiced/active frames based on RMS thresholding."""
    rms = frame_rms(y)
    thr = np.median(rms) + k * (np.median(np.abs(rms - np.median(rms))) + 1e-8)
    mask = smooth((rms > thr).astype(float), 5) > 0.5
    return mask, rms, thr

def envelope_breath_cycles(y: np.ndarray):
    """Rough breath cycle detection using amplitude envelope peaks."""
    # envelope via absolute+LPF
    env = smooth(np.abs(y), win=int(0.150*SR))
    # find local minima as cycle boundaries
    import numpy as np
    # simple heuristic: threshold crossings
    thr = np.percentile(env, 60)
    active = env > thr
    # find edges
    starts = np.where(np.logical_and(active[1:], ~active[:-1]))[0]
    ends = np.where(np.logical_and(~active[1:], active[:-1]))[0]
    if ends.size and starts.size:
        if ends[0] < starts[0]:
            ends = ends[1:]
        m = min(len(starts), len(ends))
        starts, ends = starts[:m], ends[:m]
    cycles = list(zip(starts, ends))
    return env, cycles
