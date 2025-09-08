"""
BreathSound_Analysis_Pipeline.py — Fully English, richly commented version

A self-contained, dependency-light pipeline for **breathing sound** analysis.
It favors clarity and robustness over heavy dependencies, so you can run it
almost anywhere with `numpy`, `scipy`, `matplotlib`, and `pandas` (no librosa/torchaudio).

WHAT THIS SCRIPT DOES
---------------------
1) **Load & normalize** the audio (stereo → mono, int → float in [-1, 1]).
2) **Band-pass filter** around typical breath-sound band (default 100–1800 Hz)
   to suppress low hum and very high-frequency noise while preserving wheeze energy.
3) **Time-domain plots** for raw/filtered waveforms.
4) **STFT spectrogram** (linear frequency) for time–frequency inspection.
5) **Manual Mel-spectrogram** (triangular mel filterbank on STFT bins).
6) **MFCC** (DCT of log-mel) for ML-friendly features.
7) **RMS envelope** to detect active breathing cycles (robust thresholding).
8) **Split inhale/exhale** inside each cycle using the envelope’s local peak.
9) **Spectral features** (centroid, bandwidth, flatness, rolloff, peakiness).
10) **Wheeze-candidate frames** by simple spectral heuristics, merged into
    intervals (minimum duration enforced) and overlayed on spectrogram.
11) **Export** plots (PNG) and frame-wise features/intervals (CSV).

USAGE
-----
$ python BreathSound_Analysis_Pipeline.py              # run with defaults

As a module:
>>> from BreathSound_Analysis_Pipeline import run_pipeline, Config
>>> cfg = Config(audio_path="/path/to/breath.wav", output_dir="./outputs")
>>> summary = run_pipeline(cfg)
>>> print(summary)

OUTPUTS (under output_dir)
--------------------------
- images/waveforms: raw_waveform.png, bandpassed_waveform.png
- images/spectrograms: spectrogram_linear.png, mel_spectrogram.png, mfcc.png
- images/analysis: wave_inhale_exhale.png, envelope_cycles.png, spectrogram_wheeze_overlay.png
- data: audio_features.csv (frame-wise features), intervals_inhale_exhale_wheeze.csv (inhale/exhale/wheeze intervals)

DISCLAIMER
----------
This code is a research/engineering utility, **not** a medical device. Do not
use it for diagnosis without proper validation and clinical oversight.
"""
from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, spectrogram, stft
from scipy.fftpack import dct

# =====================
# CONFIGURATION
# =====================
@dataclass
class Config:
    """User‑tunable settings for the pipeline.

    Parameters
    ----------
    audio_path : str
        Path to the input audio file. WAV is preferred for reliability.
    output_dir : str
        Directory where all plots and CSVs will be written.

    lowcut_hz, highcut_hz : float
        Band‑pass limits used to denoise while keeping breath content.
        `highcut_hz` is clamped to 0.45 * sample_rate internally for safety.

    filter_order : int
        IIR Butterworth order. 4 is a good balance (steepness vs stability).

    nperseg, noverlap : int | None
        STFT window length and hop (noverlap = window_overlap). If None, we
        pick sensible defaults based on the sample rate.

    frame_len_sec, hop_len_sec : float
        Window and hop for RMS envelope calculation. Envelope governs cycle
        detection and inhale/exhale splitting. 200 ms / 20 ms are common.

    smooth_win : int
        Smoothing window (in frames) for the envelope (simple moving average).

    thresh_mad_scale : float
        Adaptive threshold = median + thresh_mad_scale * MAD (robust to outliers).

    flatness_max, centroid_min_hz, centroid_max_hz, peakiness_min : float
        Spectral‑heuristic thresholds for wheeze‑candidate frames.

    wheeze_min_dur_sec : float
        Minimum duration for a wheeze interval after merging consecutive frames.

    n_mels, fmin, fmax, n_mfcc : int/float
        Mel‑spectrogram and MFCC configuration.
    """

    # I/O
    audio_path: str = "/mnt/data/WEBSS-002 TP 3_60sec.wav"
    output_dir: str = "/mnt/data/bs_outputs"

    # Filtering
    lowcut_hz: float = 100.0
    highcut_hz: float = 1800.0
    filter_order: int = 4

    # STFT
    nperseg: int | None = None   # if None → 1024 if sr >= 2000 else 512
    noverlap: int | None = None  # if None → nperseg // 2

    # Envelope framing
    frame_len_sec: float = 0.200
    hop_len_sec: float = 0.020
    smooth_win: int = 7

    # Active‑breathing threshold (robust)
    thresh_mad_scale: float = 0.6

    # Wheeze heuristic (더 민감한 감지를 위해 조정)
    flatness_max: float = 0.6  # 0.5 → 0.6 (더 많은 프레임이 tonal로 인식)
    centroid_min_hz: float = 100.0  # 150 → 100 (더 낮은 주파수도 포함)
    centroid_max_hz: float = 1400.0  # 1200 → 1400 (더 높은 주파수도 포함)
    peakiness_min: float = 0.15  # 0.20 → 0.15 (더 낮은 peakiness도 허용)
    wheeze_min_dur_sec: float = 0.25  # 0.30 → 0.25 (더 짧은 구간도 허용)

    # Mel / MFCC
    n_mels: int = 64
    fmin: float = 0.0
    fmax: float | None = 2000.0
    n_mfcc: int = 13


CFG = Config()

# =====================
# UTILITIES
# =====================

def ensure_dir(path: str) -> None:
    """Create the directory if it does not exist (idempotent)."""
    os.makedirs(path, exist_ok=True)


def to_mono(x: np.ndarray) -> np.ndarray:
    """Convert (N, 2) stereo arrays to mono by selecting the first channel.

    Using the first channel avoids unintended mixing artifacts that could
    change spectral content. If your recordings are dual-mic and you need
    both channels, adapt this function accordingly (e.g., average or keep both).
    """
    return x[:, 0] if x.ndim == 2 else x


def normalize_audio(x: np.ndarray) -> np.ndarray:
    """Map common PCM integer formats to float32 in [-1, 1].

    Why: downstream DSP (filtering, STFT) expects floating‑point numerics and
    this yields consistent behavior across platforms/codecs.
    """
    if x.dtype == np.int16:
        x = x.astype(np.float32) / np.iinfo(np.int16).max
    elif x.dtype == np.int32:
        x = x.astype(np.float32) / np.iinfo(np.int32).max
    elif x.dtype == np.uint8:
        # 8‑bit unsigned PCM is typically offset by 128
        x = (x.astype(np.float32) - 128) / 128.0
    else:
        x = x.astype(np.float32)
    return x


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4):
    """Design a **band‑pass** Butterworth filter.

    We clamp normalized cutoffs to (0, 1) and ensure `high > low`. If values
    come too close (e.g., very low sample rates), we widen minimally to retain
    a valid passband.
    """
    nyq = 0.5 * fs  # Nyquist frequency
    low = max(1e-3, lowcut / nyq)
    high = min(0.999, highcut / nyq)
    if high <= low:
        low = max(1e-3, min(low, 0.45))
        high = min(0.9, max(high, low + 1e-3))
    b, a = butter(order, [low, high], btype='band')
    return b, a


def frame_rms(x: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    """Compute frame‑wise RMS (root‑mean‑square) energy.

    Rationale: RMS tracks perceived loudness reasonably well and exposes
    breath activity without requiring labels.
    """
    n = len(x)
    frames = max(1, 1 + (n - frame_len) // hop_len)
    rms = np.zeros(frames, dtype=np.float32)
    for i in range(frames):
        s = i * hop_len
        e = s + frame_len
        seg = x[s:e]
        if len(seg) == 0:
            rms[i] = 0.0
        else:
            # Use float64 for numerical stability when squaring
            rms[i] = np.sqrt(np.mean(seg.astype(np.float64) ** 2) + 1e-12)
    return rms


def smooth(x: np.ndarray, w: int = 7) -> np.ndarray:
    """Simple moving‑average smoothing for a 1‑D sequence.

    Why: the raw envelope can be spiky; smoothing improves cycle grouping and
    peak localization.
    """
    if w <= 1:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode='same')


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    """Hertz → Mel scale (HTK formula)."""
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    """Mel → Hertz (inverse of HTK formula)."""
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def mel_filterbank(n_mels: int, f: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    """Construct a triangular **mel filterbank** on an arbitrary frequency grid.

    Parameters
    ----------
    n_mels : number of triangular filters.
    f : frequency grid in **Hz** (e.g., STFT bin centers).
    fmin, fmax : mel domain coverage mapped back to Hz.

    Returns
    -------
    (n_mels, len(f)) array where each row is a triangular weighting curve.

    Notes
    -----
    We also apply a simple area normalization (`enorm`) so each filter contributes
    comparably regardless of its width.
    """
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    fb = np.zeros((n_mels, len(f)), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_l, f_c, f_r = hz_points[m - 1], hz_points[m], hz_points[m + 1]
        # Rising edge
        left = np.where((f >= f_l) & (f <= f_c))[0]
        if len(left) > 0:
            fb[m - 1, left] = (f[left] - f_l) / max(1e-9, (f_c - f_l))
        # Falling edge
        right = np.where((f >= f_c) & (f <= f_r))[0]
        if len(right) > 0:
            fb[m - 1, right] = (f_r - f[right]) / max(1e-9, (f_r - f_c))

    # Lightweight area normalization so filters have similar energy
    enorm = 2.0 / (hz_points[2:n_mels + 2] - hz_points[:n_mels])
    fb = (fb.T * enorm).T
    return fb


def spectral_features_from_stft(Zxx: np.ndarray, f: np.ndarray):
    """Compute spectral descriptors from an STFT matrix.

    Parameters
    ----------
    Zxx : complex array of shape (freq_bins, frames)
        Output of `scipy.signal.stft`.
    f : array of shape (freq_bins,)
        Frequency axis in Hertz.

    Returns
    -------
    centroid, bandwidth, flatness, rolloff, peakiness, P
        Each is a 1‑D array over frames (except `P`, the power spectrogram).

    Why these features?
    - **centroid** (spectral center of mass) tends to go up during inspiration.
    - **bandwidth** increases with turbulent flow and noise.
    - **flatness** drops for tonal sounds (like wheezes), rises for noise.
    - **rolloff** (85% energy) provides a frequency bound for most energy.
    - **peakiness** quantifies how concentrated the dominant bin is.
    """
    P = (np.abs(Zxx) ** 2).astype(np.float64) + 1e-12
    Psum = np.sum(P, axis=0) + 1e-12

    centroid = np.sum(f[:, None] * P, axis=0) / Psum

    diff = f[:, None] - centroid[None, :]
    bandwidth = np.sqrt(np.sum((diff ** 2) * P, axis=0) / Psum)

    logP = np.log(P)
    flatness = np.exp(np.mean(logP, axis=0)) / np.mean(P, axis=0)

    cumsumP = np.cumsum(P, axis=0)
    targets = 0.85 * cumsumP[-1, :]
    rolloff = np.zeros(P.shape[1])
    for k in range(P.shape[1]):
        idx = np.searchsorted(cumsumP[:, k], targets[k])
        rolloff[k] = f[min(idx, len(f) - 1)]

    peakiness = np.max(P, axis=0) / Psum
    return centroid, bandwidth, flatness, rolloff, peakiness, P


def mask_to_intervals(mask: np.ndarray, times: np.ndarray, min_dur: float = 0.25) -> List[Tuple[float, float]]:
    """Convert a boolean frame mask to **merged time intervals**.

    Adjacent `True` frames are grouped; short blips below `min_dur` are dropped.
    This reduces flicker from frame‑level heuristics.
    """
    out: List[Tuple[float, float]] = []
    s = None
    for i, m in enumerate(mask):
        if m and s is None:
            s = i
        if (not m or i == len(mask) - 1) and s is not None:
            e = i if not m else i
            t0 = times[s]
            t1 = times[e]
            if t1 - t0 >= min_dur:
                out.append((float(t0), float(t1)))
            s = None
    return out

# =====================
# MAIN PIPELINE
# =====================

def run_pipeline(cfg: Config = CFG, file_index: int = None):
    """Run the full analysis pipeline with the given configuration.

    Steps
    -----
    1) Load & normalize audio
    2) Band-pass filter to focus on breath band
    3) Plot raw & filtered waveforms
    4) Compute linear spectrogram (STFT)
    5) Compute STFT features, manual mel-spectrogram, and MFCCs
    6) Build RMS envelope and detect active breathing cycles
    7) Split each cycle into inhale/exhale by the envelope peak
    8) Mark wheeze-candidate frames by a simple spectral heuristic and merge
        into time intervals
    9) Save plots and CSVs; return a compact summary

    Parameters
    ----------
    cfg : Config
        Configuration object with analysis parameters
    file_index : int, optional
        Index number to prefix to output filenames (for ordered processing)

    Returns
    -------
    dict
        Summary with counts, key parameters, and output filenames.
    """
    # Extract filename without extension for unique output directory
    base_filename = os.path.splitext(os.path.basename(cfg.audio_path))[0]
    
    # Create file-specific output directory structure
    file_output_dir = os.path.join(cfg.output_dir, base_filename)
    ensure_dir(file_output_dir)
    ensure_dir(os.path.join(file_output_dir, "images"))
    ensure_dir(os.path.join(file_output_dir, "images", "waveforms"))
    ensure_dir(os.path.join(file_output_dir, "images", "spectrograms"))
    ensure_dir(os.path.join(file_output_dir, "images", "analysis"))
    ensure_dir(os.path.join(file_output_dir, "data"))
    
    # Create filename prefix for ordered processing
    prefix = f"{file_index:03d}_" if file_index is not None else ""

    # =====================
    # 1) LOAD & PREPROCESS AUDIO
    # =====================
    print(f"1) Loading audio: {base_filename}...")
    sr, audio_raw = wavfile.read(cfg.audio_path)      # Read WAV: returns (sr, np.ndarray)
    audio = normalize_audio(to_mono(audio_raw))       # Convert to mono & float32 in [-1, 1]
    duration = len(audio) / sr                        # Audio length in seconds
    t = np.linspace(0, duration, len(audio))          # Time axis for plotting

    # =====================
    # 2) BAND-PASS FILTERING
    # =====================
    print("2) Applying band-pass filter...")
    # Clamp the highcut to 0.45 * sr to avoid instability near Nyquist.
    highcut = min(cfg.highcut_hz, 0.45 * sr)
    b, a = butter_bandpass(cfg.lowcut_hz, highcut, sr, cfg.filter_order)
    # Zero-phase filtering with filtfilt prevents phase distortion (no lag).
    audio_bp = filtfilt(b, a, audio)

    # =====================
    # 3) WAVEFORM PLOTS
    # =====================
    print("3) Plotting waveforms...")

    # Raw waveform
    plt.figure(figsize=(14, 3.2))
    plt.plot(t, audio, linewidth=0.5)
    plt.title("Raw Waveform")
    plt.xlabel("Time [sec]")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(file_output_dir, "images", "waveforms", f"{prefix}01_raw_waveform.png"), dpi=180, bbox_inches='tight')
    plt.close()

    # Band-passed waveform
    plt.figure(figsize=(14, 3.2))
    plt.plot(t, audio_bp, linewidth=0.5)
    plt.title(f"Band-passed Waveform ({cfg.lowcut_hz:.0f}-{highcut:.0f} Hz)")
    plt.xlabel("Time [sec]")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(file_output_dir, "images", "waveforms", f"{prefix}02_bandpassed_waveform.png"), dpi=180, bbox_inches='tight')
    plt.close()

    # =====================
    # 4) SPECTROGRAM (LINEAR)
    # =====================
    print("4) Computing linear spectrogram...")

    # Choose window/hop: 1024/512 at ≥2 kHz; otherwise 512/256. This gives a
    # reasonable time–frequency tradeoff for breath sounds.
    nperseg = cfg.nperseg or (1024 if sr >= 2000 else 512)
    noverlap = cfg.noverlap or (nperseg // 2)

    f_lin, t_lin, Sxx = spectrogram(
        audio_bp, fs=sr, nperseg=nperseg, noverlap=noverlap,
        scaling='spectrum', mode='magnitude'
    )
    # Convert magnitude → power (squared), then to dB for perceptual scaling.
    Sxx_db = 10.0 * np.log10(Sxx ** 2 + 1e-12)

    plt.figure(figsize=(14, 4.0))
    plt.pcolormesh(t_lin, f_lin, Sxx_db, shading='gouraud')
    plt.ylim([0, min(cfg.fmax or sr/2, sr/2)])
    plt.xlabel("Time [sec]")
    plt.ylabel("Frequency [Hz]")
    plt.title("Spectrogram (Linear)")
    plt.colorbar(label="Power [dB]")
    plt.tight_layout()
    plt.savefig(os.path.join(file_output_dir, "images", "spectrograms", f"{prefix}03_spectrogram_linear.png"), dpi=180, bbox_inches='tight')
    plt.close()

    # =====================
    # 5) STFT FEATURES, MEL, MFCC
    # =====================
    print("5) Computing STFT features, Mel-spectrogram, and MFCCs...")

    # Complex STFT for feature extraction
    f, t_stft, Zxx = stft(audio_bp, fs=sr, nperseg=nperseg, noverlap=noverlap, boundary=None)

    # Frame-wise spectral descriptors used later for a simple wheeze heuristic
    centroid, bandwidth, flatness, rolloff, peakiness, P = spectral_features_from_stft(Zxx, f)

    # Manual Mel-spectrogram on the STFT frequency grid
    fmax = min(cfg.fmax or (sr / 2), sr / 2)
    FB = mel_filterbank(cfg.n_mels, f, cfg.fmin, fmax)
    mel_spec = FB @ P                              # (n_mels, frames)
    mel_db = 10.0 * np.log10(mel_spec + 1e-10)

    plt.figure(figsize=(14, 4.0))
    extent = [t_stft[0], t_stft[-1] if len(t_stft) > 1 else duration, 0, cfg.n_mels]
    plt.imshow(mel_db, origin="lower", aspect="auto", extent=extent)
    plt.title("Mel-Spectrogram (manual)")
    plt.xlabel("Time [sec]")
    plt.ylabel("Mel bin")
    plt.colorbar(label="Power [dB]")
    plt.tight_layout()
    plt.savefig(os.path.join(file_output_dir, "images", "spectrograms", f"{prefix}04_mel_spectrogram.png"), dpi=180, bbox_inches='tight')
    plt.close()

    # MFCCs (DCT of log-mel); first 13 coefficients are common in speech/bioacoustics
    log_mel = np.log(mel_spec + 1e-10)
    mfcc = dct(log_mel, type=2, axis=0, norm='ortho')[:cfg.n_mfcc, :]

    plt.figure(figsize=(14, 4.0))
    extent = [t_stft[0], t_stft[-1] if len(t_stft) > 1 else duration, 1, cfg.n_mfcc]
    plt.imshow(mfcc, origin="lower", aspect="auto", extent=extent)
    plt.title(f"MFCC (first {cfg.n_mfcc})")
    plt.xlabel("Time [sec]")
    plt.ylabel("MFCC index")
    plt.colorbar(label="Amplitude")
    plt.tight_layout()
    plt.savefig(os.path.join(file_output_dir, "images", "spectrograms", f"{prefix}05_mfcc.png"), dpi=180, bbox_inches='tight')
    plt.close()

    # =====================
    # 6) ENVELOPE & BREATH CYCLES
    # =====================
    print("6) Building envelope and splitting breath cycles...")

    frame_len = int(cfg.frame_len_sec * sr)  # window length in samples
    hop_len   = int(cfg.hop_len_sec * sr)    # hop length in samples

    rms_env   = frame_rms(audio_bp, frame_len, hop_len)            # frame RMS
    times_env = np.arange(len(rms_env)) * (hop_len / sr)           # envelope time axis
    env_smooth = smooth(rms_env, w=cfg.smooth_win)                 # moving-average smoothing

    # Robust adaptive threshold: median + k * MAD
    med = np.median(env_smooth)
    mad = np.median(np.abs(env_smooth - med)) + 1e-12
    th  = med + cfg.thresh_mad_scale * mad
    active = env_smooth > th

    # Group consecutive active frames into coarse breathing cycles
    segments: List[Tuple[int, int]] = []
    s_idx = None
    for i, m in enumerate(active):
        if m and s_idx is None:
            s_idx = i
        if (not m or i == len(active) - 1) and s_idx is not None:
            e_idx = i if not m else i
            segments.append((s_idx, e_idx))
            s_idx = None

    # Split each cycle into inhale (rising) and exhale (falling) using the envelope peak
    inhale_intervals: List[Tuple[float, float]] = []
    exhale_intervals: List[Tuple[float, float]] = []
    cycle_intervals:  List[Tuple[float, float]] = []

    for (i0, i1) in segments:
        if i1 <= i0:
            continue
        seg_env = env_smooth[i0:i1 + 1]
        p = int(np.argmax(seg_env))                    # index of peak within the segment
        ti, tp, te = times_env[i0], times_env[i0 + p], times_env[i1]
        if tp - ti > 0.1:
            inhale_intervals.append((ti, tp))          # rising part → inspiration
        if te - tp > 0.1:
            exhale_intervals.append((tp, te))          # falling part → expiration
        cycle_intervals.append((ti, te))

    # =====================
    # 7) WHEEZE HEURISTIC
    # =====================
    print("7) Marking wheeze-candidate intervals (heuristic)...")

    # Frames that look tonal (low flatness), with spectral centroid in a plausible band
    # and sufficiently high peakiness. Thresholds are conservative and intended for
    # *candidates*, not clinical decisions.
    tonal = (
        (flatness < cfg.flatness_max)
        & (centroid > cfg.centroid_min_hz)
        & (centroid < cfg.centroid_max_hz)
        & (peakiness > cfg.peakiness_min)
    )
    wheeze_intervals = mask_to_intervals(tonal, t_stft, min_dur=cfg.wheeze_min_dur_sec)

    # =====================
    # 8) VISUALIZATION WITH OVERLAYS
    # =====================
    print("8) Rendering analysis plots...")

    # Waveform + inhale/exhale overlays
    plt.figure(figsize=(14, 3.6))
    plt.plot(t, audio_bp, linewidth=0.6)
    for a, b in inhale_intervals:
        plt.axvspan(a, b, alpha=0.30)  # inspiration (darker)
    for a, b in exhale_intervals:
        plt.axvspan(a, b, alpha=0.25)  # expiration
    plt.title("Waveform with Inhale/Exhale")
    plt.xlabel("Time [sec]")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(os.path.join(file_output_dir, "images", "analysis", f"{prefix}06_wave_inhale_exhale.png"), dpi=180, bbox_inches='tight')
    plt.close()

    # Envelope + threshold + active cycles
    plt.figure(figsize=(14, 3.6))
    plt.plot(times_env, env_smooth, linewidth=1.0)
    plt.plot([times_env[0], times_env[-1]], [th, th], linestyle='--', linewidth=1.0)
    for a, b in cycle_intervals:
        plt.axvspan(a, b, alpha=0.20)
    plt.title("Envelope with Active Breathing Cycles")
    plt.xlabel("Time [sec]")
    plt.ylabel("RMS")
    plt.tight_layout()
    plt.savefig(os.path.join(file_output_dir, "images", "analysis", f"{prefix}07_envelope_cycles.png"), dpi=180, bbox_inches='tight')
    plt.close()

    # Spectrogram + wheeze overlays
    plt.figure(figsize=(14, 4.0))
    plt.pcolormesh(t_lin, f_lin, Sxx_db, shading='gouraud')
    for a, b in wheeze_intervals:
        # 빨간색 박스로 wheeze 구간을 더 눈에 띄게 표시
        plt.axvspan(a, b, alpha=0.7, facecolor='red', edgecolor='darkred', linewidth=2)
    plt.ylim([0, min(fmax, sr / 2)])
    plt.xlabel("Time [sec]")
    plt.ylabel("Frequency [Hz]")
    plt.title("Spectrogram with Wheeze-candidate Overlays (Red Boxes)")
    plt.colorbar(label="Power [dB]")
    plt.tight_layout()
    
    # 기존 경로에 저장
    plt.savefig(os.path.join(file_output_dir, "images", "analysis", f"{prefix}08_spectrogram_wheeze_overlay.png"), dpi=180, bbox_inches='tight')
    
    # wheeze overlay만 따로 모아서 저장하는 폴더에 추가 저장
    wheeze_collection_dir = os.path.join(cfg.output_dir, "wheeze_overlays_collection")
    ensure_dir(wheeze_collection_dir)
    plt.savefig(os.path.join(wheeze_collection_dir, f"{prefix}{base_filename}_wheeze_overlay.png"), dpi=180, bbox_inches='tight')
    
    plt.close()

    # =====================
    # 9) EXPORT DATA
    # =====================
    print("9) Exporting CSV data...")

    # Frame‑wise features aligned on t_stft (good for ML training/analysis)
    df = pd.DataFrame({
        "time_sec": t_stft,
        "spectral_centroid_hz": centroid,
        "spectral_bandwidth_hz": bandwidth,
        "spectral_flatness": flatness,
        "spectral_rolloff_hz": rolloff,
        "spectral_peakiness": peakiness,
    })
    # Append MFCCs
    for i in range(cfg.n_mfcc):
        df[f"mfcc_{i + 1}"] = mfcc[i, :]
    # Interpolate envelope to the STFT timeline for easier joining
    env_interp = np.interp(t_stft, times_env, env_smooth)
    df["rms_env"] = env_interp
    df["wheeze_candidate"] = tonal.astype(int)
    df.to_csv(os.path.join(file_output_dir, "data", "audio_features.csv"), index=False, encoding='utf-8-sig')

    # Interval CSV (human‑readable summary for each detected segment)
    rows = []
    for a, b in inhale_intervals:
        rows.append({"type": "inhale", "t_start_sec": a, "t_end_sec": b, "duration_sec": b - a})
    for a, b in exhale_intervals:
        rows.append({"type": "exhale", "t_start_sec": a, "t_end_sec": b, "duration_sec": b - a})
    for a, b in wheeze_intervals:
        rows.append({"type": "wheeze_candidate", "t_start_sec": a, "t_end_sec": b, "duration_sec": b - a})

    pd.DataFrame(rows).to_csv(
        os.path.join(file_output_dir, "data", "intervals_inhale_exhale_wheeze.csv"),
        index=False, encoding='utf-8-sig'
    )

    # =====================
    # 10) SUMMARY
    # =====================
    print("10) Final summary...")

    breaths_per_min = (len(inhale_intervals) / max(duration, 1e-9)) * 60.0
    summary = {
        "sample_rate_hz": int(sr),
        "duration_sec": float(duration),
        "bandpass_hz": [float(CFG.lowcut_hz), float(highcut)],
        "estimated_breaths_per_min": float(breaths_per_min),
        "num_cycles": len(cycle_intervals),
        "num_inhale_segments": len(inhale_intervals),
        "num_exhale_segments": len(exhale_intervals),
        "num_wheeze_candidates": len(wheeze_intervals),
        "outputs": {
            # Waveform images (numbered in processing order)
            "raw_waveform": f"{base_filename}/images/waveforms/{prefix}01_raw_waveform.png",
            "bandpassed_waveform": f"{base_filename}/images/waveforms/{prefix}02_bandpassed_waveform.png",
            # Spectrogram images
            "spectrogram_linear": f"{base_filename}/images/spectrograms/{prefix}03_spectrogram_linear.png",
            "mel_spectrogram": f"{base_filename}/images/spectrograms/{prefix}04_mel_spectrogram.png",
            "mfcc": f"{base_filename}/images/spectrograms/{prefix}05_mfcc.png",
            # Analysis images
            "wave_inhale_exhale": f"{base_filename}/images/analysis/{prefix}06_wave_inhale_exhale.png",
            "envelope_cycles": f"{base_filename}/images/analysis/{prefix}07_envelope_cycles.png",
            "spectrogram_wheeze_overlay": f"{base_filename}/images/analysis/{prefix}08_spectrogram_wheeze_overlay.png",
            # Data files (saved last as final results)
            "features_csv": f"{base_filename}/data/audio_features.csv",
            "intervals_csv": f"{base_filename}/data/intervals_inhale_exhale_wheeze.csv",
        }
    }

    print("✅ Breath-sound analysis complete!")
    print(f"📊 Summary: {len(cycle_intervals)} cycles, {breaths_per_min:.1f} breaths/min, {len(wheeze_intervals)} wheeze candidates")

    return summary


def batch_process_audio_files(audio_files: List[str], output_base_dir: str = "Output", 
                             config_overrides: dict = None) -> List[dict]:
    """Process multiple audio files in batch.
    
    Parameters
    ----------
    audio_files : List[str]
        List of paths to audio files to process.
    output_base_dir : str
        Base directory where all results will be saved.
    config_overrides : dict, optional
        Dictionary of configuration parameters to override defaults.
        
    Returns
    -------
    List[dict]
        List of summary dictionaries, one for each processed file.
    """
    results = []
    
    for i, audio_path in enumerate(audio_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing file {i}/{len(audio_files)}: {os.path.basename(audio_path)}")
        print(f"{'='*60}")
        
        # Create config for this file
        cfg = Config(audio_path=audio_path, output_dir=output_base_dir)
        if config_overrides:
            for key, value in config_overrides.items():
                setattr(cfg, key, value)
        
        try:
            summary = run_pipeline(cfg, file_index=i)
            results.append(summary)
            print(f"✅ Successfully processed: {os.path.basename(audio_path)}")
        except Exception as e:
            print(f"❌ Error processing {os.path.basename(audio_path)}: {str(e)}")
            results.append({"error": str(e), "file": audio_path})
    
    return results


if __name__ == "__main__":
    # Running the script directly executes the full pipeline using CFG.
    # This makes it easy to experiment from the terminal without writing
    # another driver script.
    summary = run_pipeline(CFG)
    for k, v in summary.items():
        print(f"{k}: {v}")
