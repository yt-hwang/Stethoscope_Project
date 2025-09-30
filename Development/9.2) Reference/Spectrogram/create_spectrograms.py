#!/usr/bin/env python3
"""
Create clean mel-spectrogram images for all files listed in the JSON.
- Proper dB scaling with robust percentile clipping
- Automatic top-band cropping to remove empty mel bins
- No overlays, no colorbar, single-panel plot

Paths are fixed to your setup:
JSON:  /Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/breathing_nonbreathing_intervals.json
Audio: same folder as JSON
Output: .../Audio shared/spectrogram_outputs/
"""

import json
from pathlib import Path
import warnings

import numpy as np
import librosa
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# =========================
# Paths (your exact directories)
# =========================
JSON_FILE = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/breathing_nonbreathing_intervals.json")
AUDIO_DIR = Path('/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/RAW sound_ML test sound list')
OUTPUT_DIR = Path('/Users/yunhwang/Desktop/Stethoscope_Project/Development/9.2) Reference/Spectrogram')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Audio / Feature Params
# =========================
SR = 4000           # target sampling rate
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 64
FMAX = 4000         # show informative band for lung sounds

# Robust display scaling (ignore outliers)
SPEC_VMIN_PCT = 5.0
SPEC_VMAX_PCT = 99.5

# Auto-crop settings to remove empty top band
AUTO_CROP = True        # enable/disable automatic y-cropping
CROP_DB_REL = -40.0     # keep bins whose max is within 40 dB of global max
MIN_SHOW_BINS = 32      # never crop below this many mel bins
SMOOTH_BINS = 1         # simple smoothing of per-bin maxima; 1 = off

# Figure
FIGSIZE = (15, 8)
DPI = 300


# =========================
# Utilities
# =========================
def load_json(json_path: Path):
    with open(json_path, "r") as f:
        return json.load(f)

def load_audio(audio_path: Path, target_sr: int = SR):
    if not audio_path.exists():
        print(f"Audio missing: {audio_path}")
        return None, None
    try:
        y, sr = librosa.load(str(audio_path), sr=target_sr, mono=True)
        return y, sr
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None, None


# =========================
# Spectrogram logic
# =========================
def create_mel_spectrogram(y: np.ndarray, sr: int):
    """Return mel-spectrogram in dB, frame times, and robust vmin/vmax."""
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmax=FMAX, power=2.0, center=True
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    vmin = float(np.percentile(S_db, SPEC_VMIN_PCT))
    vmax = float(np.percentile(S_db, SPEC_VMAX_PCT))

    frames = np.arange(S_db.shape[1])
    times = librosa.frames_to_time(frames, sr=sr, hop_length=HOP_LENGTH)
    return S_db, times, vmin, vmax


def auto_crop_rows(S_db: np.ndarray):
    """Determine how many lower mel rows to display based on energy."""
    rows = S_db.shape[0]
    if not AUTO_CROP:
        return rows

    row_max = S_db.max(axis=1)  # per-bin max (dB)

    # Optional box smoothing
    if SMOOTH_BINS > 1:
        k = int(SMOOTH_BINS)
        ker = np.ones(k, dtype=float) / k
        row_max = np.convolve(row_max, ker, mode="same")

    gmax = float(row_max.max())
    keep = np.where(row_max >= (gmax + CROP_DB_REL))[0]

    if keep.size > 0:
        last_idx = int(keep.max())
        return max(last_idx + 1, MIN_SHOW_BINS)
    else:
        return MIN_SHOW_BINS


def plot_spectrogram(filename: str, entry: dict):
    """Create and save one spectrogram image for a file."""
    audio_path = AUDIO_DIR / f"{filename}.wav"
    y, sr = load_audio(audio_path, target_sr=SR)
    if y is None:
        return

    dur = len(y) / sr
    diagnosis = entry.get("diagnosis", entry.get("Diagnosis", "Unknown"))
    clean_diag = str(diagnosis).split("(")[0].strip()

    S_db, t, vmin, vmax = create_mel_spectrogram(y, sr)

    # Auto-crop the empty top band
    show_rows = auto_crop_rows(S_db)
    S_show = S_db[:show_rows, :]

    # Plot (no colorbar, no overlays)
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    t0 = float(t[0]) if t.size else 0.0
    t1 = float(t[-1]) if t.size else dur

    ax.imshow(
        S_show,
        origin="lower",
        aspect="auto",
        extent=[t0, t1, 0, show_rows],
        cmap="plasma",
        vmin=vmin,
        vmax=vmax
    )

    ax.set_title(f"Spectrogram - {filename}\nDiagnosis: {clean_diag}", fontsize=16, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Mel Frequency Bins", fontsize=12)
    ax.set_ylim(0, show_rows)

    plt.tight_layout()
    out_path = OUTPUT_DIR / f"{filename}_spectrogram.png"
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"✅ Spectrogram: {out_path}")


# =========================
# Main
# =========================
def main():
    data = load_json(JSON_FILE)
    processed, skipped = 0, 0
    for filename, entry in data.items():
        try:
            plot_spectrogram(filename, entry)
            processed += 1
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            skipped += 1

    print("\n🎉 Spectrogram generation complete!")
    print(f"✅ Processed: {processed} | ❌ Skipped: {skipped}")
    print(f"📁 Output: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
