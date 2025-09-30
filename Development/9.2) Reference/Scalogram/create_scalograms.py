#!/usr/bin/env python3
"""
Create clean scalograms (no overlays, no waveform track, no colorbar).
"""

import json
from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt
import librosa
import pywt

warnings.filterwarnings("ignore")

# =========================
# Configuration
# =========================
JSON_FILE = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/breathing_nonbreathing_intervals.json")
AUDIO_DIR = Path('/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/RAW sound_ML test sound list')
OUTPUT_DIR = Path('/Users/yunhwang/Desktop/Stethoscope_Project/Development/9.2) Reference/Scalogram')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Audio / CWT Params
SR = 4000
WAVELET = "morl"
NUM_SCALES = 128
FMAX = 4000

# Robust scaling
SCALO_VMIN_PCT = 5.0
SCALO_VMAX_PCT = 99.5


# =========================
# Utilities
# =========================
def load_breathing_json(json_path: Path):
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
# Scalogram logic
# =========================
def build_scales(sr: int, wavelet: str, fmax: float, num_scales: int):
    cf = pywt.central_frequency(wavelet)
    min_scale = max(1, int(sr * cf / max(1.0, fmax)))
    scales = np.arange(min_scale, min_scale + num_scales)
    freqs = (sr * cf) / scales
    return scales, freqs

def create_scalogram(y: np.ndarray, sr: int):
    scales, freqs = build_scales(sr, WAVELET, FMAX, NUM_SCALES)
    coefs, _ = pywt.cwt(y, scales, WAVELET, sampling_period=1.0 / sr)

    power = np.abs(coefs) ** 2
    eps = 1e-12
    p_db = 10.0 * np.log10(np.maximum(power, eps))

    vmin = float(np.percentile(p_db, SCALO_VMIN_PCT))
    vmax = float(np.percentile(p_db, SCALO_VMAX_PCT))
    return p_db, freqs, vmin, vmax

def plot_scalogram(filename: str, entry: dict):
    audio_path = AUDIO_DIR / f"{filename}.wav"
    y, sr = load_audio(audio_path, target_sr=SR)
    if y is None:
        return

    dur = len(y) / sr
    diagnosis = entry.get("diagnosis", entry.get("Diagnosis", "Unknown"))
    clean_diag = str(diagnosis).split("(")[0].strip()

    p_db, freqs, vmin, vmax = create_scalogram(y, sr)

    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    ax.imshow(
        p_db,
        origin="lower",
        aspect="auto",
        extent=[0.0, dur, float(freqs.min()), float(freqs.max())],
        cmap="plasma",
        vmin=vmin,
        vmax=vmax
    )

    ax.set_title(f"Scalogram - {filename}\nDiagnosis: {clean_diag}", fontsize=16, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Frequency (Hz)", fontsize=12)

    plt.tight_layout()
    out_path = OUTPUT_DIR / f"{filename}_scalogram.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Scalogram: {out_path}")


# =========================
# Main
# =========================
def main():
    data = load_breathing_json(JSON_FILE)
    processed, skipped = 0, 0
    for filename, entry in data.items():
        try:
            plot_scalogram(filename, entry)
            processed += 1
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            skipped += 1

    print("\n🎉 Scalogram generation complete!")
    print(f"✅ Processed: {processed} | ❌ Skipped: {skipped}")
    print(f"📁 Output: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
