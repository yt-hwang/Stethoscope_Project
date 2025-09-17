#!/usr/bin/env python3
"""
Extract features from segmented audio files with PeakNormalize + Bandpass preprocessing.
This combines amplitude normalization (PeakNormalize) with frequency focusing (Bandpass)
"""

import librosa
import librosa.display
import numpy as np
import pandas as pd
import os
import argparse
import json
from scipy.signal import butter, lfilter
import scipy.stats

# Configuration
SR = 16000
N_MELS = 64
N_MFCC = 13
HOP_LENGTH = 512
N_FFT = 2048

# --- Preprocessing Functions ---
def peak_normalize(y):
    """Normalize audio to a peak amplitude of 1.0."""
    return librosa.util.normalize(y, norm=np.inf)

def bandpass_filter(y, sr, lowcut, highcut, order=5):
    """Apply bandpass filter to focus on specific frequency range."""
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y_filtered = lfilter(b, a, y)
    return y_filtered

# --- Feature Extraction Functions ---
def extract_raw_waveform_stats(y, sr):
    """Extract raw waveform statistics."""
    rms = librosa.feature.rms(y=y).mean()
    zcr = librosa.feature.zero_crossing_rate(y=y).mean()
    
    # Spectral flatness
    spec_flatness = librosa.feature.spectral_flatness(y=y).mean()
    
    # Kurtosis and Skewness
    if np.std(y) == 0:
        kurtosis = 0.0
        skewness = 0.0
    else:
        kurtosis = scipy.stats.kurtosis(y)
        skewness = scipy.stats.skew(y)

    return {
        'rms': rms,
        'zcr': zcr,
        'spectral_flatness': spec_flatness,
        'kurtosis': kurtosis,
        'skewness': skewness
    }

def extract_logmel_mean(y, sr):
    """Extract log-mel spectrogram features (mean across time)."""
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT
    )
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return log_mel_spectrogram.mean(axis=1)  # Mean across time

def extract_mfcc_mean(y, sr):
    """Extract MFCC features (mean across time)."""
    mfccs = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT
    )
    return mfccs.mean(axis=1)  # Mean across time

def main():
    parser = argparse.ArgumentParser(
        description="Extract features from segmented audio files with PeakNormalize + Bandpass preprocessing."
    )
    parser.add_argument('--audio_dir', type=str, required=True,
                        help='Path to the root directory containing segmented audio files.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the output directory for features.')
    
    args = parser.parse_args()

    # Find all WAV files
    audio_files = []
    for root, _, files in os.walk(args.audio_dir):
        for file in files:
            if file.endswith('.wav'):
                audio_files.append(os.path.join(root, file))

    if not audio_files:
        print(f"No WAV files found in {args.audio_dir}")
        return

    print(f"Found {len(audio_files)} audio files")
    print("Preprocessing: Peak Normalization + Bandpass Filter (100-2000 Hz)")

    # Create output directories
    os.makedirs(os.path.join(args.output_dir, 'features'), exist_ok=True)

    # Initialize feature storage
    all_raw_waveform_stats = []
    all_logmel_mean = []
    all_mfcc_mean = []

    # Process each audio file
    for i, file_path in enumerate(audio_files):
        print(f"Processing {i+1}/{len(audio_files)}: {os.path.basename(file_path)}")
        
        try:
            # Load audio
            y, sr = librosa.load(file_path, sr=SR, mono=True)

            # Apply preprocessing: Peak Normalize + Bandpass Filter
            y_processed = peak_normalize(y)
            y_processed = bandpass_filter(y_processed, sr, lowcut=100, highcut=2000)

            # Extract features
            raw_stats = extract_raw_waveform_stats(y_processed, sr)
            logmel_features = extract_logmel_mean(y_processed, sr)
            mfcc_features = extract_mfcc_mean(y_processed, sr)

            # Store results
            raw_stats['file_path'] = file_path
            raw_stats['filename'] = os.path.basename(file_path)
            all_raw_waveform_stats.append(raw_stats)

            logmel_dict = {f'logmel_{j}': logmel_features[j] for j in range(len(logmel_features))}
            logmel_dict['file_path'] = file_path
            logmel_dict['filename'] = os.path.basename(file_path)
            all_logmel_mean.append(logmel_dict)

            mfcc_dict = {f'mfcc_{j}': mfcc_features[j] for j in range(len(mfcc_features))}
            mfcc_dict['file_path'] = file_path
            mfcc_dict['filename'] = os.path.basename(file_path)
            all_mfcc_mean.append(mfcc_dict)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    # Save features to CSV
    if all_raw_waveform_stats:
        pd.DataFrame(all_raw_waveform_stats).to_csv(
            os.path.join(args.output_dir, 'features', 'features_raw_waveform_stats.csv'),
            index=False
        )
        print(f"Saved raw waveform stats: {len(all_raw_waveform_stats)} samples")
    
    if all_logmel_mean:
        pd.DataFrame(all_logmel_mean).to_csv(
            os.path.join(args.output_dir, 'features', 'features_logmel_mean.csv'),
            index=False
        )
        print(f"Saved log-mel features: {len(all_logmel_mean)} samples")
    
    if all_mfcc_mean:
        pd.DataFrame(all_mfcc_mean).to_csv(
            os.path.join(args.output_dir, 'features', 'features_mfcc_mean.csv'),
            index=False
        )
        print(f"Saved MFCC features: {len(all_mfcc_mean)} samples")

    print("Feature extraction completed successfully!")

if __name__ == "__main__":
    main()
