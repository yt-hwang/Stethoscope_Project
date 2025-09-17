#!/usr/bin/env python3
"""
Feature extraction for Cycle A2: NoSeg + SpectralGating
Extracts raw_waveform_stats, logmel_mean, and mfcc_mean representations
with spectral gating applied
"""

import os
import sys
import numpy as np
import pandas as pd
import librosa
from scipy import stats
from scipy.stats import kurtosis, skew
import argparse
from pathlib import Path

# Set random seeds for reproducibility
np.random.seed(42)

def apply_spectral_gating(audio, sr, threshold_db=-20, ratio=4.0, attack_time=0.01, release_time=0.1):
    """
    Apply spectral gating to remove background noise
    
    Parameters:
    - audio: Input audio signal
    - sr: Sample rate
    - threshold_db: Threshold in dB below which to gate
    - ratio: Compression ratio (higher = more aggressive gating)
    - attack_time: Attack time in seconds
    - release_time: Release time in seconds
    """
    # Convert to dB
    audio_db = 20 * np.log10(np.abs(audio) + 1e-10)
    
    # Create gate envelope
    gate_envelope = np.ones_like(audio_db)
    
    # Apply gating based on threshold
    below_threshold = audio_db < threshold_db
    gate_envelope[below_threshold] = 1.0 / ratio
    
    # Smooth the envelope to avoid artifacts
    attack_samples = int(attack_time * sr)
    release_samples = int(release_time * sr)
    
    # Apply attack and release smoothing
    smoothed_envelope = np.copy(gate_envelope)
    for i in range(1, len(smoothed_envelope)):
        if gate_envelope[i] < smoothed_envelope[i-1]:  # Attack
            smoothed_envelope[i] = smoothed_envelope[i-1] + (gate_envelope[i] - smoothed_envelope[i-1]) / attack_samples
        else:  # Release
            smoothed_envelope[i] = smoothed_envelope[i-1] + (gate_envelope[i] - smoothed_envelope[i-1]) / release_samples
    
    # Apply the gate
    gated_audio = audio * smoothed_envelope
    
    return gated_audio

def extract_raw_waveform_stats(audio, sr):
    """Extract raw waveform statistical features"""
    # RMS (Root Mean Square)
    rms = np.sqrt(np.mean(audio**2))
    
    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    
    # Spectral flatness (Wiener entropy)
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=audio))
    
    # Kurtosis
    kurt = kurtosis(audio)
    
    # Skewness
    skewness = skew(audio)
    
    return {
        'rms': rms,
        'zcr': zcr,
        'spectral_flatness': spectral_flatness,
        'kurtosis': kurt,
        'skewness': skewness
    }

def extract_logmel_mean(audio, sr, n_mels=64):
    """Extract log-mel spectrogram features averaged over time"""
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    
    # Convert to log scale
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Average over time dimension
    logmel_mean = np.mean(log_mel, axis=1)
    
    return logmel_mean

def extract_mfcc_mean(audio, sr, n_mfcc=13):
    """Extract MFCC features averaged over time"""
    # Compute MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    
    # Average over time dimension
    mfcc_mean = np.mean(mfccs, axis=1)
    
    return mfcc_mean

def process_audio_file(file_path, target_sr=16000, threshold_db=-20, ratio=4.0):
    """Process a single audio file with spectral gating and extract all features"""
    try:
        # Load audio and resample to 16kHz mono
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        
        # Apply spectral gating
        gated_audio = apply_spectral_gating(audio, sr, threshold_db=threshold_db, ratio=ratio)
        
        # Extract features from gated audio
        raw_stats = extract_raw_waveform_stats(gated_audio, sr)
        logmel_mean = extract_logmel_mean(gated_audio, sr)
        mfcc_mean = extract_mfcc_mean(gated_audio, sr)
        
        # Create feature dictionary
        features = {
            'file_path': file_path,
            'filename': os.path.basename(file_path),
            'preprocessing': f'spectral_gating_{threshold_db}db_{ratio}ratio',
            **raw_stats
        }
        
        # Add logmel features
        for i, val in enumerate(logmel_mean):
            features[f'logmel_{i}'] = val
            
        # Add MFCC features
        for i, val in enumerate(mfcc_mean):
            features[f'mfcc_{i}'] = val
            
        return features
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Extract features for Cycle A2 with spectral gating')
    parser.add_argument('--audio_dir', type=str, 
                       default='/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/Hospital sound_raw 60sec',
                       help='Path to audio directory')
    parser.add_argument('--output_dir', type=str,
                       default='/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/A2_NoSeg_SpectralGating/outputs',
                       help='Path to output directory')
    parser.add_argument('--threshold_db', type=float, default=-20,
                       help='Threshold in dB for spectral gating')
    parser.add_argument('--ratio', type=float, default=4.0,
                       help='Compression ratio for spectral gating')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(os.path.join(args.output_dir, 'features'), exist_ok=True)
    
    # Find all WAV files
    audio_files = []
    for root, dirs, files in os.walk(args.audio_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    
    print(f"Found {len(audio_files)} audio files")
    print(f"Applying spectral gating: threshold={args.threshold_db}dB, ratio={args.ratio}")
    
    # Process all files
    all_features = []
    for i, file_path in enumerate(audio_files):
        print(f"Processing {i+1}/{len(audio_files)}: {os.path.basename(file_path)}")
        features = process_audio_file(file_path, threshold_db=args.threshold_db, ratio=args.ratio)
        if features is not None:
            all_features.append(features)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_features)
    
    # Save features
    features_path = os.path.join(args.output_dir, 'features', 'features_all.csv')
    df.to_csv(features_path, index=False)
    print(f"Saved features to {features_path}")
    
    # Create separate CSV files for each representation
    # Raw waveform stats
    raw_cols = ['file_path', 'filename', 'preprocessing', 'rms', 'zcr', 'spectral_flatness', 'kurtosis', 'skewness']
    raw_df = df[raw_cols]
    raw_path = os.path.join(args.output_dir, 'features', 'features_raw_waveform_stats.csv')
    raw_df.to_csv(raw_path, index=False)
    print(f"Saved raw waveform stats to {raw_path}")
    
    # Log-mel features
    logmel_cols = ['file_path', 'filename', 'preprocessing'] + [f'logmel_{i}' for i in range(64)]
    logmel_df = df[logmel_cols]
    logmel_path = os.path.join(args.output_dir, 'features', 'features_logmel_mean.csv')
    logmel_df.to_csv(logmel_path, index=False)
    print(f"Saved log-mel features to {logmel_path}")
    
    # MFCC features
    mfcc_cols = ['file_path', 'filename', 'preprocessing'] + [f'mfcc_{i}' for i in range(13)]
    mfcc_df = df[mfcc_cols]
    mfcc_path = os.path.join(args.output_dir, 'features', 'features_mfcc_mean.csv')
    mfcc_df.to_csv(mfcc_path, index=False)
    print(f"Saved MFCC features to {mfcc_path}")
    
    print(f"\nFeature extraction completed!")
    print(f"Total files processed: {len(all_features)}")
    print(f"Raw waveform stats shape: {raw_df.shape}")
    print(f"Log-mel features shape: {logmel_df.shape}")
    print(f"MFCC features shape: {mfcc_df.shape}")

if __name__ == "__main__":
    main()
