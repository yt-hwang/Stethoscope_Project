#!/usr/bin/env python3
"""
Feature extraction for Cycle A0: NoSeg + NoPreprocess baseline
Extracts raw_waveform_stats, logmel_mean, and mfcc_mean representations
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

def process_audio_file(file_path, target_sr=16000):
    """Process a single audio file and extract all features"""
    try:
        # Load audio and resample to 16kHz mono
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        
        # Extract features
        raw_stats = extract_raw_waveform_stats(audio, sr)
        logmel_mean = extract_logmel_mean(audio, sr)
        mfcc_mean = extract_mfcc_mean(audio, sr)
        
        # Create feature dictionary
        features = {
            'file_path': file_path,
            'filename': os.path.basename(file_path),
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
    parser = argparse.ArgumentParser(description='Extract features for Cycle A0')
    parser.add_argument('--audio_dir', type=str, 
                       default='/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/Hospital sound_raw 60sec',
                       help='Path to audio directory')
    parser.add_argument('--output_dir', type=str,
                       default='/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/A0_NoSeg_NoPre/outputs',
                       help='Path to output directory')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all WAV files
    audio_files = []
    for root, dirs, files in os.walk(args.audio_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    
    print(f"Found {len(audio_files)} audio files")
    
    # Process all files
    all_features = []
    for i, file_path in enumerate(audio_files):
        print(f"Processing {i+1}/{len(audio_files)}: {os.path.basename(file_path)}")
        features = process_audio_file(file_path)
        if features is not None:
            all_features.append(features)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_features)
    
    # Save features
    features_path = os.path.join(args.output_dir, 'features_all.csv')
    df.to_csv(features_path, index=False)
    print(f"Saved features to {features_path}")
    
    # Create separate CSV files for each representation
    # Raw waveform stats
    raw_cols = ['file_path', 'filename', 'rms', 'zcr', 'spectral_flatness', 'kurtosis', 'skewness']
    raw_df = df[raw_cols]
    raw_path = os.path.join(args.output_dir, 'features_raw_waveform_stats.csv')
    raw_df.to_csv(raw_path, index=False)
    print(f"Saved raw waveform stats to {raw_path}")
    
    # Log-mel features
    logmel_cols = ['file_path', 'filename'] + [f'logmel_{i}' for i in range(64)]
    logmel_df = df[logmel_cols]
    logmel_path = os.path.join(args.output_dir, 'features_logmel_mean.csv')
    logmel_df.to_csv(logmel_path, index=False)
    print(f"Saved log-mel features to {logmel_path}")
    
    # MFCC features
    mfcc_cols = ['file_path', 'filename'] + [f'mfcc_{i}' for i in range(13)]
    mfcc_df = df[mfcc_cols]
    mfcc_path = os.path.join(args.output_dir, 'features_mfcc_mean.csv')
    mfcc_df.to_csv(mfcc_path, index=False)
    print(f"Saved MFCC features to {mfcc_path}")
    
    print(f"\nFeature extraction completed!")
    print(f"Total files processed: {len(all_features)}")
    print(f"Raw waveform stats shape: {raw_df.shape}")
    print(f"Log-mel features shape: {logmel_df.shape}")
    print(f"MFCC features shape: {mfcc_df.shape}")

if __name__ == "__main__":
    main()
