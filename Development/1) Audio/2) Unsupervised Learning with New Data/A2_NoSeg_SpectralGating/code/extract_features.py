#!/usr/bin/env python3
"""
A2: NoSeg + SpectralGating - Feature Extraction
NEW DATA EXPERIMENT - Using RAW sound_ML test sound list
Apply spectral gating for noise reduction before feature extraction
THIS WAS THE WINNER IN ORIGINAL EXPERIMENT (0.658)!
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

def spectral_gating(y, sr, prop_decrease=0.8, freq_mask_smooth_hz=50, time_mask_smooth_ms=50, n_fft=2048, hop_length=512):
    """Apply spectral gating for noise reduction."""
    # Compute STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_full, phase = librosa.magphase(D)

    # Estimate noise profile using NMF decomposition
    S_filter = librosa.decompose.decompose(S_full, n_components=1, sort=True)[0]
    S_filter = np.minimum(S_full, S_filter)

    # Compute a noise gate mask using the updated softmask API
    mask = librosa.util.softmask(S_full, S_filter, power=2)
    
    # Apply the mask
    S_reduced = S_full * mask
    
    # Invert STFT
    y_gated = librosa.istft(S_reduced * phase, hop_length=hop_length)
    return y_gated

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
        
        # PREPROCESSING: Apply spectral gating for noise reduction
        audio_processed = spectral_gating(audio, sr)
        
        # Extract features from processed audio
        raw_stats = extract_raw_waveform_stats(audio_processed, sr)
        logmel_mean = extract_logmel_mean(audio_processed, sr)
        mfcc_mean = extract_mfcc_mean(audio_processed, sr)
        
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
    parser = argparse.ArgumentParser(description='Extract features for A2: NoSeg + SpectralGating - New Data Experiment')
    parser.add_argument('--audio_dir', type=str, 
                       default='/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/RAW sound_ML test sound list',
                       help='Path to audio directory')
    parser.add_argument('--output_dir', type=str,
                       default='/Users/yunhwang/Desktop/Stethoscope_Project/Development/Unsupervised Learning with New Data/A2_NoSeg_SpectralGating/outputs',
                       help='Path to output directory')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'features'), exist_ok=True)
    
    # Find all WAV files
    audio_files = []
    for root, dirs, files in os.walk(args.audio_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    
    print("🧪 NEW DATA EXPERIMENT - A2: NoSeg + SpectralGating")
    print("=" * 60)
    print(f"Dataset: RAW sound_ML test sound list")
    print(f"Found {len(audio_files)} audio files")
    print(f"Preprocessing: Spectral gating noise reduction")
    print(f"🏆 ORIGINAL WINNER: 0.658 quality score!")
    print()
    
    # Process all files
    all_features = []
    for i, file_path in enumerate(audio_files):
        print(f"Processing {i+1}/{len(audio_files)}: {os.path.basename(file_path)}")
        features = process_audio_file(file_path)
        if features is not None:
            all_features.append(features)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_features)
    
    # Save features to the features subdirectory
    features_dir = os.path.join(args.output_dir, 'features')
    
    # Create separate CSV files for each representation
    # Raw waveform stats
    raw_cols = ['file_path', 'filename', 'rms', 'zcr', 'spectral_flatness', 'kurtosis', 'skewness']
    raw_df = df[raw_cols]
    raw_path = os.path.join(features_dir, 'features_raw_waveform_stats.csv')
    raw_df.to_csv(raw_path, index=False)
    print(f"✅ Saved raw waveform stats: {raw_df.shape}")
    
    # Log-mel features
    logmel_cols = ['file_path', 'filename'] + [f'logmel_{i}' for i in range(64)]
    logmel_df = df[logmel_cols]
    logmel_path = os.path.join(features_dir, 'features_logmel_mean.csv')
    logmel_df.to_csv(logmel_path, index=False)
    print(f"✅ Saved log-mel features: {logmel_df.shape}")
    
    # MFCC features
    mfcc_cols = ['file_path', 'filename'] + [f'mfcc_{i}' for i in range(13)]
    mfcc_df = df[mfcc_cols]
    mfcc_path = os.path.join(features_dir, 'features_mfcc_mean.csv')
    mfcc_df.to_csv(mfcc_path, index=False)
    print(f"✅ Saved MFCC features: {mfcc_df.shape}")
    
    print(f"\n🎉 Feature extraction completed!")
    print(f"Total files processed: {len(all_features)}")
    print(f"Features saved in: {features_dir}")

if __name__ == "__main__":
    main()
