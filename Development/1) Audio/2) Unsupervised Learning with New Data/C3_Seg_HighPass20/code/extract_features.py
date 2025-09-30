#!/usr/bin/env python3
"""
C3: Seg + HighPass20 - Feature Extraction
NEW DATA EXPERIMENT - Using RAW sound_ML test sound list
Apply segmentation (10s windows) + high-pass filtering (20 Hz) before feature extraction
"""

import os
import sys
import numpy as np
import pandas as pd
import librosa
from scipy import stats
from scipy.stats import kurtosis, skew
from scipy.signal import butter, lfilter
import argparse
from pathlib import Path

# Set random seeds for reproducibility
np.random.seed(42)

def segment_audio(audio, sr, segment_length=10, overlap=0):
    """Segment audio into fixed-length windows."""
    segment_samples = int(segment_length * sr)
    overlap_samples = int(overlap * sr)
    step = segment_samples - overlap_samples
    
    segments = []
    for start in range(0, len(audio) - segment_samples + 1, step):
        end = start + segment_samples
        segment = audio[start:end]
        segments.append(segment)
    
    return segments

def highpass_filter(audio, sr, cutoff_freq=20, order=5):
    """Apply high-pass filter to audio signal."""
    nyquist = 0.5 * sr
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_audio = lfilter(b, a, audio)
    return filtered_audio

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
    """Process a single audio file and extract features from segments"""
    try:
        # Load audio and resample to 16kHz mono
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        
        # SEGMENTATION: Split into 10-second segments
        segments = segment_audio(audio, sr, segment_length=10)
        
        all_segment_features = []
        
        for i, segment in enumerate(segments):
            # PREPROCESSING: Apply high-pass filter (20 Hz)
            segment_processed = highpass_filter(segment, sr, cutoff_freq=20)
            
            # Extract features from processed segment
            raw_stats = extract_raw_waveform_stats(segment_processed, sr)
            logmel_mean = extract_logmel_mean(segment_processed, sr)
            mfcc_mean = extract_mfcc_mean(segment_processed, sr)
            
            # Create feature dictionary
            features = {
                'file_path': file_path,
                'filename': f"{os.path.splitext(os.path.basename(file_path))[0]}_seg{i+1}",
                'original_filename': os.path.basename(file_path),
                'segment_index': i,
                **raw_stats
            }
            
            # Add logmel features
            for j, val in enumerate(logmel_mean):
                features[f'logmel_{j}'] = val
                
            # Add MFCC features
            for j, val in enumerate(mfcc_mean):
                features[f'mfcc_{j}'] = val
                
            all_segment_features.append(features)
        
        return all_segment_features
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description='Extract features for C3: Seg + HighPass20 - New Data Experiment')
    parser.add_argument('--audio_dir', type=str, 
                       default='/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/RAW sound_ML test sound list',
                       help='Path to audio directory')
    parser.add_argument('--output_dir', type=str,
                       default='/Users/yunhwang/Desktop/Stethoscope_Project/Development/Unsupervised Learning with New Data/C3_Seg_HighPass20/outputs',
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
    
    print("NEW DATA EXPERIMENT - C3: Seg + HighPass20")
    print("=" * 60)
    print(f"Dataset: RAW sound_ML test sound list")
    print(f"Found {len(audio_files)} audio files")
    print(f"Segmentation: 10-second windows")
    print(f"Preprocessing: High-pass filter (20 Hz)")
    print()
    
    # Process all files
    all_features = []
    total_segments = 0
    
    for i, file_path in enumerate(audio_files):
        print(f"Processing {i+1}/{len(audio_files)}: {os.path.basename(file_path)}")
        segment_features = process_audio_file(file_path)
        if segment_features:
            all_features.extend(segment_features)
            total_segments += len(segment_features)
            print(f"  Generated {len(segment_features)} segments")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_features)
    
    print(f"\nSegmentation summary:")
    print(f"Original files: {len(audio_files)}")
    print(f"Total segments: {total_segments}")
    print(f"Average segments per file: {total_segments/len(audio_files):.1f}")
    
    # Save features to the features subdirectory
    features_dir = os.path.join(args.output_dir, 'features')
    
    # Create separate CSV files for each representation
    # Raw waveform stats
    raw_cols = ['file_path', 'filename', 'original_filename', 'segment_index', 'rms', 'zcr', 'spectral_flatness', 'kurtosis', 'skewness']
    raw_df = df[raw_cols]
    raw_path = os.path.join(features_dir, 'features_raw_waveform_stats.csv')
    raw_df.to_csv(raw_path, index=False)
    print(f"Saved raw waveform stats: {raw_df.shape}")
    
    # Log-mel features
    logmel_cols = ['file_path', 'filename', 'original_filename', 'segment_index'] + [f'logmel_{i}' for i in range(64)]
    logmel_df = df[logmel_cols]
    logmel_path = os.path.join(features_dir, 'features_logmel_mean.csv')
    logmel_df.to_csv(logmel_path, index=False)
    print(f"Saved log-mel features: {logmel_df.shape}")
    
    # MFCC features
    mfcc_cols = ['file_path', 'filename', 'original_filename', 'segment_index'] + [f'mfcc_{i}' for i in range(13)]
    mfcc_df = df[mfcc_cols]
    mfcc_path = os.path.join(features_dir, 'features_mfcc_mean.csv')
    mfcc_df.to_csv(mfcc_path, index=False)
    print(f"Saved MFCC features: {mfcc_df.shape}")
    
    print(f"\nFeature extraction completed!")
    print(f"Total segments processed: {len(all_features)}")
    print(f"Features saved in: {features_dir}")

if __name__ == "__main__":
    main()
