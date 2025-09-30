#!/usr/bin/env python3
"""
Cycle C0: Seg + NoPreprocess - Feature Extraction
Segmentation baseline using 10-second segments with no preprocessing
"""

import os
import sys
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
import argparse

def extract_raw_waveform_stats(audio):
    """Extract raw waveform statistics"""
    # RMS energy
    rms = np.sqrt(np.mean(audio**2))
    
    # Zero crossing rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio)[0])
    
    # Spectral flatness
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=audio)[0])
    
    # Kurtosis and skewness
    kurtosis = np.mean(librosa.feature.spectral_rolloff(y=audio, roll_percent=0.99)[0])
    skewness = np.mean(librosa.feature.spectral_rolloff(y=audio, roll_percent=0.01)[0])
    
    return {
        'rms': rms,
        'zcr': zcr,
        'spectral_flatness': spectral_flatness,
        'kurtosis': kurtosis,
        'skewness': skewness
    }

def extract_logmel_mean(audio, sr, n_mels=64):
    """Extract log-mel spectrogram and compute mean across time"""
    # Compute log-mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel_spec)
    
    # Compute mean across time
    logmel_mean = np.mean(log_mel, axis=1)
    
    return logmel_mean

def extract_mfcc_mean(audio, sr, n_mfcc=13):
    """Extract MFCCs and compute mean across time"""
    # Compute MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    
    # Compute mean across time
    mfcc_mean = np.mean(mfccs, axis=1)
    
    return mfcc_mean

def process_audio_file(file_path, sr=16000):
    """Process a single audio file with segmentation but no preprocessing"""
    try:
        # Load audio
        audio, orig_sr = librosa.load(file_path, sr=sr, mono=True)
        
        # No preprocessing - use raw audio directly
        
        # Extract features
        raw_stats = extract_raw_waveform_stats(audio)
        logmel_mean = extract_logmel_mean(audio, sr)
        mfcc_mean = extract_mfcc_mean(audio, sr)
        
        # Combine all features
        features = {
            'file_path': str(file_path),
            'filename': Path(file_path).name,
            'preprocessing': 'segmented_10sec_nopreprocess',
            **raw_stats
        }
        
        # Add log-mel features
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
    parser = argparse.ArgumentParser(description='Extract features for Cycle C0')
    parser.add_argument('--audio_dir', 
                       default='/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/Hospital sound_raw segmented into 10 sec',
                       help='Path to segmented audio directory')
    parser.add_argument('--output_dir', 
                       default='/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/C0_Seg_NoPre/outputs',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(os.path.join(args.output_dir, 'features'), exist_ok=True)
    
    # Find all WAV files in segmented directories
    audio_files = []
    for root, dirs, files in os.walk(args.audio_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    
    print(f"Found {len(audio_files)} segmented audio files")
    
    # Process each representation
    representations = ['raw_waveform_stats', 'logmel_mean', 'mfcc_mean']
    
    for rep in representations:
        print(f"\nProcessing {rep}...")
        
        all_features = []
        
        for i, file_path in enumerate(audio_files):
            if i % 50 == 0:  # More frequent updates due to more files
                print(f"  Processing file {i+1}/{len(audio_files)}")
            
            features = process_audio_file(file_path)
            if features is not None:
                all_features.append(features)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_features)
        
        # Save features
        output_path = os.path.join(args.output_dir, 'features', f'features_{rep}.csv')
        df.to_csv(output_path, index=False)
        print(f"  Saved {len(df)} samples to {output_path}")
    
    print(f"\nFeature extraction completed for Cycle C0: Seg + NoPreprocess")

if __name__ == "__main__":
    main()
