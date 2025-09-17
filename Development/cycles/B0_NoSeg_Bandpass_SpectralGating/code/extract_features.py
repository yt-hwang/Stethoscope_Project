#!/usr/bin/env python3
"""
Cycle B0: NoSeg + Bandpass + SpectralGating - Feature Extraction
Combines the top 2 performers from A-Series: A1 (Bandpass) + A2 (SpectralGating)
"""

import os
import sys
import numpy as np
import pandas as pd
import librosa
from scipy import signal
from pathlib import Path
import argparse

def apply_bandpass_filter(audio, sr, low_freq=100, high_freq=2000):
    """Apply bandpass filter to focus on relevant frequency range"""
    # Design bandpass filter
    nyquist = sr / 2
    low_norm = low_freq / nyquist
    high_norm = high_freq / nyquist
    
    # Use Butterworth filter
    b, a = signal.butter(4, [low_norm, high_norm], btype='band', analog=False)
    
    # Apply filter
    filtered_audio = signal.filtfilt(b, a, audio)
    
    return filtered_audio

def apply_spectral_gating(audio, sr, threshold_percentile=10, alpha=0.1):
    """Apply spectral gating for noise reduction"""
    # Compute STFT
    stft = librosa.stft(audio)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # Compute spectral threshold
    threshold = np.percentile(magnitude, threshold_percentile)
    
    # Apply gating
    gated_magnitude = magnitude * (1 - alpha * np.exp(-magnitude / threshold))
    
    # Reconstruct audio
    gated_stft = gated_magnitude * np.exp(1j * phase)
    gated_audio = librosa.istft(gated_stft)
    
    return gated_audio

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
    """Process a single audio file with combined preprocessing"""
    try:
        # Load audio
        audio, orig_sr = librosa.load(file_path, sr=sr, mono=True)
        
        # Apply combined preprocessing: Bandpass + SpectralGating
        # Step 1: Bandpass filter (100-2000 Hz)
        audio_bandpass = apply_bandpass_filter(audio, sr, low_freq=100, high_freq=2000)
        
        # Step 2: Spectral gating for noise reduction
        audio_processed = apply_spectral_gating(audio_bandpass, sr, threshold_percentile=10, alpha=0.1)
        
        # Extract features
        raw_stats = extract_raw_waveform_stats(audio_processed)
        logmel_mean = extract_logmel_mean(audio_processed, sr)
        mfcc_mean = extract_mfcc_mean(audio_processed, sr)
        
        # Combine all features
        features = {
            'file_path': str(file_path),
            'filename': Path(file_path).name,
            'preprocessing': 'bandpass_100_2000_spectral_gating',
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
    parser = argparse.ArgumentParser(description='Extract features for Cycle B0')
    parser.add_argument('--audio_dir', 
                       default='/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/Hospital sound_raw 60sec',
                       help='Path to audio directory')
    parser.add_argument('--output_dir', 
                       default='/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/B0_NoSeg_Bandpass_SpectralGating/outputs',
                       help='Output directory')
    
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
    
    # Process each representation
    representations = ['raw_waveform_stats', 'logmel_mean', 'mfcc_mean']
    
    for rep in representations:
        print(f"\nProcessing {rep}...")
        
        all_features = []
        
        for i, file_path in enumerate(audio_files):
            if i % 10 == 0:
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
    
    print(f"\nFeature extraction completed for Cycle B0: NoSeg + Bandpass + SpectralGating")

if __name__ == "__main__":
    main()
