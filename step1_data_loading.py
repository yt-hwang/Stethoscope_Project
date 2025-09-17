#!/usr/bin/env python3
"""
Step 1: Data Loading and Analysis
Load all .wav files from "Hospital sound_raw 60sec" directory
Resample to 16 kHz mono if needed
Generate summary statistics and waveform examples
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import librosa
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_and_analyze_audio_files(data_dir):
    """
    Load all .wav files from the specified directory and analyze them
    """
    audio_files = []
    file_info = []
    
    # Find all .wav files recursively
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                audio_files.append(file_path)
    
    print(f"Found {len(audio_files)} audio files")
    
    # Load and analyze each file
    for i, file_path in enumerate(audio_files):
        try:
            # Load audio file
            y, sr = librosa.load(file_path, sr=None)  # Keep original sample rate first
            
            # Resample to 16kHz if needed
            if sr != 16000:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                sr = 16000
            
            # Convert to mono if stereo
            if len(y.shape) > 1:
                y = librosa.to_mono(y)
            
            # Calculate basic statistics
            duration = len(y) / sr
            rms = np.sqrt(np.mean(y**2))
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            
            file_info.append({
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'duration_sec': duration,
                'sample_rate': sr,
                'samples': len(y),
                'rms': rms,
                'zero_crossing_rate': zcr,
                'min_amplitude': np.min(y),
                'max_amplitude': np.max(y),
                'std_amplitude': np.std(y)
            })
            
            print(f"Processed {i+1}/{len(audio_files)}: {os.path.basename(file_path)} ({duration:.2f}s)")
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
    
    return file_info, audio_files

def create_waveform_examples(file_info, output_dir, num_examples=6):
    """
    Create waveform visualization examples
    """
    # Select diverse examples based on duration and RMS
    df = pd.DataFrame(file_info)
    
    # Sort by duration and select examples
    examples = []
    
    # Short duration examples
    short_files = df[df['duration_sec'] < 30].nsmallest(2, 'duration_sec')
    examples.extend(short_files.index.tolist())
    
    # Medium duration examples
    medium_files = df[(df['duration_sec'] >= 30) & (df['duration_sec'] <= 70)].sample(min(2, len(df[(df['duration_sec'] >= 30) & (df['duration_sec'] <= 70)])))
    examples.extend(medium_files.index.tolist())
    
    # Long duration examples
    long_files = df[df['duration_sec'] > 70].nlargest(2, 'duration_sec')
    examples.extend(long_files.index.tolist())
    
    # Remove duplicates and limit to num_examples
    examples = list(set(examples))[:num_examples]
    
    # Create subplot
    fig, axes = plt.subplots(len(examples), 1, figsize=(12, 2*len(examples)))
    if len(examples) == 1:
        axes = [axes]
    
    for i, idx in enumerate(examples):
        file_path = file_info[idx]['file_path']
        y, sr = librosa.load(file_path, sr=16000)
        y = librosa.to_mono(y)
        
        time = np.linspace(0, len(y)/sr, len(y))
        
        axes[i].plot(time, y, alpha=0.7)
        axes[i].set_title(f"{file_info[idx]['file_name']} ({file_info[idx]['duration_sec']:.2f}s)")
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel('Amplitude')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'waveform_examples.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return examples

def generate_summary_statistics(file_info):
    """
    Generate summary statistics for all audio files
    """
    df = pd.DataFrame(file_info)
    
    summary = {
        'total_files': int(len(file_info)),
        'total_duration_sec': float(df['duration_sec'].sum()),
        'total_duration_min': float(df['duration_sec'].sum() / 60),
        'avg_duration_sec': float(df['duration_sec'].mean()),
        'std_duration_sec': float(df['duration_sec'].std()),
        'min_duration_sec': float(df['duration_sec'].min()),
        'max_duration_sec': float(df['duration_sec'].max()),
        'avg_rms': float(df['rms'].mean()),
        'std_rms': float(df['rms'].std()),
        'avg_zero_crossing_rate': float(df['zero_crossing_rate'].mean()),
        'std_zero_crossing_rate': float(df['zero_crossing_rate'].std()),
        'sample_rate': 16000,  # All resampled to 16kHz
        'file_distribution': {
            'WEBSS002': int(len(df[df['file_name'].str.contains('WEBSS-002')])),
            'WEBSS003': int(len(df[df['file_name'].str.contains('WEBSS-003')])),
            'WEBSS004': int(len(df[df['file_name'].str.contains('WEBSS-004')])),
            'WEBSS005': int(len(df[df['file_name'].str.contains('WEBSS-005')])),
            'WEBSS006': int(len(df[df['file_name'].str.contains('WEBSS-006')])),
            'WEBSS007': int(len(df[df['file_name'].str.contains('WEBSS-007')]))
        }
    }
    
    return summary

def main():
    # Set up paths
    data_dir = "/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/Hospital sound_raw 60sec"
    output_dir = "/Users/yunhwang/Desktop/Stethoscope_Project/outputs/step1_data_summary"
    
    print("Step 1: Data Loading and Analysis")
    print("=" * 50)
    
    # Load and analyze audio files
    print("Loading and analyzing audio files...")
    file_info, audio_files = load_and_analyze_audio_files(data_dir)
    
    if not file_info:
        print("No audio files found or processed successfully!")
        return
    
    # Generate summary statistics
    print("\nGenerating summary statistics...")
    summary = generate_summary_statistics(file_info)
    
    # Create waveform examples
    print("Creating waveform examples...")
    examples = create_waveform_examples(file_info, output_dir)
    
    # Save summary to JSON
    summary_file = os.path.join(output_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed file info to CSV
    df = pd.DataFrame(file_info)
    csv_file = os.path.join(output_dir, 'file_details.csv')
    df.to_csv(csv_file, index=False)
    
    # Print summary
    print("\n" + "="*50)
    print("DATA LOADING SUMMARY")
    print("="*50)
    print(f"Total files processed: {summary['total_files']}")
    print(f"Total duration: {summary['total_duration_min']:.2f} minutes")
    print(f"Average duration: {summary['avg_duration_sec']:.2f} seconds")
    print(f"Duration range: {summary['min_duration_sec']:.2f} - {summary['max_duration_sec']:.2f} seconds")
    print(f"Sample rate: {summary['sample_rate']} Hz")
    print(f"Average RMS: {summary['avg_rms']:.4f}")
    print(f"Average zero-crossing rate: {summary['avg_zero_crossing_rate']:.4f}")
    
    print(f"\nFile distribution by patient:")
    for patient, count in summary['file_distribution'].items():
        print(f"  {patient}: {count} files")
    
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  - summary.json: {summary_file}")
    print(f"  - file_details.csv: {csv_file}")
    print(f"  - waveform_examples.png: {os.path.join(output_dir, 'waveform_examples.png')}")
    
    return summary, file_info

if __name__ == "__main__":
    summary, file_info = main()
