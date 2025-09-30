#!/usr/bin/env python3
"""
Simple Excel Visualization Options
=================================

Creates clear visualization options showing Excel breathing data properly.
"""

import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path

def parse_excel_data(filename):
    """Parse Excel breathing data for a file."""
    
    excel_file = "/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/ML test sound list breathing info.xlsx"
    
    sheet1 = pd.read_excel(excel_file, sheet_name='Sheet1', header=None)
    healthy = pd.read_excel(excel_file, sheet_name='Healthy', header=None)
    
    print(f"Looking for {filename} in Excel...")
    
    for sheet_name, df in [('Sheet1', sheet1), ('Healthy', healthy)]:
        for idx in range(df.shape[0]):
            row = df.iloc[idx]
            
            if pd.notna(row.iloc[1]) and isinstance(row.iloc[1], str):
                excel_filename = str(row.iloc[1])
                
                if excel_filename in filename or filename.replace('.wav', '') in excel_filename:
                    print(f"Found {excel_filename} at row {idx}")
                    
                    # Get timestamps from next row
                    if idx + 1 < df.shape[0]:
                        timestamp_row = df.iloc[idx + 1]
                        header_row = row
                        
                        inhale_times = []
                        exhale_times = []
                        
                        # Parse timestamps
                        for col_idx in range(2, min(df.shape[1], 30)):
                            header_val = header_row.iloc[col_idx]
                            timestamp_val = timestamp_row.iloc[col_idx]
                            
                            if pd.notna(timestamp_val) and isinstance(timestamp_val, (int, float)):
                                timestamp = float(timestamp_val)
                                if 0 <= timestamp <= 60:
                                    if pd.notna(header_val) and isinstance(header_val, str):
                                        if 'Inhale' in header_val:
                                            inhale_times.append(timestamp)
                                        elif 'Exhale' in header_val:
                                            exhale_times.append(timestamp)
                        
                        print(f"Found {len(inhale_times)} inhales, {len(exhale_times)} exhales")
                        print(f"Inhale times: {inhale_times[:5]}")
                        print(f"Exhale times: {exhale_times[:5]}")
                        
                        return {
                            'inhale_times': sorted(inhale_times),
                            'exhale_times': sorted(exhale_times),
                            'condition': sheet_name
                        }
    
    print(f"No Excel data found for {filename}")
    return None

def create_simple_option(audio_file, output_dir):
    """Create simple, clear visualization with Excel data."""
    
    filename = audio_file.stem
    print(f"\nCreating visualization for {filename}...")
    
    # Load audio
    audio, sr = librosa.load(audio_file, sr=16000, mono=True)
    duration = len(audio) / sr
    time_axis = np.linspace(0, duration, len(audio))
    
    # Parse Excel data
    excel_data = parse_excel_data(filename)
    
    # Create demo predictions
    predictions = []
    pred_times = []
    
    current_time = 0.0
    while current_time + 2.0 <= duration:
        pred_time = current_time + 1.0
        
        # Simple demo prediction
        start_sample = int(current_time * sr)
        end_sample = int((current_time + 2.0) * sr)
        segment = audio[start_sample:end_sample]
        
        rms_energy = np.sqrt(np.mean(segment**2))
        is_breathing = rms_energy > 0.02
        
        predictions.append(1 if is_breathing else 0)
        pred_times.append(pred_time)
        
        current_time += 1.0
    
    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    title = f'Breathing Analysis - {filename}'
    if excel_data:
        title += f' (Excel: {len(excel_data["inhale_times"])} inhales, {len(excel_data["exhale_times"])} exhales)'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # 1. Waveform with Excel data
    ax = axes[0]
    ax.plot(time_axis, audio, color='navy', linewidth=0.8)
    
    if excel_data:
        # Show inhale times as green lines
        for inhale_time in excel_data['inhale_times']:
            ax.axvline(inhale_time, color='green', alpha=0.7, linewidth=3)
        
        # Show exhale times as red lines
        for exhale_time in excel_data['exhale_times']:
            ax.axvline(exhale_time, color='red', alpha=0.7, linewidth=3)
        
        # Add legend
        ax.axvline([], color='green', linewidth=3, label=f'Excel Inhale ({len(excel_data["inhale_times"])})')
        ax.axvline([], color='red', linewidth=3, label=f'Excel Exhale ({len(excel_data["exhale_times"])})')
        ax.legend()
    
    ax.set_ylabel('Amplitude')
    ax.set_title('Waveform with Excel Breathing Data')
    ax.set_xlim(0, duration)
    ax.grid(True, alpha=0.3)
    
    # 2. Spectrogram - NO LEGEND for perfect alignment
    ax = axes[1]
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio, hop_length=512)), ref=np.max)
    librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr, ax=ax, fmax=2000, hop_length=512)
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Spectrogram (0-2000 Hz)')
    ax.set_xlim(0, duration)
    
    # 3. Model predictions
    ax = axes[2]
    
    breathing_times = [t for t, p in zip(pred_times, predictions) if p == 1]
    nonbreathing_times = [t for t, p in zip(pred_times, predictions) if p == 0]
    
    if breathing_times:
        ax.scatter(breathing_times, [1]*len(breathing_times), c='darkgreen', s=100, 
                  alpha=0.8, label=f'Model Breathing ({len(breathing_times)})')
    if nonbreathing_times:
        ax.scatter(nonbreathing_times, [0]*len(nonbreathing_times), c='darkred', s=100, 
                  alpha=0.8, label=f'Model Non-breathing ({len(nonbreathing_times)})')
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Model Classification')
    ax.set_title('Model Predictions')
    ax.set_xlim(0, duration)
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Non-breathing', 'Breathing'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{filename}_WITH_EXCEL_DATA.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created visualization with Excel data for {filename}")
    
    return excel_data

def main():
    """Create options with proper Excel data."""
    
    audio_dir = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/RAW sound_ML test sound list")
    output_dir = Path("visualization_options")
    
    # Clear and recreate
    import shutil
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()
    
    print("🎨 CREATING OPTIONS WITH PROPER EXCEL BREATHING DATA")
    print("=" * 50)
    
    # Test with files that should have Excel data
    test_files = ['KP001_WWS.wav', 'H001.wav', 'KP003_WWS_1.wav']
    
    for filename in test_files:
        audio_file = audio_dir / filename
        if audio_file.exists():
            excel_data = create_simple_option(audio_file, output_dir)
        else:
            print(f"Audio file not found: {filename}")
    
    print(f"\n✅ Created visualization options with Excel breathing data!")
    print(f"📁 Check: {output_dir}/")

if __name__ == "__main__":
    main()
