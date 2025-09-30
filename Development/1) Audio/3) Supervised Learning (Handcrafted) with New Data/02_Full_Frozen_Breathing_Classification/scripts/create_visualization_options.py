#!/usr/bin/env python3
"""
Create Multiple Visualization Options
===================================

Creates several different visualization approaches for you to choose from:
1. Option A: Side-by-side comparison (Ground Truth vs Predictions)
2. Option B: Layered approach (Excel answers + Model predictions)
3. Option C: Timeline approach (Clear temporal visualization)
4. Option D: Heatmap approach (Intensity-based visualization)
"""

import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path

def parse_excel_breathing_data(filename):
    """Parse Excel breathing data for a specific file."""
    
    excel_file = "/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/ML test sound list breathing info.xlsx"
    
    # Read both sheets
    sheet1 = pd.read_excel(excel_file, sheet_name='Sheet1', header=None)
    healthy = pd.read_excel(excel_file, sheet_name='Healthy', header=None)
    
    # Find the file and extract inhale/exhale timestamps
    for sheet_name, df in [('Sheet1', sheet1), ('Healthy', healthy)]:
        for idx in range(0, df.shape[0], 3):  # Files are every 3 rows
            if idx < df.shape[0] and pd.notna(df.iloc[idx, 1]):
                excel_filename = str(df.iloc[idx, 1])
                
                if excel_filename in filename or filename.replace('.wav', '') in excel_filename:
                    print(f"   Found Excel data for {filename}")
                    
                    # Get the timestamps from the header row
                    header_row = df.iloc[idx]
                    timestamp_row = df.iloc[idx + 1] if idx + 1 < df.shape[0] else None
                    
                    if timestamp_row is None:
                        continue
                    
                    inhale_times = []
                    exhale_times = []
                    
                    # Parse inhale/exhale patterns
                    for col_idx in range(2, min(df.shape[1], 40)):
                        header_val = header_row.iloc[col_idx]
                        timestamp_val = timestamp_row.iloc[col_idx]
                        
                        if pd.notna(header_val) and pd.notna(timestamp_val):
                            if isinstance(header_val, str) and isinstance(timestamp_val, (int, float)):
                                timestamp = float(timestamp_val)
                                if 0 <= timestamp <= 60:
                                    if 'Inhale' in header_val:
                                        inhale_times.append(timestamp)
                                    elif 'Exhale' in header_val:
                                        exhale_times.append(timestamp)
                    
                    return {
                        'inhale_times': sorted(inhale_times),
                        'exhale_times': sorted(exhale_times),
                        'condition': 'Healthy' if sheet_name == 'Healthy' else 'Pathological'
                    }
    
    return None

def create_demo_predictions(audio, sr, duration):
    """Create demo model predictions."""
    
    segment_length = 2.0
    hop_length = 1.0
    
    predictions = []
    pred_times = []
    
    current_time = 0.0
    while current_time + segment_length <= duration:
        start_sample = int(current_time * sr)
        end_sample = int((current_time + segment_length) * sr)
        segment = audio[start_sample:end_sample]
        
        # Simple breathing detection
        rms_energy = np.sqrt(np.mean(segment**2))
        is_breathing = rms_energy > 0.02
        
        predictions.append(1 if is_breathing else 0)
        pred_times.append(current_time + segment_length / 2)
        
        current_time += hop_length
    
    return predictions, pred_times

def create_option_a_side_by_side(audio_file, excel_data, output_dir):
    """Option A: Side-by-side Ground Truth vs Predictions."""
    
    filename = audio_file.stem
    audio, sr = librosa.load(audio_file, sr=16000, mono=True)
    duration = len(audio) / sr
    time_axis = np.linspace(0, duration, len(audio))
    
    predictions, pred_times = create_demo_predictions(audio, sr, duration)
    
    # Create side-by-side visualization
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    fig.suptitle(f'Option A: Side-by-Side Comparison - {filename}', fontsize=16, fontweight='bold')
    
    # Left column: Ground Truth
    # Waveform with Excel annotations
    ax = axes[0, 0]
    ax.plot(time_axis, audio, color='navy', linewidth=0.8)
    
    if excel_data:
        for inhale_time in excel_data['inhale_times']:
            ax.axvline(inhale_time, color='lightgreen', alpha=0.7, linewidth=3, label='Inhale' if inhale_time == excel_data['inhale_times'][0] else "")
        for exhale_time in excel_data['exhale_times']:
            ax.axvline(exhale_time, color='lightcoral', alpha=0.7, linewidth=3, label='Exhale' if exhale_time == excel_data['exhale_times'][0] else "")
    
    ax.set_title('Ground Truth (Excel Data)')
    ax.set_ylabel('Amplitude')
    ax.set_xlim(0, duration)
    ax.grid(True, alpha=0.3)
    if excel_data and (excel_data['inhale_times'] or excel_data['exhale_times']):
        ax.legend()
    
    # Spectrogram with Excel annotations
    ax = axes[1, 0]
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio, hop_length=512)), ref=np.max)
    librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr, ax=ax, fmax=2000, hop_length=512)
    
    if excel_data:
        for inhale_time in excel_data['inhale_times']:
            ax.axvline(inhale_time, color='white', alpha=0.8, linewidth=2)
        for exhale_time in excel_data['exhale_times']:
            ax.axvline(exhale_time, color='yellow', alpha=0.8, linewidth=2)
    
    ax.set_title('Ground Truth Spectrogram')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlim(0, duration)
    
    # Right column: Model Predictions
    # Waveform with predictions
    ax = axes[0, 1]
    ax.plot(time_axis, audio, color='navy', linewidth=0.8)
    
    for pred_time, prediction in zip(pred_times, predictions):
        color = 'green' if prediction == 1 else 'red'
        ax.axvline(pred_time, color=color, alpha=0.6, linewidth=2)
    
    ax.set_title('Model Predictions')
    ax.set_ylabel('Amplitude')
    ax.set_xlim(0, duration)
    ax.grid(True, alpha=0.3)
    
    # Spectrogram with predictions
    ax = axes[1, 1]
    librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr, ax=ax, fmax=2000, hop_length=512)
    
    for pred_time, prediction in zip(pred_times, predictions):
        color = 'white' if prediction == 1 else 'orange'
        ax.axvline(pred_time, color=color, alpha=0.8, linewidth=2)
    
    ax.set_title('Prediction Spectrogram')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (seconds)')
    ax.set_xlim(0, duration)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{filename}_Option_A_Side_by_Side.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_option_b_layered(audio_file, excel_data, output_dir):
    """Option B: Layered approach with Excel + Model on same plots."""
    
    filename = audio_file.stem
    audio, sr = librosa.load(audio_file, sr=16000, mono=True)
    duration = len(audio) / sr
    time_axis = np.linspace(0, duration, len(audio))
    
    predictions, pred_times = create_demo_predictions(audio, sr, duration)
    
    # Create layered visualization
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle(f'Option B: Layered Approach - {filename}', fontsize=16, fontweight='bold')
    
    # 1. Waveform with both Excel and Model annotations
    ax = axes[0]
    ax.plot(time_axis, audio, color='navy', linewidth=0.8)
    
    # Excel data (bottom layer)
    if excel_data:
        for inhale_time in excel_data['inhale_times']:
            ax.axvspan(inhale_time - 0.5, inhale_time + 0.5, alpha=0.3, color='lightgreen', label='Excel Inhale' if inhale_time == excel_data['inhale_times'][0] else "")
        for exhale_time in excel_data['exhale_times']:
            ax.axvspan(exhale_time - 0.5, exhale_time + 0.5, alpha=0.3, color='lightcoral', label='Excel Exhale' if exhale_time == excel_data['exhale_times'][0] else "")
    
    # Model predictions (top layer)
    for pred_time, prediction in zip(pred_times, predictions):
        if prediction == 1:
            ax.axvline(pred_time, color='darkgreen', alpha=0.8, linewidth=3, label='Model Breathing' if pred_time == pred_times[0] else "")
    
    ax.set_ylabel('Amplitude')
    ax.set_title('Waveform: Excel Ground Truth + Model Predictions')
    ax.set_xlim(0, duration)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. Spectrogram - NO LEGEND for perfect alignment
    ax = axes[1]
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio, hop_length=512)), ref=np.max)
    librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr, ax=ax, fmax=2000, hop_length=512)
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Spectrogram (0-2000 Hz) - Perfect Alignment')
    ax.set_xlim(0, duration)
    
    # 3. Clear comparison timeline
    ax = axes[2]
    
    # Excel breathing periods as bars
    if excel_data:
        all_breathing_times = sorted(excel_data['inhale_times'] + excel_data['exhale_times'])
        for i in range(0, len(all_breathing_times), 2):
            if i + 1 < len(all_breathing_times):
                start = all_breathing_times[i]
                end = all_breathing_times[i + 1]
                ax.barh(1, end - start, left=start, height=0.3, color='lightblue', alpha=0.7)
    
    # Model predictions as dots
    breathing_times = [t for t, p in zip(pred_times, predictions) if p == 1]
    nonbreathing_times = [t for t, p in zip(pred_times, predictions) if p == 0]
    
    if breathing_times:
        ax.scatter(breathing_times, [0.2]*len(breathing_times), c='green', s=60, alpha=0.8, label='Model: Breathing')
    if nonbreathing_times:
        ax.scatter(nonbreathing_times, [0.2]*len(nonbreathing_times), c='red', s=60, alpha=0.8, label='Model: Non-breathing')
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Source')
    ax.set_title('Timeline Comparison')
    ax.set_xlim(0, duration)
    ax.set_ylim(0, 1.5)
    ax.set_yticks([0.2, 1.15])
    ax.set_yticklabels(['Model', 'Excel'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{filename}_Option_B_Layered.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_option_c_timeline(audio_file, excel_data, output_dir):
    """Option C: Clean timeline approach."""
    
    filename = audio_file.stem
    audio, sr = librosa.load(audio_file, sr=16000, mono=True)
    duration = len(audio) / sr
    time_axis = np.linspace(0, duration, len(audio))
    
    predictions, pred_times = create_demo_predictions(audio, sr, duration)
    
    # Create clean timeline visualization
    fig, axes = plt.subplots(4, 1, figsize=(16, 10))
    fig.suptitle(f'Option C: Clean Timeline - {filename}', fontsize=16, fontweight='bold')
    
    # 1. Waveform
    ax = axes[0]
    ax.plot(time_axis, audio, color='navy', linewidth=0.8)
    ax.set_ylabel('Amplitude')
    ax.set_title('Audio Waveform')
    ax.set_xlim(0, duration)
    ax.grid(True, alpha=0.3)
    
    # 2. Spectrogram - NO LEGEND
    ax = axes[1]
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio, hop_length=512)), ref=np.max)
    librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr, ax=ax, fmax=2000, hop_length=512)
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Spectrogram (0-2000 Hz)')
    ax.set_xlim(0, duration)
    
    # 3. Excel Ground Truth
    ax = axes[2]
    if excel_data:
        # Inhale markers
        if excel_data['inhale_times']:
            ax.scatter(excel_data['inhale_times'], [1]*len(excel_data['inhale_times']), 
                      c='lightgreen', s=100, marker='^', alpha=0.8, label='Inhale')
        # Exhale markers  
        if excel_data['exhale_times']:
            ax.scatter(excel_data['exhale_times'], [1]*len(excel_data['exhale_times']), 
                      c='lightcoral', s=100, marker='v', alpha=0.8, label='Exhale')
    
    ax.set_ylabel('Excel Data')
    ax.set_title('Ground Truth from Excel (Inhale/Exhale Markers)')
    ax.set_xlim(0, duration)
    ax.set_ylim(0.5, 1.5)
    ax.set_yticks([1])
    ax.set_yticklabels(['Breathing Events'])
    if excel_data and (excel_data['inhale_times'] or excel_data['exhale_times']):
        ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Model Predictions
    ax = axes[3]
    breathing_times = [t for t, p in zip(pred_times, predictions) if p == 1]
    nonbreathing_times = [t for t, p in zip(pred_times, predictions) if p == 0]
    
    if breathing_times:
        ax.scatter(breathing_times, [1]*len(breathing_times), c='darkgreen', s=80, 
                  alpha=0.8, label=f'Breathing ({len(breathing_times)})', marker='o')
    if nonbreathing_times:
        ax.scatter(nonbreathing_times, [0]*len(nonbreathing_times), c='darkred', s=80, 
                  alpha=0.8, label=f'Non-breathing ({len(nonbreathing_times)})', marker='s')
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Model Predictions')
    ax.set_title('Model Classification Results')
    ax.set_xlim(0, duration)
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Non-breathing', 'Breathing'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{filename}_Option_C_Timeline.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_option_d_simple_overlay(audio_file, excel_data, output_dir):
    """Option D: Simple, clean overlay approach."""
    
    filename = audio_file.stem
    audio, sr = librosa.load(audio_file, sr=16000, mono=True)
    duration = len(audio) / sr
    time_axis = np.linspace(0, duration, len(audio))
    
    predictions, pred_times = create_demo_predictions(audio, sr, duration)
    
    # Create simple overlay
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    fig.suptitle(f'Option D: Simple Clean Overlay - {filename}', fontsize=16, fontweight='bold')
    
    # 1. Waveform with clean overlays
    ax = axes[0]
    ax.plot(time_axis, audio, color='black', linewidth=1)
    
    # Excel breathing periods as background
    if excel_data:
        breathing_periods = []
        all_times = sorted(excel_data['inhale_times'] + excel_data['exhale_times'])
        for i in range(0, len(all_times), 2):
            if i + 1 < len(all_times):
                breathing_periods.append((all_times[i], all_times[i + 1]))
        
        for start, end in breathing_periods:
            ax.axvspan(start, end, alpha=0.2, color='lightblue', label='Excel Breathing' if start == breathing_periods[0][0] else "")
    
    # Model predictions as colored regions
    for pred_time, prediction in zip(pred_times, predictions):
        if prediction == 1:
            ax.axvspan(pred_time - 1, pred_time + 1, alpha=0.4, color='green')
    
    ax.set_ylabel('Amplitude')
    ax.set_title('Waveform: Excel Breathing (blue) + Model Predictions (green)')
    ax.set_xlim(0, duration)
    ax.grid(True, alpha=0.3)
    if excel_data:
        ax.legend()
    
    # 2. Spectrogram - NO LEGEND, perfect alignment
    ax = axes[1]
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio, hop_length=512)), ref=np.max)
    librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr, ax=ax, fmax=2000, hop_length=512)
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (seconds)')
    ax.set_title('Spectrogram (0-2000 Hz) - Perfect Alignment')
    ax.set_xlim(0, duration)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{filename}_Option_D_Simple.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Create all visualization options."""
    
    audio_dir = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/RAW sound_ML test sound list")
    output_dir = Path("visualization_options")
    
    print("🎨 CREATING MULTIPLE VISUALIZATION OPTIONS")
    print("=========================================")
    print("Creating 4 different approaches for you to choose from:")
    print("A. Side-by-side Ground Truth vs Predictions")
    print("B. Layered Excel + Model annotations")  
    print("C. Clean timeline approach")
    print("D. Simple overlay approach")
    print()
    
    # Use first file for demo
    audio_files = list(audio_dir.glob("*.wav"))[:1]
    
    for audio_file in audio_files:
        filename = audio_file.stem
        print(f"📊 Creating options for {filename}...")
        
        # Parse Excel data
        excel_data = parse_excel_breathing_data(filename)
        
        # Create all options
        create_option_a_side_by_side(audio_file, excel_data, output_dir)
        print("✅ Option A: Side-by-side created")
        
        create_option_b_layered(audio_file, excel_data, output_dir)
        print("✅ Option B: Layered created")
        
        create_option_c_timeline(audio_file, excel_data, output_dir)
        print("✅ Option C: Timeline created")
        
        create_option_d_simple_overlay(audio_file, excel_data, output_dir)
        print("✅ Option D: Simple overlay created")
    
    print(f"\n🎉 Created 4 visualization options!")
    print(f"📁 Saved to: {output_dir}/")
    print("\n📋 CHOOSE YOUR FAVORITE:")
    print("A. Side-by-side comparison")
    print("B. Layered annotations")
    print("C. Clean timeline")
    print("D. Simple overlay")

if __name__ == "__main__":
    main()
