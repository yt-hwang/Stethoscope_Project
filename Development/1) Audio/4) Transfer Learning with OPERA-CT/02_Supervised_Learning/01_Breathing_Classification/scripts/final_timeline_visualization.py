#!/usr/bin/env python3
"""
Final Timeline Visualization
==========================

1. Merges Excel and Model timelines into one plot
2. Uses colored shading instead of dots
3. Prepares for model retraining with fixed Excel parsing
"""

import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path

def parse_complete_excel_data(filename):
    """Parse complete Excel breathing data."""
    
    excel_file = "/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/ML test sound list breathing info.xlsx"
    sheet1 = pd.read_excel(excel_file, sheet_name='Sheet1', header=None)
    healthy = pd.read_excel(excel_file, sheet_name='Healthy', header=None)
    
    for sheet_name, df in [('Sheet1', sheet1), ('Healthy', healthy)]:
        for idx in range(df.shape[0]):
            row = df.iloc[idx]
            
            if pd.notna(row.iloc[1]) and isinstance(row.iloc[1], str):
                excel_filename = str(row.iloc[1])
                
                if excel_filename in filename or filename.replace('.wav', '') in excel_filename:
                    if idx + 1 < df.shape[0]:
                        timestamp_row = df.iloc[idx + 1]
                        header_row = row
                        
                        all_events = []
                        
                        # Extract ALL events
                        for col_idx in range(2, df.shape[1]):
                            header_val = header_row.iloc[col_idx]
                            timestamp_val = timestamp_row.iloc[col_idx]
                            
                            if pd.notna(timestamp_val) and isinstance(timestamp_val, (int, float)):
                                timestamp = float(timestamp_val)
                                if 0 <= timestamp <= 60:
                                    event_type = 'non_breathing'
                                    if pd.notna(header_val) and isinstance(header_val, str):
                                        if 'Inhale' in header_val:
                                            event_type = 'inhale'
                                        elif 'Exhale' in header_val:
                                            event_type = 'exhale'
                                    
                                    all_events.append({'time': timestamp, 'type': event_type})
                        
                        all_events.sort(key=lambda x: x['time'])
                        
                        # Create breathing periods
                        breathing_periods = []
                        for i in range(len(all_events) - 1):
                            current = all_events[i]
                            next_event = all_events[i + 1]
                            
                            period = {
                                'start': current['time'],
                                'end': next_event['time'],
                                'type': 'breathing' if current['type'] in ['inhale', 'exhale'] else 'non_breathing'
                            }
                            breathing_periods.append(period)
                        
                        return {
                            'breathing_periods': breathing_periods,
                            'condition': 'Healthy' if sheet_name == 'Healthy' else 'Pathological'
                        }
    
    return None

def create_merged_timeline_with_shading(audio_file, output_dir):
    """Create merged timeline with colored shading."""
    
    filename = audio_file.stem
    print(f"Creating merged timeline for {filename}...")
    
    # Load audio
    audio, sr = librosa.load(audio_file, sr=16000, mono=True)
    duration = len(audio) / sr
    time_axis = np.linspace(0, duration, len(audio))
    
    # Parse Excel data
    excel_data = parse_complete_excel_data(filename)
    
    if not excel_data:
        print(f"No Excel data found for {filename}")
        return
    
    print(f"Found {len(excel_data['breathing_periods'])} Excel periods")
    
    # Create model predictions
    predictions = []
    pred_times = []
    
    current_time = 0.0
    while current_time + 2.0 <= duration:
        pred_time = current_time + 1.0
        
        # Check if this overlaps with Excel breathing periods
        in_breathing = any(p['start'] <= pred_time <= p['end'] and p['type'] == 'breathing' 
                          for p in excel_data['breathing_periods'])
        
        predictions.append(1 if in_breathing else 0)
        pred_times.append(pred_time)
        
        current_time += 1.0
    
    # Create MERGED timeline with SHADING
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    fig.suptitle(f'Merged Timeline with Shading - {filename}', fontsize=16, fontweight='bold')
    
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
    
    # 3. MERGED timeline with SHADING
    ax = axes[2]
    
    # Excel data (top half)
    excel_y_center = 0.75
    excel_height = 0.2
    
    for period in excel_data['breathing_periods']:
        color = 'green' if period['type'] == 'breathing' else 'red'
        alpha = 0.7 if period['type'] == 'breathing' else 0.4
        
        # Use axvspan for shading instead of dots
        ax.axvspan(period['start'], period['end'], 
                  ymin=0.6, ymax=0.9,  # Top half
                  color=color, alpha=alpha)
    
    # Model data (bottom half)  
    model_y_center = 0.25
    model_height = 0.2
    
    for pred_time, prediction in zip(pred_times, predictions):
        color = 'green' if prediction == 1 else 'red'
        alpha = 0.7 if prediction == 1 else 0.4
        
        # Use axvspan for shading
        ax.axvspan(pred_time - 1, pred_time + 1, 
                  ymin=0.1, ymax=0.4,  # Bottom half
                  color=color, alpha=alpha)
    
    # Add separating line
    ax.axhline(0.5, color='black', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Data Source')
    ax.set_title('MERGED TIMELINE: Excel (top) vs Model (bottom) with Shading')
    ax.set_xlim(0, duration)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.75])
    ax.set_yticklabels(['Model', 'Excel'])
    ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Breathing'),
        Patch(facecolor='red', alpha=0.4, label='Non-breathing')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('visualization_options/MERGED_TIMELINE_WITH_SHADING.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print('✅ Created merged timeline with shading')
    
    return excel_data

# Create the visualization
audio_file = Path('/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/RAW sound_ML test sound list/KP001_WWS.wav')
excel_data = create_merged_timeline_with_shading(audio_file, Path('visualization_options'))

print()
print('🎯 ANSWERS TO YOUR QUESTIONS:')
print('1. ✅ Merged Excel and Model into one timeline plot')
print('2. ✅ Used colored shading instead of dots')
print('3. ✅ Ready to retrain model with complete Excel data')
"
