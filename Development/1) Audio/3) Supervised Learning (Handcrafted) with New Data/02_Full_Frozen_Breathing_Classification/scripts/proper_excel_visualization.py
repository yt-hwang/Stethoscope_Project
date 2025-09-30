#!/usr/bin/env python3
"""
Proper Excel Visualization with Complete Data
============================================

Fixes:
1. Extracts ALL breathing events (not just first few)
2. Creates start/end periods for each breathing event
3. Recreates the horizontal timeline option with proper Excel data
"""

import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path

def parse_complete_excel_data(filename):
    """Parse ALL Excel breathing data correctly."""
    
    excel_file = "/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/ML test sound list breathing info.xlsx"
    sheet1 = pd.read_excel(excel_file, sheet_name='Sheet1', header=None)
    healthy = pd.read_excel(excel_file, sheet_name='Healthy', header=None)
    
    print(f"Parsing complete Excel data for {filename}...")
    
    for sheet_name, df in [('Sheet1', sheet1), ('Healthy', healthy)]:
        for idx in range(df.shape[0]):
            row = df.iloc[idx]
            
            if pd.notna(row.iloc[1]) and isinstance(row.iloc[1], str):
                excel_filename = str(row.iloc[1])
                
                if excel_filename in filename or filename.replace('.wav', '') in excel_filename:
                    print(f"Found {excel_filename} at row {idx}")
                    
                    # Get ALL timestamps from the row
                    if idx + 1 < df.shape[0]:
                        timestamp_row = df.iloc[idx + 1]
                        header_row = row
                        
                        all_events = []
                        
                        # Parse ALL columns for breathing events
                        for col_idx in range(2, df.shape[1]):
                            header_val = header_row.iloc[col_idx]
                            timestamp_val = timestamp_row.iloc[col_idx]
                            
                            if pd.notna(timestamp_val) and isinstance(timestamp_val, (int, float)):
                                timestamp = float(timestamp_val)
                                if 0 <= timestamp <= 60:
                                    event_type = 'non_breathing'  # Default
                                    
                                    if pd.notna(header_val) and isinstance(header_val, str):
                                        if 'Inhale' in header_val:
                                            event_type = 'inhale'
                                        elif 'Exhale' in header_val:
                                            event_type = 'exhale'
                                    
                                    all_events.append({
                                        'time': timestamp,
                                        'type': event_type,
                                        'header': str(header_val) if pd.notna(header_val) else 'unknown'
                                    })
                        
                        # Sort by time
                        all_events.sort(key=lambda x: x['time'])
                        
                        print(f"Found {len(all_events)} total events")
                        
                        # Create breathing periods
                        breathing_periods = []
                        current_period = None
                        
                        for event in all_events:
                            if event['type'] == 'inhale':
                                # Start of breathing period
                                current_period = {'start': event['time'], 'type': 'breathing'}
                            elif event['type'] == 'exhale' and current_period:
                                # End of breathing period
                                current_period['end'] = event['time']
                                breathing_periods.append(current_period)
                                current_period = None
                        
                        # Extract just inhales and exhales
                        inhale_times = [e['time'] for e in all_events if e['type'] == 'inhale']
                        exhale_times = [e['time'] for e in all_events if e['type'] == 'exhale']
                        
                        print(f"Inhales: {len(inhale_times)} - {inhale_times}")
                        print(f"Exhales: {len(exhale_times)} - {exhale_times}")
                        print(f"Breathing periods: {len(breathing_periods)}")
                        
                        return {
                            'inhale_times': inhale_times,
                            'exhale_times': exhale_times,
                            'breathing_periods': breathing_periods,
                            'all_events': all_events,
                            'condition': 'Healthy' if sheet_name == 'Healthy' else 'Pathological'
                        }
    
    return None

def create_option_c_horizontal_timeline(audio_file, excel_data, output_dir):
    """Create the horizontal timeline option you wanted."""
    
    filename = audio_file.stem
    audio, sr = librosa.load(audio_file, sr=16000, mono=True)
    duration = len(audio) / sr
    time_axis = np.linspace(0, duration, len(audio))
    
    # Create demo model predictions
    predictions = []
    pred_times = []
    
    current_time = 0.0
    while current_time + 2.0 <= duration:
        pred_time = current_time + 1.0
        
        # Demo prediction
        start_sample = int(current_time * sr)
        end_sample = int((current_time + 2.0) * sr)
        segment = audio[start_sample:end_sample]
        
        rms_energy = np.sqrt(np.mean(segment**2))
        is_breathing = rms_energy > 0.02
        
        predictions.append(1 if is_breathing else 0)
        pred_times.append(pred_time)
        
        current_time += 1.0
    
    # Create horizontal timeline visualization
    fig, axes = plt.subplots(4, 1, figsize=(16, 10))
    fig.suptitle(f'OPTION C: Horizontal Timeline - {filename}', fontsize=16, fontweight='bold')
    
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
    
    # 3. Excel breathing timeline (HORIZONTAL)
    ax = axes[2]
    
    if excel_data:
        # Create breathing periods from Excel data
        all_times = sorted(excel_data['inhale_times'] + excel_data['exhale_times'])
        
        # Plot breathing periods as horizontal bars
        y_pos = 0.5
        for i in range(0, len(all_times), 2):
            if i + 1 < len(all_times):
                start = all_times[i]
                end = all_times[i + 1]
                
                # Determine if this is inhale or exhale period
                if all_times[i] in excel_data['inhale_times']:
                    color = 'lightgreen'
                    label = 'Inhale Period' if i == 0 else ''
                else:
                    color = 'lightcoral'
                    label = 'Exhale Period' if i == 0 else ''
                
                ax.barh(y_pos, end - start, left=start, height=0.3, 
                       color=color, alpha=0.8, label=label)
        
        # Mark individual events
        for inhale_time in excel_data['inhale_times']:
            ax.axvline(inhale_time, color='darkgreen', alpha=0.8, linewidth=2)
        for exhale_time in excel_data['exhale_times']:
            ax.axvline(exhale_time, color='darkred', alpha=0.8, linewidth=2)
    
    ax.set_ylabel('Excel Data')
    ax.set_title(f'EXCEL BREATHING TIMELINE ({len(excel_data[\"inhale_times\"]) if excel_data else 0} inhales, {len(excel_data[\"exhale_times\"]) if excel_data else 0} exhales)')
    ax.set_xlim(0, duration)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.5])
    ax.set_yticklabels(['Breathing Events'])
    if excel_data:
        ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Model predictions timeline (HORIZONTAL)
    ax = axes[3]
    
    # Plot model predictions as horizontal bars
    y_pos = 0.5
    for pred_time, prediction in zip(pred_times, predictions):
        color = 'green' if prediction == 1 else 'red'
        alpha = 0.7 if prediction == 1 else 0.4
        
        ax.barh(y_pos, 2.0, left=pred_time - 1, height=0.3, 
               color=color, alpha=alpha)
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Model Data')
    ax.set_title(f'MODEL PREDICTIONS ({sum(predictions)} breathing windows)')
    ax.set_xlim(0, duration)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.5])
    ax.set_yticklabels(['Predictions'])
    ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Model Breathing'),
        Patch(facecolor='red', alpha=0.4, label='Model Non-breathing')
    ]
    ax.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig('visualization_options/OPTION_C_Horizontal_Timeline.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print('✅ Created Option C: Horizontal Timeline with complete Excel data')

# Execute
audio_file = Path('/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/RAW sound_ML test sound list/KP001_WWS.wav')
excel_data = parse_complete_excel_data('KP001_WWS.wav')

if excel_data:
    create_option_c_horizontal_timeline(audio_file, excel_data, Path('visualization_options'))
else:
    print('No Excel data found')
"
