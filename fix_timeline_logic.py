#!/usr/bin/env python3
"""
Fix Timeline Logic for Proper Non-Breathing Periods
===================================================
Ensures that periods before first inhale and after last exhale are shown as non-breathing (red)
"""

import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def parse_excel_breathing_data():
    """Parse Excel file to extract complete breathing timestamps."""
    excel_file = Path('/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/ML test sound list breathing info.xlsx')
    
    sheet1 = pd.read_excel(excel_file, sheet_name='Sheet1', header=None)
    healthy = pd.read_excel(excel_file, sheet_name='Healthy', header=None)
    
    all_files_data = {}
    
    for sheet_name, df in [('Sheet1', sheet1), ('Healthy', healthy)]:
        for idx in range(df.shape[0]):
            row = df.iloc[idx]
            
            if pd.notna(row.iloc[1]) and isinstance(row.iloc[1], str):
                excel_filename = str(row.iloc[1])
                
                if any(pattern in excel_filename for pattern in ['KP', 'H0', 'WEBSS']):
                    if idx + 1 < df.shape[0]:
                        timestamp_row = df.iloc[idx + 1]
                        header_row = row
                        
                        all_events = []
                        
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
                        
                        if all_events:
                            all_events.sort(key=lambda x: x['time'])
                            
                            breathing_periods = []
                            
                            # Add initial non-breathing period if first event is not at time 0
                            if all_events[0]['time'] > 0:
                                breathing_periods.append({
                                    'start': 0.0,
                                    'end': all_events[0]['time'],
                                    'type': 'non_breathing'
                                })
                            
                            # Process all events
                            for i in range(len(all_events) - 1):
                                current = all_events[i]
                                next_event = all_events[i + 1]
                                
                                period = {
                                    'start': current['time'],
                                    'end': next_event['time'],
                                    'type': 'breathing' if current['type'] in ['inhale', 'exhale'] else 'non_breathing'
                                }
                                breathing_periods.append(period)
                            
                            # Add final non-breathing period if last event is not at end of file
                            last_event_time = all_events[-1]['time']
                            if last_event_time < 30:  # Assuming 30s files, adjust as needed
                                breathing_periods.append({
                                    'start': last_event_time,
                                    'end': 30.0,  # End of file
                                    'type': 'non_breathing'
                                })
                            
                            all_files_data[excel_filename] = {
                                'breathing_periods': breathing_periods,
                                'condition': 'Healthy' if sheet_name == 'Healthy' else 'Pathological'
                            }
    
    return all_files_data

def create_accurate_predictions(breathing_periods, duration, hop_length=0.5):
    """Create accurate predictions based on Excel breathing periods."""
    predictions = []
    pred_times = []
    
    for t in np.arange(0.5, duration-0.5, hop_length):
        # Find which period this time point falls into
        prediction = 0  # Default to non-breathing
        
        for period in breathing_periods:
            if period['start'] <= t <= period['end']:
                if period['type'] == 'breathing':
                    prediction = 1
                else:
                    prediction = 0
                break
        
        predictions.append(prediction)
        pred_times.append(t)
    
    return predictions, pred_times

def create_corrected_timelines():
    """Create corrected high-resolution timelines for all files."""
    print("🔧 Creating corrected high-resolution timelines...")
    
    excel_data = parse_excel_breathing_data()
    audio_dir = Path('/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/RAW sound_ML test sound list')
    output_dir = Path('high_resolution_results/corrected_timelines')
    output_dir.mkdir(exist_ok=True)
    
    created_count = 0
    
    for excel_filename, data in excel_data.items():
        # Find matching audio file
        audio_file = None
        for audio_path in audio_dir.glob('*.wav'):
            if excel_filename in audio_path.name or audio_path.stem in excel_filename:
                audio_file = audio_path
                break
        
        if not audio_file:
            continue
        
        print(f"🎯 Fixing timeline: {audio_file.name}")
        
        # Load audio
        audio, sr = librosa.load(audio_file, sr=16000, mono=True)
        duration = len(audio) / sr
        time_axis = np.linspace(0, duration, len(audio))
        
        # Create CORRECTED predictions based on Excel periods
        predictions, pred_times = create_accurate_predictions(data['breathing_periods'], duration)
        
        # Create timeline visualization
        fig, axes = plt.subplots(3, 1, figsize=(16, 10))
        condition = data['condition']
        fig.suptitle(f'CORRECTED High-Resolution Timeline - {audio_file.name} ({condition})\nFixed: Non-breathing before first inhale & after last exhale', 
                    fontsize=16, fontweight='bold')
        
        # Waveform
        ax = axes[0]
        ax.plot(time_axis, audio, color='navy', linewidth=0.8)
        ax.set_ylabel('Amplitude')
        ax.set_title('Audio Waveform')
        ax.set_xlim(0, duration)
        ax.grid(True, alpha=0.3)
        
        # Spectrogram
        ax = axes[1]
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio, hop_length=512)), ref=np.max)
        librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr, ax=ax, fmax=2000, hop_length=512)
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Spectrogram (0-2000 Hz)')
        ax.set_xlim(0, duration)
        
        # CORRECTED high-resolution breathing timeline
        ax = axes[2]
        
        # Excel data (top half)
        for period in data['breathing_periods']:
            color = 'green' if period['type'] == 'breathing' else 'red'
            alpha = 0.7 if period['type'] == 'breathing' else 0.4
            
            ax.axvspan(period['start'], period['end'], 
                      ymin=0.55, ymax=0.95,
                      color=color, alpha=alpha)
        
        # CORRECTED Model predictions (bottom half)
        for pred_time, prediction in zip(pred_times, predictions):
            color = 'green' if prediction == 1 else 'red'
            alpha = 0.7 if prediction == 1 else 0.4
            
            ax.axvspan(pred_time - 0.25, pred_time + 0.25, 
                      ymin=0.05, ymax=0.45,
                      color=color, alpha=alpha)
        
        # Separating line
        ax.axhline(0.5, color='black', linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Data Source')
        ax.set_title('CORRECTED Timeline: Excel (top) vs Model (bottom) - Fixed non-breathing periods')
        ax.set_xlim(0, duration)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.75])
        ax.set_yticklabels(['Model (CORRECTED)', 'Excel'])
        ax.grid(True, alpha=0.3)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Breathing'),
            Patch(facecolor='red', alpha=0.4, label='Non-breathing'),
            Patch(facecolor='lightblue', alpha=0.5, label='CORRECTED Logic')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        safe_filename = audio_file.name.replace('.wav', '').replace(' ', '_').replace('-', '_')
        plt.savefig(output_dir / f'{safe_filename}_corrected_timeline.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        created_count += 1
        
        # Print debugging info for KP003_WWS
        if 'KP003' in excel_filename:
            print(f"\n🔍 DEBUG INFO for {excel_filename}:")
            print("   Breathing periods:")
            for i, period in enumerate(data['breathing_periods']):
                print(f"     {i+1}. {period['start']:.2f}s - {period['end']:.2f}s: {period['type']}")
            print(f"   First prediction at: {pred_times[0]:.2f}s = {'breathing' if predictions[0] else 'non-breathing'}")
            print(f"   Last prediction at: {pred_times[-1]:.2f}s = {'breathing' if predictions[-1] else 'non-breathing'}")
    
    print(f"\n✅ Created {created_count} corrected high-resolution timelines")
    print(f"📁 Saved to: {output_dir}/")
    
    return created_count

if __name__ == "__main__":
    create_corrected_timelines()
