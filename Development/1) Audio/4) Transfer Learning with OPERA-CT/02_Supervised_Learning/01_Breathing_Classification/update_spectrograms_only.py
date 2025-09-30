#!/usr/bin/env python3
"""
Update Spectrograms Only
========================
Regenerates timeline images with improved mel spectrograms only
Everything else stays exactly the same
"""

import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def parse_excel_correctly():
    """Parse Excel data."""
    excel_file = "/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/ML test sound list breathing info.xlsx"
    
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
                            for i in range(len(all_events) - 1):
                                current = all_events[i]
                                next_event = all_events[i + 1]
                                period = {
                                    'start': current['time'],
                                    'end': next_event['time'],
                                    'type': 'breathing' if current['type'] in ['inhale', 'exhale'] else 'non_breathing'
                                }
                                breathing_periods.append(period)
                            
                            all_files_data[excel_filename] = {'complete_timeline': breathing_periods}
    
    return all_files_data

def update_spectrograms_for_all_models():
    """Update spectrograms for all models."""
    
    print("🎨 UPDATING SPECTROGRAMS FOR ALL MODELS")
    print("=" * 40)
    print("✅ Using improved mel spectrogram")
    print("✅ Everything else stays the same")
    print()
    
    # Load Excel data
    excel_data = parse_excel_correctly()
    audio_dir = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/RAW sound_ML test sound list")
    
    # All model directories
    model_dirs = [
        "Individual_Models/1.0s_Individual_Models/Center_Point_Labeling_Results",
        "Individual_Models/0.5s_Individual_Models/Center_Point_Labeling_Results",
        "Individual_Models/0.25s_Individual_Models/Center_Point_Labeling_Results",
        "Ensemble_Models/1.0s_Ensemble_Models/Center_Point_Labeling_Results",
        "Ensemble_Models/0.5s_Ensemble_Models/Center_Point_Labeling_Results",
        "Ensemble_Models/0.25s_Ensemble_Models/Center_Point_Labeling_Results"
    ]
    
    for model_dir in model_dirs:
        model_name = model_dir.split('/')[1]
        print(f"🎨 Updating {model_name}...")
        
        predictions_file = Path(model_dir) / "final_predictions.json"
        if not predictions_file.exists():
            print(f"  ❌ No predictions file found")
            continue
        
        with open(predictions_file, 'r') as f:
            model_predictions = json.load(f)
        
        timelines_dir = Path(model_dir) / "timelines"
        
        for filename, predictions in model_predictions.items():
            if filename not in excel_data:
                continue
            
            # Find audio file
            audio_file = None
            for audio_path in audio_dir.glob("*.wav"):
                if filename in audio_path.name or audio_path.stem in filename:
                    audio_file = audio_path
                    break
            
            if not audio_file:
                continue
            
            try:
                # Load audio
                audio, sr = librosa.load(audio_file, sr=16000, mono=True)
                duration = len(audio) / sr
                time_axis = np.linspace(0, duration, len(audio))
                
                # Calculate accuracy
                correct = sum(1 for p in predictions if p['prediction'] == p['ground_truth'])
                total = len(predictions)
                accuracy = correct / total
                
                # Create visualization with IMPROVED SPECTROGRAM
                fig, axes = plt.subplots(3, 1, figsize=(16, 10))
                fig.suptitle(f'Perfect Alignment: Excel (top) vs Model (bottom) - {correct}/{total} ({accuracy:.1%})', 
                            fontsize=16, fontweight='bold')
                
                # 1. Waveform (SAME)
                axes[0].plot(time_axis, audio, color='navy', linewidth=0.8)
                axes[0].set_ylabel('Amplitude')
                axes[0].set_title('Audio Waveform')
                axes[0].set_xlim(0, duration)
                axes[0].grid(True, alpha=0.3)
                
                # 2. IMPROVED MEL SPECTROGRAM
                mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, fmax=2000, n_mels=128)
                log_mel = librosa.power_to_db(mel_spec, ref=np.max)
                librosa.display.specshow(log_mel, y_axis='mel', x_axis='time', sr=sr, ax=axes[1], fmax=2000)
                axes[1].set_title('Spectrogram (0-2000 Hz)')
                axes[1].set_ylabel('Frequency (Hz)')
                axes[1].set_xlim(0, duration)
                
                # 3. Timeline (EXACT SAME)
                ax = axes[2]
                
                # Excel data (top half) - SAME
                for period in excel_data[filename]['complete_timeline']:
                    color = 'green' if period['type'] == 'breathing' else 'red'
                    alpha = 0.7 if period['type'] == 'breathing' else 0.4
                    ax.axvspan(period['start'], period['end'], ymin=0.55, ymax=0.95, color=color, alpha=alpha)
                
                # Model data (bottom half) - SAME
                for pred in predictions:
                    color = 'green' if pred['prediction'] == 1 else 'red'
                    alpha = 0.7 if pred['prediction'] == 1 else 0.4
                    ax.axvspan(pred['start_time'], pred['end_time'], ymin=0.05, ymax=0.45, color=color, alpha=alpha)
                
                # Formatting (SAME)
                ax.axhline(0.5, color='black', linewidth=2, alpha=0.8)
                ax.set_xlabel('Time (seconds)')
                ax.set_ylabel('Data Source')
                ax.set_title(f'Perfect Alignment: Excel (top) vs Model (bottom) - {correct}/{total} ({accuracy:.1%})')
                ax.set_xlim(0, duration)
                ax.set_ylim(0, 1)
                ax.set_yticks([0.25, 0.75])
                ax.set_yticklabels(['Model (CENTER-POINT)', 'Excel (CORRECT)'])
                ax.grid(True, alpha=0.3)
                
                # Legend (SAME)
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='green', alpha=0.7, label='Breathing'),
                    Patch(facecolor='red', alpha=0.4, label='Non-breathing')
                ]
                ax.legend(handles=legend_elements, loc='upper right')
                
                plt.tight_layout()
                plt.savefig(timelines_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                print(f"  ❌ Error with {filename}: {e}")
        
        print(f"  ✅ Updated spectrograms for {model_name}")
    
    print(f"\n🎉 ALL SPECTROGRAMS UPDATED!")
    print("✅ Better mel spectrogram visualization")
    print("✅ Everything else unchanged")

if __name__ == "__main__":
    update_spectrograms_for_all_models()
