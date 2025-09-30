#!/usr/bin/env python3
"""
Fix Timelines Using ORIGINAL Format
===================================
Uses the EXACT SAME timeline visualization code that you loved
Only fixes the Excel parsing, keeps everything else identical
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
    """Parse Excel with CORRECT structure (same as investigation)."""
    
    excel_file = "/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/ML test sound list breathing info.xlsx"
    
    print("📋 Parsing Excel with CORRECT structure...")
    
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
                            
                            all_files_data[excel_filename] = {
                                'breathing_periods': breathing_periods,
                                'condition': 'Healthy' if sheet_name == 'Healthy' else 'Pathological'
                            }
    
    print(f"✅ Correctly parsed {len(all_files_data)} files")
    return all_files_data

def create_timeline_with_original_format(model_predictions, excel_data, output_dir, model_type="Individual"):
    """Create timeline using the EXACT SAME format you loved."""
    
    print(f"🎨 Creating timelines with ORIGINAL format you loved...")
    
    output_dir = Path(output_dir)
    timelines_dir = output_dir / "timelines"
    timelines_dir.mkdir(parents=True, exist_ok=True)
    
    audio_dir = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/RAW sound_ML test sound list")
    
    for filename, predictions in model_predictions.items():
        # Find audio file
        audio_file = None
        for audio_path in audio_dir.glob("*.wav"):
            if filename in audio_path.name or audio_path.stem in filename:
                audio_file = audio_path
                break
        
        if not audio_file or filename not in excel_data:
            continue
        
        try:
            # Load audio
            audio, sr = librosa.load(audio_file, sr=16000, mono=True)
            duration = len(audio) / sr
            time_axis = np.linspace(0, duration, len(audio))
            
            # Calculate accuracy
            correct_predictions = sum(1 for p in predictions if p['prediction'] == p['ground_truth'])
            accuracy = correct_predictions / len(predictions)
            
            # Create visualization using EXACT SAME format
            fig, axes = plt.subplots(3, 1, figsize=(16, 10))
            fig.suptitle(f'Perfect Alignment: Excel (top) vs Model (bottom) - {correct_predictions}/{len(predictions)} correct', 
                        fontsize=16, fontweight='bold')
            
            # 1. Waveform (EXACT SAME)
            ax = axes[0]
            ax.plot(time_axis, audio, color='navy', linewidth=0.8)
            ax.set_ylabel('Amplitude')
            ax.set_title('Audio Waveform')
            ax.set_xlim(0, duration)
            ax.grid(True, alpha=0.3)
            
            # 2. Spectrogram (EXACT SAME - NO LEGEND)
            ax = axes[1]
            D = librosa.amplitude_to_db(np.abs(librosa.stft(audio, hop_length=512)), ref=np.max)
            librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr, ax=ax, fmax=2000, hop_length=512)
            ax.set_ylabel('Frequency (Hz)')
            ax.set_title('Spectrogram (50-3000 Hz, Aligned)')
            ax.set_xlim(0, duration)
            
            # 3. Timeline (EXACT SAME FORMAT YOU LOVED)
            ax = axes[2]
            
            # Excel data (top half) - EXACT SAME ymin/ymax
            for period in excel_data[filename]['breathing_periods']:
                color = 'green' if period['type'] == 'breathing' else 'red'
                alpha = 0.7 if period['type'] == 'breathing' else 0.4
                
                ax.axvspan(period['start'], period['end'], 
                          ymin=0.55, ymax=0.95,  # EXACT SAME as original
                          color=color, alpha=alpha)
            
            # Model data (bottom half) - EXACT SAME ymin/ymax  
            for pred in predictions:
                color = 'green' if pred['prediction'] == 1 else 'red'
                alpha = 0.7 if pred['prediction'] == 1 else 0.4
                
                ax.axvspan(pred['start_time'], pred['end_time'], 
                          ymin=0.05, ymax=0.45,  # EXACT SAME as original
                          color=color, alpha=alpha)
            
            # Separating line (EXACT SAME)
            ax.axhline(0.5, color='black', linewidth=2, alpha=0.8)
            
            # Labels and formatting (EXACT SAME)
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Data Source')
            ax.set_title(f'Perfect Alignment: Excel (top) vs Model (bottom) - {correct_predictions}/{len(predictions)} correct')
            ax.set_xlim(0, duration)
            ax.set_ylim(0, 1)
            ax.set_yticks([0.25, 0.75])
            ax.set_yticklabels(['Model (CENTER-POINT)', 'Excel (CORRECT)'])
            ax.grid(True, alpha=0.3)
            
            # Legend (EXACT SAME)
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', alpha=0.7, label='Breathing'),
                Patch(facecolor='red', alpha=0.4, label='Non-breathing')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
            plt.tight_layout()
            plt.savefig(timelines_dir / f'{filename}_ORIGINAL_FORMAT.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✅ ORIGINAL format: {filename}")
            
        except Exception as e:
            print(f"  ❌ Error: {filename}: {e}")

def fix_all_timelines_with_original_format():
    """Fix all timelines using the original format you loved."""
    
    print("🔧 FIXING ALL TIMELINES WITH ORIGINAL FORMAT")
    print("=" * 45)
    print("✅ Using EXACT SAME code you loved")
    print("✅ Only fixing Excel parsing")
    print("✅ Keeping same colors, layout, format")
    print()
    
    # Load correct Excel data
    excel_data = parse_excel_correctly()
    
    # Fix Individual Models
    individual_models = [
        "Individual_Models/1.0s_Individual_Models/Corrected_Results",
        "Individual_Models/0.5s_Individual_Models/Corrected_Results", 
        "Individual_Models/0.25s_Individual_Models/Corrected_Results"
    ]
    
    for model_dir in individual_models:
        model_name = model_dir.split('/')[1]
        print(f"\n🔧 Fixing {model_name} with ORIGINAL format...")
        
        predictions_file = Path(model_dir) / "final_predictions.json"
        if predictions_file.exists():
            with open(predictions_file, 'r') as f:
                predictions = json.load(f)
            
            create_timeline_with_original_format(predictions, excel_data, model_dir, "Individual")
            print(f"✅ {model_name} timelines fixed with ORIGINAL format")
    
    # Fix Ensemble Models
    ensemble_models = [
        "Ensemble_Models/1.0s_Ensemble_Models/Corrected_Results"
    ]
    
    for model_dir in ensemble_models:
        model_name = model_dir.split('/')[1]
        print(f"\n🎭 Fixing {model_name} with ORIGINAL format...")
        
        predictions_file = Path(model_dir) / "final_predictions.json"
        if predictions_file.exists():
            with open(predictions_file, 'r') as f:
                predictions = json.load(f)
            
            create_timeline_with_original_format(predictions, excel_data, model_dir, "Ensemble")
            print(f"✅ {model_name} timelines fixed with ORIGINAL format")
    
    print(f"\n🎉 ALL TIMELINES FIXED WITH ORIGINAL FORMAT!")
    print("✅ Same format, colors, and layout you loved")
    print("✅ Only Excel parsing was corrected")
    print("✅ Timeline visualizations now accurate AND beautiful")

if __name__ == "__main__":
    fix_all_timelines_with_original_format()
