#!/usr/bin/env python3
"""
Fix Timeline Ground Truth Display
=================================
Recreates timeline visualizations with COMPLETE ground truth display (like individual models)
"""

import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_complete_excel_data():
    """Load complete Excel breathing data for ground truth display."""
    
    excel_file = "/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/ML test sound list breathing info.xlsx"
    
    print("📋 Loading complete Excel breathing data...")
    
    sheet1 = pd.read_excel(excel_file, sheet_name='Sheet1', header=None)
    healthy = pd.read_excel(excel_file, sheet_name='Healthy', header=None)
    
    all_files_data = {}
    
    for sheet_name, df in [('Sheet1', sheet1), ('Healthy', healthy)]:
        print(f"   Processing {sheet_name}...")
        
        for idx in range(df.shape[0]):
            row = df.iloc[idx]
            
            if pd.notna(row.iloc[1]) and isinstance(row.iloc[1], str):
                excel_filename = str(row.iloc[1])
                
                # Check if this looks like a filename
                if any(pattern in excel_filename for pattern in ['KP', 'H0', 'WEBSS']):
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
                        
                        if all_events:
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
                            
                            all_files_data[excel_filename] = {
                                'breathing_periods': breathing_periods,
                                'condition': 'Healthy' if sheet_name == 'Healthy' else 'Pathological',
                                'total_events': len(all_events)
                            }
    
    print(f"✅ Loaded complete Excel data for {len(all_files_data)} files")
    return all_files_data

def create_corrected_timeline_visualizations():
    """Create corrected timeline visualizations with complete ground truth."""
    
    print("🎨 Creating Corrected Timeline Visualizations...")
    
    # Load the ensemble results
    results_dir = Path("Center_Point_Labeling_Results")
    timelines_dir = results_dir / "timelines"
    
    # Load final predictions to get test data
    with open(results_dir / "final_predictions.json", 'r') as f:
        final_predictions = json.load(f)
    
    # Load complete Excel data for ground truth
    excel_data = load_complete_excel_data()
    
    audio_dir = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/RAW sound_ML test sound list")
    
    print(f"📁 Audio directory: {audio_dir}")
    print(f"📊 Creating corrected timelines for {len(final_predictions)} files...")
    
    created_count = 0
    
    for filename, predictions in final_predictions.items():
        print(f"  🎵 Processing {filename}...")
        
        # Find audio file
        audio_file = None
        for audio_path in audio_dir.glob("*.wav"):
            if filename in audio_path.name or audio_path.stem in filename:
                audio_file = audio_path
                break
        
        if not audio_file:
            print(f"    ⚠️ Audio file not found for {filename}")
            continue
        
        # Get complete Excel data for this file
        if filename not in excel_data:
            print(f"    ⚠️ Excel data not found for {filename}")
            continue
        
        try:
            # Load audio
            y, sr = librosa.load(audio_file, sr=22050)
            duration = len(y) / sr
            
            # Calculate accuracy from predictions
            correct_predictions = sum(1 for p in predictions if p['prediction'] == p['ground_truth'])
            accuracy = correct_predictions / len(predictions)
            
            # Create visualization
            fig, axes = plt.subplots(3, 1, figsize=(15, 10))
            
            # 1. Waveform
            time_axis = np.linspace(0, duration, len(y))
            axes[0].plot(time_axis, y, color='blue', alpha=0.7)
            axes[0].set_title(f'{filename} - Waveform')
            axes[0].set_ylabel('Amplitude')
            axes[0].grid(True, alpha=0.3)
            axes[0].set_xlim(0, duration)
            
            # 2. Spectrogram (0-2000 Hz, same as individual models)
            axes[1].specgram(y, Fs=sr, vmax=0, vmin=-60, cmap='viridis')
            axes[1].set_title('Spectrogram (0-2000 Hz)')
            axes[1].set_ylabel('Frequency (Hz)')
            axes[1].set_ylim(0, 2000)
            axes[1].set_xlim(0, duration)
            
            # 3. Timeline (CORRECTED: Complete Ground Truth + Test Predictions)
            axes[2].set_title(f'Ensemble Predictions vs Complete Ground Truth\n'
                            f'Accuracy: {accuracy:.1%} ({correct_predictions}/{len(predictions)})')
            axes[2].set_xlabel('Time (seconds)')
            axes[2].set_ylabel('Track')
            axes[2].set_xlim(0, duration)
            axes[2].set_ylim(-0.5, 1.5)
            axes[2].grid(True, alpha=0.3)
            
            # Add COMPLETE ground truth from Excel (top track)
            file_excel_data = excel_data[filename]
            breathing_periods = file_excel_data['breathing_periods']
            
            # Show complete Excel breathing data
            for i, period in enumerate(breathing_periods):
                if period['type'] == 'breathing':
                    # Breathing period (green)
                    axes[2].axvspan(period['start'], period['end'], ymin=0.7, ymax=0.9,
                                  color='green', alpha=0.3, 
                                  label='Ground Truth (Breathing)' if i == 0 else "")
                else:
                    # Non-breathing period (red)
                    axes[2].axvspan(period['start'], period['end'], ymin=0.7, ymax=0.9,
                                  color='red', alpha=0.3,
                                  label='Ground Truth (Non-Breathing)' if i == 0 and period['type'] == 'non_breathing' else "")
            
            # Add ensemble predictions (bottom track - ONLY test segments)
            prediction_added = False
            for pred in predictions:
                pred_color = 'green' if pred['prediction'] == 1 else 'red'
                axes[2].axvspan(pred['start_time'], pred['end_time'], ymin=0.1, ymax=0.3,
                              color=pred_color, alpha=0.7, 
                              label='Ensemble Predictions' if not prediction_added else "")
                prediction_added = True
            
            # Set y-axis labels and legend
            axes[2].set_yticks([0.2, 0.8])
            axes[2].set_yticklabels(['Ensemble\n(Test Only)', 'Ground Truth\n(Complete)'])
            axes[2].legend(loc='upper right', fontsize=9)
            
            plt.tight_layout()
            
            # Save corrected timeline
            timeline_path = timelines_dir / f'{filename}_ensemble_timeline.png'
            plt.savefig(timeline_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"    ✅ Fixed timeline: {timeline_path.name}")
            created_count += 1
            
        except Exception as e:
            print(f"    ❌ Error creating timeline for {filename}: {e}")
    
    print(f"\n✅ Fixed {created_count} timeline visualizations")
    return created_count

def main():
    """Main execution function."""
    
    print("🔧 FIXING TIMELINE GROUND TRUTH DISPLAY")
    print("=" * 40)
    print("✅ Will show COMPLETE Excel breathing data on top track")
    print("✅ Will show ONLY test predictions on bottom track")
    print("✅ Same format as individual model timelines")
    print()
    
    created_count = create_corrected_timeline_visualizations()
    
    print(f"\n🎉 TIMELINE FIX COMPLETE!")
    print(f"✅ Fixed {created_count} timeline images")
    print(f"📁 Location: Center_Point_Labeling_Results/timelines/")
    print(f"🎯 Now matches individual model timeline format!")

if __name__ == "__main__":
    main()
