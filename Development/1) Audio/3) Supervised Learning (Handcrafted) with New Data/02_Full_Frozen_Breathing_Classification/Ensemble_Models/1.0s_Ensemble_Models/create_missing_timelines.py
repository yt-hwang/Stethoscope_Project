#!/usr/bin/env python3
"""
Create Missing Timeline Images for 1.0s Ensemble Model
======================================================
Generates the timeline visualizations that were missing from the initial run
"""

import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def create_timeline_visualizations():
    """Create timeline visualizations for all test files."""
    
    print("🎨 Creating Missing Timeline Visualizations...")
    
    # Load the ensemble results
    results_dir = Path("Center_Point_Labeling_Results")
    timelines_dir = results_dir / "timelines"
    debug_dir = results_dir / "debug_csvs"
    
    # Load final predictions to get test data
    with open(results_dir / "final_predictions.json", 'r') as f:
        final_predictions = json.load(f)
    
    audio_dir = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/RAW sound_ML test sound list")
    
    print(f"📁 Audio directory: {audio_dir}")
    print(f"📊 Creating timelines for {len(final_predictions)} files...")
    
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
            
            # 3. Timeline (Ensemble predictions vs Ground Truth)
            axes[2].set_title(f'Ensemble Predictions vs Ground Truth\n'
                            f'Accuracy: {accuracy:.1%} ({correct_predictions}/{len(predictions)})')
            axes[2].set_xlabel('Time (seconds)')
            axes[2].set_ylabel('Prediction')
            axes[2].set_xlim(0, duration)
            axes[2].set_ylim(-0.5, 1.5)
            axes[2].grid(True, alpha=0.3)
            
            # Add predictions
            for i, pred in enumerate(predictions):
                # Ground truth (top)
                gt_color = 'green' if pred['ground_truth'] == 1 else 'red'
                axes[2].axvspan(pred['start_time'], pred['end_time'], ymin=0.7, ymax=0.9,
                              color=gt_color, alpha=0.3, label='Ground Truth' if i == 0 else "")
                
                # Ensemble prediction (bottom)
                pred_color = 'green' if pred['prediction'] == 1 else 'red'
                axes[2].axvspan(pred['start_time'], pred['end_time'], ymin=0.1, ymax=0.3,
                              color=pred_color, alpha=0.7, label='Ensemble' if i == 0 else "")
            
            axes[2].legend()
            axes[2].set_yticks([0.2, 0.8])
            axes[2].set_yticklabels(['Ensemble', 'Ground Truth'])
            
            plt.tight_layout()
            
            # Save timeline
            timeline_path = timelines_dir / f'{filename}_ensemble_timeline.png'
            plt.savefig(timeline_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"    ✅ Created timeline: {timeline_path.name}")
            created_count += 1
            
        except Exception as e:
            print(f"    ❌ Error creating timeline for {filename}: {e}")
    
    print(f"\n✅ Created {created_count} timeline visualizations")
    return created_count

def main():
    """Main execution function."""
    
    print("🎨 CREATING MISSING TIMELINE IMAGES")
    print("=" * 35)
    
    created_count = create_timeline_visualizations()
    
    print(f"\n🎉 TIMELINE CREATION COMPLETE!")
    print(f"✅ Created {created_count} timeline images")
    print(f"📁 Location: Center_Point_Labeling_Results/timelines/")

if __name__ == "__main__":
    main()
