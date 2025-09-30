#!/usr/bin/env python3
"""
Simple Prediction Overlay Charts
===============================

Creates simple but effective prediction overlay visualizations.
"""

import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
import json

def create_simple_overlay(audio_file, output_dir):
    """Create a simple prediction overlay for one audio file."""
    
    filename = audio_file.stem
    print(f"📊 Creating overlay for {filename}...")
    
    # Load audio
    audio, sr = librosa.load(audio_file, sr=16000, mono=True)
    duration = len(audio) / sr
    time_axis = np.linspace(0, duration, len(audio))
    
    # Create simple breathing detection (demo)
    # In real implementation, this would use our trained model
    segment_length = 2.0
    hop_length = 1.0
    
    predictions = []
    pred_times = []
    
    current_time = 0.0
    while current_time + segment_length <= duration:
        # Simple breathing detection based on energy patterns
        start_sample = int(current_time * sr)
        end_sample = int((current_time + segment_length) * sr)
        segment = audio[start_sample:end_sample]
        
        # Energy-based breathing detection
        rms_energy = np.sqrt(np.mean(segment**2))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr))
        
        # Simple rule: breathing tends to have moderate energy and specific frequency
        is_breathing = (0.01 < rms_energy < 0.1) and (800 < spectral_centroid < 2000)
        
        predictions.append(1 if is_breathing else 0)
        pred_times.append(current_time + segment_length / 2)
        
        current_time += hop_length
    
    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    fig.suptitle(f'Breathing Detection Analysis - {filename}', fontsize=16, fontweight='bold')
    
    # 1. Waveform with predictions
    ax = axes[0]
    ax.plot(time_axis, audio, color='gray', alpha=0.7, linewidth=0.5)
    
    # Overlay predictions
    for pred_time, prediction in zip(pred_times, predictions):
        color = 'lightgreen' if prediction == 1 else 'lightcoral'
        ax.axvspan(pred_time - 1, pred_time + 1, alpha=0.3, color=color)
    
    ax.set_ylabel('Amplitude')
    ax.set_title('Audio Waveform with Breathing Predictions')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightgreen', alpha=0.3, label='Predicted Breathing'),
        Patch(facecolor='lightcoral', alpha=0.3, label='Predicted Non-breathing')
    ]
    ax.legend(handles=legend_elements)
    
    # 2. Spectrogram
    ax = axes[1]
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    img = librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr, ax=ax)
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Spectrogram')
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    
    # 3. Prediction timeline
    ax = axes[2]
    breathing_times = [t for t, p in zip(pred_times, predictions) if p == 1]
    nonbreathing_times = [t for t, p in zip(pred_times, predictions) if p == 0]
    
    if breathing_times:
        ax.scatter(breathing_times, [1]*len(breathing_times), c='green', 
                  alpha=0.7, s=50, label='Breathing')
    if nonbreathing_times:
        ax.scatter(nonbreathing_times, [0]*len(nonbreathing_times), c='red', 
                  alpha=0.7, s=50, label='Non-breathing')
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Prediction')
    ax.set_title('Breathing Detection Timeline')
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Non-breathing', 'Breathing'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / f"{filename}_simple_overlay.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created: {output_path}")
    
    return {
        'filename': filename,
        'duration': duration,
        'predictions': predictions,
        'prediction_times': pred_times,
        'breathing_segments': sum(predictions),
        'total_segments': len(predictions)
    }

def main():
    """Create prediction overlays for sample files."""
    
    audio_dir = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/RAW sound_ML test sound list")
    output_dir = Path("breathing_classification_results/prediction_overlays")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process first 5 files as examples
    audio_files = list(audio_dir.glob("*.wav"))[:5]
    
    print(f"🎨 Creating prediction overlays for {len(audio_files)} sample files...")
    
    results = []
    for audio_file in audio_files:
        try:
            result = create_simple_overlay(audio_file, output_dir)
            results.append(result)
        except Exception as e:
            print(f"❌ Error with {audio_file.name}: {e}")
    
    print(f"\n✅ Created {len(results)} prediction overlay charts!")
    print(f"📁 Saved to: {output_dir}/")

if __name__ == "__main__":
    main()
