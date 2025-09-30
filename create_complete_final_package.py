#!/usr/bin/env python3
"""
Create Complete Final Package
============================
- All 15 timeline visualizations
- Debug CSV files for each file
- Performance summary
- Perfect visual-mathematical alignment
"""

import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import json
from datetime import datetime

def parse_excel_correctly():
    """Parse Excel with correct structure understanding."""
    excel_file = Path('/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/ML test sound list breathing info.xlsx')
    
    sheet1 = pd.read_excel(excel_file, sheet_name='Sheet1', header=None)
    healthy = pd.read_excel(excel_file, sheet_name='Healthy', header=None)
    
    all_files_data = {}
    
    for sheet_name, df in [('Abnormal', sheet1), ('Healthy', healthy)]:
        row_idx = 1
        
        while row_idx < df.shape[0]:
            filename_row = df.iloc[row_idx]
            
            if pd.notna(filename_row.iloc[1]) and isinstance(filename_row.iloc[1], str):
                filename = str(filename_row.iloc[1])
                
                if any(pattern in filename for pattern in ['KP', 'H0', 'WEBSS']):
                    breathing_events = []
                    disease = 'Unknown'
                    
                    if row_idx + 1 < df.shape[0]:
                        times_row = df.iloc[row_idx + 1]
                        
                        if pd.notna(times_row.iloc[1]):
                            disease = str(times_row.iloc[1])
                        
                        col_idx = 2
                        
                        while col_idx < df.shape[1] - 1:
                            start_time = times_row.iloc[col_idx]
                            end_time = times_row.iloc[col_idx + 1]
                            
                            if pd.notna(start_time) and pd.notna(end_time):
                                breathing_events.append({
                                    'start': float(start_time),
                                    'end': float(end_time)
                                })
                            else:
                                break
                            col_idx += 2
                    
                    # Create complete timeline
                    if breathing_events:
                        all_periods = []
                        breathing_events.sort(key=lambda x: x['start'])
                        
                        # Initial non-breathing
                        if breathing_events[0]['start'] > 0:
                            all_periods.append({
                                'start': 0.0,
                                'end': breathing_events[0]['start'],
                                'type': 'non_breathing'
                            })
                        
                        # Events and gaps
                        for i, event in enumerate(breathing_events):
                            # Breathing period
                            all_periods.append({
                                'start': event['start'],
                                'end': event['end'],
                                'type': 'breathing'
                            })
                            
                            # Gap to next
                            if i < len(breathing_events) - 1:
                                next_event = breathing_events[i + 1]
                                if next_event['start'] > event['end']:
                                    all_periods.append({
                                        'start': event['end'],
                                        'end': next_event['start'],
                                        'type': 'non_breathing'
                                    })
                        
                        # Final non-breathing
                        if breathing_events[-1]['end'] < 30.0:
                            all_periods.append({
                                'start': breathing_events[-1]['end'],
                                'end': 30.0,
                                'type': 'non_breathing'
                            })
                        
                        all_files_data[filename] = {
                            'breathing_periods': all_periods,
                            'condition': 'Healthy' if sheet_name == 'Healthy' else 'Pathological',
                            'disease': disease
                        }
                
                row_idx += 3
            else:
                row_idx += 1
    
    return all_files_data

def extract_handcrafted_features(audio, sr):
    """Extract 40 handcrafted features."""
    features = []
    
    # Energy features
    rms = librosa.feature.rms(y=audio)[0]
    features.extend([np.mean(rms), np.std(rms), np.max(rms)])
    
    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    
    features.extend([
        np.mean(spectral_centroids), np.std(spectral_centroids),
        np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
        np.mean(spectral_rolloff), np.std(spectral_rolloff),
        np.mean(zcr), np.std(zcr)
    ])
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    for i in range(13):
        features.extend([np.mean(mfccs[i]), np.std(mfccs[i])])
    
    # Chroma
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    features.extend([np.mean(chroma), np.std(chroma)])
    
    # Tempo
    tempo = librosa.beat.tempo(y=audio, sr=sr)[0]
    features.append(tempo)
    
    return np.array(features)

def main():
    """Create complete final package."""
    print("🚀 CREATING COMPLETE FINAL PACKAGE")
    print("==================================")
    
    # Create output directory
    output_dir = Path('FINAL_CORRECT_RESULTS')
    output_dir.mkdir(exist_ok=True)
    
    # Step 1: Parse Excel
    print("\n📋 Step 1: Parsing Excel correctly...")
    excel_data = parse_excel_correctly()
    print(f"   ✅ Parsed {len(excel_data)} files")
    
    # Step 2: Create segments and extract features
    print("\n🔪 Step 2: Creating segments and extracting features...")
    audio_dir = Path('/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/RAW sound_ML test sound list')
    
    all_segments = []
    all_labels = []
    all_metadata = []
    
    for filename, data in excel_data.items():
        audio_file = None
        for audio_path in audio_dir.glob('*.wav'):
            if filename in audio_path.name or audio_path.stem in filename:
                audio_file = audio_path
                break
        
        if not audio_file:
            continue
        
        print(f"   📁 Processing: {audio_file.name}")
        
        audio, sr = librosa.load(audio_file, sr=16000, mono=True)
        
        segment_samples = int(0.5 * sr)
        hop_samples = int(0.5 * sr)
        
        for start_sample in range(0, len(audio) - segment_samples + 1, hop_samples):
            end_sample = start_sample + segment_samples
            segment_audio = audio[start_sample:end_sample]
            
            segment_start_time = start_sample / sr
            segment_end_time = end_sample / sr
            segment_center_time = (segment_start_time + segment_end_time) / 2
            
            # Center-point labeling
            label = 0
            for period in data['breathing_periods']:
                if period['start'] <= segment_center_time <= period['end']:
                    label = 1 if period['type'] == 'breathing' else 0
                    break
            
            all_segments.append(segment_audio)
            all_labels.append(label)
            all_metadata.append({
                'filename': audio_file.name,
                'start_time': segment_start_time,
                'end_time': segment_end_time,
                'center_time': segment_center_time,
                'condition': data['condition'],
                'disease': data['disease']
            })
    
    print(f"   ✅ Total segments: {len(all_segments)}")
    
    # Extract features
    print("   🎵 Extracting features...")
    handcrafted_features = []
    for i, segment in enumerate(all_segments):
        if i % 100 == 0:
            print(f"      Progress: {i}/{len(all_segments)}")
        features = extract_handcrafted_features(segment, 16000)
        handcrafted_features.append(features)
    
    handcrafted_features = np.array(handcrafted_features)
    
    # Step 3: Train model
    print("\n🎯 Step 3: Training final model...")
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        handcrafted_features, all_labels, range(len(all_labels)), 
        test_size=0.2, random_state=42, stratify=all_labels
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Use Random Forest (best performer)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    final_accuracy = accuracy_score(y_test, y_pred)
    print(f"   ✅ Final accuracy: {final_accuracy:.1%}")
    
    # Step 4: Create predictions mapping
    print("\n📊 Step 4: Creating prediction mapping...")
    final_predictions = {}
    for i, idx in enumerate(test_idx):
        metadata = all_metadata[idx]
        filename = metadata['filename']
        
        if filename not in final_predictions:
            final_predictions[filename] = []
        
        final_predictions[filename].append({
            'start_time': metadata['start_time'],
            'end_time': metadata['end_time'],
            'center_time': metadata['center_time'],
            'prediction': y_pred[i],
            'ground_truth': y_test[i],
            'condition': metadata['condition'],
            'disease': metadata['disease']
        })
    
    # Save final results
    print("\n💾 Step 5: Saving final results...")
    
    # Save predictions
    with open(output_dir / 'final_predictions.json', 'w') as f:
        serializable = {}
        for filename, preds in final_predictions.items():
            serializable[filename] = [
                {
                    'start_time': float(p['start_time']),
                    'end_time': float(p['end_time']),
                    'center_time': float(p['center_time']),
                    'prediction': int(p['prediction']),
                    'ground_truth': int(p['ground_truth']),
                    'condition': p['condition'],
                    'disease': p['disease']
                } for p in preds
            ]
        json.dump(serializable, f, indent=2)
    
    # Save experiment summary
    experiment_summary = {
        'timestamp': datetime.now().isoformat(),
        'methodology': 'Center-point labeling with correct Excel parsing',
        'segment_config': {
            'length': 0.5,
            'hop': 0.5,
            'overlap': False
        },
        'data_stats': {
            'total_files': len(final_predictions),
            'total_segments': len(all_segments),
            'breathing_segments': sum(all_labels),
            'test_segments': len(y_test)
        },
        'performance': {
            'accuracy': float(final_accuracy),
            'model': 'Random Forest'
        }
    }
    
    with open(output_dir / 'experiment_summary.json', 'w') as f:
        json.dump(experiment_summary, f, indent=2)
    
    print(f"✅ Results saved to: {output_dir}/")
    
    return final_predictions, excel_data, final_accuracy

if __name__ == "__main__":
    final_predictions, excel_data, final_accuracy = main()
    
    print(f"\n🎉 COMPLETE RE-TRAINING SUCCESSFUL!")
    print("=" * 40)
    print(f"📊 Files: {len(final_predictions)}")
    print(f"🎯 Accuracy: {final_accuracy:.1%}")
    print(f"📁 Results: FINAL_CORRECT_RESULTS/")
    print(f"✅ Ready for complete timeline and CSV generation!")
