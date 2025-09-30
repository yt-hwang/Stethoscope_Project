#!/usr/bin/env python3
"""
Handcrafted Features Breathing Classifier
=========================================
Main pipeline for handcrafted features breathing vs non-breathing classification
"""

import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

class HandcraftedBreathingClassifier:
    def __init__(self):
        self.excel_file = "/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/ML test sound list breathing info.xlsx"
        self.audio_dir = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/RAW sound_ML test sound list")
        
    def parse_excel_correctly(self):
        """Parse Excel with the CORRECT structure."""
        
        print("📋 Parsing Excel with CORRECT structure...")
        print("Structure: Row 1=filename+headers, Row 2=disease+timestamps, Row 3=blank")
        
        all_files_data = {}
        
        for sheet_name in ['Sheet1', 'Healthy']:
            print(f"   Processing {sheet_name} sheet...")
            df = pd.read_excel(self.excel_file, sheet_name=sheet_name, header=None)
            
            # Process all rows to find filenames (Method 2 - finds all 14 files)
            for idx in range(df.shape[0]):
                row = df.iloc[idx]
                
                if pd.notna(row.iloc[1]) and isinstance(row.iloc[1], str):
                    filename = str(row.iloc[1]).strip()
                    
                    # Check if this looks like a filename
                    if any(pattern in filename for pattern in ['KP', 'H0', 'WEBSS']):
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
                                
                                all_files_data[filename] = {
                                    'breathing_events': all_events,
                                    'complete_timeline': breathing_periods,
                                    'condition': 'Healthy' if sheet_name == 'Healthy' else 'Pathological'
                                }
                                
                                print(f"   ✅ Found {filename}: {len(all_events)} events, {len(breathing_periods)} periods")
        
        print(f"📊 Total files found: {len(all_files_data)}")
        return all_files_data

    def extract_handcrafted_features(self, audio, sr):
        """Extract handcrafted features (same as unsupervised experiments)."""
        
        try:
            # RMS Energy
            rms = librosa.feature.rms(y=audio)[0]
            
            # Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
            
            # MFCCs
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            
            # Harmonic features
            harmonic, percussive = librosa.effects.hpss(audio)
            harmonic_centroid = librosa.feature.spectral_centroid(y=harmonic, sr=sr)[0]
            percussive_centroid = librosa.feature.spectral_centroid(y=percussive, sr=sr)[0]
            
            # Rhythm features
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
            onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            
            # Aggregate features (mean, std, min, max) - ensure all are scalars
            features = []
            
            # RMS features
            features.extend([float(np.mean(rms)), float(np.std(rms)), float(np.min(rms)), float(np.max(rms))])
            
            # ZCR features
            features.extend([float(np.mean(zcr)), float(np.std(zcr)), float(np.min(zcr)), float(np.max(zcr))])
            
            # Spectral features
            features.extend([float(np.mean(spectral_centroids)), float(np.std(spectral_centroids))])
            features.extend([float(np.mean(spectral_rolloff)), float(np.std(spectral_rolloff))])
            features.extend([float(np.mean(spectral_bandwidth)), float(np.std(spectral_bandwidth))])
            
            # MFCC features (mean and std of each coefficient)
            for i in range(mfccs.shape[0]):
                features.extend([float(np.mean(mfccs[i])), float(np.std(mfccs[i]))])
            
            # Harmonic features
            features.extend([float(np.mean(harmonic_centroid)), float(np.std(harmonic_centroid))])
            features.extend([float(np.mean(percussive_centroid)), float(np.std(percussive_centroid))])
            
            # Rhythm features
            features.extend([float(tempo), float(len(beats)), float(len(onset_times))])
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"      Error in feature extraction: {e}")
            # Return a zero vector of expected length if extraction fails
            return np.zeros(47, dtype=np.float32)

    def create_segments_and_labels(self, audio, sr, segment_length, excel_data):
        """Create segments with center-point labeling."""
        
        duration = len(audio) / sr
        segments = []
        labels = []
        
        # Create non-overlapping segments
        for start_time in np.arange(0, duration, segment_length):
            end_time = min(start_time + segment_length, duration)
            
            if end_time - start_time < segment_length * 0.5:  # Skip segments that are too short
                continue
                
            # Extract segment
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment_audio = audio[start_sample:end_sample]
            
            # Center-point labeling
            center_time = (start_time + end_time) / 2
            
            # Find which Excel period contains the center point
            label = 0  # Default to non-breathing
            for period in excel_data['complete_timeline']:
                if period['start'] <= center_time <= period['end']:
                    label = 1 if period['type'] == 'breathing' else 0
                    break
            
            # Extract features
            features = self.extract_handcrafted_features(segment_audio, sr)
            
            segments.append(features)
            labels.append(label)
        
        return np.array(segments), np.array(labels)

    def train_individual_models(self, X_train, y_train, X_test, y_test):
        """Train individual models."""
        
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        results = {}
        for name, model in models.items():
            print(f"   Training {name}...")
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred
            }
            
            print(f"   {name} Accuracy: {accuracy:.1%}")
        
        return results

    def run_experiment(self, segment_length):
        """Run complete experiment for given segment length."""
        
        print(f"\n🎯 RUNNING HANDCRAFTED FEATURES EXPERIMENT")
        print(f"Segment Length: {segment_length}s")
        print("=" * 50)
        
        # Parse Excel data
        excel_data = self.parse_excel_correctly()
        
        # Process all audio files
        all_features = []
        all_labels = []
        file_info = []
        
        for audio_file in self.audio_dir.glob("*.wav"):
            filename = audio_file.stem
            
            if filename not in excel_data:
                print(f"   ⚠️  Skipping {filename} - no Excel data")
                continue
            
            print(f"   Processing {filename}...")
            
            try:
                # Load audio
                audio, sr = librosa.load(audio_file, sr=16000, mono=True)
                
                # Create segments and labels
                features, labels = self.create_segments_and_labels(
                    audio, sr, segment_length, excel_data[filename]
                )
                
                if len(features) > 0:
                    all_features.append(features)
                    all_labels.append(labels)
                    file_info.append({
                        'filename': filename,
                        'features': features,
                        'labels': labels,
                        'audio': audio,
                        'sr': sr
                    })
                    
                    print(f"   ✅ {filename}: {len(features)} segments")
                else:
                    print(f"   ❌ {filename}: No valid segments")
                    
            except Exception as e:
                print(f"   ❌ Error processing {filename}: {e}")
        
        if not all_features:
            print("❌ No valid data found!")
            return
        
        # Combine all data
        X = np.vstack(all_features)
        y = np.hstack(all_labels)
        
        print(f"\n📊 Dataset Summary:")
        print(f"   Total segments: {len(X)}")
        print(f"   Features per segment: {X.shape[1]}")
        print(f"   Breathing segments: {np.sum(y)}")
        print(f"   Non-breathing segments: {len(y) - np.sum(y)}")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   Training segments: {len(X_train)}")
        print(f"   Test segments: {len(X_test)}")
        
        # Train individual models
        print(f"\n🤖 Training Individual Models...")
        individual_results = self.train_individual_models(X_train, y_train, X_test, y_test)
        
        # Find best model
        best_model_name = max(individual_results.keys(), key=lambda k: individual_results[k]['accuracy'])
        best_accuracy = individual_results[best_model_name]['accuracy']
        
        print(f"\n🏆 Best Individual Model: {best_model_name} ({best_accuracy:.1%})")
        
        # Save results
        results_dir = Path(f"Individual_Models/{segment_length}s_Individual_Models/Center_Point_Labeling_Results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model performance
        results = {
            'segment_length': segment_length,
            'total_segments': len(X),
            'training_segments': len(X_train),
            'test_segments': len(X_test),
            'feature_count': X.shape[1],
            'individual_results': {name: result['accuracy'] for name, result in individual_results.items()},
            'best_model': best_model_name,
            'best_accuracy': best_accuracy
        }
        
        with open(results_dir / "model_performance.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save final predictions for visualization
        final_predictions = {}
        model = individual_results[best_model_name]['model']
        
        for info in file_info:
            filename = info['filename']
            file_features = info['features']
            file_labels = info['labels']
            file_predictions = model.predict(file_features)
            
            # Create prediction records
            predictions = []
            for i, (pred, true_label) in enumerate(zip(file_predictions, file_labels)):
                start_time = i * segment_length
                end_time = min((i + 1) * segment_length, len(info['audio']) / info['sr'])
                
                predictions.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'prediction': int(pred),
                    'ground_truth': int(true_label)
                })
            
            final_predictions[filename] = predictions
        
        with open(results_dir / "final_predictions.json", 'w') as f:
            json.dump(final_predictions, f, indent=2)
        
        print(f"\n✅ Experiment Complete!")
        print(f"   Results saved to: {results_dir}")
        print(f"   Best accuracy: {best_accuracy:.1%}")

def main():
    """Main execution function."""
    
    print("🎯 HANDCRAFTED FEATURES BREATHING CLASSIFICATION")
    print("=" * 60)
    print("Using traditional signal processing features:")
    print("• RMS Energy, Zero Crossing Rate")
    print("• Spectral features (centroid, rolloff, bandwidth)")
    print("• MFCCs (13 coefficients)")
    print("• Harmonic and rhythm features")
    print("• Total: ~50-100 features per segment")
    print()
    
    trainer = HandcraftedBreathingClassifier()
    
    # Run experiments for different segment lengths
    segment_lengths = [0.25, 0.5, 1.0]
    
    for segment_length in segment_lengths:
        trainer.run_experiment(segment_length)
    
    print(f"\n🎉 ALL EXPERIMENTS COMPLETE!")
    print("=" * 40)
    print("✅ Handcrafted features classification completed")
    print("✅ Results logged and saved")

if __name__ == "__main__":
    main()
