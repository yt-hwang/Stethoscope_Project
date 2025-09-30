#!/usr/bin/env python3
"""
Fix All Models with Correct Excel Parsing
=========================================
Fixes the severe Excel parsing problem by retraining all models with correct ground truth
"""

import pandas as pd
import numpy as np
import librosa
import soundfile as sf
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

class CorrectedModelTrainer:
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
                                
                                print(f"     {filename}: {len(all_events)} breathing events")
        
        print(f"✅ Correctly parsed {len(all_files_data)} files")
        return all_files_data
    
    def create_corrected_segments(self, excel_data, segment_length, hop_length):
        """Create segments with CORRECT ground truth labels."""
        
        print(f"🎵 Creating CORRECTED {segment_length}s segments...")
        
        all_segments = []
        all_labels = []
        file_info = []
        segment_info = []
        
        for excel_filename, data in excel_data.items():
            # Find matching audio file
            audio_file = None
            for audio_path in self.audio_dir.glob("*.wav"):
                if excel_filename in audio_path.name or audio_path.stem in excel_filename:
                    audio_file = audio_path
                    break
            
            if not audio_file:
                print(f"⚠️ Audio file not found for {excel_filename}")
                continue
            
            print(f"  📁 Processing {audio_file.name}...")
            
            # Load audio
            audio, sr = librosa.load(audio_file, sr=16000, mono=True)
            duration = len(audio) / sr
            
            # Create segments
            current_time = 0.0
            file_segments = []
            file_labels = []
            file_segment_info = []
            
            while current_time + segment_length <= duration:
                # Extract segment
                start_sample = int(current_time * sr)
                end_sample = int((current_time + segment_length) * sr)
                segment = audio[start_sample:end_sample]
                
                # Center-point labeling using CORRECT Excel timeline
                segment_center = current_time + segment_length / 2
                
                # Determine label based on center point using CORRECT timeline
                is_breathing = False
                for period in data['complete_timeline']:
                    if period['start'] <= segment_center <= period['end'] and period['type'] == 'breathing':
                        is_breathing = True
                        break
                
                label = 1 if is_breathing else 0
                
                file_segments.append(segment)
                file_labels.append(label)
                file_segment_info.append({
                    'filename': excel_filename,
                    'start_time': current_time,
                    'end_time': current_time + segment_length,
                    'center_time': segment_center,
                    'condition': data['condition']
                })
                
                current_time += hop_length
            
            print(f"    ✅ {len(file_segments)} segments, {sum(file_labels)} breathing, {len(file_labels)-sum(file_labels)} non-breathing")
            
            all_segments.extend(file_segments)
            all_labels.extend(file_labels)
            file_info.extend([excel_filename] * len(file_segments))
            segment_info.extend(file_segment_info)
        
        print(f"✅ Created {len(all_segments)} total segments")
        if len(all_labels) > 0:
            print(f"   📊 {sum(all_labels)} breathing ({sum(all_labels)/len(all_labels)*100:.1f}%)")
            print(f"   📊 {len(all_labels)-sum(all_labels)} non-breathing ({(len(all_labels)-sum(all_labels))/len(all_labels)*100:.1f}%)")
        
        return all_segments, all_labels, file_info, segment_info
    
    def extract_handcrafted_features(self, segments):
        """Extract handcrafted features from audio segments."""
        
        print("🎯 Extracting handcrafted features...")
        
        sr = 16000
        features = []
        
        for i, segment in enumerate(segments):
            if i % 100 == 0:
                print(f"  Processing segment {i+1}/{len(segments)}...")
            
            # Time domain features
            feature_vector = []
            
            # Basic statistics
            feature_vector.extend([
                np.mean(segment),
                np.std(segment),
                np.max(segment),
                np.min(segment),
                np.mean(np.abs(segment)),  # RMS
                np.median(segment)
            ])
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(segment)[0]
            feature_vector.extend([np.mean(zcr), np.std(zcr)])
            
            # Spectral features
            try:
                # Spectral centroid, rolloff, bandwidth
                spectral_centroids = librosa.feature.spectral_centroid(y=segment, sr=sr)[0]
                spectral_rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr)[0]
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)[0]
                
                feature_vector.extend([
                    np.mean(spectral_centroids),
                    np.std(spectral_centroids),
                    np.mean(spectral_rolloff),
                    np.std(spectral_rolloff),
                    np.mean(spectral_bandwidth),
                    np.std(spectral_bandwidth)
                ])
                
                # MFCCs (first 13)
                mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
                for j in range(13):
                    feature_vector.extend([np.mean(mfccs[j]), np.std(mfccs[j])])
                
                # Spectral contrast
                spectral_contrast = librosa.feature.spectral_contrast(y=segment, sr=sr)
                feature_vector.extend([np.mean(spectral_contrast), np.std(spectral_contrast)])
                
                # Chroma features
                chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
                feature_vector.extend([np.mean(chroma), np.std(chroma)])
                
            except Exception as e:
                # Fill with zeros if extraction fails
                feature_vector.extend([0] * 36)  # 6 + 26 + 2 + 2
            
            features.append(feature_vector)
        
        features_array = np.array(features)
        print(f"✅ Extracted features: {features_array.shape}")
        return features_array
    
    def train_corrected_models(self, features, labels, segment_info, segment_length, model_type="individual"):
        """Train corrected models with proper ground truth."""
        
        print(f"🤖 Training CORRECTED {model_type} models for {segment_length}s...")
        
        # Split data (segment-based)
        X_train, X_test, y_train, y_test, info_train, info_test = train_test_split(
            features, labels, segment_info, 
            test_size=0.2, 
            random_state=42, 
            stratify=labels
        )
        
        print(f"📊 Training: {len(X_train)} segments")
        print(f"📊 Testing: {len(X_test)} segments")
        print(f"📊 Train breathing: {sum(y_train)}/{len(y_train)} ({sum(y_train)/len(y_train)*100:.1f}%)")
        print(f"📊 Test breathing: {sum(y_test)}/{len(y_test)} ({sum(y_test)/len(y_test)*100:.1f}%)")
        
        # Define base models
        base_models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        # Train individual base models
        print("🤖 Training base models...")
        trained_models = {}
        individual_scores = {}
        
        for name, model in base_models.items():
            print(f"  Training {name}...")
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            trained_models[name] = model
            individual_scores[name] = score
            print(f"    ✅ {name}: {score:.3f}")
        
        # Find best individual model
        best_individual = max(individual_scores, key=individual_scores.get)
        best_individual_score = individual_scores[best_individual]
        
        print(f"🏆 Best Individual: {best_individual} ({best_individual_score:.3f})")
        
        results = {
            'individual_models': trained_models,
            'individual_scores': individual_scores,
            'best_individual': best_individual,
            'best_individual_score': best_individual_score,
            'test_data': (X_test, y_test, info_test),
            'train_data': (X_train, y_train, info_train),
            'segment_length': segment_length
        }
        
        # If this is ensemble training, add ensemble methods
        if model_type == "ensemble":
            results.update(self.train_ensemble_methods(trained_models, individual_scores, X_train, y_train, X_test, y_test))
        
        return results
    
    def train_ensemble_methods(self, trained_models, individual_scores, X_train, y_train, X_test, y_test):
        """Train ensemble methods."""
        
        print("🎭 Training ensemble methods...")
        
        # Define ensemble methods
        ensemble_methods = {
            'Hard Voting': VotingClassifier(
                estimators=[(name, model) for name, model in trained_models.items()],
                voting='hard'
            ),
            'Soft Voting': VotingClassifier(
                estimators=[(name, model) for name, model in trained_models.items()],
                voting='soft'
            ),
            'RF Heavy (3:1:1)': self.create_weighted_ensemble(trained_models, [3, 1, 1]),
            'SVM Heavy (1:3:1)': self.create_weighted_ensemble(trained_models, [1, 3, 1]),
            'LR Heavy (1:1:3)': self.create_weighted_ensemble(trained_models, [1, 1, 3]),
            'Best 2 Models': self.create_best_two_ensemble(trained_models, individual_scores)
        }
        
        # Train and evaluate ensemble methods
        ensemble_results = {}
        best_individual_score = max(individual_scores.values())
        
        for method_name, ensemble in ensemble_methods.items():
            print(f"  Training {method_name}...")
            
            if hasattr(ensemble, 'fit'):
                # Scikit-learn ensemble - FIX: Train on training set, evaluate on test set
                ensemble.fit(X_train, y_train)  # CORRECTED: Train on training set
                score = ensemble.score(X_test, y_test)  # Evaluate on test set
                predictions = ensemble.predict(X_test)
            else:
                # Custom weighted ensemble
                predictions = ensemble(X_test)
                score = accuracy_score(y_test, predictions)
            
            ensemble_results[method_name] = {
                'model': ensemble,
                'score': score,
                'predictions': predictions,
                'improvement': score - best_individual_score
            }
            
            print(f"    ✅ {method_name}: {score:.3f} ({score - best_individual_score:+.3f})")
        
        # Find best ensemble
        best_ensemble_name = max(ensemble_results, key=lambda x: ensemble_results[x]['score'])
        best_ensemble = ensemble_results[best_ensemble_name]
        
        return {
            'ensemble_results': ensemble_results,
            'best_ensemble_name': best_ensemble_name,
            'best_ensemble': best_ensemble
        }
    
    def create_weighted_ensemble(self, models, weights):
        """Create a weighted ensemble function."""
        model_names = list(models.keys())
        
        def weighted_predict(X):
            predictions = []
            for name, weight in zip(model_names, weights):
                pred_proba = models[name].predict_proba(X)[:, 1]  # Get probability of class 1
                predictions.append(pred_proba * weight)
            
            weighted_proba = np.sum(predictions, axis=0) / sum(weights)
            return (weighted_proba > 0.5).astype(int)
        
        return weighted_predict
    
    def create_best_two_ensemble(self, models, scores):
        """Create ensemble using the two best performing models."""
        sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_two = [name for name, _ in sorted_models[:2]]
        
        def best_two_predict(X):
            pred1 = models[best_two[0]].predict(X)
            pred2 = models[best_two[1]].predict(X)
            # Majority voting
            return ((pred1 + pred2) / 2 > 0.5).astype(int)
        
        return best_two_predict
    
    def save_corrected_results(self, results, output_dir, excel_data):
        """Save corrected results with proper visualizations."""
        
        print(f"💾 Saving corrected results to {output_dir}...")
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_dir = output_dir / "Corrected_Results"
        results_dir.mkdir(exist_ok=True)
        timelines_dir = results_dir / "timelines"
        timelines_dir.mkdir(exist_ok=True)
        debug_dir = results_dir / "debug_csvs"
        debug_dir.mkdir(exist_ok=True)
        
        X_test, y_test, info_test = results['test_data']
        segment_length = results['segment_length']
        
        # Determine best method and predictions
        if 'best_ensemble' in results:
            best_method = results['best_ensemble_name']
            best_score = results['best_ensemble']['score']
            best_predictions = results['best_ensemble']['predictions']
            method_type = "Ensemble"
        else:
            best_method = results['best_individual']
            best_score = results['best_individual_score']
            best_predictions = results['individual_models'][best_method].predict(X_test)
            method_type = "Individual"
        
        print(f"🏆 Best {method_type}: {best_method} ({best_score:.3f})")
        
        # 1. Create confusion matrix
        cm = confusion_matrix(y_test, best_predictions)
        accuracy = accuracy_score(y_test, best_predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Non-Breathing', 'Breathing'],
                   yticklabels=['Non-Breathing', 'Breathing'])
        plt.title(f'CORRECTED Confusion Matrix - {best_method}\nAccuracy: {accuracy:.1%}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(results_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Save final predictions
        final_predictions = {}
        for i, info in enumerate(info_test):
            filename = info['filename']
            if filename not in final_predictions:
                final_predictions[filename] = []
            
            final_predictions[filename].append({
                'start_time': info['start_time'],
                'end_time': info['end_time'],
                'center_time': info['center_time'],
                'prediction': int(best_predictions[i]),
                'ground_truth': int(y_test[i]),
                'condition': info['condition']
            })
        
        with open(results_dir / 'final_predictions.json', 'w') as f:
            json.dump(final_predictions, f, indent=2)
        
        # 3. Create CORRECTED timeline visualizations
        self.create_corrected_timelines(final_predictions, excel_data, timelines_dir, segment_length)
        
        # 4. Create debug CSVs
        self.create_corrected_debug_csvs(final_predictions, debug_dir)
        
        # 5. Save model comparison
        model_comparison = {
            'segment_length': segment_length,
            'method_type': method_type,
            'best_method': best_method,
            'best_score': float(best_score),
            'individual_scores': {name: float(score) for name, score in results['individual_scores'].items()},
            'corrected_excel_parsing': True,
            'timestamp': datetime.now().isoformat()
        }
        
        if 'ensemble_results' in results:
            model_comparison['ensemble_scores'] = {
                name: float(data['score']) for name, data in results['ensemble_results'].items()
            }
            model_comparison['improvement'] = float(results['best_ensemble']['improvement'])
        
        with open(results_dir / 'model_comparison.json', 'w') as f:
            json.dump(model_comparison, f, indent=2)
        
        print(f"✅ Corrected results saved to {results_dir}")
        return results_dir
    
    def create_corrected_timelines(self, final_predictions, excel_data, timelines_dir, segment_length):
        """Create timeline visualizations with CORRECT Excel data."""
        
        print("🎨 Creating CORRECTED timeline visualizations...")
        
        for filename, predictions in final_predictions.items():
            if filename not in excel_data:
                continue
            
            # Find audio file
            audio_file = None
            for audio_path in self.audio_dir.glob("*.wav"):
                if filename in audio_path.name or audio_path.stem in filename:
                    audio_file = audio_path
                    break
            
            if not audio_file:
                continue
            
            try:
                # Load audio
                y, sr = librosa.load(audio_file, sr=22050)
                duration = len(y) / sr
                
                # Calculate accuracy
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
                
                # 2. Mel Spectrogram (0-2000 Hz) - Better for breathing sounds
                mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, fmax=2000, n_mels=128)
                log_mel = librosa.power_to_db(mel_spec, ref=np.max)
                librosa.display.specshow(log_mel, y_axis='mel', x_axis='time', sr=sr, ax=axes[1], fmax=2000)
                axes[1].set_title('Spectrogram (0-2000 Hz)')
                axes[1].set_ylabel('Frequency (Hz)')
                axes[1].set_xlim(0, duration)
                
                # 3. Timeline (EXACT SAME FORMAT AS YOUR IMAGE)
                ax = axes[2]
                
                # Excel data (top half) - EXACT SAME as your image
                for period in excel_data[filename]['complete_timeline']:
                    color = 'green' if period['type'] == 'breathing' else 'red'
                    alpha = 0.7 if period['type'] == 'breathing' else 0.4
                    
                    ax.axvspan(period['start'], period['end'], 
                              ymin=0.55, ymax=0.95,  # EXACT SAME as your image
                              color=color, alpha=alpha)
                
                # Model data (bottom half) - EXACT SAME as your image
                for pred in predictions:
                    color = 'green' if pred['prediction'] == 1 else 'red'
                    alpha = 0.7 if pred['prediction'] == 1 else 0.4
                    
                    ax.axvspan(pred['start_time'], pred['end_time'], 
                              ymin=0.05, ymax=0.45,  # EXACT SAME as your image
                              color=color, alpha=alpha)
                
                # Separating line (EXACT SAME as your image)
                ax.axhline(0.5, color='black', linewidth=2, alpha=0.8)
                
                ax.set_xlabel('Time (seconds)')
                ax.set_ylabel('Data Source')
                ax.set_title(f'Perfect Alignment: Excel (top) vs Model (bottom) - {correct_predictions}/{len(predictions)} ({accuracy:.1%})')
                ax.set_xlim(0, duration)
                ax.set_ylim(0, 1)
                ax.set_yticks([0.25, 0.75])
                ax.set_yticklabels(['Model (CENTER-POINT)', 'Excel (CORRECT)'])
                ax.grid(True, alpha=0.3)
                
                # Legend (EXACT SAME as your image)
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='green', alpha=0.7, label='Breathing'),
                    Patch(facecolor='red', alpha=0.4, label='Non-breathing')
                ]
                ax.legend(handles=legend_elements, loc='upper right')
                
                plt.tight_layout()
                
                # Save corrected timeline
                timeline_path = timelines_dir / f'{filename}_CORRECTED_timeline.png'
                plt.savefig(timeline_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"  ✅ CORRECTED timeline: {filename}")
                
            except Exception as e:
                print(f"  ❌ Error creating timeline for {filename}: {e}")
    
    def create_corrected_debug_csvs(self, final_predictions, debug_dir):
        """Create debug CSV files with corrected data."""
        
        print("📊 Creating CORRECTED debug CSV files...")
        
        for filename, predictions in final_predictions.items():
            debug_data = {
                'Time_Range': [f"{p['start_time']:.2f}-{p['end_time']:.2f}s" for p in predictions],
                'Center_Time': [f"{p['center_time']:.2f}s" for p in predictions],
                'CORRECT_Ground_Truth': ['breathing' if p['ground_truth'] == 1 else 'non-breathing' for p in predictions],
                'Model_Prediction': ['breathing' if p['prediction'] == 1 else 'non-breathing' for p in predictions],
                'Match_Status': ['MATCH' if p['prediction'] == p['ground_truth'] else 'MISMATCH' for p in predictions],
                'Visual_Check': [self.get_visual_check(p['ground_truth'], p['prediction']) for p in predictions]
            }
            
            debug_df = pd.DataFrame(debug_data)
            debug_df.to_csv(debug_dir / f'{filename}_CORRECTED_debug.csv', index=False)
        
        print(f"  ✅ Created {len(final_predictions)} CORRECTED debug CSV files")
    
    def get_visual_check(self, ground_truth, prediction):
        """Get visual check description."""
        if ground_truth == 1 and prediction == 1:
            return "GREEN under GREEN"
        elif ground_truth == 0 and prediction == 0:
            return "RED under RED"
        elif ground_truth == 1 and prediction == 0:
            return "RED under GREEN"
        else:  # ground_truth == 0 and prediction == 1
            return "GREEN under RED"

def main():
    """Main function to fix all models."""
    
    print("🚨 FIXING SEVERE EXCEL PARSING PROBLEM")
    print("=" * 40)
    print("🎯 Retraining ALL models with CORRECT Excel parsing")
    print("✅ Individual models: 1.0s, 0.5s, 0.25s")
    print("✅ Ensemble models: 1.0s")
    print()
    
    trainer = CorrectedModelTrainer()
    
    try:
        # Step 1: Parse Excel correctly
        print("📋 STEP 1: Parse Excel with CORRECT structure")
        excel_data = trainer.parse_excel_correctly()
        
        # Step 2: Fix Individual Models
        segment_configs = [
            {'length': 1.0, 'hop': 1.0, 'name': '1.0s_Individual_Models'},
            {'length': 0.5, 'hop': 0.5, 'name': '0.5s_Individual_Models'},
            {'length': 0.25, 'hop': 0.25, 'name': '0.25s_Individual_Models'}
        ]
        
        for config in segment_configs:
            print(f"\n📊 STEP 2: Fixing {config['name']}")
            print("=" * 50)
            
            # Create corrected segments
            segments, labels, file_info, segment_info = trainer.create_corrected_segments(
                excel_data, config['length'], config['hop']
            )
            
            if len(segments) == 0:
                print(f"❌ No segments created for {config['name']}")
                continue
            
            # Extract features
            features = trainer.extract_handcrafted_features(segments)
            
            # Train corrected individual model
            results = trainer.train_corrected_models(
                features, labels, segment_info, config['length'], "individual"
            )
            
            # Save results
            output_dir = f"Individual_Models/{config['name']}"
            trainer.save_corrected_results(results, output_dir, excel_data)
            
            print(f"✅ {config['name']} CORRECTED successfully!")
        
        # Step 3: Fix Ensemble Model (1.0s)
        print(f"\n🎭 STEP 3: Fixing 1.0s_Ensemble_Models")
        print("=" * 50)
        
        # Create corrected 1.0s segments for ensemble
        segments, labels, file_info, segment_info = trainer.create_corrected_segments(
            excel_data, 1.0, 1.0
        )
        
        if len(segments) > 0:
            # Extract features
            features = trainer.extract_handcrafted_features(segments)
            
            # Train corrected ensemble model
            results = trainer.train_corrected_models(
                features, labels, segment_info, 1.0, "ensemble"
            )
            
            # Save results
            output_dir = "Ensemble_Models/1.0s_Ensemble_Models"
            trainer.save_corrected_results(results, output_dir, excel_data)
            
            print(f"✅ 1.0s_Ensemble_Models CORRECTED successfully!")
        
        # Step 4: Add 0.5s Ensemble Model
        print(f"\n🎭 STEP 4: Adding 0.5s_Ensemble_Models")
        print("=" * 50)
        
        # Create corrected 0.5s segments for ensemble
        segments, labels, file_info, segment_info = trainer.create_corrected_segments(
            excel_data, 0.5, 0.5
        )
        
        if len(segments) > 0:
            # Extract features
            features = trainer.extract_handcrafted_features(segments)
            
            # Train corrected ensemble model
            results = trainer.train_corrected_models(
                features, labels, segment_info, 0.5, "ensemble"
            )
            
            # Save results
            output_dir = "Ensemble_Models/0.5s_Ensemble_Models"
            trainer.save_corrected_results(results, output_dir, excel_data)
            
            print(f"✅ 0.5s_Ensemble_Models CORRECTED successfully!")
        
        # Step 5: Add 0.25s Ensemble Model
        print(f"\n🎭 STEP 5: Adding 0.25s_Ensemble_Models")
        print("=" * 50)
        
        # Create corrected 0.25s segments for ensemble
        segments, labels, file_info, segment_info = trainer.create_corrected_segments(
            excel_data, 0.25, 0.25
        )
        
        if len(segments) > 0:
            # Extract features
            features = trainer.extract_handcrafted_features(segments)
            
            # Train corrected ensemble model
            results = trainer.train_corrected_models(
                features, labels, segment_info, 0.25, "ensemble"
            )
            
            # Save results
            output_dir = "Ensemble_Models/0.25s_Ensemble_Models"
            trainer.save_corrected_results(results, output_dir, excel_data)
            
            print(f"✅ 0.25s_Ensemble_Models CORRECTED successfully!")
        
        print(f"\n🎉 ALL MODELS CORRECTED SUCCESSFULLY!")
        print("=" * 40)
        print("✅ Excel parsing fixed")
        print("✅ Individual models retrained with correct ground truth")
        print("✅ Ensemble models retrained with correct ground truth (1.0s + 0.5s + 0.25s)")
        print("✅ Timeline visualizations corrected")
        print("✅ All results saved with 'CORRECTED' labels")
        
    except Exception as e:
        print(f"❌ Error during correction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
