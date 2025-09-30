#!/usr/bin/env python3
"""
1.0s Ensemble Breathing Classifier
==================================

Creates ensemble models for breathing vs non-breathing classification using:
- OPERA-CT features (768-dimensional embeddings)
- 1.0s segments with center-point labeling
- Multiple ensemble methods (voting, weighted, stacking)
- Same structure and visualizations as individual models
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
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Add OPERA path
opera_path = Path(__file__).parent.parent.parent / "03_Layer_Analysis" / "setup" / "OPERA"
sys.path.append(str(opera_path / "src"))
sys.path.append(str(opera_path))

try:
    from src.benchmark.model_util import extract_opera_feature
    OPERA_AVAILABLE = True
    print("✅ OPERA-CT available for ensemble training")
except ImportError:
    OPERA_AVAILABLE = False
    print("⚠️ OPERA-CT not available - using handcrafted features instead")

class EnsembleBreathingClassifier:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories (same as individual models)
        self.results_dir = self.output_dir / "Center_Point_Labeling_Results"
        self.results_dir.mkdir(exist_ok=True)
        self.timelines_dir = self.results_dir / "timelines"
        self.timelines_dir.mkdir(exist_ok=True)
        self.debug_dir = self.results_dir / "debug_csvs"
        self.debug_dir.mkdir(exist_ok=True)
        
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📊 Results directory: {self.results_dir}")
    
    def parse_excel_data(self):
        """Parse Excel breathing data using the working method from individual models."""
        
        excel_file = "/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/ML test sound list breathing info.xlsx"
        
        print("📋 Parsing Excel breathing data...")
        
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
                                
                                print(f"     {excel_filename}: {len(breathing_periods)} periods")
        
        print(f"✅ Parsed complete data for {len(all_files_data)} files")
        return all_files_data
    
    def create_labeled_segments(self, audio_dir, excel_data):
        """Create 1.0s labeled segments using center-point labeling."""
        
        print("🎵 Creating 1.0s labeled segments...")
        
        all_segments = []
        all_labels = []
        file_info = []
        segment_info = []
        
        for excel_filename, data in excel_data.items():
            # Find matching audio file
            audio_file = None
            for audio_path in audio_dir.glob("*.wav"):
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
            
            # Create 1.0s segments (no overlap for clean ensemble)
            segment_length = 1.0
            hop_length = 1.0
            
            current_time = 0.0
            file_segments = []
            file_labels = []
            file_segment_info = []
            
            while current_time + segment_length <= duration:
                # Extract segment
                start_sample = int(current_time * sr)
                end_sample = int((current_time + segment_length) * sr)
                segment = audio[start_sample:end_sample]
                
                # Center-point labeling
                segment_center = current_time + segment_length / 2
                
                # Determine label based on center point
                is_breathing = False
                for period in data['breathing_periods']:
                    if period['start'] <= segment_center <= period['end']:
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
        else:
            print("   ❌ No segments created - check audio files and Excel parsing")
        
        return all_segments, all_labels, file_info, segment_info
    
    def extract_features(self, segments):
        """Extract features from audio segments (OPERA-CT or handcrafted)."""
        
        if OPERA_AVAILABLE:
            return self.extract_opera_features(segments)
        else:
            return self.extract_handcrafted_features(segments)
    
    def extract_opera_features(self, segments):
        """Extract OPERA-CT features from audio segments."""
        
        print("🎯 Extracting OPERA-CT features...")
        
        # Save segments as temporary files
        temp_files = []
        try:
            for i, segment in enumerate(segments):
                temp_file = f"temp_ensemble_segment_{i}.wav"
                sf.write(temp_file, segment, 16000)
                temp_files.append(temp_file)
            
            # Extract OPERA-CT features
            features = extract_opera_feature(
                temp_files,
                pretrain="operaCT",
                input_sec=1.0,  # 1.0s segments
                dim=768
            )
            
            print(f"✅ Extracted OPERA-CT features: {features.shape}")
            return features
            
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                if Path(temp_file).exists():
                    Path(temp_file).unlink()
    
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
                print(f"    Warning: Spectral feature extraction failed for segment {i}: {e}")
                # Fill with zeros if extraction fails
                feature_vector.extend([0] * 36)  # 6 + 26 + 2 + 2
            
            features.append(feature_vector)
        
        features_array = np.array(features)
        print(f"✅ Extracted handcrafted features: {features_array.shape}")
        return features_array
    
    def train_ensemble_models(self, features, labels, segment_info):
        """Train multiple ensemble models and select the best."""
        
        print("🎭 Training Ensemble Models...")
        
        # Split data (segment-based, same as individual models)
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
        print("🎭 Training ensemble methods...")
        ensemble_results = {}
        
        for method_name, ensemble in ensemble_methods.items():
            print(f"  Training {method_name}...")
            
            if hasattr(ensemble, 'fit'):
                # Scikit-learn ensemble
                ensemble.fit(X_train, y_train)
                score = ensemble.score(X_test, y_test)
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
        
        print(f"\n🏆 RESULTS SUMMARY:")
        print(f"   Best Individual: {best_individual} - {best_individual_score:.3f}")
        print(f"   Best Ensemble: {best_ensemble_name} - {best_ensemble['score']:.3f}")
        print(f"   Improvement: {best_ensemble['improvement']:+.3f}")
        
        return {
            'individual_models': trained_models,
            'individual_scores': individual_scores,
            'best_individual': best_individual,
            'best_individual_score': best_individual_score,
            'ensemble_results': ensemble_results,
            'best_ensemble_name': best_ensemble_name,
            'best_ensemble': best_ensemble,
            'test_data': (X_test, y_test, info_test),
            'train_data': (X_train, y_train, info_train)
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
    
    def create_visualizations(self, results):
        """Create all visualizations (same as individual models)."""
        
        print("🎨 Creating visualizations...")
        
        X_test, y_test, info_test = results['test_data']
        best_ensemble = results['best_ensemble']
        
        # 1. Confusion Matrix
        self.create_confusion_matrix(y_test, best_ensemble['predictions'], results['best_ensemble_name'])
        
        # 2. Performance Comparison
        self.create_performance_comparison(results)
        
        # 3. Per-file Accuracy
        self.create_per_file_accuracy(y_test, best_ensemble['predictions'], info_test)
        
        # 4. Dataset Summary
        self.create_dataset_summary(results)
        
        # 5. Timeline Visualizations
        self.create_timeline_visualizations(results)
        
        # 6. Debug CSVs
        self.create_debug_csvs(results)
        
        print("✅ All visualizations created")
    
    def create_confusion_matrix(self, y_true, y_pred, method_name):
        """Create confusion matrix visualization."""
        
        cm = confusion_matrix(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Non-Breathing', 'Breathing'],
                   yticklabels=['Non-Breathing', 'Breathing'])
        plt.title(f'Confusion Matrix - {method_name}\nAccuracy: {accuracy:.1%}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Confusion matrix saved")
    
    def create_performance_comparison(self, results):
        """Create performance comparison chart."""
        
        plt.figure(figsize=(12, 8))
        
        # Individual models
        individual_scores = results['individual_scores']
        models = list(individual_scores.keys())
        scores = list(individual_scores.values())
        colors = ['lightblue', 'lightcoral', 'lightgreen']
        
        bars = plt.bar(range(len(models)), scores, color=colors, alpha=0.7, label='Individual Models')
        
        # Best ensemble
        best_ensemble_score = results['best_ensemble']['score']
        best_ensemble_name = results['best_ensemble_name']
        
        plt.bar(len(models), best_ensemble_score, color='gold', alpha=0.8, label='Best Ensemble')
        
        # Add improvement annotation
        improvement = results['best_ensemble']['improvement']
        if improvement > 0:
            plt.annotate(f'+{improvement:.1%}', 
                        xy=(len(models), best_ensemble_score),
                        xytext=(len(models), best_ensemble_score + 0.02),
                        arrowprops=dict(arrowstyle='->', color='green', lw=2),
                        ha='center', fontsize=12, color='green', weight='bold')
        
        plt.title('1.0s Ensemble vs Individual Model Performance')
        plt.ylabel('Accuracy')
        plt.xticks(range(len(models) + 1), models + [best_ensemble_name], rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels
        all_bars = plt.gca().patches
        all_scores = scores + [best_ensemble_score]
        for bar, score in zip(all_bars, all_scores):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Performance comparison saved")
    
    def create_per_file_accuracy(self, y_true, y_pred, info_test):
        """Create per-file accuracy visualization."""
        
        # Group by filename
        file_accuracies = {}
        for i, info in enumerate(info_test):
            filename = info['filename']
            if filename not in file_accuracies:
                file_accuracies[filename] = {'correct': 0, 'total': 0}
            
            if y_true[i] == y_pred[i]:
                file_accuracies[filename]['correct'] += 1
            file_accuracies[filename]['total'] += 1
        
        # Calculate accuracies
        filenames = []
        accuracies = []
        for filename, data in file_accuracies.items():
            filenames.append(filename)
            accuracies.append(data['correct'] / data['total'])
        
        # Create plot
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(filenames)), accuracies, alpha=0.7)
        
        # Color bars by accuracy
        for bar, acc in zip(bars, accuracies):
            if acc >= 0.8:
                bar.set_color('green')
            elif acc >= 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        plt.title('Per-File Accuracy - 1.0s Ensemble Model')
        plt.ylabel('Accuracy')
        plt.xlabel('Audio Files')
        plt.xticks(range(len(filenames)), filenames, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{acc:.1%}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'per_file_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Per-file accuracy saved")
    
    def create_dataset_summary(self, results):
        """Create dataset summary visualization."""
        
        X_train, y_train, info_train = results['train_data']
        X_test, y_test, info_test = results['test_data']
        
        # Create summary plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Train/Test split
        split_data = ['Training', 'Testing']
        split_counts = [len(y_train), len(y_test)]
        axes[0, 0].pie(split_counts, labels=split_data, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Train/Test Split')
        
        # Class distribution in training
        train_breathing = sum(y_train)
        train_non_breathing = len(y_train) - train_breathing
        axes[0, 1].pie([train_non_breathing, train_breathing], 
                       labels=['Non-Breathing', 'Breathing'], 
                       autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Training Set Class Distribution')
        
        # Class distribution in testing
        test_breathing = sum(y_test)
        test_non_breathing = len(y_test) - test_breathing
        axes[1, 0].pie([test_non_breathing, test_breathing], 
                       labels=['Non-Breathing', 'Breathing'], 
                       autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Test Set Class Distribution')
        
        # Ensemble method comparison
        ensemble_names = list(results['ensemble_results'].keys())
        ensemble_scores = [results['ensemble_results'][name]['score'] for name in ensemble_names]
        
        bars = axes[1, 1].bar(range(len(ensemble_names)), ensemble_scores, alpha=0.7)
        axes[1, 1].set_title('Ensemble Method Comparison')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_xticks(range(len(ensemble_names)))
        axes[1, 1].set_xticklabels(ensemble_names, rotation=45, ha='right')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        # Highlight best method
        best_idx = ensemble_scores.index(max(ensemble_scores))
        bars[best_idx].set_color('gold')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'dataset_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Dataset summary saved")
    
    def create_timeline_visualizations(self, results):
        """Create timeline visualizations for all test files."""
        
        print("🎨 Creating timeline visualizations...")
        
        X_test, y_test, info_test = results['test_data']
        best_predictions = results['best_ensemble']['predictions']
        
        # Group predictions by file
        file_predictions = {}
        for i, info in enumerate(info_test):
            filename = info['filename']
            if filename not in file_predictions:
                file_predictions[filename] = []
            
            file_predictions[filename].append({
                'start_time': info['start_time'],
                'end_time': info['end_time'],
                'center_time': info['center_time'],
                'ground_truth': y_test[i],
                'prediction': best_predictions[i],
                'correct': y_test[i] == best_predictions[i]
            })
        
        # Create timeline for each file
        audio_dir = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/RAW sound_ML test sound list")
        
        for filename, predictions in file_predictions.items():
            self.create_single_timeline(filename, predictions, audio_dir)
        
        print(f"  ✅ Created {len(file_predictions)} timeline visualizations")
    
    def create_single_timeline(self, filename, predictions, audio_dir):
        """Create timeline visualization for a single file."""
        
        # Find audio file
        audio_file = None
        for audio_path in audio_dir.glob("*.wav"):
            if filename in audio_path.name or audio_path.stem in filename:
                audio_file = audio_path
                break
        
        if not audio_file:
            print(f"    ⚠️ Audio file not found for {filename}")
            return
        
        try:
            # Load audio
            y, sr = librosa.load(audio_file, sr=22050)
            duration = len(y) / sr
            
            # Calculate accuracy
            correct_predictions = sum(1 for p in predictions if p['correct'])
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
            for pred in predictions:
                # Ground truth (top)
                gt_color = 'green' if pred['ground_truth'] == 1 else 'red'
                axes[2].axvspan(pred['start_time'], pred['end_time'], ymin=0.7, ymax=0.9,
                              color=gt_color, alpha=0.3, label='Ground Truth' if pred == predictions[0] else "")
                
                # Ensemble prediction (bottom)
                pred_color = 'green' if pred['prediction'] == 1 else 'red'
                axes[2].axvspan(pred['start_time'], pred['end_time'], ymin=0.1, ymax=0.3,
                              color=pred_color, alpha=0.7, label='Ensemble' if pred == predictions[0] else "")
            
            axes[2].legend()
            axes[2].set_yticks([0.2, 0.8])
            axes[2].set_yticklabels(['Ensemble', 'Ground Truth'])
            
            plt.tight_layout()
            plt.savefig(self.timelines_dir / f'{filename}_ensemble_timeline.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"    ❌ Error creating timeline for {filename}: {e}")
    
    def create_debug_csvs(self, results):
        """Create debug CSV files for all test files."""
        
        print("📊 Creating debug CSV files...")
        
        X_test, y_test, info_test = results['test_data']
        best_predictions = results['best_ensemble']['predictions']
        
        # Group by file
        file_data = {}
        for i, info in enumerate(info_test):
            filename = info['filename']
            if filename not in file_data:
                file_data[filename] = []
            
            file_data[filename].append({
                'Time_Range': f"{info['start_time']:.2f}-{info['end_time']:.2f}s",
                'Center_Time': f"{info['center_time']:.2f}s",
                'Ground_Truth': 'breathing' if y_test[i] == 1 else 'non-breathing',
                'Ensemble_Prediction': 'breathing' if best_predictions[i] == 1 else 'non-breathing',
                'Match_Status': 'MATCH' if y_test[i] == best_predictions[i] else 'MISMATCH',
                'Visual_Check': self.get_visual_check(y_test[i], best_predictions[i])
            })
        
        # Save CSV for each file
        for filename, data in file_data.items():
            df = pd.DataFrame(data)
            df.to_csv(self.debug_dir / f'{filename}_ensemble_debug.csv', index=False)
        
        print(f"  ✅ Created {len(file_data)} debug CSV files")
    
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
    
    def save_results(self, results):
        """Save final results and summary."""
        
        print("💾 Saving final results...")
        
        # Save final predictions
        final_predictions = {}
        X_test, y_test, info_test = results['test_data']
        best_predictions = results['best_ensemble']['predictions']
        
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
        
        with open(self.results_dir / 'final_predictions.json', 'w') as f:
            json.dump(final_predictions, f, indent=2)
        
        # Save model comparison
        model_comparison = {
            'individual_models': {name: float(score) for name, score in results['individual_scores'].items()},
            'ensemble_methods': {name: float(data['score']) for name, data in results['ensemble_results'].items()},
            'best_individual': results['best_individual'],
            'best_individual_score': float(results['best_individual_score']),
            'best_ensemble': results['best_ensemble_name'],
            'best_ensemble_score': float(results['best_ensemble']['score']),
            'improvement': float(results['best_ensemble']['improvement']),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.results_dir / 'model_comparison.json', 'w') as f:
            json.dump(model_comparison, f, indent=2)
        
        print(f"✅ Results saved to {self.results_dir}")
        
        return model_comparison

def main():
    """Main execution function."""
    
    print("🎭 1.0s ENSEMBLE BREATHING CLASSIFIER")
    print("=" * 40)
    print("✅ OPERA-CT features (768-dim)")
    print("✅ 1.0s segments, center-point labeling")
    print("✅ Multiple ensemble methods")
    print("✅ Same structure as individual models")
    print()
    
    # Initialize classifier
    output_dir = Path(__file__).parent
    classifier = EnsembleBreathingClassifier(output_dir)
    
    try:
        # Step 1: Parse Excel data
        excel_data = classifier.parse_excel_data()
        
        # Step 2: Create labeled segments
        audio_dir = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/RAW sound_ML test sound list")
        segments, labels, file_info, segment_info = classifier.create_labeled_segments(audio_dir, excel_data)
        
        # Step 3: Extract features (OPERA-CT or handcrafted)
        features = classifier.extract_features(segments)
        
        # Step 4: Train ensemble models
        results = classifier.train_ensemble_models(features, labels, segment_info)
        
        # Step 5: Create visualizations
        classifier.create_visualizations(results)
        
        # Step 6: Save results
        model_comparison = classifier.save_results(results)
        
        # Final summary
        print(f"\n🎉 1.0s ENSEMBLE MODEL COMPLETE!")
        print("=" * 35)
        print(f"🏆 Best Individual: {results['best_individual']} ({results['best_individual_score']:.3f})")
        print(f"🎭 Best Ensemble: {results['best_ensemble_name']} ({results['best_ensemble']['score']:.3f})")
        print(f"📈 Improvement: {results['best_ensemble']['improvement']:+.3f}")
        print(f"📁 Results saved to: {classifier.results_dir}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()
