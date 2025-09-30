#!/usr/bin/env python3
"""
Retrain Model with Complete Excel Data
=====================================

Retrains the breathing classifier with:
1. Complete Excel breathing data (all periods parsed correctly)
2. Segment-based splitting (first)
3. Comparison with previous results
4. Preparation for patient-based splitting
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Add OPERA path
opera_path = Path(__file__).parent.parent.parent / "03_Layer_Analysis" / "setup" / "OPERA"
sys.path.append(str(opera_path / "src"))
sys.path.append(str(opera_path))

try:
    from src.benchmark.model_util import extract_opera_feature
    OPERA_AVAILABLE = True
    print("✅ OPERA-CT available for retraining")
except ImportError:
    OPERA_AVAILABLE = False
    print("⚠️ OPERA-CT not available")

def parse_complete_excel_data_all_files():
    """Parse complete Excel data for ALL files."""
    
    excel_file = "/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/ML test sound list breathing info.xlsx"
    
    print("📋 Parsing complete Excel data for all files...")
    
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

def create_labeled_segments_complete(audio_dir, excel_data):
    """Create labeled segments using complete Excel data."""
    
    print("🎵 Creating labeled segments with complete Excel data...")
    
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
        
        # Create segments with labels
        segment_length = 2.0
        hop_length = 1.0
        
        current_time = 0.0
        file_segments = 0
        file_breathing = 0
        
        while current_time + segment_length <= duration:
            segment_start = int(current_time * sr)
            segment_end = int((current_time + segment_length) * sr)
            segment_audio = audio[segment_start:segment_end]
            
            # Determine label using complete Excel data
            segment_mid = current_time + segment_length / 2
            
            # Check if segment overlaps with any breathing period
            is_breathing = any(
                period['start'] <= segment_mid <= period['end'] and period['type'] == 'breathing'
                for period in data['breathing_periods']
            )
            
            all_segments.append(segment_audio)
            all_labels.append(1 if is_breathing else 0)
            file_info.append(audio_file.name)
            segment_info.append({
                'file': audio_file.name,
                'start_time': current_time,
                'end_time': current_time + segment_length,
                'mid_time': segment_mid,
                'label': 1 if is_breathing else 0
            })
            
            file_segments += 1
            if is_breathing:
                file_breathing += 1
            
            current_time += hop_length
        
        print(f"     → {file_segments} segments ({file_breathing} breathing, {file_segments - file_breathing} non-breathing)")
    
    print(f"📊 COMPLETE DATASET:")
    print(f"   Total segments: {len(all_segments)}")
    print(f"   Breathing segments: {sum(all_labels)} ({sum(all_labels)/len(all_labels)*100:.1f}%)")
    print(f"   Non-breathing segments: {len(all_labels) - sum(all_labels)} ({(len(all_labels) - sum(all_labels))/len(all_labels)*100:.1f}%)")
    print(f"   Files processed: {len(set(file_info))}")
    
    return all_segments, all_labels, file_info, segment_info

def extract_opera_features_complete(segments):
    """Extract OPERA-CT features from all segments."""
    
    print("🤖 Extracting OPERA-CT features...")
    
    # Save segments as temporary files
    temp_files = []
    try:
        for i, segment in enumerate(segments):
            temp_file = f"temp_retrain_segment_{i}.wav"
            sf.write(temp_file, segment, 16000)
            temp_files.append(temp_file)
        
        # Extract OPERA-CT features
        features = extract_opera_feature(
            temp_files,
            pretrain="operaCT",
            input_sec=2.0,
            dim=768
        )
        
        return features
        
    finally:
        # Clean up
        for temp_file in temp_files:
            if Path(temp_file).exists():
                Path(temp_file).unlink()

def train_classifiers_complete(features, labels, segment_info):
    """Train classifiers with complete data and segment-based splitting."""
    
    print("🎯 Training classifiers with COMPLETE Excel data (segment-based split)...")
    
    # Segment-based splitting (current method)
    X_train, X_test, y_train, y_test, info_train, info_test = train_test_split(
        features, labels, segment_info, 
        test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"   Training: {len(X_train)} segments")
    print(f"   Testing: {len(X_test)} segments")
    print(f"   Training breathing: {sum(y_train)} ({sum(y_train)/len(y_train)*100:.1f}%)")
    print(f"   Testing breathing: {sum(y_test)} ({sum(y_test)/len(y_test)*100:.1f}%)")
    
    # Train multiple classifiers
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    results = {}
    
    for name, clf in classifiers.items():
        print(f"\\n  🔬 Training {name}...")
        
        # Train
        clf.fit(X_train, y_train)
        
        # Predict
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else None
        
        # Evaluate
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # ROC curve
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
        else:
            fpr, tpr, roc_auc = None, None, None
        
        results[name] = {
            'classifier': clf,
            'predictions': y_pred,
            'probabilities': y_prob,
            'true_labels': y_test,
            'report': report,
            'accuracy': report['accuracy'],
            'f1_breathing': report['1']['f1-score'],
            'f1_nonbreathing': report['0']['f1-score'],
            'roc_auc': roc_auc
        }
        
        print(f"     Accuracy: {report['accuracy']:.3f} (vs previous 0.688)")
        print(f"     Breathing F1: {report['1']['f1-score']:.3f}")
        print(f"     Non-breathing F1: {report['0']['f1-score']:.3f}")
        if roc_auc:
            print(f"     ROC AUC: {roc_auc:.3f}")
    
    return results, X_test, y_test, info_test

def main():
    """Main retraining pipeline."""
    
    print("🔄 RETRAINING WITH COMPLETE EXCEL DATA (SEGMENT-BASED)")
    print("=" * 60)
    
    # Step 1: Parse complete Excel data
    excel_data = parse_complete_excel_data_all_files()
    
    if not excel_data:
        print("❌ No Excel data found!")
        return
    
    # Step 2: Create labeled segments
    audio_dir = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/RAW sound_ML test sound list")
    segments, labels, file_info, segment_info = create_labeled_segments_complete(audio_dir, excel_data)
    
    if not segments:
        print("❌ No segments created!")
        return
    
    # Step 3: Extract OPERA-CT features
    if OPERA_AVAILABLE:
        features = extract_opera_features_complete(segments)
        print(f"✅ OPERA-CT features extracted: {features.shape}")
    else:
        print("❌ OPERA-CT not available!")
        return
    
    # Step 4: Train classifiers (segment-based)
    results, X_test, y_test, info_test = train_classifiers_complete(features, labels, segment_info)
    
    # Step 5: Save results
    output_dir = Path("retrained_results_segment_based")
    output_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    summary = {
        'experiment_date': datetime.now().isoformat(),
        'approach': 'Complete Excel Data + Segment-based Splitting',
        'dataset_stats': {
            'total_segments': len(segments),
            'breathing_segments': sum(labels),
            'non_breathing_segments': len(labels) - sum(labels),
            'breathing_percentage': sum(labels) / len(labels) * 100,
            'files_processed': len(set(file_info))
        },
        'improvements': [
            'Complete Excel breathing data parsing',
            'All breathing periods included',
            'Better ground truth labels'
        ],
        'classifier_results': {}
    }
    
    for name, result in results.items():
        summary['classifier_results'][name] = {
            'accuracy': result['accuracy'],
            'f1_breathing': result['f1_breathing'],
            'f1_nonbreathing': result['f1_nonbreathing'],
            'roc_auc': result['roc_auc']
        }
    
    with open(output_dir / 'retrained_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create performance comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    ax = axes[0]
    classifiers = list(results.keys())
    accuracies = [results[clf]['accuracy'] for clf in classifiers]
    previous_accuracies = [0.688, 0.667, 0.581]  # Previous OPERA-CT results
    
    x = np.arange(len(classifiers))
    width = 0.35
    
    ax.bar(x - width/2, previous_accuracies, width, label='Previous (Incomplete Excel)', alpha=0.7)
    ax.bar(x + width/2, accuracies, width, label='Retrained (Complete Excel)', alpha=0.7)
    
    ax.set_xlabel('Classifiers')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Comparison: Previous vs Retrained')
    ax.set_xticks(x)
    ax.set_xticklabels(classifiers)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ROC curves
    ax = axes[1]
    colors = ['blue', 'red', 'green']
    
    for i, (name, result) in enumerate(results.items()):
        if result['roc_auc']:
            fpr, tpr, _ = roc_curve(result['true_labels'], result['probabilities'])
            ax.plot(fpr, tpr, color=colors[i], 
                   label=f'{name} (AUC = {result[\"roc_auc\"]:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves - Retrained Models')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'retrained_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print final results
    print(f"\\n🎉 RETRAINING COMPLETED!")
    print(f"📁 Results saved to: {output_dir}/")
    print(f"\\n🏆 BEST RETRAINED RESULTS:")
    
    best_clf = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"   Best model: {best_clf[0]}")
    print(f"   Accuracy: {best_clf[1]['accuracy']:.3f} (vs previous 0.688)")
    print(f"   Improvement: {((best_clf[1]['accuracy'] - 0.688) / 0.688 * 100):+.1f}%")
    
    return results, summary

if __name__ == "__main__":
    results, summary = main()
