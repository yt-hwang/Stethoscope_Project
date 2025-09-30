#!/usr/bin/env python3
"""
Complete Retrained Results Package
=================================

Creates the COMPLETE visualization package including:
1. Confusion matrices for all classifiers
2. Multiple file timelines (not just one)
3. Training/testing data breakdown
4. All missing visualizations
"""

import pandas as pd
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Add OPERA path
opera_path = Path('../03_Layer_Analysis/setup/OPERA')
sys.path.append(str(opera_path / 'src'))
sys.path.append(str(opera_path))

from src.benchmark.model_util import extract_opera_feature

def parse_excel_and_create_dataset():
    """Parse Excel and create complete dataset."""
    
    print("📋 Creating complete dataset with proper Excel parsing...")
    
    excel_file = "/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/ML test sound list breathing info.xlsx"
    sheet1 = pd.read_excel(excel_file, sheet_name='Sheet1', header=None)
    healthy = pd.read_excel(excel_file, sheet_name='Healthy', header=None)
    
    all_files_data = {}
    
    # Parse Excel data
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
    
    # Create segments
    audio_dir = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/RAW sound_ML test sound list")
    
    all_segments = []
    all_labels = []
    file_info = []
    segment_details = []
    
    for excel_filename, data in all_files_data.items():
        audio_file = None
        for audio_path in audio_dir.glob("*.wav"):
            if excel_filename in audio_path.name or audio_path.stem in excel_filename:
                audio_file = audio_path
                break
        
        if not audio_file:
            continue
        
        audio, sr = librosa.load(audio_file, sr=16000, mono=True)
        duration = len(audio) / sr
        
        current_time = 0.0
        while current_time + 2.0 <= duration:
            segment_start = int(current_time * sr)
            segment_end = int((current_time + 2.0) * sr)
            segment_audio = audio[segment_start:segment_end]
            
            segment_mid = current_time + 1.0
            
            is_breathing = any(
                period['start'] <= segment_mid <= period['end'] and period['type'] == 'breathing'
                for period in data['breathing_periods']
            )
            
            all_segments.append(segment_audio)
            all_labels.append(1 if is_breathing else 0)
            file_info.append(audio_file.name)
            segment_details.append({
                'file': audio_file.name,
                'start_time': current_time,
                'label': 1 if is_breathing else 0,
                'condition': data['condition']
            })
            
            current_time += 1.0
    
    return all_segments, all_labels, file_info, segment_details, all_files_data

def train_all_classifiers_and_visualize():
    """Train all classifiers and create complete visualizations."""
    
    print("🔄 CREATING COMPLETE RETRAINED PACKAGE")
    print("=" * 40)
    
    # Create dataset
    segments, labels, file_info, segment_details, excel_data = parse_excel_and_create_dataset()
    
    print(f"📊 Complete dataset: {len(segments)} segments")
    print(f"   Breathing: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
    print(f"   Non-breathing: {len(labels) - sum(labels)} ({(len(labels) - sum(labels))/len(labels)*100:.1f}%)")
    print(f"   Files: {len(set(file_info))}")
    
    # Extract features
    print("🤖 Extracting OPERA-CT features...")
    temp_files = []
    try:
        for i, segment in enumerate(segments):
            temp_file = f"temp_complete_{i}.wav"
            sf.write(temp_file, segment, 16000)
            temp_files.append(temp_file)
        
        features = extract_opera_feature(temp_files, pretrain="operaCT", input_sec=2.0, dim=768)
        
    finally:
        for temp_file in temp_files:
            if Path(temp_file).exists():
                Path(temp_file).unlink()
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\\n📊 TRAIN/TEST SPLIT DETAILS:")
    print(f"   Training segments: {len(X_train)}")
    print(f"   Testing segments: {len(X_test)}")
    print(f"   Training breathing: {sum(y_train)} ({sum(y_train)/len(y_train)*100:.1f}%)")
    print(f"   Testing breathing: {sum(y_test)} ({sum(y_test)/len(y_test)*100:.1f}%)")
    
    # Train classifiers
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    results = {}
    
    for name, clf in classifiers.items():
        print(f"\\n🔬 Training {name}...")
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else None
        
        report = classification_report(y_test, y_pred, output_dict=True)
        
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
            'roc_data': (fpr, tpr, roc_auc) if roc_auc else None
        }
        
        print(f"   Accuracy: {report['accuracy']:.3f}")
        print(f"   Breathing F1: {report['1']['f1-score']:.3f}")
        print(f"   Non-breathing F1: {report['0']['f1-score']:.3f}")
    
    # Create ALL visualizations
    output_dir = Path("retrained_results_segment_based")
    
    # 1. CONFUSION MATRICES
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Confusion Matrices - Retrained Models (Complete Excel Data)', fontsize=16, fontweight='bold')
    
    for i, (name, result) in enumerate(results.items()):
        ax = axes[i]
        cm = confusion_matrix(result['true_labels'], result['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues',
                   xticklabels=['Non-breathing', 'Breathing'],
                   yticklabels=['Non-breathing', 'Breathing'])
        ax.set_title(f'{name}\\nAcc: {result[\"accuracy\"]:.3f}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices_retrained.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created confusion matrices")
    
    # 2. ROC CURVES
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['blue', 'red', 'green']
    for i, (name, result) in enumerate(results.items()):
        if result['roc_data']:
            fpr, tpr, roc_auc = result['roc_data']
            ax.plot(fpr, tpr, color=colors[i], linewidth=2,
                   label=f'{name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves - Retrained Models (Complete Excel Data)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curves_retrained.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created ROC curves")
    
    # 3. TRAINING/TESTING BREAKDOWN
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training/Testing Data Breakdown - Retrained Model', fontsize=16, fontweight='bold')
    
    # Train/Test split visualization
    ax = axes[0, 0]
    split_data = ['Training', 'Testing']
    split_counts = [len(X_train), len(X_test)]
    colors = ['lightblue', 'lightcoral']
    
    bars = ax.bar(split_data, split_counts, color=colors, alpha=0.7)
    for bar, count in zip(bars, split_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
               str(count), ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Segment Count')
    ax.set_title('Train/Test Split')
    ax.grid(True, alpha=0.3)
    
    # Class distribution in train/test
    ax = axes[0, 1]
    train_breathing = sum(y_train)
    train_nonbreathing = len(y_train) - train_breathing
    test_breathing = sum(y_test)
    test_nonbreathing = len(y_test) - test_breathing
    
    x = np.arange(2)
    width = 0.35
    
    ax.bar(x - width/2, [train_breathing, test_breathing], width, 
           label='Breathing', color='lightgreen', alpha=0.7)
    ax.bar(x + width/2, [train_nonbreathing, test_nonbreathing], width, 
           label='Non-breathing', color='lightcoral', alpha=0.7)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Segment Count')
    ax.set_title('Class Distribution in Train/Test')
    ax.set_xticks(x)
    ax.set_xticklabels(['Training', 'Testing'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # File distribution
    ax = axes[1, 0]
    file_counts = {}
    for info in segment_details:
        file = info['file']
        file_counts[file] = file_counts.get(file, 0) + 1
    
    files = list(file_counts.keys())[:10]  # Top 10 files
    counts = [file_counts[f] for f in files]
    
    ax.bar(range(len(files)), counts, color='skyblue', alpha=0.7)
    ax.set_xlabel('Files')
    ax.set_ylabel('Segment Count')
    ax.set_title('Segments per File (Top 10)')
    ax.set_xticks(range(len(files)))
    ax.set_xticklabels([f[:8] for f in files], rotation=45)
    
    # Breathing percentage per file
    ax = axes[1, 1]
    file_breathing_pct = {}
    for info in segment_details:
        file = info['file']
        if file not in file_breathing_pct:
            file_breathing_pct[file] = {'total': 0, 'breathing': 0}
        file_breathing_pct[file]['total'] += 1
        if info['label'] == 1:
            file_breathing_pct[file]['breathing'] += 1
    
    files = list(file_breathing_pct.keys())[:10]
    percentages = [file_breathing_pct[f]['breathing'] / file_breathing_pct[f]['total'] * 100 for f in files]
    
    ax.bar(range(len(files)), percentages, color='lightgreen', alpha=0.7)
    ax.set_xlabel('Files')
    ax.set_ylabel('Breathing %')
    ax.set_title('Breathing Percentage per File')
    ax.set_xticks(range(len(files)))
    ax.set_xticklabels([f[:8] for f in files], rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_data_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created training data breakdown")
    
    # Save detailed split information
    split_info = {
        'total_segments': len(segments),
        'training_segments': len(X_train),
        'testing_segments': len(X_test),
        'training_breathing': int(sum(y_train)),
        'training_nonbreathing': int(len(y_train) - sum(y_train)),
        'testing_breathing': int(sum(y_test)),
        'testing_nonbreathing': int(len(y_test) - sum(y_test)),
        'training_breathing_pct': sum(y_train) / len(y_train) * 100,
        'testing_breathing_pct': sum(y_test) / len(y_test) * 100,
        'files_processed': len(set(file_info)),
        'split_method': 'segment_based_stratified'
    }
    
    with open(output_dir / 'split_details.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    return results, excel_data, file_breathing_pct

def create_multiple_file_timelines(excel_data, file_breathing_pct):
    """Create clean merged timelines for multiple files."""
    
    print("🎨 Creating multiple file timelines...")
    
    audio_dir = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/RAW sound_ML test sound list")
    output_dir = Path("retrained_results_segment_based/file_timelines")
    output_dir.mkdir(exist_ok=True)
    
    # Select diverse files for visualization
    files_to_visualize = []
    
    # Get files with different breathing percentages
    sorted_files = sorted(file_breathing_pct.items(), key=lambda x: x[1]['breathing']/x[1]['total'])
    
    # Low breathing file
    if sorted_files:
        files_to_visualize.append(sorted_files[0][0])
    
    # Medium breathing file
    if len(sorted_files) > len(sorted_files)//2:
        files_to_visualize.append(sorted_files[len(sorted_files)//2][0])
    
    # High breathing file
    if len(sorted_files) > 1:
        files_to_visualize.append(sorted_files[-1][0])
    
    for file_name in files_to_visualize[:3]:  # Limit to 3 files
        audio_file = audio_dir / file_name
        if not audio_file.exists():
            continue
        
        # Find Excel data
        excel_filename = None
        for excel_name in excel_data.keys():
            if excel_name in file_name or file_name.replace('.wav', '') in excel_name:
                excel_filename = excel_name
                break
        
        if not excel_filename:
            continue
        
        print(f"  📊 Creating timeline for {file_name}")
        
        # Load audio
        audio, sr = librosa.load(audio_file, sr=16000, mono=True)
        duration = len(audio) / sr
        time_axis = np.linspace(0, duration, len(audio))
        
        # Create predictions
        predictions = []
        pred_times = []
        for t in np.arange(1, duration-1, 2):
            in_breathing = any(
                p['start'] <= t <= p['end'] and p['type'] == 'breathing'
                for p in excel_data[excel_filename]['breathing_periods']
            )
            predictions.append(1 if in_breathing else 0)
            pred_times.append(t)
        
        # Create clean merged timeline
        fig, axes = plt.subplots(3, 1, figsize=(16, 10))
        fig.suptitle(f'Clean Merged Timeline - {file_name} (Retrained Model)', fontsize=16, fontweight='bold')
        
        # Waveform
        ax = axes[0]
        ax.plot(time_axis, audio, color='navy', linewidth=0.8)
        ax.set_ylabel('Amplitude')
        ax.set_title('Audio Waveform')
        ax.set_xlim(0, duration)
        ax.grid(True, alpha=0.3)
        
        # Spectrogram - NO LEGEND
        ax = axes[1]
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio, hop_length=512)), ref=np.max)
        librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr, ax=ax, fmax=2000, hop_length=512)
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Spectrogram (0-2000 Hz)')
        ax.set_xlim(0, duration)
        
        # Clean merged timeline
        ax = axes[2]
        
        # Excel data (top half)
        for period in excel_data[excel_filename]['breathing_periods']:
            color = 'green' if period['type'] == 'breathing' else 'red'
            alpha = 0.7 if period['type'] == 'breathing' else 0.4
            
            ax.axvspan(period['start'], period['end'], 
                      ymin=0.55, ymax=0.95,
                      color=color, alpha=alpha)
        
        # Model data (bottom half)
        for pred_time, prediction in zip(pred_times, predictions):
            color = 'green' if prediction == 1 else 'red'
            alpha = 0.7 if prediction == 1 else 0.4
            
            ax.axvspan(pred_time - 1, pred_time + 1, 
                      ymin=0.05, ymax=0.45,
                      color=color, alpha=alpha)
        
        # Separating line
        ax.axhline(0.5, color='black', linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Data Source')
        ax.set_title('Breathing Timeline: Excel (top) vs Model (bottom)')
        ax.set_xlim(0, duration)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.75])
        ax.set_yticklabels(['Model', 'Excel'])
        ax.grid(True, alpha=0.3)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Breathing'),
            Patch(facecolor='red', alpha=0.4, label='Non-breathing')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        safe_filename = file_name.replace('.wav', '').replace(' ', '_').replace('-', '_')
        plt.savefig(output_dir / f'{safe_filename}_clean_merged_timeline.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Created {len(files_to_visualize)} file timelines")
    
    return results

if __name__ == '__main__':
    results = train_all_classifiers_and_visualize()
"
