#!/usr/bin/env python3
"""
Data Audit Script - Discover dataset structure without assumptions
Infers label sources safely and creates comprehensive data inventory
"""

import argparse
import os
import sys
import json
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from excel_logger import ExcelLogger

def find_audio_files(root_path):
    """Recursively find all audio files."""
    audio_extensions = ['.wav', '.flac', '.mp3', '.m4a', '.aac']
    audio_files = []
    
    root_path = Path(root_path)
    for ext in audio_extensions:
        # Case insensitive search
        audio_files.extend(root_path.glob(f"**/*{ext}"))
        audio_files.extend(root_path.glob(f"**/*{ext.upper()}"))
    
    return sorted(list(set(audio_files)))  # Remove duplicates and sort

def infer_labels_from_folders(audio_files, root_path):
    """Infer labels from folder structure (ImageFolder style)."""
    root_path = Path(root_path)
    labels = {}
    folder_structure = {}
    
    for file_path in audio_files:
        # Get relative path from root
        try:
            rel_path = file_path.relative_to(root_path)
            parent_folder = rel_path.parent.name
            
            if parent_folder not in folder_structure:
                folder_structure[parent_folder] = []
            folder_structure[parent_folder].append(file_path)
            
            labels[str(file_path)] = parent_folder
            
        except ValueError:
            # File is not under root_path
            labels[str(file_path)] = "unknown"
    
    return labels, folder_structure

def find_sidecar_files(root_path):
    """Look for CSV/TSV files that might contain labels."""
    root_path = Path(root_path)
    sidecar_candidates = []
    
    # Common label file names
    label_names = [
        'labels.csv', 'metadata.csv', 'annotations.csv', 'info.csv',
        'labels.tsv', 'metadata.tsv', 'annotations.tsv', 'info.tsv',
        'dataset.csv', 'data.csv', 'index.csv'
    ]
    
    for name in label_names:
        candidates = list(root_path.glob(f"**/{name}"))
        sidecar_candidates.extend(candidates)
    
    return sidecar_candidates

def analyze_filename_patterns(audio_files):
    """Analyze filename patterns for potential labels."""
    patterns = {}
    
    for file_path in audio_files:
        filename = file_path.stem
        
        # Look for common patterns
        if '_' in filename:
            parts = filename.split('_')
            for i, part in enumerate(parts):
                pattern_key = f"underscore_part_{i}"
                if pattern_key not in patterns:
                    patterns[pattern_key] = set()
                patterns[pattern_key].add(part)
        
        if '-' in filename:
            parts = filename.split('-')
            for i, part in enumerate(parts):
                pattern_key = f"dash_part_{i}"
                if pattern_key not in patterns:
                    patterns[pattern_key] = set()
                patterns[pattern_key].add(part)
    
    # Filter patterns that might be labels (2-20 unique values)
    potential_labels = {}
    for pattern, values in patterns.items():
        if 2 <= len(values) <= 20:
            potential_labels[pattern] = sorted(list(values))
    
    return potential_labels

def get_audio_info(file_path):
    """Get audio file information safely."""
    try:
        # Get basic file info
        file_info = {
            'file_path': str(file_path),
            'filename': file_path.name,
            'file_size_mb': file_path.stat().st_size / (1024 * 1024),
        }
        
        # Get audio properties
        audio, sr = librosa.load(file_path, sr=None, duration=None)
        
        file_info.update({
            'duration_sec': len(audio) / sr,
            'sample_rate': sr,
            'channels': 1,  # librosa loads as mono by default
            'samples': len(audio)
        })
        
        # Try to get bit depth (approximate)
        try:
            import soundfile as sf
            with sf.SoundFile(file_path) as f:
                file_info['bit_depth'] = f.subtype_info.name if hasattr(f, 'subtype_info') else 'unknown'
        except:
            file_info['bit_depth'] = 'unknown'
        
        return file_info, None
        
    except Exception as e:
        return {
            'file_path': str(file_path),
            'filename': file_path.name,
            'error': str(e)
        }, str(e)

def create_preview_plots(audio_files, labels, save_dir, max_per_label=3):
    """Create preview waveform and mel-spectrogram plots."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Group files by label
    label_groups = {}
    for file_path in audio_files:
        label = labels.get(str(file_path), 'unknown')
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(file_path)
    
    preview_paths = []
    
    for label, files in label_groups.items():
        # Sample up to max_per_label files
        sample_files = files[:max_per_label] if len(files) >= max_per_label else files
        
        for i, file_path in enumerate(sample_files):
            try:
                # Load audio
                audio, sr = librosa.load(file_path, sr=16000, duration=10)  # First 10 seconds
                
                # Create figure
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                
                # Waveform plot
                time = np.linspace(0, len(audio)/sr, len(audio))
                ax1.plot(time, audio)
                ax1.set_title(f'Waveform: {file_path.name} (Label: {label})')
                ax1.set_xlabel('Time (s)')
                ax1.set_ylabel('Amplitude')
                ax1.grid(True, alpha=0.3)
                
                # Mel spectrogram
                mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
                log_mel = librosa.power_to_db(mel_spec, ref=np.max)
                
                img = librosa.display.specshow(log_mel, sr=sr, x_axis='time', y_axis='mel', ax=ax2)
                ax2.set_title(f'Mel Spectrogram: {file_path.name}')
                plt.colorbar(img, ax=ax2, format='%+2.0f dB')
                
                plt.tight_layout()
                
                # Save plot
                plot_path = save_dir / f'preview_{label}_{i:02d}_{file_path.stem}.png'
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                preview_paths.append(str(plot_path))
                
            except Exception as e:
                print(f"Warning: Could not create preview for {file_path}: {e}")
                continue
    
    return preview_paths

def detect_ambiguity(folder_labels, sidecar_files, filename_patterns):
    """Detect conflicting label sources."""
    ambiguity_report = []
    
    # Check if multiple label sources exist
    sources_found = []
    if len(set(folder_labels.values())) > 1:
        sources_found.append("folder_structure")
    if sidecar_files:
        sources_found.append("sidecar_files")
    if filename_patterns:
        sources_found.append("filename_patterns")
    
    if len(sources_found) > 1:
        ambiguity_report.append(f"Multiple label sources detected: {', '.join(sources_found)}")
    
    # Check folder structure consistency
    folder_counts = {}
    for label in folder_labels.values():
        folder_counts[label] = folder_counts.get(label, 0) + 1
    
    if len(folder_counts) > 1:
        # Check if distribution makes sense for classification
        min_count = min(folder_counts.values())
        max_count = max(folder_counts.values())
        if min_count == 1 and max_count > 10:
            ambiguity_report.append("Highly imbalanced folder distribution - may not be labels")
    
    return ambiguity_report

def main():
    parser = argparse.ArgumentParser(description="Audit dataset structure and infer labels")
    parser.add_argument("--run_id", required=True, help="Run ID for logging")
    parser.add_argument("--root", required=True, help="Path to data root directory")
    
    args = parser.parse_args()
    
    print(f"Data audit for run: {args.run_id}")
    print(f"Root directory: {args.root}")
    
    # Verify root exists
    root_path = Path(args.root)
    if not root_path.exists():
        print(f"Error: Root directory does not exist: {root_path}")
        sys.exit(1)
    
    # Initialize logging
    excel_logger = ExcelLogger()
    
    print("Step 1: Finding audio files...")
    audio_files = find_audio_files(root_path)
    print(f"Found {len(audio_files)} audio files")
    
    if len(audio_files) == 0:
        print("No audio files found. Creating ambiguity report...")
        
        # Log to Excel and exit
        excel_data = {
            'run_id': args.run_id,
            'n_files': 0,
            'n_labels': 'unknown',
            'median_duration': 0,
            'sr_set': 'none',
            'label_source': 'none',
            'notes': 'No audio files found in directory',
            'ambiguity': True
        }
        
        excel_logger.append_row('runs', excel_data)
        print("Logged to Excel and exiting.")
        return
    
    print("Step 2: Inferring labels from folder structure...")
    folder_labels, folder_structure = infer_labels_from_folders(audio_files, root_path)
    
    print("Step 3: Looking for sidecar CSV/TSV files...")
    sidecar_files = find_sidecar_files(root_path)
    
    print("Step 4: Analyzing filename patterns...")
    filename_patterns = analyze_filename_patterns(audio_files)
    
    print("Step 5: Detecting ambiguity...")
    ambiguity_issues = detect_ambiguity(folder_labels, sidecar_files, filename_patterns)
    
    if ambiguity_issues:
        print("AMBIGUITY DETECTED:")
        for issue in ambiguity_issues:
            print(f"  - {issue}")
        
        # Create ambiguity report and stop
        ambiguity_report = {
            'timestamp': datetime.now().isoformat(),
            'root_path': str(root_path),
            'n_files': len(audio_files),
            'issues': ambiguity_issues,
            'folder_structure': {k: len(v) for k, v in folder_structure.items()},
            'sidecar_files': [str(f) for f in sidecar_files],
            'filename_patterns': filename_patterns
        }
        
        # Save ambiguity report
        run_dir = Path("results/experiments") / args.run_id
        report_path = run_dir / "artifacts" / "ambiguity_report.json"
        with open(report_path, 'w') as f:
            json.dump(ambiguity_report, f, indent=2)
        
        # Log to Excel
        excel_data = {
            'run_id': args.run_id,
            'n_files': len(audio_files),
            'n_labels': 'ambiguous',
            'notes': f"Ambiguity detected: {'; '.join(ambiguity_issues)}",
            'ambiguity_report_path': str(report_path)
        }
        
        excel_logger.append_row('runs', excel_data)
        print(f"Ambiguity report saved: {report_path}")
        print("STOPPING due to ambiguity.")
        return
    
    print("Step 6: Analyzing audio files...")
    file_data = []
    durations = []
    sample_rates = set()
    errors = []
    
    for i, file_path in enumerate(audio_files):
        print(f"Processing {i+1}/{len(audio_files)}: {file_path.name}")
        
        info, error = get_audio_info(file_path)
        
        if error:
            errors.append(f"{file_path.name}: {error}")
            info['candidate_label'] = 'error'
        else:
            info['candidate_label'] = folder_labels.get(str(file_path), 'unknown')
            durations.append(info['duration_sec'])
            sample_rates.add(info['sample_rate'])
        
        file_data.append(info)
    
    print("Step 7: Creating data audit CSV...")
    df = pd.DataFrame(file_data)
    
    # Save data audit CSV
    run_dir = Path("results/experiments") / args.run_id
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    audit_csv_path = artifacts_dir / "data_audit.csv"
    df.to_csv(audit_csv_path, index=False)
    print(f"Data audit CSV saved: {audit_csv_path}")
    
    print("Step 8: Creating preview visualizations...")
    preview_dir = run_dir / "01_features" / "preview"
    preview_paths = create_preview_plots(audio_files, folder_labels, preview_dir)
    print(f"Created {len(preview_paths)} preview plots")
    
    print("Step 9: Computing summary statistics...")
    # Compute summary stats
    median_duration = np.median(durations) if durations else 0
    unique_labels = set(folder_labels.values())
    sr_list = sorted(list(sample_rates))
    
    # Determine label source
    if len(unique_labels) > 1 and 'unknown' not in unique_labels:
        label_source = 'folder'
    elif sidecar_files:
        label_source = 'csv'
    elif filename_patterns:
        label_source = 'filename'
    else:
        label_source = 'unknown'
    
    print("Step 10: Logging to Excel...")
    # Prepare Excel data
    excel_data = {
        'run_id': args.run_id,
        'n_files': len(audio_files),
        'n_labels': len(unique_labels) if len(unique_labels) > 1 else 'unknown',
        'median_duration': median_duration,
        'sr_set': ','.join(map(str, sr_list)),
        'label_source': label_source,
        'preview_image_paths': preview_paths[0] if preview_paths else '',
        'audit_csv_path': str(audit_csv_path),
        'n_errors': len(errors)
    }
    
    if errors:
        excel_data['notes'] = f"{len(errors)} files had errors"
    
    try:
        row_num = excel_logger.append_row('runs', excel_data)
        print(f"Excel updated: row {row_num}")
    except Exception as e:
        print(f"Warning: Excel logging failed: {e}")
    
    print("SUMMARY:")
    print(f"  Files found: {len(audio_files)}")
    print(f"  Labels detected: {len(unique_labels)} ({label_source})")
    print(f"  Median duration: {median_duration:.1f}s")
    print(f"  Sample rates: {sr_list}")
    print(f"  Errors: {len(errors)}")
    print(f"  Preview plots: {len(preview_paths)}")
    
    print("STOP - Data audit complete.")

if __name__ == "__main__":
    main()
