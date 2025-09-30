#!/usr/bin/env python3
"""
Feature Extraction Script - Segment audio and extract OPERA-CT embeddings
Handles train/val/test splits and creates parquet files with embeddings
"""

import argparse
import sys
import os
import json
import time
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Add utils to path  
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from excel_logger import ExcelLogger

def setup_opera_environment():
    """Set up OPERA-CT environment."""
    opera_path = Path.cwd() / "setup" / "OPERA"
    
    # Add to Python path
    if str(opera_path) not in sys.path:
        sys.path.append(str(opera_path))
    
    # Set environment variables
    os.environ['PYTHONPATH'] = f"{os.environ.get('PYTHONPATH', '')}:{opera_path}"
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    return opera_path

def check_opera_availability():
    """Check if OPERA-CT is available and working."""
    try:
        from src.benchmark.model_util import extract_opera_feature, initialize_pretrained_model
        return True, "OPERA-CT available"
    except ImportError as e:
        return False, f"OPERA-CT not available: {e}"

def segment_audio_file(audio, sr, seg_sec, hop_sec):
    """Segment audio file into fixed-length segments."""
    seg_samples = int(seg_sec * sr)
    hop_samples = int(hop_sec * sr)
    
    segments = []
    for start in range(0, len(audio) - seg_samples + 1, hop_samples):
        end = start + seg_samples
        segment = audio[start:end]
        
        # Pad if necessary (shouldn't happen with proper calculation)
        if len(segment) < seg_samples:
            segment = np.pad(segment, (0, seg_samples - len(segment)), mode='constant')
        
        segments.append(segment)
    
    return segments

def extract_opera_features(audio_segments, sr):
    """Extract OPERA-CT features from audio segments."""
    try:
        from src.benchmark.model_util import extract_opera_feature
        import tempfile
        import soundfile as sf
        
        # Save segments as temporary files for OPERA-CT
        temp_files = []
        temp_dir = Path(tempfile.mkdtemp())
        
        for i, segment in enumerate(audio_segments):
            temp_path = temp_dir / f"segment_{i:04d}.wav"
            sf.write(temp_path, segment, sr)
            temp_files.append(str(temp_path))
        
        # Extract features using OPERA-CT
        features = extract_opera_feature(temp_files, pretrain="operaCT", input_sec=len(audio_segments[0])/sr, dim=768)
        
        # Clean up temporary files
        for temp_file in temp_files:
            os.remove(temp_file)
        temp_dir.rmdir()
        
        return features, None
        
    except Exception as e:
        return None, str(e)

def extract_mel_features_fallback(audio_segments, sr):
    """Extract mel-spectrogram features as fallback."""
    features = []
    
    for segment in audio_segments:
        # Extract mel spectrogram (matching OPERA-CT format)
        mel_spec = librosa.feature.melspectrogram(
            y=segment, sr=sr, n_mels=64, fmin=50, fmax=2000, 
            n_fft=1024, hop_length=512
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to 0-1 range (like OPERA-CT)
        if log_mel.max() != log_mel.min():
            mel_normalized = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min())
        else:
            mel_normalized = log_mel
        
        # Average over time dimension to get fixed-size features
        features.append(np.mean(mel_normalized, axis=1))
    
    return np.array(features), None

def create_splits(file_data, split_mode="file_level", test_size=0.2, val_size=0.2, random_state=42):
    """Create train/val/test splits."""
    if split_mode == "file_level":
        # Split by files to avoid data leakage
        unique_files = file_data['original_file'].unique()
        
        if len(unique_files) < 3:
            # Too few files for proper splitting
            return {
                'train': file_data,
                'val': pd.DataFrame(),
                'test': pd.DataFrame()
            }
        
        # Split files
        train_files, temp_files = train_test_split(
            unique_files, test_size=(test_size + val_size), random_state=random_state
        )
        
        if len(temp_files) >= 2:
            val_files, test_files = train_test_split(
                temp_files, test_size=(test_size / (test_size + val_size)), random_state=random_state
            )
        else:
            val_files = temp_files[:len(temp_files)//2] if len(temp_files) > 1 else []
            test_files = temp_files[len(temp_files)//2:] if len(temp_files) > 1 else temp_files
        
        # Create splits
        splits = {
            'train': file_data[file_data['original_file'].isin(train_files)],
            'val': file_data[file_data['original_file'].isin(val_files)],
            'test': file_data[file_data['original_file'].isin(test_files)]
        }
        
    else:  # segment_level
        # Split by segments (may cause data leakage)
        train_data, temp_data = train_test_split(
            file_data, test_size=(test_size + val_size), random_state=random_state
        )
        
        if len(temp_data) >= 2:
            val_data, test_data = train_test_split(
                temp_data, test_size=(test_size / (test_size + val_size)), random_state=random_state
            )
        else:
            val_data = temp_data[:len(temp_data)//2] if len(temp_data) > 1 else pd.DataFrame()
            test_data = temp_data[len(temp_data)//2:] if len(temp_data) > 1 else temp_data
        
        splits = {
            'train': train_data,
            'val': val_data, 
            'test': test_data
        }
    
    return splits

def create_preview_images(segments_sample, sr, save_dir, max_samples=6):
    """Create preview images for random segments."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample random segments
    n_samples = min(max_samples, len(segments_sample))
    indices = np.random.choice(len(segments_sample), n_samples, replace=False)
    
    preview_paths = []
    
    for i, idx in enumerate(indices):
        segment = segments_sample[idx]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        
        # Waveform
        time = np.linspace(0, len(segment)/sr, len(segment))
        ax1.plot(time, segment)
        ax1.set_title(f'Segment {idx} Waveform')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        
        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=64)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        img = librosa.display.specshow(log_mel, sr=sr, x_axis='time', y_axis='mel', ax=ax2)
        ax2.set_title(f'Segment {idx} Mel Spectrogram')
        plt.colorbar(img, ax=ax2, format='%+2.0f dB')
        
        plt.tight_layout()
        
        # Save
        plot_path = save_dir / f'segment_preview_{i:02d}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        preview_paths.append(str(plot_path))
    
    return preview_paths

def main():
    parser = argparse.ArgumentParser(description="Extract features with segmentation")
    parser.add_argument("--run_id", required=True, help="Run ID")
    parser.add_argument("--root", required=True, help="Data root path")
    parser.add_argument("--seg_sec", type=float, default=7.5, help="Segment length in seconds")
    parser.add_argument("--hop_sec", type=float, default=7.5, help="Hop length in seconds (for overlap)")
    parser.add_argument("--split_mode", choices=["file_level", "segment_level"], default="file_level",
                       help="How to create train/val/test splits")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resample_sr", default="auto", help="Resample rate (auto or integer)")
    
    args = parser.parse_args()
    
    print(f"Feature extraction for run: {args.run_id}")
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Setup OPERA-CT environment
    print("Step 1: Setting up OPERA-CT environment...")
    opera_path = setup_opera_environment()
    
    print("Step 2: Checking OPERA-CT availability...")
    opera_available, opera_status = check_opera_availability()
    print(f"OPERA-CT status: {opera_status}")
    
    # Load data audit
    print("Step 3: Loading data audit...")
    run_dir = Path("results/experiments") / args.run_id
    audit_csv_path = run_dir / "artifacts" / "data_audit.csv"
    
    if not audit_csv_path.exists():
        print(f"Error: Data audit not found: {audit_csv_path}")
        sys.exit(1)
    
    df_audit = pd.read_csv(audit_csv_path)
    print(f"Loaded {len(df_audit)} file records")
    
    # Filter out error files
    valid_files = df_audit[df_audit['candidate_label'] != 'error']
    print(f"Valid files: {len(valid_files)}")
    
    if len(valid_files) == 0:
        print("No valid audio files found. Recording in Excel and stopping.")
        
        excel_logger = ExcelLogger()
        excel_data = {
            'run_id': args.run_id,
            'seg_sec': args.seg_sec,
            'hop_sec': args.hop_sec,
            'n_train': 0,
            'n_val': 0,
            'n_test': 0,
            'emb_dim': 0,
            'op_model': False,
            'notes': 'No valid audio files found'
        }
        excel_logger.append_row('features', excel_data)
        return
    
    print("Step 4: Processing audio files and segmenting...")
    
    # Determine target sample rate
    if args.resample_sr == "auto":
        target_sr = 16000  # OPERA-CT standard
    else:
        target_sr = int(args.resample_sr)
    
    all_segments_data = []
    segments_for_preview = []
    
    start_time = time.time()
    
    for _, row in valid_files.iterrows():
        file_path = Path(row['file_path'])
        print(f"Processing: {file_path.name}")
        
        try:
            # Load audio
            audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
            
            # Segment audio
            segments = segment_audio_file(audio, sr, args.seg_sec, args.hop_sec)
            print(f"  Created {len(segments)} segments")
            
            # Store segment metadata
            for seg_idx, segment in enumerate(segments):
                segment_data = {
                    'file_path': str(file_path),
                    'file_id': file_path.stem,
                    'original_file': file_path.name,
                    'segment_idx': seg_idx,
                    'duration': len(segment) / sr,
                    'sr_used': sr,
                    'label': row.get('candidate_label', 'unknown')
                }
                all_segments_data.append(segment_data)
                
                # Store segments for preview and feature extraction
                segments_for_preview.append(segment)
        
        except Exception as e:
            print(f"  Error processing {file_path.name}: {e}")
            continue
    
    processing_time = time.time() - start_time
    print(f"Audio processing completed in {processing_time:.1f}s")
    print(f"Total segments created: {len(all_segments_data)}")
    
    if len(all_segments_data) == 0:
        print("No segments created. Recording in Excel and stopping.")
        
        excel_logger = ExcelLogger()
        excel_data = {
            'run_id': args.run_id,
            'seg_sec': args.seg_sec,
            'hop_sec': args.hop_sec,
            'n_train': 0,
            'n_val': 0,
            'n_test': 0,
            'emb_dim': 0,
            'op_model': opera_available,
            'notes': 'No segments created from audio files'
        }
        excel_logger.append_row('features', excel_data)
        return
    
    print("Step 5: Creating train/val/test splits...")
    df_segments = pd.DataFrame(all_segments_data)
    
    # Check if we have labels for supervised splitting
    unique_labels = df_segments['label'].unique()
    has_labels = len(unique_labels) > 1 and 'unknown' not in unique_labels
    
    if has_labels:
        print(f"Labels detected: {unique_labels}")
        splits = create_splits(df_segments, args.split_mode, random_state=args.seed)
    else:
        print("No clear labels detected - creating unsupervised split")
        splits = {'unsup': df_segments, 'train': pd.DataFrame(), 'val': pd.DataFrame(), 'test': pd.DataFrame()}
    
    print("Step 6: Extracting features...")
    
    if opera_available:
        print("Using OPERA-CT for feature extraction...")
        features, feature_error = extract_opera_features(segments_for_preview, target_sr)
        feature_backend = "opera_ct"
        emb_dim = 768
    else:
        print("Using mel-spectrogram fallback...")
        features, feature_error = extract_mel_features_fallback(segments_for_preview, target_sr)
        feature_backend = "mel_fallback"
        emb_dim = 64
    
    if feature_error:
        print(f"Feature extraction failed: {feature_error}")
        
        excel_logger = ExcelLogger()
        excel_data = {
            'run_id': args.run_id,
            'seg_sec': args.seg_sec,
            'hop_sec': args.hop_sec,
            'n_train': 0,
            'n_val': 0,
            'n_test': 0,
            'emb_dim': 0,
            'op_model': opera_available,
            'notes': f'Feature extraction failed: {feature_error}'
        }
        excel_logger.append_row('features', excel_data)
        return
    
    feature_extraction_time = time.time() - start_time - processing_time
    print(f"Feature extraction completed in {feature_extraction_time:.1f}s")
    print(f"Feature shape: {features.shape}")
    
    print("Step 7: Creating feature DataFrames...")
    
    # Add features to segment data
    for i, (_, row) in enumerate(df_segments.iterrows()):
        if i < len(features):
            for j in range(features.shape[1]):
                df_segments.loc[df_segments.index[i], f'emb_{j}'] = features[i, j]
    
    print("Step 8: Saving parquet files...")
    features_dir = run_dir / "01_features"
    features_dir.mkdir(parents=True, exist_ok=True)
    
    parquet_paths = []
    split_counts = {}
    
    for split_name, split_data in splits.items():
        if len(split_data) > 0:
            parquet_path = features_dir / f"{split_name}.parquet"
            split_data.to_parquet(parquet_path, index=False)
            parquet_paths.append(str(parquet_path))
            split_counts[split_name] = len(split_data)
            print(f"Saved {split_name}: {len(split_data)} segments -> {parquet_path}")
    
    print("Step 9: Creating preview images...")
    preview_dir = features_dir / "preview"
    preview_paths = create_preview_images(segments_for_preview, target_sr, preview_dir)
    print(f"Created {len(preview_paths)} preview images")
    
    print("Step 10: Saving statistics...")
    stats = {
        'embedding_dim': int(emb_dim),
        'n_segments_by_split': split_counts,
        'seg_sec': args.seg_sec,
        'hop_sec': args.hop_sec,
        'resample_sr': target_sr,
        'op_model_detected': opera_available,
        'feature_backend': feature_backend,
        'total_cpu_time': processing_time + feature_extraction_time,
        'total_gpu_time': 0,  # CPU only
        'n_files_processed': len(valid_files),
        'n_segments_created': len(all_segments_data)
    }
    
    stats_path = features_dir / "stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("Step 11: Logging to Excel...")
    excel_logger = ExcelLogger()
    
    excel_data = {
        'run_id': args.run_id,
        'seg_sec': args.seg_sec,
        'hop_sec': args.hop_sec,
        'n_train': split_counts.get('train', 0),
        'n_val': split_counts.get('val', 0),
        'n_test': split_counts.get('test', 0),
        'emb_dim': emb_dim,
        'op_model': opera_available,
        'parquet_paths': ','.join(parquet_paths),
        'preview_paths': ','.join(preview_paths[:2])  # First 2 paths
    }
    
    try:
        row_num = excel_logger.append_row('features', excel_data)
        print(f"Excel features sheet updated: row {row_num}")
    except Exception as e:
        print(f"Warning: Excel logging failed: {e}")
    
    print("SUMMARY:")
    print(f"  Segments created: {len(all_segments_data)}")
    print(f"  Feature backend: {feature_backend}")
    print(f"  Embedding dimension: {emb_dim}")
    print(f"  Train/Val/Test: {split_counts.get('train', 0)}/{split_counts.get('val', 0)}/{split_counts.get('test', 0)}")
    print(f"  Processing time: {processing_time:.1f}s")
    print(f"  Feature extraction time: {feature_extraction_time:.1f}s")
    
    print("STOP - Feature extraction complete.")

if __name__ == "__main__":
    main()