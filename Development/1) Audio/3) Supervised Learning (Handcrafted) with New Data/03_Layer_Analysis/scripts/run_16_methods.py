#!/usr/bin/env python3
"""
OPERA-CT 16-Method Experiment Runner
===================================

Tests all 16 preprocessing methods (A0-D2) with OPERA-CT features
to compare against our previous handcrafted feature results.

Usage:
    python scripts/run_16_methods.py --method A0  # Run single method
    python scripts/run_16_methods.py --series A   # Run entire A-series
"""

import argparse
import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import librosa
import soundfile as sf
import torch
import scipy.signal
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add OPERA path
opera_path = Path(__file__).parent.parent / "setup" / "OPERA"
sys.path.append(str(opera_path / "src"))
sys.path.append(str(opera_path))
from src.benchmark.model_util import extract_opera_feature

# Method definitions matching our unsupervised analysis
METHOD_CONFIGS = {
    # A-Series: Individual NoSeg methods
    'A0': {'segmentation': False, 'name': 'NoSeg_NoPre'},
    'A1': {'segmentation': False, 'bandpass': True, 'name': 'NoSeg_Bandpass'},
    'A2': {'segmentation': False, 'spectral_gating': True, 'name': 'NoSeg_SpectralGating'},
    'A3': {'segmentation': False, 'highpass': True, 'name': 'NoSeg_HighPass'},
    'A4': {'segmentation': False, 'peak_normalize': True, 'name': 'NoSeg_PeakNormalize'},
    
    # B-Series: Combination NoSeg methods
    'B0': {'segmentation': False, 'bandpass': True, 'spectral_gating': True, 'name': 'NoSeg_Bandpass_SpectralGating'},
    'B1': {'segmentation': False, 'peak_normalize': True, 'bandpass': True, 'name': 'NoSeg_PeakNormalize_Bandpass'},
    'B2': {'segmentation': False, 'full_pipeline': True, 'name': 'NoSeg_FullPipeline'},
    
    # C-Series: Individual Seg methods  
    'C0': {'segmentation': True, 'name': 'Seg_NoPre'},
    'C1': {'segmentation': True, 'bandpass': True, 'name': 'Seg_Bandpass'},
    'C2': {'segmentation': True, 'spectral_gating': True, 'name': 'Seg_SpectralGating'},
    'C3': {'segmentation': True, 'highpass': True, 'name': 'Seg_HighPass'},
    'C4': {'segmentation': True, 'peak_normalize': True, 'name': 'Seg_PeakNormalize'},
    
    # D-Series: Combination Seg methods
    'D0': {'segmentation': True, 'highpass': True, 'peak_normalize': True, 'name': 'Seg_HighPass_PeakNormalize'},
    'D1': {'segmentation': True, 'highpass': True, 'bandpass': True, 'name': 'Seg_HighPass_Bandpass'},
    'D2': {'segmentation': True, 'full_pipeline': True, 'name': 'Seg_FullPipeline'},
}

def apply_preprocessing(audio, sr, config):
    """Apply preprocessing based on config."""
    processed_audio = audio.copy()
    
    # Peak normalization
    if config.get('peak_normalize', False) or config.get('full_pipeline', False):
        max_val = np.max(np.abs(processed_audio))
        if max_val > 0:
            processed_audio = processed_audio / max_val
    
    # High-pass filter (20 Hz)
    if config.get('highpass', False) or config.get('full_pipeline', False):
        from scipy.signal import butter, filtfilt
        nyquist = sr / 2
        high = 20 / nyquist
        b, a = butter(5, high, btype='high')
        processed_audio = filtfilt(b, a, processed_audio)
    
    # Bandpass filter (100-2000 Hz)
    if config.get('bandpass', False) or config.get('full_pipeline', False):
        from scipy.signal import butter, filtfilt
        nyquist = sr / 2
        low = 100 / nyquist
        high = 2000 / nyquist
        b, a = butter(5, [low, high], btype='band')
        processed_audio = filtfilt(b, a, processed_audio)
    
    # Spectral gating (noise reduction)
    if config.get('spectral_gating', False) or config.get('full_pipeline', False):
        processed_audio = apply_spectral_gating(processed_audio, sr)
    
    return processed_audio

def apply_spectral_gating(audio, sr, alpha=2.0, beta=0.15):
    """Apply spectral gating noise reduction."""
    # Compute STFT
    stft = librosa.stft(audio, hop_length=512, n_fft=2048)
    S_mag, S_phase = np.abs(stft), np.angle(stft)
    
    # Estimate noise floor (bottom 10% of magnitude)
    noise_floor = np.percentile(S_mag, 10, axis=1, keepdims=True)
    
    # Create mask
    mask = (S_mag > alpha * noise_floor).astype(float)
    mask = np.maximum(mask, beta)  # Ensure minimum value
    
    # Apply soft masking
    S_filtered = S_mag * mask
    
    # Reconstruct
    stft_filtered = S_filtered * np.exp(1j * S_phase)
    audio_filtered = librosa.istft(stft_filtered, hop_length=512)
    
    return audio_filtered

def segment_audio(audio, sr, segment_length=10.0):
    """Segment audio into fixed-length chunks."""
    segment_samples = int(segment_length * sr)
    segments = []
    
    for start in range(0, len(audio), segment_samples):
        end = min(start + segment_samples, len(audio))
        segment = audio[start:end]
        
        # Pad if necessary
        if len(segment) < segment_samples:
            segment = np.pad(segment, (0, segment_samples - len(segment)), mode='constant')
        
        segments.append(segment)
    
    return segments

def extract_opera_features(audio_segments, sr):
    """Extract OPERA-CT features from audio segments."""
    # Save temporary files for OPERA-CT
    temp_files = []
    features_list = []
    
    try:
        for i, segment in enumerate(tqdm(audio_segments, desc="Extracting OPERA-CT features")):
            temp_file = f"temp_segment_{i}.wav"
            sf.write(temp_file, segment, sr)
            temp_files.append(temp_file)
        
        # Extract features using OPERA-CT
        features = extract_opera_feature(
            temp_files,
            pretrain="operaCT",
            input_sec=len(audio_segments[0]) / sr,
            dim=768
        )
        
        return features
    
    finally:
        # Clean up temp files
        for temp_file in temp_files:
            if Path(temp_file).exists():
                Path(temp_file).unlink()

def run_clustering(features, k_values=[3, 4, 5]):
    """Run K-Means clustering and compute metrics."""
    results = {}
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        
        # Compute silhouette score
        if len(np.unique(labels)) > 1:
            sil_score = silhouette_score(features, labels)
        else:
            sil_score = 0.0
        
        results[f'k{k}'] = {
            'silhouette_score': sil_score,
            'labels': labels,
            'n_clusters': len(np.unique(labels))
        }
    
    return results

def create_umap_visualization(features, clustering_results, method_name, output_dir):
    """Create UMAP visualization."""
    # Reduce dimensions with UMAP
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(features)
    
    # Create subplots for different k values
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'UMAP Visualization - {method_name} (OPERA-CT Features)', fontsize=16)
    
    for i, k in enumerate([3, 4, 5]):
        ax = axes[i]
        labels = clustering_results[f'k{k}']['labels']
        sil_score = clustering_results[f'k{k}']['silhouette_score']
        
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab10', alpha=0.7)
        ax.set_title(f'k={k}, Silhouette={sil_score:.3f}')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        plt.colorbar(scatter, ax=ax)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{method_name}_umap_opera.png', dpi=300, bbox_inches='tight')
    plt.close()

def run_single_method(method_code, data_root):
    """Run a single preprocessing method with OPERA-CT features."""
    print(f"\n🔬 Running {method_code}: {METHOD_CONFIGS[method_code]['name']}")
    print("=" * 60)
    
    config = METHOD_CONFIGS[method_code]
    output_dir = Path(f"OPERA_16_Methods/{method_code}_{config['name']}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load audio files
    data_path = Path(data_root)
    audio_files = list(data_path.glob("*.wav"))
    print(f"📁 Found {len(audio_files)} audio files")
    
    if not audio_files:
        print("❌ No audio files found!")
        return None
    
    # Process all audio files
    all_segments = []
    file_info = []
    
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        try:
            # Load audio
            audio, sr = librosa.load(audio_file, sr=16000, mono=True)
            
            # Apply preprocessing
            processed_audio = apply_preprocessing(audio, sr, config)
            
            # Segment if required
            if config.get('segmentation', False):
                segments = segment_audio(processed_audio, sr, segment_length=10.0)
            else:
                segments = [processed_audio]
            
            all_segments.extend(segments)
            file_info.extend([audio_file.name] * len(segments))
            
        except Exception as e:
            print(f"⚠️ Error processing {audio_file.name}: {e}")
            continue
    
    print(f"🎵 Total segments: {len(all_segments)}")
    
    if not all_segments:
        print("❌ No segments to process!")
        return None
    
    # Extract OPERA-CT features
    print("🤖 Extracting OPERA-CT features...")
    start_time = time.time()
    
    try:
        features = extract_opera_features(all_segments, sr)
        extraction_time = time.time() - start_time
        print(f"✅ Features extracted: {features.shape} in {extraction_time:.1f}s")
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        return None
    
    # Run clustering
    print("🔍 Running K-Means clustering...")
    clustering_results = run_clustering(features, k_values=[3, 4, 5])
    
    # Create visualization
    print("📊 Creating UMAP visualization...")
    create_umap_visualization(features, clustering_results, config['name'], output_dir)
    
    # Save results
    results = {
        'method_code': method_code,
        'method_name': config['name'],
        'config': config,
        'n_files': len(audio_files),
        'n_segments': len(all_segments),
        'feature_shape': features.shape,
        'extraction_time': extraction_time,
        'clustering_results': {k: {
            'silhouette_score': v['silhouette_score'],
            'n_clusters': v['n_clusters']
        } for k, v in clustering_results.items()},
        'best_silhouette': max([v['silhouette_score'] for v in clustering_results.values()]),
        'timestamp': datetime.now().isoformat()
    }
    
    # Save to JSON
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save features
    np.save(output_dir / 'features.npy', features)
    
    print(f"✅ {method_code} completed!")
    print(f"📊 Best silhouette score: {results['best_silhouette']:.3f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run OPERA-CT 16-method experiment")
    parser.add_argument("--method", type=str, choices=list(METHOD_CONFIGS.keys()), 
                       help="Single method to run (e.g., A0, A1, ...)")
    parser.add_argument("--series", type=str, choices=['A', 'B', 'C', 'D'], 
                       help="Run entire series (A, B, C, or D)")
    parser.add_argument("--data_root", type=str, 
                       default="/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/RAW sound_ML test sound list",
                       help="Path to audio data")
    
    args = parser.parse_args()
    
    if not args.method and not args.series:
        print("❌ Please specify either --method or --series")
        return
    
    # Determine methods to run
    if args.method:
        methods_to_run = [args.method]
    else:
        series_methods = {
            'A': ['A0', 'A1', 'A2', 'A3', 'A4'],
            'B': ['B0', 'B1', 'B2'],
            'C': ['C0', 'C1', 'C2', 'C3', 'C4'],
            'D': ['D0', 'D1', 'D2']
        }
        methods_to_run = series_methods[args.series]
    
    print(f"🚀 Starting OPERA-CT experiment with methods: {methods_to_run}")
    print(f"📁 Data root: {args.data_root}")
    
    # Run experiments
    all_results = []
    for method in methods_to_run:
        result = run_single_method(method, args.data_root)
        if result:
            all_results.append(result)
    
    # Create summary
    if all_results:
        summary_df = pd.DataFrame([{
            'Method': r['method_code'],
            'Name': r['method_name'],
            'Files': r['n_files'],
            'Segments': r['n_segments'],
            'Best_Silhouette': r['best_silhouette'],
            'K3_Silhouette': r['clustering_results']['k3']['silhouette_score'],
            'K4_Silhouette': r['clustering_results']['k4']['silhouette_score'],
            'K5_Silhouette': r['clustering_results']['k5']['silhouette_score'],
        } for r in all_results])
        
        summary_df = summary_df.sort_values('Best_Silhouette', ascending=False)
        
        print("\n📊 RESULTS SUMMARY")
        print("=" * 80)
        print(summary_df.to_string(index=False, float_format='%.3f'))
        
        # Save summary
        summary_df.to_csv('OPERA_16_Methods/results_summary.csv', index=False)
        
        print(f"\n✅ Experiment completed! Results saved to OPERA_16_Methods/")
        print(f"🏆 Best method: {summary_df.iloc[0]['Method']} ({summary_df.iloc[0]['Best_Silhouette']:.3f})")

if __name__ == "__main__":
    main()
