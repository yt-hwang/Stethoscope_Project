#!/usr/bin/env python3
"""
Layer Experiment Runner - Sequential testing of different OPERA-CT layers
Tests layers 0-3 systematically to find optimal feature extraction point
"""

import argparse
import sys
import os
import json
import time
import pandas as pd
import numpy as np
import librosa
from pathlib import Path
from datetime import datetime

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from excel_logger import ExcelLogger
from run_manager import RunManager
from layer_extractor import extract_features_by_layer

def load_audio_segments(data_root, seg_sec=7.5, max_files=None):
    """Load and segment audio files for layer testing."""
    data_root = Path(data_root)
    
    # Find audio files
    audio_files = []
    for ext in ['.wav', '.flac', '.mp3']:
        audio_files.extend(data_root.glob(f"**/*{ext}"))
    
    if max_files:
        audio_files = audio_files[:max_files]
    
    print(f"Loading {len(audio_files)} audio files...")
    
    all_segments = []
    file_info = []
    
    for file_path in audio_files:
        try:
            # Load audio
            audio, sr = librosa.load(file_path, sr=16000, mono=True)
            
            # Segment audio
            seg_samples = int(seg_sec * sr)
            n_segments = len(audio) // seg_samples
            
            for i in range(n_segments):
                start = i * seg_samples
                end = start + seg_samples
                segment = audio[start:end]
                
                all_segments.append(segment)
                file_info.append({
                    'file_path': str(file_path),
                    'filename': file_path.name,
                    'segment_idx': i
                })
        
        except Exception as e:
            print(f"Warning: Could not process {file_path}: {e}")
            continue
    
    print(f"Created {len(all_segments)} segments from {len(audio_files)} files")
    return all_segments, file_info, 16000

def run_single_layer_experiment(layer_name, layer_dir, audio_segments, sr, file_info):
    """Run experiment for a single layer."""
    print(f"\n{'='*60}")
    print(f"LAYER EXPERIMENT: {layer_name}")
    print(f"{'='*60}")
    
    # Create layer-specific directories
    layer_path = Path(layer_dir)
    results_dir = layer_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Extract features from specific layer
    print(f"Step 1: Extracting features from {layer_name}...")
    start_time = time.time()
    
    features, feature_dim, error = extract_features_by_layer(audio_segments, sr, layer_name)
    
    extraction_time = time.time() - start_time
    
    if error or features is None:
        error_msg = error or "Unknown error - features is None"
        print(f"❌ Feature extraction failed: {error_msg}")
        
        # Save error report
        error_report = {
            'layer_name': layer_name,
            'error': error_msg,
            'timestamp': datetime.now().isoformat()
        }
        
        error_path = results_dir / "extraction_error.json"
        with open(error_path, 'w') as f:
            json.dump(error_report, f, indent=2)
        
        return None
    
    print(f"✅ Features extracted: {features.shape} in {extraction_time:.1f}s")
    
    # Step 2: Run clustering analysis
    print(f"Step 2: Running clustering analysis...")
    
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    import umap
    import matplotlib.pyplot as plt
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Test different k values
    k_range = range(2, min(11, len(features) // 2))
    silhouette_scores = []
    k_values = []
    
    for k in k_range:
        if k >= len(features):
            continue
            
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_scaled)
        
        if len(np.unique(labels)) < 2:
            continue
            
        sil_score = silhouette_score(features_scaled, labels)
        silhouette_scores.append(sil_score)
        k_values.append(k)
        
        print(f"  k={k}: silhouette={sil_score:.3f}")
    
    if not silhouette_scores:
        print(f"❌ No valid clustering found for {layer_name}")
        return None
    
    # Choose best k
    best_idx = np.argmax(silhouette_scores)
    best_k = k_values[best_idx]
    best_silhouette = silhouette_scores[best_idx]
    
    print(f"✅ Best clustering: k={best_k}, silhouette={best_silhouette:.3f}")
    
    # Step 3: Create UMAP visualization
    print(f"Step 3: Creating UMAP visualization...")
    
    # Final clustering with best k
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    final_labels = kmeans.fit_predict(features_scaled)
    
    # UMAP visualization
    n_neighbors = min(15, len(features) - 1)
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors)
    embedding = reducer.fit_transform(features_scaled)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=final_labels, cmap='viridis', alpha=0.7, s=50)
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'UMAP: {layer_name} Features (k={best_k}, silhouette={best_silhouette:.3f})', 
              fontsize=14, fontweight='bold')
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add info
    plt.figtext(0.02, 0.02, f'Samples: {len(features)}, Features: {feature_dim}, Layer: {layer_name}', 
                fontsize=10, style='italic')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = results_dir / f"umap_{layer_name}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ UMAP saved: {plot_path}")
    
    # Step 4: Save results
    print(f"Step 4: Saving results...")
    
    results = {
        'layer_name': layer_name,
        'feature_dim': int(feature_dim),
        'n_samples': len(features),
        'best_k': int(best_k),
        'best_silhouette': float(best_silhouette),
        'extraction_time': extraction_time,
        'timestamp': datetime.now().isoformat(),
        'umap_path': str(plot_path),
        'all_k_results': [{'k': int(k), 'silhouette': float(s)} for k, s in zip(k_values, silhouette_scores)]
    }
    
    results_path = results_dir / f"layer_{layer_name}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved: {results_path}")
    
    # Save features for comparison
    features_df = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(feature_dim)])
    
    # Add metadata
    for i, info in enumerate(file_info[:len(features)]):
        for key, value in info.items():
            features_df.loc[i, key] = value
    
    features_path = results_dir / f"features_{layer_name}.parquet"
    features_df.to_parquet(features_path, index=False)
    
    print(f"✅ Features saved: {features_path}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run layer-specific OPERA-CT experiments")
    parser.add_argument("--data_root", required=True, help="Path to audio data")
    parser.add_argument("--layers", nargs='+', 
                       choices=['layer_0', 'layer_1', 'layer_2', 'layer_3', 'final'],
                       default=['layer_0', 'layer_1', 'layer_2', 'layer_3', 'final'],
                       help="Layers to test")
    parser.add_argument("--seg_sec", type=float, default=7.5, help="Segment length")
    parser.add_argument("--max_files", type=int, help="Maximum files to process (for testing)")
    
    args = parser.parse_args()
    
    print("OPERA-CT LAYER ANALYSIS EXPERIMENT")
    print("=" * 50)
    print(f"Data root: {args.data_root}")
    print(f"Layers to test: {args.layers}")
    print(f"Segment length: {args.seg_sec}s")
    
    # Load audio data
    print("\nStep 1: Loading and segmenting audio...")
    audio_segments, file_info, sr = load_audio_segments(args.data_root, args.seg_sec, args.max_files)
    
    if len(audio_segments) == 0:
        print("❌ No audio segments created")
        return
    
    # Initialize Excel logging
    excel_logger = ExcelLogger()
    
    # Layer mapping for folder names
    layer_folders = {
        'layer_0': 'Layer_0_Early',
        'layer_1': 'Layer_1_Middle', 
        'layer_2': 'Layer_2_Late',
        'layer_3': 'Layer_3_Final',
        'final': 'Layer_3_Final'  # Final uses same folder as layer_3
    }
    
    all_results = []
    
    # Run experiments for each layer
    for layer_name in args.layers:
        layer_dir = layer_folders[layer_name]
        
        print(f"\n🔍 Testing {layer_name}...")
        result = run_single_layer_experiment(layer_name, layer_dir, audio_segments, sr, file_info)
        
        if result:
            all_results.append(result)
            
            # Log to Excel
            excel_data = {
                'layer_name': layer_name,
                'feature_dim': result['feature_dim'],
                'n_samples': result['n_samples'],
                'best_k': result['best_k'],
                'best_silhouette': result['best_silhouette'],
                'extraction_time': result['extraction_time'],
                'umap_path': result['umap_path'],
                'results_path': str(Path(layer_dir) / "results" / f"layer_{layer_name}_results.json")
            }
            
            try:
                excel_logger.append_row('cluster', excel_data)
                print(f"✅ Excel logged for {layer_name}")
            except Exception as e:
                print(f"⚠️ Excel logging failed for {layer_name}: {e}")
    
    # Step 5: Create comparison summary
    print(f"\n📊 LAYER COMPARISON SUMMARY:")
    print("=" * 50)
    
    if all_results:
        # Sort by silhouette score
        sorted_results = sorted(all_results, key=lambda x: x['best_silhouette'], reverse=True)
        
        print(f"{'Rank':<6} {'Layer':<12} {'Feature Dim':<12} {'Best k':<8} {'Silhouette':<12} {'Time (s)':<10}")
        print("-" * 70)
        
        for rank, result in enumerate(sorted_results, 1):
            print(f"{rank:<6} {result['layer_name']:<12} {result['feature_dim']:<12} {result['best_k']:<8} {result['best_silhouette']:<12.3f} {result['extraction_time']:<10.1f}")
        
        # Save comparison results
        comparison_dir = Path("Layer_Comparison")
        comparison_dir.mkdir(exist_ok=True)
        
        comparison_results = {
            'timestamp': datetime.now().isoformat(),
            'data_root': args.data_root,
            'n_segments': len(audio_segments),
            'seg_sec': args.seg_sec,
            'layer_results': sorted_results,
            'best_layer': sorted_results[0]['layer_name'],
            'best_silhouette': sorted_results[0]['best_silhouette'],
            'performance_ranking': [r['layer_name'] for r in sorted_results]
        }
        
        comparison_path = comparison_dir / "layer_comparison_results.json"
        with open(comparison_path, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        print(f"\n🏆 BEST PERFORMING LAYER: {sorted_results[0]['layer_name']}")
        print(f"🏆 BEST SILHOUETTE SCORE: {sorted_results[0]['best_silhouette']:.3f}")
        print(f"📊 Comparison saved: {comparison_path}")
        
        # Compare with Full Frozen baseline
        full_frozen_baseline = 0.255  # From our previous experiment
        best_score = sorted_results[0]['best_silhouette']
        improvement = ((best_score - full_frozen_baseline) / full_frozen_baseline * 100)
        
        print(f"\n📈 COMPARISON WITH FULL FROZEN BASELINE:")
        print(f"Full Frozen (final layer): {full_frozen_baseline:.3f}")
        print(f"Best layer ({sorted_results[0]['layer_name']}): {best_score:.3f}")
        print(f"Improvement: {improvement:+.1f}%")
        
        if improvement > 10:
            print("🎉 SIGNIFICANT IMPROVEMENT FOUND!")
        elif improvement > 0:
            print("✅ Modest improvement found")
        else:
            print("❌ No improvement over final layer")
    
    else:
        print("❌ No successful layer experiments")
    
    print("\nSTOP - Layer analysis complete.")

if __name__ == "__main__":
    main()
