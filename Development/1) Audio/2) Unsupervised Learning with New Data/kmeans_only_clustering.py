#!/usr/bin/env python3
"""
K-Means Only Clustering Script
Ensures exactly 9 results per cycle (3 reps × 3 k-values)
Logs failures as 0.000 quality score
"""

import pandas as pd
import numpy as np
import os
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import argparse

# Configuration - K-MEANS ONLY
KMEANS_K_VALUES = [3, 4, 5]
RANDOM_SEED = 42
N_SEEDS = 7

# Cluster Quality Constraints
MIN_CLUSTER_SIZE = 3
MAX_GINI_COEFFICIENT = 0.8
MIN_SILHOUETTE_THRESHOLD = -0.5

def calculate_gini_coefficient(cluster_sizes):
    """Calculate Gini coefficient for cluster size distribution."""
    if len(cluster_sizes) <= 1:
        return 0.0
    cluster_sizes = np.array(cluster_sizes)
    cluster_sizes = cluster_sizes[cluster_sizes > 0]
    if len(cluster_sizes) <= 1:
        return 0.0
    
    sorted_sizes = np.sort(cluster_sizes)
    n = len(sorted_sizes)
    cumsum = np.cumsum(sorted_sizes)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

def validate_cluster_quality(labels, n_samples):
    """Validate cluster quality."""
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    n_clusters = len(unique_labels)
    min_cluster_size = np.min(counts)
    gini_coeff = calculate_gini_coefficient(counts)
    
    # Validation rules
    valid = True
    if n_clusters < 2 or min_cluster_size < MIN_CLUSTER_SIZE or gini_coeff > MAX_GINI_COEFFICIENT:
        valid = False
    
    return valid

def run_kmeans_only(features, representation, cycle_name):
    """Run K-Means clustering only - always returns 3 results."""
    all_results = {}
    
    print(f"🔍 K-Means Only: {representation}")
    print(f"Dataset: {len(features)} samples, {features.shape[1]} features")
    print("-" * 40)
    
    for k in KMEANS_K_VALUES:
        print(f"  K-Means k={k} (N={N_SEEDS} seeds)...")
        
        silhouette_scores = []
        calinski_harabasz_scores = []
        davies_bouldin_scores = []
        valid_runs = 0
        
        for seed in range(N_SEEDS):
            kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED + seed, n_init=10)
            labels = kmeans.fit_predict(features)
            
            # Validate cluster quality
            if validate_cluster_quality(labels, len(features)):
                silhouette = silhouette_score(features, labels)
                if silhouette >= MIN_SILHOUETTE_THRESHOLD:
                    calinski_harabasz = calinski_harabasz_score(features, labels)
                    davies_bouldin = davies_bouldin_score(features, labels)
                    
                    silhouette_scores.append(silhouette)
                    calinski_harabasz_scores.append(calinski_harabasz)
                    davies_bouldin_scores.append(davies_bouldin)
                    valid_runs += 1
        
        # ALWAYS create result entry (even for failures)
        if valid_runs > 0:
            # Valid results
            mean_silhouette = np.mean(silhouette_scores)
            std_silhouette = np.std(silhouette_scores)
            stability = 1.0 - min(std_silhouette, 1.0)
            quality_score = mean_silhouette * stability
            
            all_results[f'kmeans_k{k}'] = {
                'algorithm': 'kmeans',
                'params': {'n_clusters': k},
                'silhouette_mean': float(mean_silhouette),
                'silhouette_std': float(std_silhouette),
                'calinski_harabasz_mean': float(np.mean(calinski_harabasz_scores)),
                'davies_bouldin_mean': float(np.mean(davies_bouldin_scores)),
                'stability': float(stability),
                'quality_score': float(quality_score),
                'valid_runs': int(valid_runs),
                'total_runs': int(N_SEEDS)
            }
            
            print(f"    ✅ Valid: {valid_runs}/{N_SEEDS}, Quality: {quality_score:.3f}")
        else:
            # Failed results - log as 0.000
            all_results[f'kmeans_k{k}'] = {
                'algorithm': 'kmeans',
                'params': {'n_clusters': k},
                'silhouette_mean': 0.0,
                'silhouette_std': 0.0,
                'calinski_harabasz_mean': 0.0,
                'davies_bouldin_mean': 0.0,
                'stability': 0.0,
                'quality_score': 0.0,
                'valid_runs': 0,
                'total_runs': int(N_SEEDS)
            }
            print(f"    ❌ Failed: 0/{N_SEEDS}, Quality: 0.000")
    
    return all_results

def load_features(output_dir, representation):
    """Load features for a given representation."""
    features_path = os.path.join(output_dir, 'features', f'features_{representation}.csv')
    df = pd.read_csv(features_path)
    
    feature_cols = [col for col in df.columns if col not in ['file_path', 'filename', 'original_filename', 'segment_index']]
    features = df[feature_cols].values
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    return features_scaled, df['filename'].tolist()

def main():
    parser = argparse.ArgumentParser(description='K-Means only clustering with complete logging')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to cycle output directory')
    parser.add_argument('--cycle_name', type=str, required=True, help='Cycle name for logging')
    
    args = parser.parse_args()
    
    clustering_dir = os.path.join(args.output_dir, 'clustering')
    os.makedirs(clustering_dir, exist_ok=True)
    
    print(f"🧪 K-MEANS ONLY CLUSTERING: {args.cycle_name}")
    print("=" * 60)
    
    # Process each representation - ALWAYS 3 results per representation
    representations = ['raw_waveform_stats', 'logmel_mean', 'mfcc_mean']
    
    for representation in representations:
        try:
            features, filenames = load_features(args.output_dir, representation)
            results = run_kmeans_only(features, representation, args.cycle_name)
            
            # Save results
            results_path = os.path.join(clustering_dir, f'clustering_{representation}_improved_metrics.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✅ Saved: {os.path.basename(results_path)} ({len(results)} results)")
            
        except Exception as e:
            print(f"❌ Error processing {representation}: {e}")
        
        print()
    
    print("🎉 K-Means clustering completed!")
    print(f"Expected: 9 results (3 reps × 3 k-values)")

if __name__ == "__main__":
    main()
