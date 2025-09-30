#!/usr/bin/env python3
"""
Run clustering algorithms on extracted features for D1_new: Seg + PeakNormalize + Bandpass
"""

import pandas as pd
import numpy as np
import os
import json
import argparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import hdbscan

# Configuration
KMEANS_K_VALUES = [3, 4, 5]
HDBSCAN_MIN_CLUSTER_SIZE = 3
RANDOM_SEED = 42

def convert_numpy(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def run_clustering_on_representation(features_df, representation_name, output_dir):
    """Run clustering on a specific representation."""
    print(f"\nProcessing {representation_name}...")
    
    # Prepare features (exclude file paths and filenames)
    feature_cols = [col for col in features_df.columns if col not in ['file_path', 'filename']]
    X = features_df[feature_cols].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {}
    
    # K-Means clustering
    for k in KMEANS_K_VALUES:
        print(f"  Running K-Means with k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # Calculate metrics
        silhouette = silhouette_score(X_scaled, labels)
        calinski_harabasz = calinski_harabasz_score(X_scaled, labels)
        davies_bouldin = davies_bouldin_score(X_scaled, labels)
        
        results[f'kmeans_k{k}'] = {
            'algorithm': 'kmeans',
            'k': k,
            'silhouette': convert_numpy(silhouette),
            'calinski_harabasz': convert_numpy(calinski_harabasz),
            'davies_bouldin': convert_numpy(davies_bouldin),
            'num_clusters': k,
            'num_noise_points': 0,
            'labels': convert_numpy(labels)
        }
        
        print(f"    Silhouette: {silhouette:.4f}, CH: {calinski_harabasz:.4f}, DB: {davies_bouldin:.4f}")
    
    # HDBSCAN clustering
    print(f"  Running HDBSCAN with min_cluster_size={HDBSCAN_MIN_CLUSTER_SIZE}...")
    hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE)
    labels = hdbscan_clusterer.fit_predict(X_scaled)
    
    # Calculate metrics (excluding noise points for silhouette)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = np.sum(labels == -1)
    
    if n_clusters > 1:
        # Calculate silhouette score excluding noise points
        non_noise_mask = labels != -1
        if np.sum(non_noise_mask) > 1 and len(np.unique(labels[non_noise_mask])) > 1:
            silhouette = silhouette_score(X_scaled[non_noise_mask], labels[non_noise_mask])
        else:
            silhouette = -1.0
        
        calinski_harabasz = calinski_harabasz_score(X_scaled, labels)
        davies_bouldin = davies_bouldin_score(X_scaled, labels)
    else:
        silhouette = -1.0
        calinski_harabasz = 0.0
        davies_bouldin = float('inf')
    
    results[f'hdbscan_min_cluster_size{HDBSCAN_MIN_CLUSTER_SIZE}'] = {
        'algorithm': 'hdbscan',
        'min_cluster_size': HDBSCAN_MIN_CLUSTER_SIZE,
        'silhouette': convert_numpy(silhouette),
        'calinski_harabasz': convert_numpy(calinski_harabasz),
        'davies_bouldin': convert_numpy(davies_bouldin),
        'num_clusters': n_clusters,
        'num_noise_points': int(n_noise),
        'labels': convert_numpy(labels)
    }
    
    print(f"    Silhouette: {silhouette:.4f}, CH: {calinski_harabasz:.4f}, DB: {davies_bouldin:.4f}")
    print(f"    Clusters: {n_clusters}, Noise points: {n_noise}")
    
    # Save results
    os.makedirs(os.path.join(output_dir, 'clustering'), exist_ok=True)
    output_file = os.path.join(output_dir, 'clustering', f'clustering_{representation_name}_metrics.json')
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  Saved clustering results to {output_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run clustering on extracted features.")
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the output directory.')
    
    args = parser.parse_args()
    
    print("="*60)
    print("D1_new: Seg + PeakNormalize + Bandpass - Clustering")
    print("="*60)
    
    # Process each representation
    representations = ['raw_waveform_stats', 'logmel_mean', 'mfcc_mean']
    
    for rep in representations:
        features_file = os.path.join(args.output_dir, 'features', f'features_{rep}.csv')
        
        if not os.path.exists(features_file):
            print(f"Warning: Features file not found: {features_file}")
            continue
        
        # Load features
        features_df = pd.read_csv(features_file)
        print(f"\nLoaded {len(features_df)} samples for {rep}")
        
        # Run clustering
        run_clustering_on_representation(features_df, rep, args.output_dir)
    
    print("\n" + "="*60)
    print("Clustering completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
