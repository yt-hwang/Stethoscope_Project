#!/usr/bin/env python3
"""
Cycle C2: Seg + SpectralGating - Clustering and Evaluation
"""

import os
import sys
import numpy as np
import pandas as pd
import json
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import hdbscan
import argparse
from pathlib import Path

def load_features(features_path):
    """Load features from CSV file"""
    df = pd.read_csv(features_path)
    
    # Extract feature columns (exclude file_path, filename, and preprocessing)
    feature_cols = [col for col in df.columns if col not in ['file_path', 'filename', 'preprocessing']]
    features = df[feature_cols].values
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    return features_scaled, df, scaler

def run_kmeans(features, k, random_state=42):
    """Run KMeans clustering"""
    print(f"Running KMeans with k={k}")
    
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(features)
    
    # Calculate metrics
    silhouette = silhouette_score(features, labels)
    calinski_harabasz = calinski_harabasz_score(features, labels)
    davies_bouldin = davies_bouldin_score(features, labels)
    
    return {
        f'kmeans_k{k}': {
            'Silhouette': silhouette,
            'Calinski-Harabasz': calinski_harabasz,
            'Davies-Bouldin': davies_bouldin,
            'n_clusters': k,
            'n_noise': 0,
            'params': {'n_clusters': k, 'random_state': random_state}
        }
    }

def run_hdbscan(features, min_cluster_size=25):
    """Run HDBSCAN clustering"""
    print(f"Running HDBSCAN with min_cluster_size={min_cluster_size}")
    
    # Initialize and fit HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(features)
    
    # Calculate metrics (only for non-noise points)
    non_noise_mask = labels != -1
    if np.sum(non_noise_mask) > 1 and len(np.unique(labels[non_noise_mask])) > 1:
        silhouette = silhouette_score(features[non_noise_mask], labels[non_noise_mask])
        calinski_harabasz = calinski_harabasz_score(features[non_noise_mask], labels[non_noise_mask])
        davies_bouldin = davies_bouldin_score(features[non_noise_mask], labels[non_noise_mask])
    else:
        silhouette = -1.0
        calinski_harabasz = 0.0
        davies_bouldin = np.inf
    
    n_clusters = len(np.unique(labels[non_noise_mask])) if np.sum(non_noise_mask) > 0 else 0
    n_noise = np.sum(labels == -1)
    
    return {
        f'hdbscan': {
            'Silhouette': silhouette,
            'Calinski-Harabasz': calinski_harabasz,
            'Davies-Bouldin': davies_bouldin,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'params': {'min_cluster_size': min_cluster_size}
        }
    }

def main():
    parser = argparse.ArgumentParser(description='Run clustering for Cycle C2')
    parser.add_argument('--output_dir', 
                       default='/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/C2_Seg_SpectralGating/outputs',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(os.path.join(args.output_dir, 'clustering'), exist_ok=True)
    
    representations = ['raw_waveform_stats', 'logmel_mean', 'mfcc_mean']
    
    for rep in representations:
        print(f"\nProcessing {rep}...")
        
        # Load features
        features_path = os.path.join(args.output_dir, 'features', f'features_{rep}.csv')
        if not os.path.exists(features_path):
            print(f"  Warning: {features_path} not found")
            continue
        
        features, df, scaler = load_features(features_path)
        print(f"  Loaded {features.shape[0]} samples with {features.shape[1]} features")
        
        # Run clustering algorithms
        all_results = {}
        
        # KMeans with different k values
        for k in [3, 4, 5]:
            results = run_kmeans(features, k)
            all_results.update(results)
        
        # HDBSCAN
        results = run_hdbscan(features)
        all_results.update(results)
        
        # Save results (convert numpy types to Python types)
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Convert all numpy types to Python types
        converted_results = {}
        for key, value in all_results.items():
            if isinstance(value, dict):
                converted_results[key] = {k: convert_numpy(v) for k, v in value.items()}
            else:
                converted_results[key] = convert_numpy(value)
        
        output_path = os.path.join(args.output_dir, 'clustering', f'clustering_{rep}_metrics.json')
        with open(output_path, 'w') as f:
            json.dump(converted_results, f, indent=2)
        
        print(f"  Saved clustering results to {output_path}")
        
        # Print summary
        print(f"  Summary for {rep}:")
        for algo, metrics in all_results.items():
            print(f"    {algo}: Silhouette={metrics['Silhouette']:.3f}, "
                  f"CH={metrics['Calinski-Harabasz']:.1f}, "
                  f"DB={metrics['Davies-Bouldin']:.3f}, "
                  f"Clusters={metrics['n_clusters']}, "
                  f"Noise={metrics['n_noise']}")
    
    print(f"\nClustering completed for Cycle C2: Seg + SpectralGating")

if __name__ == "__main__":
    main()
