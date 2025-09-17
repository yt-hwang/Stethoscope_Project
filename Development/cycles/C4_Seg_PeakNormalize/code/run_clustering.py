#!/usr/bin/env python3
"""
Cycle C4: Seg + PeakNormalize - Clustering and Evaluation
"""

import pandas as pd
import numpy as np
import os
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import hdbscan
import argparse

# Configuration
KMEANS_K_VALUES = [3, 4, 5]
HDBSCAN_MIN_CLUSTER_SIZE = 25
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

def load_features(output_dir, representation):
    """Load features for a given representation."""
    features_path = os.path.join(output_dir, 'features', f'features_{representation}.csv')
    df = pd.read_csv(features_path)
    
    # Exclude non-feature columns
    feature_cols = [col for col in df.columns if col not in ['file_path', 'filename']]
    features = df[feature_cols].values
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    return features_scaled, df['filename'].tolist()

def run_clustering(features, filenames, representation, output_dir):
    """Run KMeans and HDBSCAN clustering and return metrics."""
    all_results = {}
    
    # KMeans clustering
    for k in KMEANS_K_VALUES:
        print(f"  Running KMeans with k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = kmeans.fit_predict(features)
        
        # Calculate metrics
        if len(np.unique(labels)) < 2:
            print(f"    Warning: Less than 2 clusters found for k={k}")
            continue
            
        silhouette = silhouette_score(features, labels)
        calinski_harabasz = calinski_harabasz_score(features, labels)
        davies_bouldin = davies_bouldin_score(features, labels)
        
        all_results[f'kmeans_k{k}'] = {
            'representation': representation,
            'algorithm': 'kmeans',
            'params': {'n_clusters': k},
            'Silhouette': silhouette,
            'Calinski-Harabasz': calinski_harabasz,
            'Davies-Bouldin': davies_bouldin,
            'n_clusters': len(np.unique(labels)),
            'n_noise': 0,
            'labels': labels.tolist()
        }
        
        print(f"    Silhouette: {silhouette:.3f}, CH: {calinski_harabasz:.1f}, DB: {davies_bouldin:.3f}")

    # HDBSCAN clustering
    print(f"  Running HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE)
    labels = clusterer.fit_predict(features)
    
    # Calculate metrics (excluding noise points)
    non_noise_indices = labels != -1
    if np.sum(non_noise_indices) > 1 and len(np.unique(labels[non_noise_indices])) > 1:
        silhouette = silhouette_score(features[non_noise_indices], labels[non_noise_indices])
        calinski_harabasz = calinski_harabasz_score(features[non_noise_indices], labels[non_noise_indices])
        davies_bouldin = davies_bouldin_score(features[non_noise_indices], labels[non_noise_indices])
        n_clusters = len(np.unique(labels[non_noise_indices]))
    else:
        print(f"    Warning: Not enough non-noise points for HDBSCAN metrics")
        silhouette, calinski_harabasz, davies_bouldin = np.nan, np.nan, np.nan
        n_clusters = 0
    
    n_noise = np.sum(labels == -1)
    
    all_results['hdbscan'] = {
        'representation': representation,
        'algorithm': 'hdbscan',
        'params': {'min_cluster_size': HDBSCAN_MIN_CLUSTER_SIZE},
        'Silhouette': silhouette,
        'Calinski-Harabasz': calinski_harabasz,
        'Davies-Bouldin': davies_bouldin,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'labels': labels.tolist()
    }
    
    print(f"    Silhouette: {silhouette:.3f}, CH: {calinski_harabasz:.1f}, DB: {davies_bouldin:.3f}")
    print(f"    Clusters: {n_clusters}, Noise points: {n_noise}")
    
    # Save results
    output_path = os.path.join(output_dir, 'clustering', f'clustering_{representation}_metrics.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert numpy types for JSON serialization
    converted_results = {}
    for key, value in all_results.items():
        if isinstance(value, dict):
            converted_results[key] = {k: convert_numpy(v) for k, v in value.items()}
        else:
            converted_results[key] = convert_numpy(value)
    
    with open(output_path, 'w') as f:
        json.dump(converted_results, f, indent=2)
    
    print(f"  Saved results to {output_path}")
    return all_results

def main():
    parser = argparse.ArgumentParser(description="Run clustering algorithms and evaluate metrics.")
    parser.add_argument('--output_dir', type=str, 
                       default='/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/C4_Seg_PeakNormalize/outputs',
                       help='Path to the output directory for this cycle.')
    
    args = parser.parse_args()

    representations = ['raw_waveform_stats', 'logmel_mean', 'mfcc_mean']
    
    for rep in representations:
        print(f"\nEvaluating {rep}...")
        try:
            features, filenames = load_features(args.output_dir, rep)
            print(f"  Loaded {len(features)} samples with {features.shape[1]} features")
            run_clustering(features, filenames, rep, args.output_dir)
        except Exception as e:
            print(f"  Error processing {rep}: {e}")
            continue

    print(f"\nClustering evaluation completed!")

if __name__ == "__main__":
    main()
