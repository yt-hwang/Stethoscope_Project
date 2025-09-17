#!/usr/bin/env python3
"""
Clustering for Cycle A0: NoSeg + NoPreprocess baseline
Runs KMeans and HDBSCAN on all three representations
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

# Set random seeds for reproducibility
np.random.seed(42)

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

def run_kmeans(features, k_values=[3, 4, 5]):
    """Run KMeans clustering with different k values"""
    results = {}
    
    for k in k_values:
        print(f"Running KMeans with k={k}")
        
        # Initialize and fit KMeans
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        
        # Calculate metrics
        silhouette = silhouette_score(features, labels)
        calinski_harabasz = calinski_harabasz_score(features, labels)
        davies_bouldin = davies_bouldin_score(features, labels)
        
        results[f'k{k}'] = {
            'algorithm': 'kmeans',
            'params': {'n_clusters': k},
            'labels': labels.tolist(),
            'silhouette': float(silhouette),
            'calinski_harabasz': float(calinski_harabasz),
            'davies_bouldin': float(davies_bouldin),
            'n_clusters': k,
            'n_noise': 0
        }
        
        print(f"  Silhouette: {silhouette:.4f}")
        print(f"  Calinski-Harabasz: {calinski_harabasz:.4f}")
        print(f"  Davies-Bouldin: {davies_bouldin:.4f}")
    
    return results

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
        davies_bouldin = float('inf')
    
    n_clusters = len(np.unique(labels[non_noise_mask])) if np.sum(non_noise_mask) > 0 else 0
    n_noise = np.sum(labels == -1)
    
    results = {
        'hdbscan': {
            'algorithm': 'hdbscan',
            'params': {'min_cluster_size': min_cluster_size},
            'labels': labels.tolist(),
            'silhouette': float(silhouette),
            'calinski_harabasz': float(calinski_harabasz),
            'davies_bouldin': float(davies_bouldin),
            'n_clusters': int(n_clusters),
            'n_noise': int(n_noise)
        }
    }
    
    print(f"  Silhouette: {silhouette:.4f}")
    print(f"  Calinski-Harabasz: {calinski_harabasz:.4f}")
    print(f"  Davies-Bouldin: {davies_bouldin:.4f}")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Number of noise points: {n_noise}")
    
    return results

def cluster_representation(features_path, representation_name, output_dir):
    """Cluster a specific representation"""
    print(f"\nClustering {representation_name}...")
    
    # Load features
    features, df, scaler = load_features(features_path)
    print(f"Feature matrix shape: {features.shape}")
    
    # Run KMeans
    kmeans_results = run_kmeans(features)
    
    # Run HDBSCAN
    hdbscan_results = run_hdbscan(features)
    
    # Combine results
    all_results = {**kmeans_results, **hdbscan_results}
    
    # Save results
    os.makedirs(os.path.join(output_dir, 'clustering'), exist_ok=True)
    output_file = os.path.join(output_dir, 'clustering', f'clustering_{representation_name}_metrics.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Saved clustering results to {output_file}")
    
    return all_results, df

def main():
    parser = argparse.ArgumentParser(description='Run clustering for Cycle A0')
    parser.add_argument('--output_dir', type=str,
                       default='/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/A2_NoSeg_SpectralGating/outputs',
                       help='Path to output directory')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define representations
    representations = {
        'raw_waveform_stats': 'features/features_raw_waveform_stats.csv',
        'logmel_mean': 'features/features_logmel_mean.csv',
        'mfcc_mean': 'features/features_mfcc_mean.csv'
    }
    
    all_results = {}
    
    # Cluster each representation
    for rep_name, csv_file in representations.items():
        features_path = os.path.join(args.output_dir, csv_file)
        
        if not os.path.exists(features_path):
            print(f"Warning: {features_path} not found. Skipping {rep_name}.")
            continue
            
        results, df = cluster_representation(features_path, rep_name, args.output_dir)
        all_results[rep_name] = results
    
    # Print summary
    print("\n" + "="*60)
    print("CLUSTERING SUMMARY")
    print("="*60)
    
    for rep_name, results in all_results.items():
        print(f"\n{rep_name.upper()}:")
        print("-" * 40)
        
        for algo_name, metrics in results.items():
            print(f"{algo_name}:")
            print(f"  Silhouette: {metrics['silhouette']:.4f}")
            print(f"  Calinski-Harabasz: {metrics['calinski_harabasz']:.4f}")
            print(f"  Davies-Bouldin: {metrics['davies_bouldin']:.4f}")
            print(f"  Clusters: {metrics['n_clusters']}")
            if metrics['n_noise'] > 0:
                print(f"  Noise points: {metrics['n_noise']}")
    
    print(f"\nClustering completed! Results saved in {args.output_dir}")

if __name__ == "__main__":
    main()
