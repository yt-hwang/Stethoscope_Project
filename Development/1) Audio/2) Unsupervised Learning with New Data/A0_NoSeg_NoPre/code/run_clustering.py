#!/usr/bin/env python3
"""
A0: NoSeg + NoPreprocess - Clustering Analysis
NEW DATA EXPERIMENT - Using RAW sound_ML test sound list
Robust evaluation with improved validation system
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
HDBSCAN_MIN_CLUSTER_SIZE = 3  # Adjusted for smaller dataset
RANDOM_SEED = 42
N_SEEDS = 7  # For robust evaluation

# Cluster Quality Constraints (improved validation system)
MIN_CLUSTER_SIZE = 3  # Minimum samples per cluster
MAX_GINI_COEFFICIENT = 0.8  # Maximum imbalance
MIN_SILHOUETTE_THRESHOLD = -0.5  # Minimum silhouette score
MAX_HDBSCAN_NOISE_FRACTION = 0.8  # Maximum noise fraction for HDBSCAN

def calculate_gini_coefficient(cluster_sizes):
    """Calculate Gini coefficient for cluster size distribution."""
    if len(cluster_sizes) <= 1:
        return 0.0
    cluster_sizes = np.array(cluster_sizes)
    cluster_sizes = cluster_sizes[cluster_sizes > 0]  # Remove empty clusters
    if len(cluster_sizes) <= 1:
        return 0.0
    
    # Calculate Gini coefficient manually
    sorted_sizes = np.sort(cluster_sizes)
    n = len(sorted_sizes)
    cumsum = np.cumsum(sorted_sizes)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

def validate_cluster_quality(labels, n_samples, algorithm_name):
    """Validate cluster quality and return validation results."""
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Handle noise points for HDBSCAN
    if -1 in unique_labels:
        noise_count = counts[unique_labels == -1][0]
        valid_labels = labels[labels != -1]
        if len(valid_labels) == 0:
            return {'valid': False, 'reason': 'All points classified as noise'}
        unique_labels = np.unique(valid_labels)
        counts = np.array([np.sum(valid_labels == label) for label in unique_labels])
        n_valid_samples = len(valid_labels)
    else:
        noise_count = 0
        n_valid_samples = n_samples
    
    n_clusters = len(unique_labels)
    min_cluster_size = np.min(counts)
    gini_coeff = calculate_gini_coefficient(counts)
    noise_ratio = noise_count / n_samples
    
    # Validation rules
    valid = True
    reasons = []
    
    if n_clusters < 2:
        valid = False
        reasons.append(f"Too few clusters: {n_clusters}")
    
    if min_cluster_size < MIN_CLUSTER_SIZE:
        valid = False
        reasons.append(f"Cluster too small: {min_cluster_size} samples")
    
    if gini_coeff > MAX_GINI_COEFFICIENT:
        valid = False
        reasons.append(f"Too imbalanced: Gini={gini_coeff:.3f}")
    
    if algorithm_name == 'hdbscan' and noise_ratio > MAX_HDBSCAN_NOISE_FRACTION:
        valid = False
        reasons.append(f"Too much noise: {noise_ratio:.3f}")
    
    return {
        'valid': valid,
        'reason': '; '.join(reasons) if reasons else 'Valid',
        'n_clusters': n_clusters,
        'min_cluster_size': min_cluster_size,
        'gini_coefficient': gini_coeff,
        'noise_ratio': noise_ratio
    }

def run_robust_clustering(features, filenames, representation, output_dir):
    """Run robust clustering with multiple seeds and quality validation."""
    all_results = {}
    
    print(f"🔍 Clustering Analysis: {representation}")
    print(f"Dataset: {len(features)} samples, {features.shape[1]} features")
    print("-" * 40)
    
    # K-Means with multiple seeds
    for k in KMEANS_K_VALUES:
        print(f"  Running K-Means k={k} (N={N_SEEDS} seeds)...")
        
        silhouette_scores = []
        calinski_harabasz_scores = []
        davies_bouldin_scores = []
        valid_runs = 0
        
        for seed in range(N_SEEDS):
            kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED + seed, n_init=10)
            labels = kmeans.fit_predict(features)
            
            # Validate cluster quality
            validation = validate_cluster_quality(labels, len(features), 'kmeans')
            
            if validation['valid']:
                silhouette = silhouette_score(features, labels)
                if silhouette >= MIN_SILHOUETTE_THRESHOLD:
                    calinski_harabasz = calinski_harabasz_score(features, labels)
                    davies_bouldin = davies_bouldin_score(features, labels)
                    
                    silhouette_scores.append(silhouette)
                    calinski_harabasz_scores.append(calinski_harabasz)
                    davies_bouldin_scores.append(davies_bouldin)
                    valid_runs += 1
        
        if valid_runs > 0:
            # Calculate statistics for valid runs
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
            # Still log failed results with 0 quality score
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
            print(f"    ❌ No valid runs - logged as 0.000")
    
    # HDBSCAN with multiple seeds
    print(f"  Running HDBSCAN (N={N_SEEDS} seeds)...")
    
    silhouette_scores = []
    calinski_harabasz_scores = []
    davies_bouldin_scores = []
    valid_runs = 0
    
    for seed in range(N_SEEDS):
        np.random.seed(RANDOM_SEED + seed)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE)
        labels = clusterer.fit_predict(features)
        
        validation = validate_cluster_quality(labels, len(features), 'hdbscan')
        
        if validation['valid']:
            non_noise_indices = labels != -1
            if np.sum(non_noise_indices) > 1 and len(np.unique(labels[non_noise_indices])) > 1:
                silhouette = silhouette_score(features[non_noise_indices], labels[non_noise_indices])
                if silhouette >= MIN_SILHOUETTE_THRESHOLD:
                    calinski_harabasz = calinski_harabasz_score(features[non_noise_indices], labels[non_noise_indices])
                    davies_bouldin = davies_bouldin_score(features[non_noise_indices], labels[non_noise_indices])
                    
                    silhouette_scores.append(silhouette)
                    calinski_harabasz_scores.append(calinski_harabasz)
                    davies_bouldin_scores.append(davies_bouldin)
                    valid_runs += 1
    
    if valid_runs > 0:
        mean_silhouette = np.mean(silhouette_scores)
        std_silhouette = np.std(silhouette_scores)
        stability = 1.0 - min(std_silhouette, 1.0)
        quality_score = mean_silhouette * stability
        
        all_results['hdbscan'] = {
            'algorithm': 'hdbscan',
            'params': {'min_cluster_size': HDBSCAN_MIN_CLUSTER_SIZE},
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
        print(f"    ❌ No valid runs")
    
    return all_results

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

def main():
    parser = argparse.ArgumentParser(description='Run clustering for A0: NoSeg + NoPreprocess - New Data Experiment')
    parser.add_argument('--output_dir', type=str,
                       default='/Users/yunhwang/Desktop/Stethoscope_Project/Development/Unsupervised Learning with New Data/A0_NoSeg_NoPre/outputs',
                       help='Path to output directory')
    
    args = parser.parse_args()
    
    # Create clustering output directory
    clustering_dir = os.path.join(args.output_dir, 'clustering')
    os.makedirs(clustering_dir, exist_ok=True)
    
    print("🧪 NEW DATA EXPERIMENT - A0: Clustering Analysis")
    print("=" * 60)
    
    # Process each representation
    representations = ['raw_waveform_stats', 'logmel_mean', 'mfcc_mean']
    
    for representation in representations:
        try:
            # Load features
            features, filenames = load_features(args.output_dir, representation)
            
            # Run clustering
            results = run_robust_clustering(features, filenames, representation, args.output_dir)
            
            # Save results
            results_path = os.path.join(clustering_dir, f'clustering_{representation}_improved_metrics.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✅ Saved: {os.path.basename(results_path)}")
            
        except Exception as e:
            print(f"❌ Error processing {representation}: {e}")
            continue
        
        print()
    
    print("🎉 Clustering analysis completed!")
    print(f"Results saved in: {clustering_dir}")

if __name__ == "__main__":
    main()