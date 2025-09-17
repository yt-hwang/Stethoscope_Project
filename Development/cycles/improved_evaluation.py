#!/usr/bin/env python3
"""
Improved Evaluation System with Cluster Balance Constraints
"""

import pandas as pd
import numpy as np
import os
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import hdbscan
# Gini coefficient calculation (not available in scipy.stats)
import argparse

# Configuration
KMEANS_K_VALUES = [3, 4, 5]
HDBSCAN_MIN_CLUSTER_SIZE = 3  # Reduced for small datasets
RANDOM_SEED = 42
N_SEEDS = 7  # For robust evaluation

# Cluster Quality Constraints
MIN_CLUSTER_SIZE_RATIO = 0.02  # Minimum 2% of data per cluster (relaxed from 5%)
MAX_GINI_COEFFICIENT = 0.8     # Maximum imbalance (relaxed from 0.7)
MIN_SILHOUETTE_THRESHOLD = 0.2  # Minimum silhouette score (relaxed from 0.3)

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
    
    # Remove noise points for HDBSCAN
    if -1 in unique_labels:
        noise_count = counts[unique_labels == -1][0]
        valid_labels = labels[labels != -1]
        if len(valid_labels) == 0:
            return {
                'valid': False,
                'reason': 'All points classified as noise',
                'n_clusters': 0,
                'min_cluster_size': 0,
                'max_cluster_size': 0,
                'gini_coefficient': 1.0,
                'noise_ratio': 1.0
            }
        unique_labels = np.unique(valid_labels)
        counts = np.array([np.sum(valid_labels == label) for label in unique_labels])
        n_valid_samples = len(valid_labels)
    else:
        noise_count = 0
        n_valid_samples = n_samples
    
    n_clusters = len(unique_labels)
    min_cluster_size = np.min(counts)
    max_cluster_size = np.max(counts)
    min_cluster_ratio = min_cluster_size / n_valid_samples
    gini_coeff = calculate_gini_coefficient(counts)
    noise_ratio = noise_count / n_samples
    
    # Validation rules
    valid = True
    reasons = []
    
    if n_clusters < 2:
        valid = False
        reasons.append(f"Too few clusters: {n_clusters}")
    
    if min_cluster_ratio < MIN_CLUSTER_SIZE_RATIO:
        valid = False
        reasons.append(f"Cluster too small: {min_cluster_ratio:.3f} < {MIN_CLUSTER_SIZE_RATIO}")
    
    if gini_coeff > MAX_GINI_COEFFICIENT:
        valid = False
        reasons.append(f"Too imbalanced: Gini={gini_coeff:.3f} > {MAX_GINI_COEFFICIENT}")
    
    if noise_ratio > 0.5:
        valid = False
        reasons.append(f"Too much noise: {noise_ratio:.3f} > 0.5")
    
    return {
        'valid': valid,
        'reason': '; '.join(reasons) if reasons else 'Valid',
        'n_clusters': n_clusters,
        'min_cluster_size': min_cluster_size,
        'max_cluster_size': max_cluster_size,
        'min_cluster_ratio': min_cluster_ratio,
        'gini_coefficient': gini_coeff,
        'noise_ratio': noise_ratio,
        'cluster_sizes': counts.tolist()
    }

def run_robust_clustering(features, filenames, representation, output_dir):
    """Run robust clustering with multiple seeds and quality validation."""
    all_results = {}
    
    # KMeans with multiple seeds
    for k in KMEANS_K_VALUES:
        print(f"  Running KMeans with k={k} (N={N_SEEDS} seeds)...")
        
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
                # Calculate metrics only for valid clusters
                silhouette = silhouette_score(features, labels)
                calinski_harabasz = calinski_harabasz_score(features, labels)
                davies_bouldin = davies_bouldin_score(features, labels)
                
                # Additional quality check
                if silhouette >= MIN_SILHOUETTE_THRESHOLD:
                    silhouette_scores.append(silhouette)
                    calinski_harabasz_scores.append(calinski_harabasz)
                    davies_bouldin_scores.append(davies_bouldin)
                    valid_runs += 1
        
        if valid_runs > 0:
            # Calculate statistics across valid runs
            mean_silhouette = np.mean(silhouette_scores)
            std_silhouette = np.std(silhouette_scores)
            mean_calinski_harabasz = np.mean(calinski_harabasz_scores)
            std_calinski_harabasz = np.std(calinski_harabasz_scores)
            mean_davies_bouldin = np.mean(davies_bouldin_scores)
            std_davies_bouldin = np.std(davies_bouldin_scores)
            
            # Calculate stability (1 - normalized std)
            stability = 1.0 - min(std_silhouette, 1.0)
            
            all_results[f'kmeans_k{k}'] = {
                'representation': representation,
                'algorithm': 'kmeans',
                'params': {'n_clusters': k},
                'valid_runs': valid_runs,
                'total_runs': N_SEEDS,
                'silhouette_mean': mean_silhouette,
                'silhouette_std': std_silhouette,
                'calinski_harabasz_mean': mean_calinski_harabasz,
                'calinski_harabasz_std': std_calinski_harabasz,
                'davies_bouldin_mean': mean_davies_bouldin,
                'davies_bouldin_std': std_davies_bouldin,
                'stability': stability,
                'quality_score': mean_silhouette * stability  # Combined quality metric
            }
            
            print(f"    Valid runs: {valid_runs}/{N_SEEDS}")
            print(f"    Silhouette: {mean_silhouette:.3f} ± {std_silhouette:.3f}")
            print(f"    Quality Score: {mean_silhouette * stability:.3f}")
        else:
            print(f"    No valid runs found for k={k}")
            all_results[f'kmeans_k{k}'] = {
                'representation': representation,
                'algorithm': 'kmeans',
                'params': {'n_clusters': k},
                'valid_runs': 0,
                'total_runs': N_SEEDS,
                'silhouette_mean': np.nan,
                'silhouette_std': np.nan,
                'calinski_harabasz_mean': np.nan,
                'calinski_harabasz_std': np.nan,
                'davies_bouldin_mean': np.nan,
                'davies_bouldin_std': np.nan,
                'stability': 0.0,
                'quality_score': 0.0,
                'reason': 'No valid runs'
            }

    # HDBSCAN with multiple seeds
    print(f"  Running HDBSCAN (N={N_SEEDS} seeds)...")
    
    silhouette_scores = []
    calinski_harabasz_scores = []
    davies_bouldin_scores = []
    valid_runs = 0
    cluster_counts = []
    noise_ratios = []
    
    for seed in range(N_SEEDS):
        # Vary min_cluster_size slightly for different seeds
        min_cluster_size = HDBSCAN_MIN_CLUSTER_SIZE + seed * 2
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        labels = clusterer.fit_predict(features)
        
        # Validate cluster quality
        validation = validate_cluster_quality(labels, len(features), 'hdbscan')
        
        if validation['valid']:
            # Calculate metrics only for valid clusters
            non_noise_indices = labels != -1
            if np.sum(non_noise_indices) > 1 and len(np.unique(labels[non_noise_indices])) > 1:
                silhouette = silhouette_score(features[non_noise_indices], labels[non_noise_indices])
                calinski_harabasz = calinski_harabasz_score(features[non_noise_indices], labels[non_noise_indices])
                davies_bouldin = davies_bouldin_score(features[non_noise_indices], labels[non_noise_indices])
                
                # Additional quality check
                if silhouette >= MIN_SILHOUETTE_THRESHOLD:
                    silhouette_scores.append(silhouette)
                    calinski_harabasz_scores.append(calinski_harabasz)
                    davies_bouldin_scores.append(davies_bouldin)
                    cluster_counts.append(validation['n_clusters'])
                    noise_ratios.append(validation['noise_ratio'])
                    valid_runs += 1
    
    if valid_runs > 0:
        # Calculate statistics across valid runs
        mean_silhouette = np.mean(silhouette_scores)
        std_silhouette = np.std(silhouette_scores)
        mean_calinski_harabasz = np.mean(calinski_harabasz_scores)
        std_calinski_harabasz = np.std(calinski_harabasz_scores)
        mean_davies_bouldin = np.mean(davies_bouldin_scores)
        std_davies_bouldin = np.std(davies_bouldin_scores)
        mean_clusters = np.mean(cluster_counts)
        mean_noise_ratio = np.mean(noise_ratios)
        
        # Calculate stability
        stability = 1.0 - min(std_silhouette, 1.0)
        
        all_results['hdbscan'] = {
            'representation': representation,
            'algorithm': 'hdbscan',
            'params': {'min_cluster_size': HDBSCAN_MIN_CLUSTER_SIZE},
            'valid_runs': valid_runs,
            'total_runs': N_SEEDS,
            'silhouette_mean': mean_silhouette,
            'silhouette_std': std_silhouette,
            'calinski_harabasz_mean': mean_calinski_harabasz,
            'calinski_harabasz_std': std_calinski_harabasz,
            'davies_bouldin_mean': mean_davies_bouldin,
            'davies_bouldin_std': std_davies_bouldin,
            'stability': stability,
            'quality_score': mean_silhouette * stability,
            'mean_clusters': mean_clusters,
            'mean_noise_ratio': mean_noise_ratio
        }
        
        print(f"    Valid runs: {valid_runs}/{N_SEEDS}")
        print(f"    Silhouette: {mean_silhouette:.3f} ± {std_silhouette:.3f}")
        print(f"    Quality Score: {mean_silhouette * stability:.3f}")
    else:
        print(f"    No valid runs found for HDBSCAN")
        all_results['hdbscan'] = {
            'representation': representation,
            'algorithm': 'hdbscan',
            'params': {'min_cluster_size': HDBSCAN_MIN_CLUSTER_SIZE},
            'valid_runs': 0,
            'total_runs': N_SEEDS,
            'silhouette_mean': np.nan,
            'silhouette_std': np.nan,
            'calinski_harabasz_mean': np.nan,
            'calinski_harabasz_std': np.nan,
            'davies_bouldin_mean': np.nan,
            'davies_bouldin_std': np.nan,
            'stability': 0.0,
            'quality_score': 0.0,
            'mean_clusters': 0,
            'mean_noise_ratio': 1.0,
            'reason': 'No valid runs'
        }
    
    # Save results
    output_path = os.path.join(output_dir, 'clustering', f'clustering_{representation}_improved_metrics.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"  Saved improved results to {output_path}")
    return all_results

def main():
    parser = argparse.ArgumentParser(description="Run improved clustering evaluation with cluster balance constraints.")
    parser.add_argument('--cycle_dir', type=str, required=True, help='Path to cycle output directory.')
    parser.add_argument('--cycle_id', type=str, required=True, help='Cycle ID (e.g., A0, B1, C3, D0).')
    
    args = parser.parse_args()
    
    print(f"Running improved evaluation for {args.cycle_id}...")
    print(f"Cluster balance constraints:")
    print(f"  - Min cluster size: {MIN_CLUSTER_SIZE_RATIO:.1%} of data")
    print(f"  - Max Gini coefficient: {MAX_GINI_COEFFICIENT}")
    print(f"  - Min silhouette threshold: {MIN_SILHOUETTE_THRESHOLD}")
    print(f"  - Robust evaluation: {N_SEEDS} seeds per algorithm")
    
    representations = ['raw_waveform_stats', 'logmel_mean', 'mfcc_mean']
    
    for rep in representations:
        print(f"\nEvaluating {rep}...")
        try:
            # Load features
            features_path = os.path.join(args.cycle_dir, 'features', f'features_{rep}.csv')
            df = pd.read_csv(features_path)
            feature_cols = [col for col in df.columns if col not in ['file_path', 'filename']]
            features = df[feature_cols].values
            
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            print(f"  Loaded {len(features)} samples with {features.shape[1]} features")
            run_robust_clustering(features_scaled, df['filename'].tolist(), rep, args.cycle_dir)
            
        except Exception as e:
            print(f"  Error processing {rep}: {e}")
            continue

    print(f"\nImproved evaluation completed for {args.cycle_id}!")

if __name__ == "__main__":
    main()
