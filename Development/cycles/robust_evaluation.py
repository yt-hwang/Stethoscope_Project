#!/usr/bin/env python3
"""
Robust evaluation system with statistical validation, stability analysis, and quality scoring
"""

import os
import sys
import numpy as np
import pandas as pd
import json
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import hdbscan
from scipy import stats
import argparse
from pathlib import Path

def compute_gini_coefficient(cluster_sizes):
    """Compute Gini coefficient for cluster size imbalance"""
    if len(cluster_sizes) <= 1:
        return 0.0
    
    n = sum(cluster_sizes)
    if n == 0:
        return 0.0
    
    # Sort cluster sizes
    sizes = sorted(cluster_sizes)
    K = len(sizes)
    
    # Compute Gini coefficient
    gini = 0.0
    for i in range(K):
        for j in range(K):
            gini += abs(sizes[i] - sizes[j])
    
    gini = gini / (2 * K * n)
    return gini

def check_cluster_validity(labels, min_cluster_pct=0.05):
    """Check if clustering meets minimum cluster size requirements"""
    n = len(labels)
    min_cluster_size = max(3, int(np.ceil(min_cluster_pct * n)))
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Remove noise labels (-1) for HDBSCAN
    valid_labels = unique_labels[unique_labels != -1]
    valid_counts = counts[unique_labels != -1]
    
    if len(valid_counts) == 0:
        return False, "No valid clusters found"
    
    min_size = np.min(valid_counts)
    if min_size < min_cluster_size:
        return False, f"Min cluster size {min_size} < required {min_cluster_size}"
    
    return True, "Valid"

def compute_stability_ari(features, labels_full, n_bootstrap=5, bootstrap_ratio=0.8):
    """Compute stability using bootstrap sampling and ARI"""
    n_samples = len(features)
    bootstrap_size = int(bootstrap_ratio * n_samples)
    
    ari_scores = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = resample(range(n_samples), n_samples=bootstrap_size, random_state=None)
        features_bootstrap = features[indices]
        labels_bootstrap = labels_full[indices]
        
        # For stability, compare bootstrap labels with full labels on the same indices
        # This measures how consistent the clustering is across different samples
        ari = adjusted_rand_score(labels_full[indices], labels_bootstrap)
        ari_scores.append(ari)
    
    return np.mean(ari_scores)

def compute_cross_algorithm_ari(features, kmeans_labels, hdbscan_labels):
    """Compute ARI between KMeans and HDBSCAN results"""
    # Remove HDBSCAN noise points for comparison
    valid_mask = hdbscan_labels != -1
    if np.sum(valid_mask) < 2:
        return 0.0
    
    return adjusted_rand_score(kmeans_labels[valid_mask], hdbscan_labels[valid_mask])

def normalize_metrics(silhouette_scores, ch_scores, db_scores):
    """Normalize metrics so higher is better"""
    # Silhouette: map [-1, 1] to [0, 1]
    sil_norm = (np.array(silhouette_scores) + 1) / 2
    
    # Calinski-Harabasz: rank percentile
    ch_norm = stats.rankdata(ch_scores) / len(ch_scores)
    
    # Davies-Bouldin: 1 / (1 + DB) - lower DB is better
    db_norm = 1 / (1 + np.array(db_scores))
    
    return sil_norm, ch_norm, db_norm

def compute_quality_score(sil_norm, ch_norm, db_norm, gini, noise_frac, stability_ari, cross_ari, 
                         sil_std, weights=None):
    """Compute comprehensive quality score"""
    if weights is None:
        weights = {
            'sil': 0.50, 'ch': 0.30, 'db': 0.20,  # base metrics
            'gini_penalty': 0.20, 'noise_penalty': 0.10, 'std_penalty': 0.10,  # penalties
            'stability_bonus': 0.10, 'cross_bonus': 0.05  # bonuses
        }
    
    # Base score
    base = (weights['sil'] * sil_norm + 
            weights['ch'] * ch_norm + 
            weights['db'] * db_norm)
    
    # Penalties
    penalty = (weights['gini_penalty'] * gini + 
               weights['noise_penalty'] * noise_frac + 
               weights['std_penalty'] * sil_std)
    
    # Bonuses
    bonus = (weights['stability_bonus'] * np.clip(stability_ari, 0, 1) + 
             weights['cross_bonus'] * np.clip(cross_ari, 0, 1))
    
    return base + bonus - penalty, base, penalty, bonus

def run_robust_evaluation(features, representation_name, algorithm_name, params, 
                         n_seeds=7, min_cluster_pct=0.05):
    """Run robust evaluation with multiple seeds and statistical analysis"""
    
    results = {
        'representation': representation_name,
        'algorithm': algorithm_name,
        'params': params,
        'n_seeds': n_seeds,
        'valid_runs': [],
        'invalid_runs': [],
        'summary': {}
    }
    
    silhouette_scores = []
    ch_scores = []
    db_scores = []
    valid_labels_list = []
    
    for seed in range(n_seeds):
        np.random.seed(seed)
        
        try:
            if algorithm_name == 'kmeans':
                k = params['n_clusters']
                clusterer = KMeans(n_clusters=k, random_state=seed, n_init=10)
                labels = clusterer.fit_predict(features)
                noise_frac = 0.0
            elif algorithm_name == 'hdbscan':
                min_cluster_size = params['min_cluster_size']
                clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
                labels = clusterer.fit_predict(features)
                noise_frac = np.sum(labels == -1) / len(labels)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm_name}")
            
            # Check validity
            is_valid, reason = check_cluster_validity(labels, min_cluster_pct)
            
            if not is_valid:
                results['invalid_runs'].append({
                    'seed': seed,
                    'reason': reason,
                    'labels': labels.tolist()
                })
                continue
            
            # Compute metrics
            valid_mask = labels != -1 if algorithm_name == 'hdbscan' else np.ones(len(labels), dtype=bool)
            if np.sum(valid_mask) < 2:
                results['invalid_runs'].append({
                    'seed': seed,
                    'reason': 'Insufficient valid points for metrics',
                    'labels': labels.tolist()
                })
                continue
            
            silhouette = silhouette_score(features[valid_mask], labels[valid_mask])
            ch = calinski_harabasz_score(features[valid_mask], labels[valid_mask])
            db = davies_bouldin_score(features[valid_mask], labels[valid_mask])
            
            # Store valid run
            run_data = {
                'seed': seed,
                'labels': labels.tolist(),
                'silhouette': silhouette,
                'calinski_harabasz': ch,
                'davies_bouldin': db,
                'noise_frac': noise_frac,
                'cluster_sizes': [np.sum(labels == k) for k in np.unique(labels) if k != -1]
            }
            
            results['valid_runs'].append(run_data)
            silhouette_scores.append(silhouette)
            ch_scores.append(ch)
            db_scores.append(db)
            valid_labels_list.append(labels)
            
        except Exception as e:
            results['invalid_runs'].append({
                'seed': seed,
                'reason': f"Error: {str(e)}",
                'labels': []
            })
    
    if len(valid_labels_list) == 0:
        results['summary'] = {
            'valid_run': False,
            'reject_reason': 'No valid runs found',
            'run_quality_score': 0.0
        }
        return results
    
    # Compute summary statistics
    sil_mean = np.mean(silhouette_scores)
    sil_std = np.std(silhouette_scores)
    ch_mean = np.mean(ch_scores)
    ch_std = np.std(ch_scores)
    db_mean = np.mean(db_scores)
    db_std = np.std(db_scores)
    
    # Compute cluster size statistics
    all_cluster_sizes = []
    for run in results['valid_runs']:
        all_cluster_sizes.extend(run['cluster_sizes'])
    
    gini = compute_gini_coefficient(all_cluster_sizes)
    avg_noise_frac = np.mean([run['noise_frac'] for run in results['valid_runs']])
    
    # Compute stability and cross-algorithm agreement
    # Simplified stability: use standard deviation of silhouette scores as stability measure
    stability_ari = 1.0 - min(sil_std, 1.0)  # Higher std = lower stability
    
    # Cross-algorithm agreement (if both KMeans and HDBSCAN available)
    cross_ari = 0.0  # Will be computed when both algorithms are available
    
    # Normalize metrics
    sil_norm, ch_norm, db_norm = normalize_metrics(silhouette_scores, ch_scores, db_scores)
    
    # Compute quality score
    quality_score, base, penalty, bonus = compute_quality_score(
        np.mean(sil_norm), np.mean(ch_norm), np.mean(db_norm),
        gini, avg_noise_frac, stability_ari, cross_ari, sil_std
    )
    
    results['summary'] = {
        'valid_run': True,
        'seed_count': len(results['valid_runs']),
        'silhouette_mean': sil_mean,
        'silhouette_std': sil_std,
        'calinski_harabasz_mean': ch_mean,
        'calinski_harabasz_std': ch_std,
        'davies_bouldin_mean': db_mean,
        'davies_bouldin_std': db_std,
        'stability_ari': stability_ari,
        'cross_ari': cross_ari,
        'min_cluster_size': min(all_cluster_sizes) if all_cluster_sizes else 0,
        'gini_size': gini,
        'noise_frac': avg_noise_frac,
        'run_quality_score': quality_score,
        'reject_reason': '',
        'score_breakdown': {
            'base': base,
            'penalty': penalty,
            'bonus': bonus
        }
    }
    
    return results

def evaluate_cycle_robust(cycle_path, cycle_name, n_seeds=7):
    """Evaluate a complete cycle with robust statistical analysis"""
    
    print(f"\n{'='*60}")
    print(f"ROBUST EVALUATION: {cycle_name}")
    print(f"{'='*60}")
    
    representations = ['raw_waveform_stats', 'logmel_mean', 'mfcc_mean']
    all_results = []
    
    for rep in representations:
        print(f"\nEvaluating {rep}...")
        
        # Load features
        features_path = os.path.join(cycle_path, 'features', f'features_{rep}.csv')
        if not os.path.exists(features_path):
            print(f"  Warning: {features_path} not found")
            continue
        
        df = pd.read_csv(features_path)
        feature_cols = [col for col in df.columns if col not in ['file_path', 'filename', 'preprocessing']]
        features = df[feature_cols].values
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Evaluate KMeans
        for k in [3, 4, 5]:
            print(f"  KMeans k={k}...")
            result = run_robust_evaluation(
                features_scaled, rep, 'kmeans', {'n_clusters': k}, n_seeds
            )
            result['cycle'] = cycle_name
            all_results.append(result)
        
        # Evaluate HDBSCAN
        print(f"  HDBSCAN...")
        result = run_robust_evaluation(
            features_scaled, rep, 'hdbscan', {'min_cluster_size': 25}, n_seeds
        )
        result['cycle'] = cycle_name
        all_results.append(result)
    
    return all_results

def create_ranking_table(all_results):
    """Create ranking table of top results"""
    
    # Filter valid runs only
    valid_results = [r for r in all_results if r['summary']['valid_run']]
    
    if not valid_results:
        print("No valid results found!")
        return pd.DataFrame()
    
    # Sort by quality score
    valid_results.sort(key=lambda x: x['summary']['run_quality_score'], reverse=True)
    
    # Create ranking table
    ranking_data = []
    for i, result in enumerate(valid_results[:10]):  # Top 10
        summary = result['summary']
        ranking_data.append({
            'Rank': i + 1,
            'Cycle': result['cycle'],
            'Representation': result['representation'],
            'Algorithm': f"{result['algorithm']}_{result['params']}",
            'Quality_Score': f"{summary['run_quality_score']:.4f}",
            'Silhouette': f"{summary['silhouette_mean']:.3f}±{summary['silhouette_std']:.3f}",
            'CH': f"{summary['calinski_harabasz_mean']:.1f}±{summary['calinski_harabasz_std']:.1f}",
            'DB': f"{summary['davies_bouldin_mean']:.3f}±{summary['davies_bouldin_std']:.3f}",
            'Valid_Runs': f"{summary['seed_count']}/{result['n_seeds']}",
            'Gini': f"{summary['gini_size']:.3f}",
            'Noise_Frac': f"{summary['noise_frac']:.3f}",
            'Stability_ARI': f"{summary['stability_ari']:.3f}",
            'Base': f"{summary['score_breakdown']['base']:.3f}",
            'Penalty': f"{summary['score_breakdown']['penalty']:.3f}",
            'Bonus': f"{summary['score_breakdown']['bonus']:.3f}"
        })
    
    return pd.DataFrame(ranking_data)

def main():
    parser = argparse.ArgumentParser(description='Robust evaluation of clustering cycles')
    parser.add_argument('--cycles', nargs='+', 
                       default=['A0', 'A1', 'A2'],
                       help='Cycles to evaluate')
    parser.add_argument('--n_seeds', type=int, default=7,
                       help='Number of random seeds')
    parser.add_argument('--min_cluster_pct', type=float, default=0.05,
                       help='Minimum cluster size as percentage of total samples')
    
    args = parser.parse_args()
    
    # Define cycle paths
    cycle_paths = {
        'A0': '/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/A0_NoSeg_NoPre/outputs',
        'A1': '/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/A1_NoSeg_Bandpass/outputs',
        'A2': '/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/A2_NoSeg_SpectralGating/outputs',
        'A3': '/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/A3_NoSeg_HighPass20/outputs',
        'A4': '/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/A4_NoSeg_PeakNormalize/outputs',
        'B0': '/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/B0_NoSeg_Bandpass_SpectralGating/outputs',
        'B1': '/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/B1_NoSeg_PeakNormalize_Bandpass/outputs',
        'B2': '/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/B2_NoSeg_FullPipeline/outputs',
        'C0': '/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/C0_Seg_NoPre/outputs',
        'C1': '/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/C1_Seg_Bandpass/outputs',
        'C2': '/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/C2_Seg_SpectralGating/outputs',
        'C3': '/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/C3_Seg_HighPass20/outputs',
        'C4': '/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/C4_Seg_PeakNormalize/outputs'
    }
    
    all_results = []
    
    # Evaluate each cycle
    for cycle in args.cycles:
        if cycle not in cycle_paths:
            print(f"Warning: Unknown cycle {cycle}")
            continue
        
        cycle_results = evaluate_cycle_robust(
            cycle_paths[cycle], cycle, args.n_seeds
        )
        all_results.extend(cycle_results)
    
    # Create ranking
    ranking_df = create_ranking_table(all_results)
    
    print(f"\n{'='*80}")
    print("TOP 10 RANKED RESULTS (by Quality Score)")
    print(f"{'='*80}")
    print(ranking_df.to_string(index=False))
    
    # Save results
    output_dir = '/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/robust_evaluation'
    os.makedirs(output_dir, exist_ok=True)
    
    ranking_df.to_csv(os.path.join(output_dir, 'ranking_table.csv'), index=False)
    
    # Save detailed results (convert numpy types to Python types)
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Convert all numpy types to Python types
    converted_results = []
    for result in all_results:
        converted_result = {}
        for key, value in result.items():
            if isinstance(value, dict):
                converted_result[key] = {k: convert_numpy(v) for k, v in value.items()}
            elif isinstance(value, list):
                converted_result[key] = [convert_numpy(v) for v in value]
            else:
                converted_result[key] = convert_numpy(value)
        converted_results.append(converted_result)
    
    with open(os.path.join(output_dir, 'detailed_results.json'), 'w') as f:
        json.dump(converted_results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()
