#!/usr/bin/env python3
"""
Visualization for Cycle A0: NoSeg + NoPreprocess baseline
Creates UMAP visualizations for all representations and clustering algorithms
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import umap
import argparse
from pathlib import Path

# Set random seeds for reproducibility
np.random.seed(42)

def load_features_and_labels(features_path, metrics_path):
    """Load features and clustering labels"""
    # Load features
    df = pd.read_csv(features_path)
    feature_cols = [col for col in df.columns if col not in ['file_path', 'filename']]
    features = df[feature_cols].values
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Load clustering results
    with open(metrics_path, 'r') as f:
        clustering_results = json.load(f)
    
    return features_scaled, df, clustering_results

def create_umap_visualization(features, labels, representation_name, algorithm_name, output_dir):
    """Create UMAP visualization for a specific representation and algorithm"""
    print(f"Creating UMAP for {representation_name} - {algorithm_name}")
    
    # Create UMAP embedding
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(features)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Handle noise points (label -1) for HDBSCAN
    if algorithm_name == 'hdbscan' and -1 in labels:
        # Plot noise points in gray
        noise_mask = np.array(labels) == -1
        if np.any(noise_mask):
            plt.scatter(embedding[noise_mask, 0], embedding[noise_mask, 1], 
                       c='gray', alpha=0.6, s=20, label='Noise')
        
        # Plot clusters
        cluster_mask = ~noise_mask
        if np.any(cluster_mask):
            unique_labels = np.unique(labels[cluster_mask])
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = np.array(labels) == label
                plt.scatter(embedding[mask, 0], embedding[mask, 1], 
                           c=[colors[i]], alpha=0.7, s=50, label=f'Cluster {label}')
    else:
        # Regular clustering (KMeans)
        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            plt.scatter(embedding[mask, 0], embedding[mask, 1], 
                       c=[colors[i]], alpha=0.7, s=50, label=f'Cluster {label}')
    
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title(f'UMAP Visualization: {representation_name} - {algorithm_name}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(output_dir, f'umap_{representation_name}_{algorithm_name}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved UMAP plot to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Create visualizations for Cycle A0')
    parser.add_argument('--output_dir', type=str,
                       default='/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/A0_NoSeg_NoPre/outputs',
                       help='Path to output directory')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define representations
    representations = {
        'raw_waveform_stats': 'features_raw_waveform_stats.csv',
        'logmel_mean': 'features_logmel_mean.csv',
        'mfcc_mean': 'features_mfcc_mean.csv'
    }
    
    # Create visualizations for each representation
    for rep_name, csv_file in representations.items():
        features_path = os.path.join(args.output_dir, csv_file)
        metrics_path = os.path.join(args.output_dir, f'clustering_{rep_name}_metrics.json')
        
        if not os.path.exists(features_path) or not os.path.exists(metrics_path):
            print(f"Warning: Missing files for {rep_name}. Skipping.")
            continue
        
        # Load features and clustering results
        features, df, clustering_results = load_features_and_labels(features_path, metrics_path)
        
        # Create UMAP for each algorithm
        for algo_name, metrics in clustering_results.items():
            labels = metrics['labels']
            create_umap_visualization(features, labels, rep_name, algo_name, args.output_dir)
    
    print(f"\nVisualization completed! Plots saved in {args.output_dir}")

if __name__ == "__main__":
    main()
