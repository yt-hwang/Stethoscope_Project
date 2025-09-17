#!/usr/bin/env python3
"""
Cycle B2: NoSeg + PeakNormalize + Bandpass + SpectralGating - Visualization
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

def load_features_and_labels(features_path, metrics_path):
    """Load features and clustering labels"""
    # Load features
    df = pd.read_csv(features_path)
    feature_cols = [col for col in df.columns if col not in ['file_path', 'filename', 'preprocessing']]
    features = df[feature_cols].values
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Load clustering results
    with open(metrics_path, 'r') as f:
        clustering_results = json.load(f)
        
    return features_scaled, df, clustering_results

def create_umap_plot(features, labels, title, output_path):
    """Create UMAP visualization"""
    print(f"  Creating UMAP plot: {title}")
    
    # Fit UMAP
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(features)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Handle noise points (label = -1) for HDBSCAN
    if -1 in labels:
        # Plot noise points in gray
        noise_mask = labels == -1
        plt.scatter(embedding[noise_mask, 0], embedding[noise_mask, 1], 
                   c='gray', alpha=0.5, s=20, label='Noise')
        
        # Plot clusters
        cluster_mask = ~noise_mask
        unique_labels = np.unique(labels[cluster_mask])
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(embedding[mask, 0], embedding[mask, 1], 
                       c=[colors[i]], label=f'Cluster {label}', s=50, alpha=0.7)
    else:
        # No noise points, plot all clusters
        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(embedding[mask, 0], embedding[mask, 1], 
                       c=[colors[i]], label=f'Cluster {label}', s=50, alpha=0.7)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('UMAP 1', fontsize=12)
    plt.ylabel('UMAP 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Create visualizations for Cycle B2')
    parser.add_argument('--output_dir', 
                       default='/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/B2_NoSeg_FullPipeline/outputs',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)
    
    representations = ['raw_waveform_stats', 'logmel_mean', 'mfcc_mean']
    
    for rep in representations:
        print(f"\nProcessing {rep}...")
        
        # Load features and clustering results
        features_path = os.path.join(args.output_dir, 'features', f'features_{rep}.csv')
        metrics_path = os.path.join(args.output_dir, 'clustering', f'clustering_{rep}_metrics.json')
        
        if not os.path.exists(features_path) or not os.path.exists(metrics_path):
            print(f"  Warning: Missing files for {rep}")
            continue
        
        features, df, clustering_results = load_features_and_labels(features_path, metrics_path)
        
        # Create UMAP plots for each algorithm
        for algo_name, metrics in clustering_results.items():
            # Skip if no valid clusters
            if metrics['n_clusters'] == 0:
                print(f"  Skipping {algo_name}: No valid clusters")
                continue
            
            # For this visualization, we'll use the metrics to create a mock clustering
            # In practice, you'd want to save the actual labels from the clustering step
            n_clusters = metrics['n_clusters']
            
            # Create mock labels for visualization (in practice, save actual labels)
            if 'kmeans' in algo_name:
                from sklearn.cluster import KMeans
                k = metrics['params']['n_clusters']
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(features)
            else:  # HDBSCAN
                import hdbscan
                clusterer = hdbscan.HDBSCAN(min_cluster_size=metrics['params']['min_cluster_size'])
                labels = clusterer.fit_predict(features)
            
            # Create UMAP plot
            title = f"Cycle B2: {rep} - {algo_name.upper()}\nFull Pipeline: PeakNormalize + Bandpass + SpectralGating"
            output_path = os.path.join(args.output_dir, 'visualizations', f'umap_{rep}_{algo_name}.png')
            create_umap_plot(features, labels, title, output_path)
    
    print(f"\nVisualization completed for Cycle B2: NoSeg + PeakNormalize + Bandpass + SpectralGating")

if __name__ == "__main__":
    main()
