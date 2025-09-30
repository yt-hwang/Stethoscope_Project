#!/usr/bin/env python3
"""
Generate UMAP visualizations for clustering results for D0_new: Seg + PeakNormalize + HighPass
"""

import pandas as pd
import numpy as np
import os
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.preprocessing import StandardScaler

# Configuration
RANDOM_SEED = 42
plt.style.use('default')
sns.set_palette("husl")

def create_umap_visualization(features_df, representation_name, clustering_results, output_dir):
    """Create UMAP visualization for a specific representation and clustering result."""
    
    # Prepare features
    feature_cols = [col for col in features_df.columns if col not in ['file_path', 'filename']]
    X = features_df[feature_cols].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create UMAP embedding
    reducer = umap.UMAP(n_components=2, random_state=RANDOM_SEED, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(X_scaled)
    
    # Create visualizations for each clustering result
    for algo_key, result in clustering_results.items():
        labels = np.array(result['labels'])
        algorithm = result['algorithm']
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Plot points
        if algorithm == 'hdbscan' and -1 in labels:
            # Separate noise points
            noise_mask = labels == -1
            cluster_mask = ~noise_mask
            
            if np.any(cluster_mask):
                scatter = ax.scatter(embedding[cluster_mask, 0], embedding[cluster_mask, 1], 
                                   c=labels[cluster_mask], cmap='tab10', alpha=0.7, s=50)
            
            if np.any(noise_mask):
                ax.scatter(embedding[noise_mask, 0], embedding[noise_mask, 1], 
                          c='black', marker='x', alpha=0.5, s=20, label='Noise')
        else:
            scatter = ax.scatter(embedding[:, 0], embedding[:, 1], 
                               c=labels, cmap='tab10', alpha=0.7, s=50)
        
        # Customize plot
        if algorithm == 'kmeans':
            title = f"D0_new: {representation_name} - K-Means (k={result['k']})"
            filename = f"umap_{representation_name}_kmeans_k{result['k']}.png"
        else:  # hdbscan
            title = f"D0_new: {representation_name} - HDBSCAN (min_cluster_size={result['min_cluster_size']})"
            filename = f"umap_{representation_name}_hdbscan_min_cluster_size{result['min_cluster_size']}.png"
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)
        
        # Add metrics to plot
        silhouette = result['silhouette']
        calinski_harabasz = result['calinski_harabasz']
        davies_bouldin = result['davies_bouldin']
        n_clusters = result['num_clusters']
        n_noise = result['num_noise_points']
        
        metrics_text = f"Silhouette: {silhouette:.3f}\nCH: {calinski_harabasz:.1f}\nDB: {davies_bouldin:.3f}\nClusters: {n_clusters}"
        if n_noise > 0:
            metrics_text += f"\nNoise: {n_noise}"
        
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10, fontfamily='monospace')
        
        # Add colorbar
        if 'scatter' in locals():
            plt.colorbar(scatter, ax=ax, label='Cluster')
        
        # Add legend for noise points if present
        if algorithm == 'hdbscan' and -1 in labels and np.any(noise_mask):
            ax.legend()
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
        output_path = os.path.join(output_dir, 'visualizations', filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved visualization: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Generate UMAP visualizations for clustering results.")
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the output directory.')
    
    args = parser.parse_args()
    
    print("="*60)
    print("D0_new: Seg + PeakNormalize + HighPass - Visualizations")
    print("="*60)
    
    # Process each representation
    representations = ['raw_waveform_stats', 'logmel_mean', 'mfcc_mean']
    
    for rep in representations:
        print(f"\nProcessing {rep}...")
        
        # Load features
        features_file = os.path.join(args.output_dir, 'features', f'features_{rep}.csv')
        if not os.path.exists(features_file):
            print(f"  Warning: Features file not found: {features_file}")
            continue
        
        features_df = pd.read_csv(features_file)
        print(f"  Loaded {len(features_df)} samples")
        
        # Load clustering results
        clustering_file = os.path.join(args.output_dir, 'clustering', f'clustering_{rep}_metrics.json')
        if not os.path.exists(clustering_file):
            print(f"  Warning: Clustering results not found: {clustering_file}")
            continue
        
        with open(clustering_file, 'r') as f:
            clustering_results = json.load(f)
        
        # Create visualizations
        create_umap_visualization(features_df, rep, clustering_results, args.output_dir)
    
    print("\n" + "="*60)
    print("Visualization completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
