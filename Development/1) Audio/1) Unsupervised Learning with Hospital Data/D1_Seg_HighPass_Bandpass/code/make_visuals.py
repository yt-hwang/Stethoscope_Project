#!/usr/bin/env python3
"""
Cycle D1: Seg + HighPass + Bandpass - UMAP Visualizations
"""

import pandas as pd
import numpy as np
import os
import json
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from sklearn.preprocessing import StandardScaler

def load_features_and_labels(output_dir, representation):
    """Load features and clustering labels for a given representation."""
    features_path = os.path.join(output_dir, 'features', f'features_{representation}.csv')
    metrics_path = os.path.join(output_dir, 'clustering', f'clustering_{representation}_metrics.json')

    # Load features
    df_features = pd.read_csv(features_path)
    feature_cols = [col for col in df_features.columns if col not in ['file_path', 'filename']]
    features = df_features[feature_cols].values
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Load clustering results
    with open(metrics_path, 'r') as f:
        metrics_data = json.load(f)
    
    return features_scaled, metrics_data, df_features['filename'].tolist()

def create_umap_plot(features, labels, filenames, representation, algorithm, params_str, output_path):
    """Create and save a 2D UMAP plot."""
    # Create UMAP embedding
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(features)

    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Handle HDBSCAN noise points (-1 label)
    unique_labels = np.unique(labels)
    if -1 in unique_labels:
        # Plot noise points in grey
        noise_mask = labels == -1
        plt.scatter(embedding[noise_mask, 0], embedding[noise_mask, 1], 
                    color='grey', s=20, alpha=0.6, label='Noise (-1)')
        
        # Plot actual clusters
        cluster_mask = labels != -1
        if np.sum(cluster_mask) > 0:
            sns.scatterplot(x=embedding[cluster_mask, 0], y=embedding[cluster_mask, 1], 
                           hue=labels[cluster_mask], palette='viridis', s=60, alpha=0.8, 
                           legend='full')
    else:
        # No noise points, plot all clusters
        sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], 
                       hue=labels, palette='viridis', s=60, alpha=0.8, legend='full')

    plt.title(f'UMAP: {representation} with {algorithm} ({params_str})\nCycle D1: Seg + HighPass + Bandpass', 
              fontsize=14, fontweight='bold')
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved UMAP plot: {os.path.basename(output_path)}")

def main():
    parser = argparse.ArgumentParser(description="Generate UMAP visualizations for clustering results.")
    parser.add_argument('--output_dir', type=str, 
                       default='/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/D1_Seg_HighPass_Bandpass/outputs',
                       help='Path to the output directory for this cycle.')
    
    args = parser.parse_args()

    # Create visualizations directory
    os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)

    representations = ['raw_waveform_stats', 'logmel_mean', 'mfcc_mean']

    for rep in representations:
        print(f"\nCreating visualizations for {rep}...")
        
        try:
            features, metrics_data, filenames = load_features_and_labels(args.output_dir, rep)
            print(f"  Loaded {len(features)} samples for visualization")
            
            for algo_key, data in metrics_data.items():
                algorithm = data['algorithm']
                labels = np.array(data['labels'])
                
                # Create parameter string for filename
                if algorithm == 'kmeans':
                    params_str = f"k{data['params']['n_clusters']}"
                elif algorithm == 'hdbscan':
                    params_str = f"min_cluster_size{data['params']['min_cluster_size']}"
                else:
                    params_str = "unknown"
                
                # Create output filename
                output_filename = f"umap_{rep}_{algorithm}_{params_str}.png"
                output_path = os.path.join(args.output_dir, 'visualizations', output_filename)
                
                # Create UMAP plot
                create_umap_plot(features, labels, filenames, rep, algorithm, params_str, output_path)
                
        except Exception as e:
            print(f"  Error creating visualizations for {rep}: {e}")
            continue

    print(f"\nVisualization generation completed!")

if __name__ == "__main__":
    main()
