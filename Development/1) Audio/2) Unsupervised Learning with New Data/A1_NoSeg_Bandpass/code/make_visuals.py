#!/usr/bin/env python3
"""
A1: NoSeg + Bandpass - UMAP Visualizations
NEW DATA EXPERIMENT - Using RAW sound_ML test sound list
K-Means only (HDBSCAN excluded)
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
from sklearn.cluster import KMeans

def load_features_and_results(output_dir, representation):
    """Load features and clustering results for visualization."""
    features_path = os.path.join(output_dir, 'features', f'features_{representation}.csv')
    results_path = os.path.join(output_dir, 'clustering', f'clustering_{representation}_improved_metrics.json')

    # Load features
    df_features = pd.read_csv(features_path)
    feature_cols = [col for col in df_features.columns if col not in ['file_path', 'filename']]
    features = df_features[feature_cols].values
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Load clustering results
    with open(results_path, 'r') as f:
        results_data = json.load(f)
    
    return features_scaled, results_data, df_features['filename'].tolist()

def create_umap_plot(features, labels, filenames, representation, algorithm, params_str, output_path):
    """Create and save a 2D UMAP plot."""
    # Create UMAP embedding
    n_neighbors = min(15, len(features)-1)  # Adjust for small dataset
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=0.1)
    embedding = reducer.fit_transform(features)

    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Plot clusters (K-Means only, no noise points)
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], 
                   hue=labels, palette='viridis', s=60, alpha=0.8)

    plt.title(f'UMAP: {representation} with {algorithm} ({params_str})\nA1: NoSeg + Bandpass - NEW DATA EXPERIMENT', 
              fontsize=14, fontweight='bold')
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    ✅ Saved: {os.path.basename(output_path)}")

def main():
    parser = argparse.ArgumentParser(description="Generate UMAP visualizations for A1 clustering results - New Data Experiment")
    parser.add_argument('--output_dir', type=str, 
                       default='/Users/yunhwang/Desktop/Stethoscope_Project/Development/Unsupervised Learning with New Data/A1_NoSeg_Bandpass/outputs',
                       help='Path to the output directory')
    
    args = parser.parse_args()

    # Create visualizations directory
    vis_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    print("🧪 NEW DATA EXPERIMENT - A1: Creating Visualizations")
    print("=" * 60)

    representations = ['raw_waveform_stats', 'logmel_mean', 'mfcc_mean']

    for rep in representations:
        print(f"\n📊 Creating visualizations for {rep}...")
        
        try:
            features, results_data, filenames = load_features_and_results(args.output_dir, rep)
            print(f"  Loaded {len(features)} samples for visualization")
            
            for algo_key, data in results_data.items():
                if data.get('valid_runs', 0) == 0:
                    print(f"  ⚠️  Skipping {algo_key} - no valid runs")
                    continue
                    
                algorithm = data['algorithm']
                
                # Generate labels using K-Means only
                if algorithm == 'kmeans':
                    k = data['params']['n_clusters']
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(features)
                    params_str = f"k={k}"
                else:
                    continue  # Skip non-kmeans algorithms
                
                # Create UMAP visualization
                output_filename = f"umap_{rep}_{algo_key}.png"
                output_path = os.path.join(vis_dir, output_filename)
                
                create_umap_plot(features, labels, filenames, rep, algorithm, 
                               params_str, output_path)
                
        except Exception as e:
            print(f"  ❌ Error creating visualizations for {rep}: {e}")
            continue

    print(f"\n🎉 Visualizations completed!")
    print(f"Saved in: {vis_dir}")

if __name__ == "__main__":
    main()
