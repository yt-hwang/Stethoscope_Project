#!/usr/bin/env python3
"""
Create ALL visualizations for all cycles
Includes failed results (quality_score = 0) to show why they failed
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
from pathlib import Path

def create_umap_plot(features, labels, representation, algorithm, params_str, output_path, cycle_name, quality_score):
    """Create and save a 2D UMAP plot."""
    n_neighbors = min(15, len(features)-1)
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=0.1)
    embedding = reducer.fit_transform(features)

    plt.figure(figsize=(12, 10))
    
    # Plot clusters
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], 
                   hue=labels, palette='viridis', s=60, alpha=0.8)

    # Add quality score to title
    status = "VALID" if quality_score > 0 else "FAILED"
    title = f'UMAP: {representation} with {algorithm} ({params_str})\\n{cycle_name} - Quality: {quality_score:.3f} ({status})'
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_visuals_for_cycle(cycle_name, base_dir):
    """Create visualizations for a specific cycle."""
    cycle_dir = Path(base_dir) / cycle_name
    output_dir = cycle_dir / "outputs"
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    print(f"\\n📊 Creating visualizations for {cycle_name}...")
    
    representations = ['raw_waveform_stats', 'logmel_mean', 'mfcc_mean']
    total_visuals = 0
    
    for rep in representations:
        try:
            # Load features
            features_path = output_dir / "features" / f"features_{rep}.csv"
            df_features = pd.read_csv(features_path)
            feature_cols = [col for col in df_features.columns if col not in ['file_path', 'filename']]
            features = df_features[feature_cols].values
            
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Load results
            results_path = output_dir / "clustering" / f"clustering_{rep}_improved_metrics.json"
            with open(results_path, 'r') as f:
                results_data = json.load(f)
            
            print(f"  {rep}: {len(results_data)} visualizations")
            
            for algo_key, data in results_data.items():
                algorithm = data['algorithm']
                quality_score = data['quality_score']
                
                # Generate labels
                if algorithm == 'kmeans':
                    k = data['params']['n_clusters']
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(features_scaled)
                    params_str = f"k={k}"
                else:
                    continue
                
                # Create visualization
                output_filename = f"umap_{rep}_{algo_key}.png"
                output_path = vis_dir / output_filename
                
                create_umap_plot(features_scaled, labels, rep, algorithm, params_str, 
                               output_path, cycle_name, quality_score)
                
                status = "✅" if quality_score > 0 else "❌"
                print(f"    {status} {algo_key}: {quality_score:.3f}")
                total_visuals += 1
                
        except Exception as e:
            print(f"  ❌ Error with {rep}: {e}")
    
    print(f"  Total: {total_visuals} visualizations created")
    return total_visuals

def main():
    base_dir = "/Users/yunhwang/Desktop/Stethoscope_Project/Development/Unsupervised Learning with New Data"
    
    cycles = ['A0_NoSeg_NoPre', 'A1_NoSeg_Bandpass', 'A2_NoSeg_SpectralGating']
    
    print("🎨 CREATING ALL VISUALIZATIONS")
    print("=" * 50)
    
    total_visuals = 0
    for cycle in cycles:
        visuals = create_visuals_for_cycle(cycle, base_dir)
        total_visuals += visuals
    
    print(f"\\n🎉 ALL VISUALIZATIONS COMPLETED!")
    print(f"Total visualizations created: {total_visuals}")
    print(f"Expected: {len(cycles) * 3 * 3} (3 cycles × 3 reps × 3 k-values)")

if __name__ == "__main__":
    main()
