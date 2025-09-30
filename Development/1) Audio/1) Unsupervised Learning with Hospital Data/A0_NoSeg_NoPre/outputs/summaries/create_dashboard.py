#!/usr/bin/env python3
"""
Create a visual dashboard for Cycle A0 results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_results():
    """Load all clustering results"""
    results = {}
    representations = ['raw_waveform_stats', 'logmel_mean', 'mfcc_mean']
    
    for rep in representations:
        with open(f'clustering_{rep}_metrics.json', 'r') as f:
            results[rep] = json.load(f)
    
    return results

def create_metrics_comparison(results):
    """Create a comparison plot of metrics across representations and algorithms"""
    data = []
    
    for rep_name, rep_results in results.items():
        for algo_name, metrics in rep_results.items():
            if algo_name == 'hdbscan' and metrics['n_clusters'] == 0:
                continue  # Skip HDBSCAN with no clusters
                
            data.append({
                'Representation': rep_name.replace('_', ' ').title(),
                'Algorithm': algo_name.upper(),
                'Silhouette': metrics['silhouette'],
                'Calinski-Harabasz': metrics['calinski_harabasz'],
                'Davies-Bouldin': metrics['davies_bouldin'],
                'Clusters': metrics['n_clusters']
            })
    
    df = pd.DataFrame(data)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Cycle A0: Clustering Performance Comparison', fontsize=16, fontweight='bold')
    
    # Silhouette scores
    sns.barplot(data=df, x='Representation', y='Silhouette', hue='Algorithm', ax=axes[0,0])
    axes[0,0].set_title('Silhouette Score (Higher is Better)')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Calinski-Harabasz scores
    sns.barplot(data=df, x='Representation', y='Calinski-Harabasz', hue='Algorithm', ax=axes[0,1])
    axes[0,1].set_title('Calinski-Harabasz Index (Higher is Better)')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Davies-Bouldin scores
    sns.barplot(data=df, x='Representation', y='Davies-Bouldin', hue='Algorithm', ax=axes[1,0])
    axes[1,0].set_title('Davies-Bouldin Index (Lower is Better)')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Number of clusters
    sns.barplot(data=df, x='Representation', y='Clusters', hue='Algorithm', ax=axes[1,1])
    axes[1,1].set_title('Number of Clusters')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('cycle_a0_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return df

def create_best_results_table(results):
    """Create a table of best results for each representation"""
    best_results = []
    
    for rep_name, rep_results in results.items():
        # Find best KMeans result (skip HDBSCAN)
        kmeans_results = {k: v for k, v in rep_results.items() if k.startswith('k')}
        if kmeans_results:
            best_kmeans = max(kmeans_results.items(), key=lambda x: x[1]['silhouette'])
            best_results.append({
                'Representation': rep_name.replace('_', ' ').title(),
                'Best Algorithm': best_kmeans[0].upper(),
                'Silhouette': best_kmeans[1]['silhouette'],
                'Calinski-Harabasz': best_kmeans[1]['calinski_harabasz'],
                'Davies-Bouldin': best_kmeans[1]['davies_bouldin'],
                'Clusters': best_kmeans[1]['n_clusters']
            })
    
    return pd.DataFrame(best_results)

def main():
    print("Creating Cycle A0 Dashboard...")
    
    # Load results
    results = load_results()
    
    # Create comparison plot
    df = create_metrics_comparison(results)
    
    # Create best results table
    best_df = create_best_results_table(results)
    
    print("\nBest Results by Representation:")
    print("=" * 80)
    print(best_df.to_string(index=False, float_format='%.4f'))
    
    print(f"\nDashboard saved as 'cycle_a0_dashboard.png'")
    print(f"Total combinations tested: {len(df)}")

if __name__ == "__main__":
    main()
