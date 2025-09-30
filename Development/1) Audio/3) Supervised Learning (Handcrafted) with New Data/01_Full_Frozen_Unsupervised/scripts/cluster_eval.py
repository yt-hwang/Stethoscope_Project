#!/usr/bin/env python3
"""
Clustering Evaluation Script - UMAP visualization and KMeans clustering
Runs unsupervised clustering diagnostics on extracted features
"""

import argparse
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import umap
import warnings
warnings.filterwarnings('ignore')

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from excel_logger import ExcelLogger

def load_features_from_parquet(parquet_path):
    """Load features and metadata from parquet file."""
    df = pd.read_parquet(parquet_path)
    
    # Extract embedding columns
    emb_cols = [col for col in df.columns if col.startswith('emb_')]
    
    if not emb_cols:
        raise ValueError("No embedding columns found in parquet file")
    
    features = df[emb_cols].values
    metadata = df[[col for col in df.columns if not col.startswith('emb_')]]
    
    return features, metadata

def create_umap_visualization(features, labels, split_name, save_path, title_info=""):
    """Create UMAP 2D visualization."""
    print(f"Creating UMAP for {split_name}...")
    
    # Configure UMAP parameters based on dataset size
    n_neighbors = min(15, len(features) - 1)
    min_dist = 0.1
    
    # Fit UMAP
    reducer = umap.UMAP(
        n_components=2, 
        random_state=42, 
        n_neighbors=n_neighbors,
        min_dist=min_dist
    )
    
    embedding = reducer.fit_transform(features)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    if labels is not None:
        # Colored by cluster labels
        scatter = plt.scatter(
            embedding[:, 0], embedding[:, 1], 
            c=labels, cmap='viridis', alpha=0.7, s=50
        )
        plt.colorbar(scatter, label='Cluster')
        title = f'UMAP Visualization - {split_name} (Colored by K-Means Clusters)'
    else:
        # Single color
        plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7, s=50)
        title = f'UMAP Visualization - {split_name}'
    
    if title_info:
        title += f' - {title_info}'
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add dataset info
    plt.figtext(0.02, 0.02, f'Samples: {len(features)}, Features: {features.shape[1]}', 
                fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"UMAP saved: {save_path}")
    return str(save_path)

def perform_kmeans_clustering(features, metadata):
    """Perform K-Means clustering with optimal k selection."""
    print("Performing K-Means clustering...")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Try different k values
    k_range = range(2, min(11, len(features) // 2))  # k from 2 to 10, but not more than half the data
    
    silhouette_scores = []
    k_values = []
    
    for k in k_range:
        if k >= len(features):
            continue
            
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_scaled)
        
        # Check if we have valid clusters
        if len(np.unique(labels)) < 2:
            continue
            
        sil_score = silhouette_score(features_scaled, labels)
        silhouette_scores.append(sil_score)
        k_values.append(k)
        
        print(f"  k={k}: silhouette={sil_score:.3f}")
    
    if not silhouette_scores:
        return None, None, "No valid clustering found"
    
    # Choose best k
    best_idx = np.argmax(silhouette_scores)
    best_k = k_values[best_idx]
    best_silhouette = silhouette_scores[best_idx]
    
    print(f"Best k: {best_k} (silhouette: {best_silhouette:.3f})")
    
    # Perform final clustering with best k
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    final_labels = kmeans.fit_predict(features_scaled)
    
    # Calculate additional metrics if labels exist
    metrics = {
        'chosen_k': int(best_k),
        'silhouette': float(best_silhouette),
        'n_samples': len(features),
        'n_features': features.shape[1]
    }
    
    # Try to calculate NMI and ARI if true labels exist
    if 'label' in metadata.columns:
        true_labels = metadata['label'].values
        unique_true_labels = pd.Series(true_labels).nunique()
        
        if unique_true_labels > 1:
            try:
                # Encode string labels to integers
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                true_labels_encoded = le.fit_transform(true_labels)
                
                nmi = normalized_mutual_info_score(true_labels_encoded, final_labels)
                ari = adjusted_rand_score(true_labels_encoded, final_labels)
                
                metrics['nmi'] = float(nmi)
                metrics['ari'] = float(ari)
                metrics['n_true_labels'] = int(unique_true_labels)
                
                print(f"  NMI: {nmi:.3f}, ARI: {ari:.3f}")
                
            except Exception as e:
                print(f"  Warning: Could not compute NMI/ARI: {e}")
    
    return final_labels, metrics, None

def main():
    parser = argparse.ArgumentParser(description="Run clustering evaluation on extracted features")
    parser.add_argument("--run_id", required=True, help="Run ID")
    parser.add_argument("--split", choices=["train", "val", "test", "unsup"], default="unsup",
                       help="Which split to cluster")
    
    args = parser.parse_args()
    
    print(f"Clustering evaluation for run: {args.run_id}, split: {args.split}")
    
    # Load features
    print("Step 1: Loading features...")
    run_dir = Path("results/experiments") / args.run_id
    features_dir = run_dir / "01_features"
    
    parquet_path = features_dir / f"{args.split}.parquet"
    
    if not parquet_path.exists():
        print(f"Error: Features file not found: {parquet_path}")
        print("Available files:")
        for f in features_dir.glob("*.parquet"):
            print(f"  {f.name}")
        sys.exit(1)
    
    features, metadata = load_features_from_parquet(parquet_path)
    print(f"Loaded {len(features)} samples with {features.shape[1]} features")
    
    # Create cluster output directory
    cluster_dir = run_dir / "04_cluster"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    
    print("Step 2: Performing K-Means clustering...")
    cluster_labels, cluster_metrics, cluster_error = perform_kmeans_clustering(features, metadata)
    
    if cluster_error:
        print(f"Clustering failed: {cluster_error}")
        
        # Save error report
        error_report = {
            'run_id': args.run_id,
            'split': args.split,
            'error': cluster_error,
            'timestamp': datetime.now().isoformat()
        }
        
        error_path = cluster_dir / "clustering_error.json"
        with open(error_path, 'w') as f:
            json.dump(error_report, f, indent=2)
        
        # Log to Excel
        excel_logger = ExcelLogger()
        excel_data = {
            'run_id': args.run_id,
            'split': args.split,
            'chosen_k': 0,
            'silhouette': 0,
            'nmi': 0,
            'ari': 0,
            'notes': cluster_error
        }
        excel_logger.append_row('cluster', excel_data)
        
        print("STOP - Clustering failed.")
        return
    
    print("Step 3: Creating UMAP visualization...")
    umap_path = cluster_dir / f"umap_{args.split}.png"
    
    title_info = f"k={cluster_metrics['chosen_k']}, silhouette={cluster_metrics['silhouette']:.3f}"
    umap_file = create_umap_visualization(
        features, cluster_labels, args.split, umap_path, title_info
    )
    
    print("Step 4: Saving clustering report...")
    
    # Add paths to metrics
    cluster_metrics.update({
        'run_id': args.run_id,
        'split': args.split,
        'timestamp': datetime.now().isoformat(),
        'umap_path': umap_file,
        'parquet_path': str(parquet_path)
    })
    
    report_path = cluster_dir / "clustering_report.json"
    with open(report_path, 'w') as f:
        json.dump(cluster_metrics, f, indent=2)
    
    print(f"Clustering report saved: {report_path}")
    
    print("Step 5: Logging to Excel...")
    excel_logger = ExcelLogger()
    
    excel_data = {
        'run_id': args.run_id,
        'split': args.split,
        'chosen_k': cluster_metrics['chosen_k'],
        'silhouette': cluster_metrics['silhouette'],
        'nmi': cluster_metrics.get('nmi', 0),
        'ari': cluster_metrics.get('ari', 0),
        'umap_path': umap_file,
        'report_path': str(report_path)
    }
    
    try:
        row_num = excel_logger.append_row('cluster', excel_data)
        print(f"Excel cluster sheet updated: row {row_num}")
    except Exception as e:
        print(f"Warning: Excel logging failed: {e}")
    
    print("RESULTS SUMMARY:")
    print(f"  Split: {args.split}")
    print(f"  Samples: {cluster_metrics['n_samples']}")
    print(f"  Optimal k: {cluster_metrics['chosen_k']}")
    print(f"  Silhouette score: {cluster_metrics['silhouette']:.3f}")
    
    if 'nmi' in cluster_metrics:
        print(f"  NMI: {cluster_metrics['nmi']:.3f}")
        print(f"  ARI: {cluster_metrics['ari']:.3f}")
    
    print(f"  UMAP plot: {umap_file}")
    print(f"  Report: {report_path}")
    
    print("STOP - Clustering evaluation complete.")

if __name__ == "__main__":
    main()
