#!/usr/bin/env python3
"""
Cluster Membership Analysis - Who's in Each Group?
================================================

Creates clear visualizations and tables showing exactly which files 
are assigned to each cluster group for each method and k-value.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import re
from collections import Counter, defaultdict

def extract_file_identifier(filename):
    """Extract meaningful identifier from filename."""
    base_name = Path(filename).stem
    
    patterns = [
        r'(H\d+)',           # H001, H002, etc.
        r'(KP\d+-?\d*)',     # KP003, KP003-2, etc.
        r'(WEBBS?-?\d+)',    # WEBB-002, WEBBS-002, etc.
        r'(WEBSS-?\d+)',     # WEBSS-002, etc.
        r'(P\d+)',           # P001, P002, etc.
        r'(S\d+)',           # S001, S002, etc.
    ]
    
    for pattern in patterns:
        match = re.search(pattern, base_name, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    return base_name[:8].upper()

def get_real_filenames():
    """Get actual filenames from the dataset."""
    data_root = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/RAW sound_ML test sound list")
    
    if not data_root.exists():
        return None
    
    audio_files = list(data_root.glob("*.wav"))
    audio_files.sort()
    return [f.name for f in audio_files]

def create_file_info_mapping(audio_filenames, n_segments, is_segmented):
    """Create mapping from segments to original filenames."""
    file_info = []
    
    if is_segmented:
        # For segmented methods, each file contributes ~3 segments (30s/10s)
        segments_per_file = max(1, n_segments // len(audio_filenames))
        
        for audio_file in audio_filenames:
            file_info.extend([audio_file] * segments_per_file)
        
        # Adjust to exact segment count
        while len(file_info) < n_segments:
            file_info.append(audio_filenames[len(file_info) % len(audio_filenames)])
        file_info = file_info[:n_segments]
    else:
        # For non-segmented methods, one segment per file
        file_info = audio_filenames[:n_segments]
        while len(file_info) < n_segments:
            file_info.append(audio_filenames[len(file_info) % len(audio_filenames)])
    
    return file_info

def create_cluster_membership_table(features, file_info, method_name, k_values=[3, 4, 5]):
    """Create detailed cluster membership tables."""
    
    results = {}
    
    for k in k_values:
        # Perform clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        
        # Calculate silhouette score
        if len(np.unique(labels)) > 1:
            sil_score = silhouette_score(features, labels)
        else:
            sil_score = 0.0
        
        # Create membership dataframe
        df = pd.DataFrame({
            'Segment_ID': range(len(file_info)),
            'Original_Filename': file_info,
            'File_ID': [extract_file_identifier(f) for f in file_info],
            'Cluster': labels
        })
        
        # Group by cluster
        cluster_groups = {}
        for cluster_id in range(k):
            cluster_mask = df['Cluster'] == cluster_id
            cluster_files = df[cluster_mask]['File_ID'].tolist()
            cluster_groups[f'Cluster_{cluster_id}'] = {
                'files': cluster_files,
                'file_counts': Counter(cluster_files),
                'size': len(cluster_files)
            }
        
        results[f'k{k}'] = {
            'silhouette_score': sil_score,
            'dataframe': df,
            'cluster_groups': cluster_groups,
            'n_clusters': len(np.unique(labels))
        }
    
    return results

def create_cluster_membership_visualization(membership_results, method_name, output_dir):
    """Create comprehensive cluster membership visualizations."""
    
    # 1. Cluster Composition Heatmap
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.suptitle(f'Cluster Membership Analysis - {method_name}', fontsize=16, fontweight='bold')
    
    for i, k in enumerate([3, 4, 5]):
        ax = axes[i]
        result = membership_results[f'k{k}']
        sil_score = result['silhouette_score']
        
        # Create file-by-cluster matrix
        df = result['dataframe']
        unique_files = sorted(df['File_ID'].unique())
        
        # Count matrix: rows = files, cols = clusters
        matrix = np.zeros((len(unique_files), k))
        
        for cluster_id in range(k):
            cluster_files = df[df['Cluster'] == cluster_id]['File_ID']
            file_counts = Counter(cluster_files)
            
            for j, file_id in enumerate(unique_files):
                matrix[j, cluster_id] = file_counts.get(file_id, 0)
        
        # Create heatmap
        im = ax.imshow(matrix, cmap='Blues', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(k))
        ax.set_xticklabels([f'C{i}' for i in range(k)])
        ax.set_yticks(range(len(unique_files)))
        ax.set_yticklabels(unique_files, fontsize=8)
        
        ax.set_xlabel('Cluster')
        ax.set_ylabel('File ID')
        ax.set_title(f'k={k}, Silhouette={sil_score:.3f}')
        
        # Add text annotations
        for j in range(len(unique_files)):
            for cluster_id in range(k):
                count = int(matrix[j, cluster_id])
                if count > 0:
                    ax.text(cluster_id, j, str(count), 
                           ha='center', va='center', 
                           color='white' if count > matrix.max()/2 else 'black',
                           fontweight='bold')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Segment Count')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{method_name}_cluster_heatmap.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Cluster Composition Bar Charts
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle(f'Cluster Composition Details - {method_name}', fontsize=16, fontweight='bold')
    
    for i, k in enumerate([3, 4, 5]):
        ax = axes[i]
        result = membership_results[f'k{k}']
        cluster_groups = result['cluster_groups']
        
        # Prepare data for stacked bar chart
        cluster_names = [f'Cluster {j}' for j in range(k)]
        all_files = sorted(set().union(*[group['file_counts'].keys() for group in cluster_groups.values()]))
        
        # Create matrix for stacked bar chart
        data_matrix = []
        for file_id in all_files:
            file_row = []
            for cluster_id in range(k):
                count = cluster_groups[f'Cluster_{cluster_id}']['file_counts'].get(file_id, 0)
                file_row.append(count)
            data_matrix.append(file_row)
        
        data_matrix = np.array(data_matrix).T  # Transpose for plotting
        
        # Create stacked bar chart
        bottom = np.zeros(len(all_files))
        colors = plt.cm.Set3(np.linspace(0, 1, k))
        
        for cluster_id in range(k):
            ax.bar(all_files, data_matrix[cluster_id], bottom=bottom, 
                  label=f'Cluster {cluster_id}', color=colors[cluster_id], alpha=0.8)
            bottom += data_matrix[cluster_id]
        
        ax.set_xlabel('File ID')
        ax.set_ylabel('Segment Count')
        ax.set_title(f'k={k} - Segment Distribution Across Clusters')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{method_name}_cluster_bars.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_cluster_membership_tables(membership_results, method_name, output_dir):
    """Create detailed text tables showing cluster membership."""
    
    for k in [3, 4, 5]:
        result = membership_results[f'k{k}']
        cluster_groups = result['cluster_groups']
        sil_score = result['silhouette_score']
        
        # Create detailed table
        table_content = []
        table_content.append(f"CLUSTER MEMBERSHIP ANALYSIS - {method_name}")
        table_content.append(f"k={k}, Silhouette Score={sil_score:.3f}")
        table_content.append("=" * 60)
        table_content.append("")
        
        for cluster_id in range(k):
            group = cluster_groups[f'Cluster_{cluster_id}']
            table_content.append(f"CLUSTER {cluster_id} ({group['size']} segments):")
            table_content.append("-" * 30)
            
            # Sort files by count (most frequent first)
            sorted_files = sorted(group['file_counts'].items(), 
                                key=lambda x: x[1], reverse=True)
            
            for file_id, count in sorted_files:
                percentage = (count / group['size']) * 100
                table_content.append(f"  {file_id:12} : {count:2d} segments ({percentage:5.1f}%)")
            
            table_content.append("")
        
        # Save table
        table_path = output_dir / f'{method_name}_k{k}_membership.txt'
        with open(table_path, 'w') as f:
            f.write('\n'.join(table_content))

def analyze_method_cluster_membership(method_folder):
    """Analyze cluster membership for a single method."""
    method_folder = Path(method_folder)
    
    # Load existing results
    results_file = method_folder / 'results.json'
    features_file = method_folder / 'features.npy'
    
    if not results_file.exists() or not features_file.exists():
        print(f"❌ Missing files in {method_folder.name}")
        return None
    
    # Load data
    with open(results_file) as f:
        results = json.load(f)
    
    features = np.load(features_file)
    method_name = results['method_name']
    n_segments = results['n_segments']
    is_segmented = 'Seg_' in method_name
    
    print(f"🔍 Analyzing {method_name}...")
    
    # Get real filenames
    audio_filenames = get_real_filenames()
    if audio_filenames is None:
        print("❌ Could not load audio filenames")
        return None
    
    # Create file info mapping
    file_info = create_file_info_mapping(audio_filenames, n_segments, is_segmented)
    
    # Create cluster membership analysis
    membership_results = create_cluster_membership_table(features, file_info, method_name)
    
    # Create visualizations
    create_cluster_membership_visualization(membership_results, method_name, method_folder)
    
    # Create detailed tables
    create_cluster_membership_tables(membership_results, method_name, method_folder)
    
    print(f"✅ {method_name} - Created heatmap, bar charts, and membership tables")
    
    return membership_results

def analyze_all_methods():
    """Analyze cluster membership for all methods."""
    methods_dir = Path("OPERA_16_Methods")
    
    if not methods_dir.exists():
        print("❌ OPERA_16_Methods directory not found!")
        return
    
    method_folders = [d for d in methods_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    method_folders.sort()
    
    print(f"🎯 Analyzing cluster membership for {len(method_folders)} methods")
    print("🔍 Creating: heatmaps, bar charts, and detailed membership tables")
    print()
    
    for method_folder in method_folders:
        try:
            analyze_method_cluster_membership(method_folder)
        except Exception as e:
            print(f"❌ Error processing {method_folder.name}: {e}")
    
    print(f"\n✅ Completed cluster membership analysis for {len(method_folders)} methods")
    print("\n📁 New files created in each method folder:")
    print("   📊 *_cluster_heatmap.png (file-by-cluster heatmap)")
    print("   📊 *_cluster_bars.png (stacked bar charts)")
    print("   📄 *_k3_membership.txt (detailed k=3 membership table)")
    print("   📄 *_k4_membership.txt (detailed k=4 membership table)")
    print("   📄 *_k5_membership.txt (detailed k=5 membership table)")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze cluster membership - who's in each group")
    parser.add_argument("--method", type=str, help="Specific method to analyze (e.g., A0, C1)")
    parser.add_argument("--all", action="store_true", help="Analyze all methods")
    
    args = parser.parse_args()
    
    if args.all:
        analyze_all_methods()
    elif args.method:
        method_folder = f"OPERA_16_Methods/{args.method}_*"
        import glob
        folders = glob.glob(method_folder)
        if folders:
            analyze_method_cluster_membership(folders[0])
        else:
            print(f"❌ Method folder not found: {method_folder}")
    else:
        print("❌ Please specify --all or --method <method_code>")

if __name__ == "__main__":
    main()
