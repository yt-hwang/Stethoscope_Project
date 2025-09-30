#!/usr/bin/env python3
"""
Apply Cluster Membership Analysis to Original Cycles (A-D)
=========================================================

Applies the same cluster membership analysis we created for OPERA-CT
to the original handcrafted features cycles for comparison.
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
from collections import Counter
import glob

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

def load_original_cycle_data(cycle_path):
    """Load data from original cycle folder."""
    cycle_path = Path(cycle_path)
    
    # Look for features files (CSV format)
    features_files = list(cycle_path.glob("**/features_all.csv"))
    if not features_files:
        features_files = list(cycle_path.glob("**/*features*.csv"))
    
    if not features_files:
        print(f"❌ No features file found in {cycle_path}")
        return None, None, None
    
    features_file = features_files[0]  # Take the first one found
    
    # Load features from CSV
    try:
        df = pd.read_csv(features_file)
        # Remove non-numeric columns (like filename)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        features = df[numeric_cols].values
        print(f"✅ Loaded features: {features.shape} from {features_file.name}")
    except Exception as e:
        print(f"❌ Error loading {features_file}: {e}")
        return None, None, None
    
    # Determine if segmented based on folder name
    is_segmented = 'Seg_' in cycle_path.name or any(['Seg_' in part for part in cycle_path.parts])
    
    # Get cycle name
    cycle_name = cycle_path.name
    
    return features, is_segmented, cycle_name

def create_cluster_membership_analysis_original(features, file_info, method_name, output_dir, k_values=[3, 4, 5]):
    """Create cluster membership analysis for original cycles."""
    
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

def create_visualizations_for_original(membership_results, method_name, output_dir):
    """Create visualizations for original cycles."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Cluster Composition Heatmap
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.suptitle(f'Cluster Membership Analysis - {method_name} (Original Handcrafted Features)', fontsize=16, fontweight='bold')
    
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
        im = ax.imshow(matrix, cmap='Reds', aspect='auto')
        
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
    plt.savefig(output_dir / f'{method_name}_original_cluster_heatmap.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Cluster Composition Bar Charts (ADDED)
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle(f'Cluster Composition Details - {method_name} (Original Handcrafted Features)', fontsize=16, fontweight='bold')
    
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
        ax.set_title(f'k={k} - Segment Distribution Across Clusters (Silhouette={result["silhouette_score"]:.3f})')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{method_name}_original_cluster_bars.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_membership_tables_original(membership_results, method_name, output_dir):
    """Create detailed text tables for original cycles."""
    output_dir = Path(output_dir)
    
    for k in [3, 4, 5]:
        result = membership_results[f'k{k}']
        cluster_groups = result['cluster_groups']
        sil_score = result['silhouette_score']
        
        # Create detailed table
        table_content = []
        table_content.append(f"CLUSTER MEMBERSHIP ANALYSIS - {method_name} (Original Handcrafted Features)")
        table_content.append(f"k={k}, Silhouette Score={sil_score:.3f}")
        table_content.append("=" * 80)
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
        table_path = output_dir / f'{method_name}_original_k{k}_membership.txt'
        with open(table_path, 'w') as f:
            f.write('\n'.join(table_content))

def analyze_original_cycle(cycle_path):
    """Analyze a single original cycle."""
    cycle_path = Path(cycle_path)
    
    print(f"🔍 Analyzing original cycle: {cycle_path.name}")
    
    # Load cycle data
    features, is_segmented, cycle_name = load_original_cycle_data(cycle_path)
    if features is None:
        return
    
    # Get real filenames
    audio_filenames = get_real_filenames()
    if audio_filenames is None:
        print("❌ Could not load audio filenames")
        return
    
    # Create file info mapping
    n_segments = features.shape[0]
    file_info = create_file_info_mapping(audio_filenames, n_segments, is_segmented)
    
    # Create cluster membership analysis
    membership_results = create_cluster_membership_analysis_original(features, file_info, cycle_name, cycle_path)
    
    # Create output directory
    output_dir = cycle_path / "cluster_analysis"
    
    # Create visualizations and tables
    create_visualizations_for_original(membership_results, cycle_name, output_dir)
    create_membership_tables_original(membership_results, cycle_name, output_dir)
    
    print(f"✅ {cycle_name} - Created cluster membership analysis in cluster_analysis/")

def find_and_analyze_original_cycles():
    """Find and analyze all original cycles."""
    
    # Look for original cycles in the Development directory
    base_paths = [
        "/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles",
        "/Users/yunhwang/Desktop/Stethoscope_Project/Development/Unsupervised Learning with New Data"
    ]
    
    all_cycles = []
    
    for base_path in base_paths:
        base_path = Path(base_path)
        if base_path.exists():
            # Look for cycle directories
            cycle_dirs = [d for d in base_path.iterdir() if d.is_dir() and 
                         any(c in d.name for c in ['A0', 'A1', 'A2', 'A3', 'A4', 'B0', 'B1', 'B2', 
                                                  'C0', 'C1', 'C2', 'C3', 'C4', 'D0', 'D1', 'D2'])]
            all_cycles.extend(cycle_dirs)
    
    if not all_cycles:
        print("❌ No original cycle directories found!")
        return
    
    all_cycles.sort()
    print(f"🎯 Found {len(all_cycles)} original cycle directories:")
    for cycle in all_cycles:
        print(f"   📁 {cycle}")
    
    print(f"\n🔄 Analyzing cluster membership for original cycles...")
    
    successful = 0
    for cycle_path in all_cycles:
        try:
            analyze_original_cycle(cycle_path)
            successful += 1
        except Exception as e:
            print(f"❌ Error analyzing {cycle_path.name}: {e}")
    
    print(f"\n✅ Successfully analyzed {successful}/{len(all_cycles)} original cycles")
    print("\n📁 New files created in each cycle folder:")
    print("   📂 cluster_analysis/")
    print("      📊 *_original_cluster_heatmap.png (file-by-cluster matrix)")
    print("      📊 *_original_cluster_bars.png (stacked bar charts)")
    print("      📄 *_original_k3_membership.txt")
    print("      📄 *_original_k4_membership.txt")
    print("      📄 *_original_k5_membership.txt")

if __name__ == "__main__":
    find_and_analyze_original_cycles()
