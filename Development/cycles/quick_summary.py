#!/usr/bin/env python3
"""
Quick Summary of All Cycle Results
"""

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path

def load_cycle_results(cycle_path, cycle_id):
    """Load results from a single cycle"""
    results = []
    
    representations = ['raw_waveform_stats', 'logmel_mean', 'mfcc_mean']
    
    for rep in representations:
        metrics_path = os.path.join(cycle_path, 'clustering', f'clustering_{rep}_metrics.json')
        
        if not os.path.exists(metrics_path):
            continue
        
        try:
            with open(metrics_path, 'r') as f:
                data = json.load(f)
            
            for algo_key, metrics in data.items():
                # Handle both naming conventions
                silhouette = metrics.get('Silhouette', metrics.get('silhouette', np.nan))
                calinski_harabasz = metrics.get('Calinski-Harabasz', metrics.get('calinski_harabasz', np.nan))
                davies_bouldin = metrics.get('Davies-Bouldin', metrics.get('davies_bouldin', np.nan))
                n_clusters = metrics.get('n_clusters', 0)
                n_noise = metrics.get('n_noise', 0)
                
                # Skip invalid results
                if silhouette == -1.0 or silhouette == np.inf or np.isnan(silhouette):
                    continue
                    
                results.append({
                    'Cycle': cycle_id,
                    'Representation': rep,
                    'Algorithm': algo_key,
                    'Silhouette': silhouette,
                    'Calinski_Harabasz': calinski_harabasz,
                    'Davies_Bouldin': davies_bouldin,
                    'N_Clusters': n_clusters,
                    'N_Noise': n_noise
                })
        except Exception as e:
            print(f"Error loading {cycle_id} {rep}: {e}")
            continue
    
    return results

def main():
    # Define cycle paths
    cycle_paths = {
        'A0': '/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/A0_NoSeg_NoPre/outputs',
        'A1': '/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/A1_NoSeg_Bandpass/outputs',
        'A2': '/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/A2_NoSeg_SpectralGating/outputs',
        'A3': '/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/A3_NoSeg_HighPass20/outputs',
        'A4': '/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/A4_NoSeg_PeakNormalize/outputs',
        'B0': '/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/B0_NoSeg_Bandpass_SpectralGating/outputs',
        'B1': '/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/B1_NoSeg_PeakNormalize_Bandpass/outputs',
        'B2': '/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/B2_NoSeg_FullPipeline/outputs',
        'C0': '/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/C0_Seg_NoPre/outputs',
        'C1': '/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/C1_Seg_Bandpass/outputs',
        'C2': '/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/C2_Seg_SpectralGating/outputs',
        'C3': '/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/C3_Seg_HighPass20/outputs',
        'C4': '/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/C4_Seg_PeakNormalize/outputs',
        'D0': '/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/D0_Seg_HighPass_PeakNormalize/outputs',
        'D1': '/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/D1_Seg_HighPass_Bandpass/outputs'
    }
    
    all_results = []
    
    # Load results from each cycle
    for cycle_id, cycle_path in cycle_paths.items():
        print(f"Loading {cycle_id}...")
        cycle_results = load_cycle_results(cycle_path, cycle_id)
        all_results.extend(cycle_results)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    if df.empty:
        print("No results found!")
        return
    
    # Calculate a simple quality score (normalized silhouette)
    df['Quality_Score'] = df['Silhouette'].fillna(0)
    
    # Sort by quality score
    df_sorted = df.sort_values('Quality_Score', ascending=False)
    
    print("\n" + "="*80)
    print("TOP 20 RESULTS (by Silhouette Score)")
    print("="*80)
    
    # Show top 20 results
    top_results = df_sorted.head(20)
    
    for idx, row in top_results.iterrows():
        print(f"{row['Cycle']:3s} | {row['Representation']:18s} | {row['Algorithm']:15s} | "
              f"Silhouette: {row['Silhouette']:6.3f} | CH: {row['Calinski_Harabasz']:7.1f} | "
              f"DB: {row['Davies_Bouldin']:6.3f} | Clusters: {row['N_Clusters']:2d}")
    
    print("\n" + "="*80)
    print("CYCLE SUMMARY (Best Result per Cycle)")
    print("="*80)
    
    # Group by cycle and show best result
    cycle_best = df_sorted.groupby('Cycle').first().reset_index()
    
    for idx, row in cycle_best.iterrows():
        print(f"{row['Cycle']:3s} | {row['Representation']:18s} | {row['Algorithm']:15s} | "
              f"Silhouette: {row['Silhouette']:6.3f} | CH: {row['Calinski_Harabasz']:7.1f} | "
              f"DB: {row['Davies_Bouldin']:6.3f} | Clusters: {row['N_Clusters']:2d}")
    
    print(f"\nTotal results loaded: {len(df)}")
    print(f"Cycles with data: {df['Cycle'].nunique()}")

if __name__ == "__main__":
    main()
