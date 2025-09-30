#!/usr/bin/env python3
"""
Cycle B1: NoSeg + PeakNormalize + Bandpass - Excel Logging
"""

import os
import sys
import pandas as pd
import json
from pathlib import Path
import argparse

def load_excel_file(excel_path):
    """Load existing Excel file or create new one"""
    if os.path.exists(excel_path):
        # Load existing sheets
        cycle_summary = pd.read_excel(excel_path, sheet_name='CycleSummary')
        metrics_by_run = pd.read_excel(excel_path, sheet_name='MetricsByRun')
        return cycle_summary, metrics_by_run
    else:
        # Create new DataFrames
        cycle_summary = pd.DataFrame(columns=[
            'Cycle_ID', 'Phase', 'Segmentation', 'Preprocessing', 
            'Representations', 'Clustering_Algos', 'Output_Root', 'Notes'
        ])
        metrics_by_run = pd.DataFrame(columns=[
            'Cycle_ID', 'Representation', 'Clustering', 'Params',
            'Silhouette', 'CalinskiHarabasz', 'DaviesBouldin',
            'NumClusters', 'NumNoise', 'Features_CSV_Path',
            'Metrics_JSON_Path', 'UMAP_PNG_Path'
        ])
        return cycle_summary, metrics_by_run

def create_cycle_summary_row():
    """Create cycle summary row for B1"""
    return {
        'Cycle_ID': 'B1',
        'Phase': 'B_NoSeg_Combinations',
        'Segmentation': 'none',
        'Preprocessing': 'peak_normalize_bandpass_100_2000',
        'Representations': 'raw_waveform_stats, logmel_mean, mfcc_mean',
        'Clustering_Algos': 'KMeans(k=3,4,5), HDBSCAN',
        'Output_Root': '/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/B1_NoSeg_PeakNormalize_Bandpass/outputs',
        'Notes': 'Combination of A4 (PeakNormalize) + A1 (Bandpass) - consistency + performance'
    }

def create_metrics_rows(output_dir):
    """Create metrics rows for all representations and algorithms"""
    rows = []
    
    representations = ['raw_waveform_stats', 'logmel_mean', 'mfcc_mean']
    
    for rep in representations:
        metrics_path = os.path.join(output_dir, 'clustering', f'clustering_{rep}_metrics.json')
        
        if not os.path.exists(metrics_path):
            print(f"Warning: {metrics_path} not found")
            continue
        
        # Load clustering results
        with open(metrics_path, 'r') as f:
            clustering_results = json.load(f)
        
        for algo_name, metrics in clustering_results.items():
            # Determine clustering algorithm name
            if 'kmeans' in algo_name:
                clustering_name = f"KMeans_k{metrics['params']['n_clusters']}"
            else:
                clustering_name = "HDBSCAN"
            
            # Create row
            row = {
                'Cycle_ID': 'B1',
                'Representation': rep,
                'Clustering': clustering_name,
                'Params': str(metrics['params']),
                'Silhouette': metrics['Silhouette'],
                'CalinskiHarabasz': metrics['Calinski-Harabasz'],
                'DaviesBouldin': metrics['Davies-Bouldin'],
                'NumClusters': metrics['n_clusters'],
                'NumNoise': metrics['n_noise'],
                'Features_CSV_Path': os.path.join(output_dir, 'features', f'features_{rep}.csv'),
                'Metrics_JSON_Path': metrics_path,
                'UMAP_PNG_Path': os.path.join(output_dir, 'visualizations', f'umap_{rep}_{algo_name}.png')
            }
            rows.append(row)
    
    return rows

def main():
    parser = argparse.ArgumentParser(description='Log Cycle B1 results to Excel')
    parser.add_argument('--output_dir', 
                       default='/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/B1_NoSeg_PeakNormalize_Bandpass/outputs',
                       help='Output directory')
    parser.add_argument('--excel_path',
                       default='/Users/yunhwang/Desktop/Stethoscope_Project/Experiment_Tracking_System_Final.xlsx',
                       help='Path to Excel file')
    
    args = parser.parse_args()
    
    # Load existing Excel data
    cycle_summary, metrics_by_run = load_excel_file(args.excel_path)
    
    # Create new rows
    cycle_summary_row = create_cycle_summary_row()
    metrics_rows = create_metrics_rows(args.output_dir)
    
    # Add new rows
    cycle_summary = pd.concat([cycle_summary, pd.DataFrame([cycle_summary_row])], ignore_index=True)
    metrics_by_run = pd.concat([metrics_by_run, pd.DataFrame(metrics_rows)], ignore_index=True)
    
    # Save to Excel
    with pd.ExcelWriter(args.excel_path, engine='openpyxl') as writer:
        cycle_summary.to_excel(writer, sheet_name='CycleSummary', index=False)
        metrics_by_run.to_excel(writer, sheet_name='MetricsByRun', index=False)
    
    print(f"Logged Cycle B1 results to {args.excel_path}")
    print(f"  Cycle Summary: 1 row added")
    print(f"  Metrics by Run: {len(metrics_rows)} rows added")

if __name__ == "__main__":
    main()
