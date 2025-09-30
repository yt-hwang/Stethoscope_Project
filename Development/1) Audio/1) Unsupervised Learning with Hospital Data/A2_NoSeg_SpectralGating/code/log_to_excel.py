#!/usr/bin/env python3
"""
Excel logging for Cycle A0: NoSeg + NoPreprocess baseline
Updates Experiment_Tracking_System_Final.xlsx with cycle results
"""

import os
import sys
import pandas as pd
import json
import argparse
from pathlib import Path

def load_clustering_results(output_dir):
    """Load clustering results from JSON files"""
    representations = ['raw_waveform_stats', 'logmel_mean', 'mfcc_mean']
    all_results = {}
    
    for rep in representations:
        metrics_path = os.path.join(output_dir, 'clustering', f'clustering_{rep}_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                all_results[rep] = json.load(f)
        else:
            print(f"Warning: {metrics_path} not found")
    
    return all_results

def create_cycle_summary_row(cycle_id, phase, segmentation, preprocessing, representations, clustering_algos, output_root, notes=""):
    """Create a row for CycleSummary sheet"""
    return {
        'Cycle_ID': cycle_id,
        'Phase': phase,
        'Segmentation': segmentation,
        'Preprocessing': preprocessing,
        'Representations': ', '.join(representations),
        'Clustering_Algos': ', '.join(clustering_algos),
        'Output_Root': output_root,
        'Notes': notes
    }

def create_metrics_rows(cycle_id, all_results, output_dir):
    """Create rows for MetricsByRun sheet"""
    rows = []
    
    for rep_name, results in all_results.items():
        for algo_name, metrics in results.items():
            # Determine file paths
            features_csv = os.path.join(output_dir, 'features', f'features_{rep_name}.csv')
            metrics_json = os.path.join(output_dir, 'clustering', f'clustering_{rep_name}_metrics.json')
            umap_png = os.path.join(output_dir, 'visualizations', f'umap_{rep_name}_{algo_name}.png')
            
            # Create params string
            params = json.dumps(metrics['params'])
            
            row = {
                'Cycle_ID': cycle_id,
                'Representation': rep_name,
                'Clustering': algo_name,
                'Params': params,
                'Silhouette': metrics['silhouette'],
                'CalinskiHarabasz': metrics['calinski_harabasz'],
                'DaviesBouldin': metrics['davies_bouldin'],
                'NumClusters': metrics['n_clusters'],
                'NumNoise': metrics['n_noise'],
                'Features_CSV_Path': features_csv,
                'Metrics_JSON_Path': metrics_json,
                'UMAP_PNG_Path': umap_png
            }
            rows.append(row)
    
    return rows

def update_excel_file(excel_path, cycle_summary_row, metrics_rows):
    """Update the Excel file with new data"""
    try:
        # Try to read existing file
        with pd.ExcelWriter(excel_path, mode='a', if_sheet_exists='overlay') as writer:
            # Update CycleSummary sheet
            try:
                cycle_df = pd.read_excel(excel_path, sheet_name='CycleSummary')
                new_cycle_df = pd.concat([cycle_df, pd.DataFrame([cycle_summary_row])], ignore_index=True)
            except:
                # Create new sheet if it doesn't exist
                new_cycle_df = pd.DataFrame([cycle_summary_row])
            
            new_cycle_df.to_excel(writer, sheet_name='CycleSummary', index=False)
            
            # Update MetricsByRun sheet
            try:
                metrics_df = pd.read_excel(excel_path, sheet_name='MetricsByRun')
                new_metrics_df = pd.concat([metrics_df, pd.DataFrame(metrics_rows)], ignore_index=True)
            except:
                # Create new sheet if it doesn't exist
                new_metrics_df = pd.DataFrame(metrics_rows)
            
            new_metrics_df.to_excel(writer, sheet_name='MetricsByRun', index=False)
            
    except Exception as e:
        print(f"Error updating Excel file: {e}")
        # Create new file if it doesn't exist or can't be updated
        with pd.ExcelWriter(excel_path, mode='w') as writer:
            pd.DataFrame([cycle_summary_row]).to_excel(writer, sheet_name='CycleSummary', index=False)
            pd.DataFrame(metrics_rows).to_excel(writer, sheet_name='MetricsByRun', index=False)

def main():
    parser = argparse.ArgumentParser(description='Log Cycle A0 results to Excel')
    parser.add_argument('--output_dir', type=str,
                       default='/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/A2_NoSeg_SpectralGating/outputs',
                       help='Path to output directory')
    parser.add_argument('--excel_path', type=str,
                       default='/Users/yunhwang/Desktop/Stethoscope_Project/Experiment_Tracking_System_Final.xlsx',
                       help='Path to Excel tracking file')
    
    args = parser.parse_args()
    
    # Load clustering results
    all_results = load_clustering_results(args.output_dir)
    
    if not all_results:
        print("No clustering results found. Exiting.")
        return
    
    # Create cycle summary row
    cycle_summary_row = create_cycle_summary_row(
        cycle_id='A2',
        phase='A_NoSeg',
        segmentation='none',
        preprocessing='spectral_gating',
        representations=list(all_results.keys()),
        clustering_algos=['kmeans_k3', 'kmeans_k4', 'kmeans_k5', 'hdbscan'],
        output_root=args.output_dir,
        notes='Spectral gating applied for noise reduction'
    )
    
    # Create metrics rows
    metrics_rows = create_metrics_rows('A2', all_results, args.output_dir)
    
    # Update Excel file
    update_excel_file(args.excel_path, cycle_summary_row, metrics_rows)
    
    print(f"Excel logging completed!")
    print(f"Cycle A0 results logged to {args.excel_path}")
    print(f"Cycle summary: {cycle_summary_row}")
    print(f"Number of metric rows: {len(metrics_rows)}")

if __name__ == "__main__":
    main()
