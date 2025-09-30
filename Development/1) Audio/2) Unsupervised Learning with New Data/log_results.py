#!/usr/bin/env python3
"""
Results Logging System for New Data Experiment
Creates Excel and CSV logs just like the original experiment
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment

def collect_all_results(base_dir):
    """Collect results from all completed cycles."""
    results = []
    
    cycles = [
        'A0_NoSeg_NoPre', 'A1_NoSeg_Bandpass', 'A2_NoSeg_SpectralGating', 'A3_NoSeg_HighPass20', 'A4_NoSeg_PeakNormalize',
        'B0_NoSeg_Bandpass_SpectralGating', 'B1_NoSeg_PeakNormalize_Bandpass', 'B2_NoSeg_FullPipeline',
        'C0_Seg_NoPre', 'C1_Seg_Bandpass', 'C2_Seg_SpectralGating', 'C3_Seg_HighPass20', 'C4_Seg_PeakNormalize',
        'D0_Seg_HighPass_PeakNormalize', 'D1_Seg_HighPass_Bandpass', 'D2_Seg_FullPipeline'
    ]
    
    for cycle in cycles:
        cycle_dir = Path(base_dir) / cycle / "outputs" / "clustering"
        if cycle_dir.exists():
            for rep in ['raw_waveform_stats', 'logmel_mean', 'mfcc_mean']:
                metrics_file = cycle_dir / f'clustering_{rep}_improved_metrics.json'
                if metrics_file.exists():
                    try:
                        with open(metrics_file, 'r') as f:
                            data = json.load(f)
                            for algo, metrics in data.items():
                                # Include ALL results, even failures (quality_score = 0)
                                results.append({
                                        'Cycle': cycle,
                                        'Representation': rep,
                                        'Algorithm': algo,
                                        'Quality_Score': metrics['quality_score'],
                                        'Silhouette_Mean': metrics['silhouette_mean'],
                                        'Silhouette_Std': metrics['silhouette_std'],
                                        'Stability': metrics['stability'],
                                        'Valid_Runs': metrics['valid_runs'],
                                        'Total_Runs': metrics['total_runs'],
                                        'Calinski_Harabasz_Mean': metrics.get('calinski_harabasz_mean', 0),
                                        'Davies_Bouldin_Mean': metrics.get('davies_bouldin_mean', 0),
                                    })
                    except Exception as e:
                        print(f"Error reading {metrics_file}: {e}")
    
    return pd.DataFrame(results)

def create_excel_log(results_df, base_dir):
    """Create comprehensive Excel log."""
    excel_path = Path(base_dir) / "New_Data_Experiment_Tracking.xlsx"
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Summary sheet - best result per cycle
        summary_data = []
        for cycle in results_df['Cycle'].unique():
            cycle_data = results_df[results_df['Cycle'] == cycle]
            if len(cycle_data) > 0:
                best_result = cycle_data.loc[cycle_data['Quality_Score'].idxmax()]
                
                # Determine preprocessing methods
                if 'NoSeg' in cycle:
                    segmentation = 'No'
                else:
                    segmentation = 'Yes (10s)'
                
                if 'NoPre' in cycle:
                    preprocessing = 'None'
                elif 'Bandpass' in cycle and 'SpectralGating' in cycle:
                    preprocessing = 'Bandpass + SpectralGating'
                elif 'PeakNormalize' in cycle and 'Bandpass' in cycle:
                    preprocessing = 'PeakNormalize + Bandpass'  
                elif 'HighPass' in cycle and 'PeakNormalize' in cycle:
                    preprocessing = 'HighPass + PeakNormalize'
                elif 'FullPipeline' in cycle:
                    preprocessing = 'Full Pipeline'
                elif 'Bandpass' in cycle:
                    preprocessing = 'Bandpass Filter'
                elif 'SpectralGating' in cycle:
                    preprocessing = 'Spectral Gating'
                elif 'HighPass' in cycle:
                    preprocessing = 'High-pass Filter'
                elif 'PeakNormalize' in cycle:
                    preprocessing = 'Peak Normalization'
                else:
                    preprocessing = 'Unknown'
                
                summary_data.append({
                    'Cycle': cycle,
                    'Series': cycle[0],
                    'Segmentation': segmentation,
                    'Preprocessing': preprocessing,
                    'Best_Algorithm': best_result['Algorithm'],
                    'Best_Representation': best_result['Representation'],
                    'Best_Quality_Score': round(best_result['Quality_Score'], 3),
                    'Best_Silhouette': round(best_result['Silhouette_Mean'], 3),
                    'Best_Stability': round(best_result['Stability'], 3),
                    'Valid_Runs': f"{best_result['Valid_Runs']}/{best_result['Total_Runs']}",
                    'Status': 'Completed'
                })
        
        summary_df = pd.DataFrame(summary_data).sort_values(['Series', 'Cycle'])
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Detailed results
        detailed_df = results_df.copy()
        detailed_df['Quality_Score'] = detailed_df['Quality_Score'].round(3)
        detailed_df['Silhouette_Mean'] = detailed_df['Silhouette_Mean'].round(3)
        detailed_df['Stability'] = detailed_df['Stability'].round(3)
        detailed_df.to_excel(writer, sheet_name='Detailed_Results', index=False)
        
        # Experiment info
        info_data = {
            'Parameter': [
                'Experiment Name',
                'Dataset', 
                'Dataset Path',
                'File Count',
                'Date Created',
                'Total Cycles Planned',
                'Cycles Completed',
                'Best Overall Result',
                'Best Cycle',
                'Comparison Baseline (Original)',
                'Improvement vs Original'
            ],
            'Value': [
                'Unsupervised Learning with New Data - Validation Experiment',
                'RAW sound_ML test sound list',
                '/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/RAW sound_ML test sound list',
                29,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                16,
                len(summary_df),
                f"{summary_df['Best_Quality_Score'].max():.3f}" if len(summary_df) > 0 else 'N/A',
                summary_df.loc[summary_df['Best_Quality_Score'].idxmax(), 'Cycle'] if len(summary_df) > 0 else 'N/A',
                '0.343 (A0, Hospital data, 330 files)',
                f"{((summary_df['Best_Quality_Score'].max() - 0.343) / 0.343 * 100):+.1f}%" if len(summary_df) > 0 else 'N/A'
            ]
        }
        info_df = pd.DataFrame(info_data)
        info_df.to_excel(writer, sheet_name='Experiment_Info', index=False)
        
        # Comparison with original
        original_data = {
            'Cycle': ['A0', 'A1', 'A2', 'A3', 'A4'],
            'Original_Best_Score': [0.343, 'No data', 0.658, 'No data', 'No data'],
            'New_Best_Score': [
                summary_df[summary_df['Cycle'] == 'A0_NoSeg_NoPre']['Best_Quality_Score'].iloc[0] if len(summary_df[summary_df['Cycle'] == 'A0_NoSeg_NoPre']) > 0 else 'Not completed',
                summary_df[summary_df['Cycle'] == 'A1_NoSeg_Bandpass']['Best_Quality_Score'].iloc[0] if len(summary_df[summary_df['Cycle'] == 'A1_NoSeg_Bandpass']) > 0 else 'Not completed',
                summary_df[summary_df['Cycle'] == 'A2_NoSeg_SpectralGating']['Best_Quality_Score'].iloc[0] if len(summary_df[summary_df['Cycle'] == 'A2_NoSeg_SpectralGating']) > 0 else 'Not completed',
                summary_df[summary_df['Cycle'] == 'A3_NoSeg_HighPass20']['Best_Quality_Score'].iloc[0] if len(summary_df[summary_df['Cycle'] == 'A3_NoSeg_HighPass20']) > 0 else 'Not completed',
                summary_df[summary_df['Cycle'] == 'A4_NoSeg_PeakNormalize']['Best_Quality_Score'].iloc[0] if len(summary_df[summary_df['Cycle'] == 'A4_NoSeg_PeakNormalize']) > 0 else 'Not completed'
            ]
        }
        comparison_df = pd.DataFrame(original_data)
        comparison_df.to_excel(writer, sheet_name='Original_vs_New', index=False)
    
    print(f"📊 Excel log created: {excel_path}")
    return excel_path

def create_csv_summary(results_df, base_dir):
    """Create CSV summary for easy analysis."""
    csv_path = Path(base_dir) / "new_data_comprehensive_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"📋 CSV summary created: {csv_path}")
    return csv_path

def main():
    base_dir = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Development/Unsupervised Learning with New Data")
    
    print("📊 CREATING COMPREHENSIVE LOGS")
    print("=" * 50)
    
    # Collect all results
    results_df = collect_all_results(base_dir)
    
    if results_df.empty:
        print("❌ No results found to log!")
        return
    
    print(f"✅ Found {len(results_df)} results from {results_df['Cycle'].nunique()} cycles")
    
    # Create logs
    excel_path = create_excel_log(results_df, base_dir)
    csv_path = create_csv_summary(results_df, base_dir)
    
    # Show summary
    print(f"\n🏆 CURRENT RESULTS SUMMARY:")
    print("-" * 30)
    top_results = results_df.nlargest(5, 'Quality_Score')
    for i, (_, row) in enumerate(top_results.iterrows(), 1):
        print(f"{i}. {row['Cycle']}: {row['Quality_Score']:.3f} ({row['Algorithm']}, {row['Representation']})")
    
    print(f"\n✅ LOGGING COMPLETED!")
    print(f"Excel: {excel_path}")
    print(f"CSV: {csv_path}")

if __name__ == "__main__":
    main()
