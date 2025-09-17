#!/usr/bin/env python3
"""
Re-evaluate All Cycles with Improved Cluster Balance Constraints
"""

import os
import sys
import subprocess
import pandas as pd
import json
from pathlib import Path
import argparse

def run_improved_evaluation(cycle_id, cycle_dir):
    """Run improved evaluation for a single cycle."""
    print(f"\n{'='*60}")
    print(f"RE-EVALUATING CYCLE {cycle_id}")
    print(f"{'='*60}")
    print(f"Directory: {cycle_dir}")
    
    if not os.path.exists(cycle_dir):
        print(f"❌ Cycle directory not found: {cycle_dir}")
        return False
    
    # Check if features exist
    features_dir = os.path.join(cycle_dir, 'features')
    if not os.path.exists(features_dir):
        print(f"❌ Features directory not found: {features_dir}")
        return False
    
    # Run improved evaluation
    cmd = [
        sys.executable, 
        'improved_evaluation.py',
        '--cycle_dir', cycle_dir,
        '--cycle_id', cycle_id
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Cycle {cycle_id} re-evaluation completed successfully")
        return True
    else:
        print(f"❌ Cycle {cycle_id} re-evaluation failed")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False

def create_comprehensive_summary():
    """Create comprehensive summary of all re-evaluated cycles."""
    print(f"\n{'='*80}")
    print("CREATING COMPREHENSIVE SUMMARY")
    print(f"{'='*80}")
    
    # Define all cycles
    cycles = {
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
    valid_cycles = []
    
    for cycle_id, cycle_dir in cycles.items():
        print(f"\nLoading results for {cycle_id}...")
        
        representations = ['raw_waveform_stats', 'logmel_mean', 'mfcc_mean']
        cycle_has_valid_results = False
        
        for rep in representations:
            metrics_path = os.path.join(cycle_dir, 'clustering', f'clustering_{rep}_improved_metrics.json')
            
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, 'r') as f:
                        metrics_data = json.load(f)
                    
                    for algo_name, metrics in metrics_data.items():
                        if metrics.get('valid_runs', 0) > 0:  # Only include valid results
                            cycle_has_valid_results = True
                            
                            result = {
                                'Cycle_ID': cycle_id,
                                'Representation': rep,
                                'Algorithm': algo_name,
                                'Valid_Runs': metrics.get('valid_runs', 0),
                                'Total_Runs': metrics.get('total_runs', 0),
                                'Silhouette_Mean': metrics.get('silhouette_mean', np.nan),
                                'Silhouette_Std': metrics.get('silhouette_std', np.nan),
                                'Calinski_Harabasz_Mean': metrics.get('calinski_harabasz_mean', np.nan),
                                'Calinski_Harabasz_Std': metrics.get('calinski_harabasz_std', np.nan),
                                'Davies_Bouldin_Mean': metrics.get('davies_bouldin_mean', np.nan),
                                'Davies_Bouldin_Std': metrics.get('davies_bouldin_std', np.nan),
                                'Stability': metrics.get('stability', 0.0),
                                'Quality_Score': metrics.get('quality_score', 0.0),
                                'Params': str(metrics.get('params', {}))
                            }
                            
                            # Add HDBSCAN-specific metrics
                            if 'hdbscan' in algo_name:
                                result['Mean_Clusters'] = metrics.get('mean_clusters', 0)
                                result['Mean_Noise_Ratio'] = metrics.get('mean_noise_ratio', 0.0)
                            else:
                                result['Mean_Clusters'] = metrics.get('params', {}).get('n_clusters', 0)
                                result['Mean_Noise_Ratio'] = 0.0
                            
                            all_results.append(result)
                
                except Exception as e:
                    print(f"  Error loading {rep} for {cycle_id}: {e}")
                    continue
        
        if cycle_has_valid_results:
            valid_cycles.append(cycle_id)
            print(f"  ✅ {cycle_id}: Found valid results")
        else:
            print(f"  ❌ {cycle_id}: No valid results found")
    
    # Create summary DataFrame
    df_results = pd.DataFrame(all_results)
    
    if len(df_results) == 0:
        print("\n❌ No valid results found across all cycles!")
        return
    
    # Sort by quality score (descending)
    df_results = df_results.sort_values('Quality_Score', ascending=False)
    
    # Create cycle summary (best result per cycle)
    cycle_summary = []
    for cycle_id in valid_cycles:
        cycle_data = df_results[df_results['Cycle_ID'] == cycle_id]
        if len(cycle_data) > 0:
            best_result = cycle_data.iloc[0]  # Already sorted by quality score
            cycle_summary.append({
                'Cycle_ID': cycle_id,
                'Best_Representation': best_result['Representation'],
                'Best_Algorithm': best_result['Algorithm'],
                'Quality_Score': best_result['Quality_Score'],
                'Silhouette_Mean': best_result['Silhouette_Mean'],
                'Silhouette_Std': best_result['Silhouette_Std'],
                'Valid_Runs': best_result['Valid_Runs'],
                'Total_Runs': best_result['Total_Runs'],
                'Stability': best_result['Stability'],
                'Mean_Clusters': best_result['Mean_Clusters'],
                'Mean_Noise_Ratio': best_result['Mean_Noise_Ratio']
            })
    
    # Sort cycle summary by quality score
    cycle_summary_df = pd.DataFrame(cycle_summary)
    cycle_summary_df = cycle_summary_df.sort_values('Quality_Score', ascending=False)
    
    # Print results
    print(f"\n{'='*80}")
    print("COMPREHENSIVE RE-EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"Total valid results: {len(df_results)}")
    print(f"Cycles with valid results: {len(valid_cycles)}")
    print(f"Cycles without valid results: {len(cycles) - len(valid_cycles)}")
    
    print(f"\n{'='*80}")
    print("CYCLE RANKING (Best Result per Cycle)")
    print(f"{'='*80}")
    print(f"{'Rank':<4} {'Cycle':<6} {'Representation':<18} {'Algorithm':<15} {'Quality':<8} {'Silhouette':<12} {'Valid':<8} {'Stability':<10}")
    print(f"{'-'*80}")
    
    for i, (_, row) in enumerate(cycle_summary_df.iterrows(), 1):
        print(f"{i:<4} {row['Cycle_ID']:<6} {row['Best_Representation']:<18} {row['Best_Algorithm']:<15} "
              f"{row['Quality_Score']:<8.3f} {row['Silhouette_Mean']:<8.3f}±{row['Silhouette_Std']:<3.3f} "
              f"{row['Valid_Runs']:<3}/{row['Total_Runs']:<3} {row['Stability']:<10.3f}")
    
    print(f"\n{'='*80}")
    print("TOP 10 OVERALL RESULTS")
    print(f"{'='*80}")
    print(f"{'Rank':<4} {'Cycle':<6} {'Representation':<18} {'Algorithm':<15} {'Quality':<8} {'Silhouette':<12} {'Valid':<8}")
    print(f"{'-'*80}")
    
    for i, (_, row) in enumerate(df_results.head(10).iterrows(), 1):
        print(f"{i:<4} {row['Cycle_ID']:<6} {row['Representation']:<18} {row['Algorithm']:<15} "
              f"{row['Quality_Score']:<8.3f} {row['Silhouette_Mean']:<8.3f}±{row['Silhouette_Std']:<3.3f} "
              f"{row['Valid_Runs']:<3}/{row['Total_Runs']:<3}")
    
    # Save detailed results
    output_path = '/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/comprehensive_re_evaluation_results.csv'
    df_results.to_csv(output_path, index=False)
    print(f"\n💾 Detailed results saved to: {output_path}")
    
    # Save cycle summary
    summary_path = '/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/cycle_ranking_summary.csv'
    cycle_summary_df.to_csv(summary_path, index=False)
    print(f"💾 Cycle ranking saved to: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="Re-evaluate all cycles with improved cluster balance constraints.")
    parser.add_argument('--skip_evaluation', action='store_true', 
                       help='Skip re-evaluation and only create summary from existing results.')
    
    args = parser.parse_args()
    
    if not args.skip_evaluation:
        # Define all cycles to re-evaluate
        cycles = {
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
        
        print(f"Starting re-evaluation of {len(cycles)} cycles...")
        print("Cluster balance constraints:")
        print("  - Min cluster size: 5% of data")
        print("  - Max Gini coefficient: 0.7")
        print("  - Min silhouette threshold: 0.3")
        print("  - Robust evaluation: 7 seeds per algorithm")
        
        successful_cycles = 0
        failed_cycles = []
        
        for cycle_id, cycle_dir in cycles.items():
            if run_improved_evaluation(cycle_id, cycle_dir):
                successful_cycles += 1
            else:
                failed_cycles.append(cycle_id)
        
        print(f"\n{'='*60}")
        print("RE-EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Successful cycles: {successful_cycles}/{len(cycles)}")
        if failed_cycles:
            print(f"Failed cycles: {', '.join(failed_cycles)}")
    
    # Create comprehensive summary
    create_comprehensive_summary()

if __name__ == "__main__":
    import numpy as np
    main()
