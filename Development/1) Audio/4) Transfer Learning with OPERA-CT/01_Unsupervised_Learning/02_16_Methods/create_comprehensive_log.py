#!/usr/bin/env python3
"""
Create comprehensive logging for OPERA-CT 16 methods experiment.
Extract all results from individual JSON files and create a complete summary.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import glob

def extract_all_results():
    """Extract results from all 16 methods."""
    
    # Define all 16 methods
    methods = [
        'A0_NoSeg_NoPre', 'A1_NoSeg_Bandpass', 'A2_NoSeg_SpectralGating', 
        'A3_NoSeg_HighPass', 'A4_NoSeg_PeakNormalize',
        'B0_NoSeg_Bandpass_SpectralGating', 'B1_NoSeg_PeakNormalize_Bandpass', 
        'B2_NoSeg_FullPipeline',
        'C0_Seg_NoPre', 'C1_Seg_Bandpass', 'C2_Seg_SpectralGating', 
        'C3_Seg_HighPass', 'C4_Seg_PeakNormalize',
        'D0_Seg_HighPass_PeakNormalize', 'D1_Seg_HighPass_Bandpass', 
        'D2_Seg_FullPipeline'
    ]
    
    results = []
    
    for method in methods:
        method_dir = Path(method)
        results_file = method_dir / 'results.json'
        
        if results_file.exists():
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            # Extract key information
            result = {
                'Method': data['method_code'],
                'Name': data['method_name'],
                'Files': data['n_files'],
                'Segments': data['n_segments'],
                'Feature_Shape': f"{data['feature_shape'][0]}x{data['feature_shape'][1]}",
                'Extraction_Time': f"{data['extraction_time']:.1f}s",
                'Best_Silhouette': data['best_silhouette'],
                'K3_Silhouette': data['clustering_results']['k3']['silhouette_score'],
                'K4_Silhouette': data['clustering_results']['k4']['silhouette_score'],
                'K5_Silhouette': data['clustering_results']['k5']['silhouette_score'],
                'Timestamp': data['timestamp']
            }
            
            results.append(result)
        else:
            print(f"⚠️  Missing results for {method}")
    
    return results

def create_comprehensive_summary(results):
    """Create comprehensive summary and analysis."""
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Sort by best silhouette score
    df_sorted = df.sort_values('Best_Silhouette', ascending=False).reset_index(drop=True)
    df_sorted['Rank'] = range(1, len(df_sorted) + 1)
    
    # Reorder columns
    columns = ['Rank', 'Method', 'Name', 'Files', 'Segments', 'Feature_Shape', 
               'Extraction_Time', 'Best_Silhouette', 'K3_Silhouette', 'K4_Silhouette', 
               'K5_Silhouette', 'Timestamp']
    df_sorted = df_sorted[columns]
    
    # Calculate statistics
    baseline_score = df_sorted[df_sorted['Method'] == 'A0']['Best_Silhouette'].iloc[0]
    best_score = df_sorted['Best_Silhouette'].max()
    improvement = ((best_score - baseline_score) / baseline_score) * 100
    
    # Create summary statistics
    summary_stats = {
        'Total_Methods': len(df_sorted),
        'Baseline_Score': baseline_score,
        'Best_Score': best_score,
        'Improvement_Percent': improvement,
        'Best_Method': df_sorted.iloc[0]['Method'],
        'Best_Method_Name': df_sorted.iloc[0]['Name'],
        'Segmentation_Impact': df_sorted[df_sorted['Name'].str.contains('Seg')]['Best_Silhouette'].mean() - 
                              df_sorted[~df_sorted['Name'].str.contains('Seg')]['Best_Silhouette'].mean(),
        'Bandpass_Impact': df_sorted[df_sorted['Name'].str.contains('Bandpass')]['Best_Silhouette'].mean() - 
                          df_sorted[~df_sorted['Name'].str.contains('Bandpass')]['Best_Silhouette'].mean()
    }
    
    return df_sorted, summary_stats

def main():
    """Main execution function."""
    
    print("🔍 Extracting results from all 16 OPERA-CT methods...")
    
    # Extract results
    results = extract_all_results()
    
    if not results:
        print("❌ No results found!")
        return
    
    print(f"✅ Found results for {len(results)} methods")
    
    # Create comprehensive summary
    df_sorted, summary_stats = create_comprehensive_summary(results)
    
    # Save detailed CSV
    csv_path = 'OPERA_CT_16_Methods_Complete_Results.csv'
    df_sorted.to_csv(csv_path, index=False)
    print(f"📊 Detailed results saved to: {csv_path}")
    
    # Save summary statistics
    summary_path = 'OPERA_CT_16_Methods_Summary_Stats.json'
    with open(summary_path, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    print(f"📈 Summary statistics saved to: {summary_path}")
    
    # Print key findings
    print(f"\n🏆 KEY FINDINGS:")
    print(f"   Best Method: {summary_stats['Best_Method']} ({summary_stats['Best_Method_Name']})")
    print(f"   Best Score: {summary_stats['Best_Score']:.3f}")
    print(f"   Improvement: +{summary_stats['Improvement_Percent']:.1f}% over baseline")
    print(f"   Segmentation Impact: +{summary_stats['Segmentation_Impact']:.3f}")
    print(f"   Bandpass Impact: +{summary_stats['Bandpass_Impact']:.3f}")
    
    # Print top 5 methods
    print(f"\n🥇 TOP 5 METHODS:")
    for _, row in df_sorted.head(5).iterrows():
        print(f"   {row['Rank']}. {row['Method']} ({row['Name']}): {row['Best_Silhouette']:.3f}")
    
    print(f"\n✅ Comprehensive logging complete!")

if __name__ == "__main__":
    main()
