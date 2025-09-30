#!/usr/bin/env python3
"""
Extract results from handcrafted features unsupervised learning experiments.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import glob

def extract_handcrafted_results():
    """Extract results from all 16 handcrafted methods."""
    
    # Define all 16 methods
    methods = [
        'A0_NoSeg_NoPre', 'A1_NoSeg_Bandpass', 'A2_NoSeg_SpectralGating', 
        'A3_NoSeg_HighPass20', 'A4_NoSeg_PeakNormalize',
        'B0_NoSeg_Bandpass_SpectralGating', 'B1_NoSeg_PeakNormalize_Bandpass', 
        'B2_NoSeg_FullPipeline',
        'C0_Seg_NoPre', 'C1_Seg_Bandpass', 'C2_Seg_SpectralGating', 
        'C3_Seg_HighPass20', 'C4_Seg_PeakNormalize',
        'D0_Seg_HighPass_PeakNormalize', 'D1_Seg_HighPass_Bandpass', 
        'D2_Seg_FullPipeline'
    ]
    
    results = []
    
    for method in methods:
        method_dir = Path(method)
        metrics_file = method_dir / 'outputs' / 'clustering' / 'clustering_logmel_mean_improved_metrics.json'
        
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                data = json.load(f)
            
            # Extract key information from the correct structure
            k3_data = data.get('kmeans_k3', {})
            k4_data = data.get('kmeans_k4', {})
            k5_data = data.get('kmeans_k5', {})
            
            # Find best silhouette score
            k3_silhouette = k3_data.get('silhouette_mean', 0)
            k4_silhouette = k4_data.get('silhouette_mean', 0)
            k5_silhouette = k5_data.get('silhouette_mean', 0)
            
            best_silhouette = max(k3_silhouette, k4_silhouette, k5_silhouette)
            best_k = 3 if k3_silhouette == best_silhouette else (4 if k4_silhouette == best_silhouette else 5)
            
            result = {
                'Method': method.split('_')[0] + method.split('_')[1] if len(method.split('_')) > 1 else method.split('_')[0],
                'Name': method,
                'Best_Silhouette': best_silhouette,
                'K3_Silhouette': k3_silhouette,
                'K4_Silhouette': k4_silhouette,
                'K5_Silhouette': k5_silhouette,
                'Best_K': best_k,
                'Calinski_Harabasz': k3_data.get('calinski_harabasz_mean', 0),
                'Davies_Bouldin': k3_data.get('davies_bouldin_mean', 0),
                'Quality_Score': k3_data.get('quality_score', 0)
            }
            
            results.append(result)
        else:
            print(f"⚠️  Missing results for {method}")
    
    return results

def create_comparison():
    """Create comparison between handcrafted and OPERA-CT results."""
    
    print("🔍 Extracting handcrafted features results...")
    handcrafted_results = extract_handcrafted_results()
    
    if not handcrafted_results:
        print("❌ No handcrafted results found!")
        return
    
    print(f"✅ Found results for {len(handcrafted_results)} handcrafted methods")
    
    # Convert to DataFrame
    df_handcrafted = pd.DataFrame(handcrafted_results)
    df_handcrafted = df_handcrafted.sort_values('Best_Silhouette', ascending=False).reset_index(drop=True)
    df_handcrafted['Rank'] = range(1, len(df_handcrafted) + 1)
    df_handcrafted['Feature_Type'] = 'Handcrafted'
    
    # Load OPERA-CT results
    opera_ct_path = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Development/4) Transfer Learning with OPERA-CT/01_Unsupervised_Learning/02_16_Methods/OPERA_CT_16_Methods_Complete_Results.csv")
    
    if opera_ct_path.exists():
        df_opera_ct = pd.read_csv(opera_ct_path)
        df_opera_ct['Feature_Type'] = 'OPERA-CT'
        
        # Create comparison
        comparison_data = []
        
        # Add handcrafted results
        for _, row in df_handcrafted.iterrows():
            comparison_data.append({
                'Rank': row['Rank'],
                'Method': row['Method'],
                'Name': row['Name'],
                'Feature_Type': 'Handcrafted',
                'Best_Silhouette': row['Best_Silhouette'],
                'K3_Silhouette': row['K3_Silhouette'],
                'K4_Silhouette': row['K4_Silhouette'],
                'K5_Silhouette': row['K5_Silhouette'],
                'Quality_Score': row.get('Quality_Score', 0)
            })
        
        # Add OPERA-CT results
        for _, row in df_opera_ct.iterrows():
            comparison_data.append({
                'Rank': row['Rank'],
                'Method': row['Method'],
                'Name': row['Name'],
                'Feature_Type': 'OPERA-CT',
                'Best_Silhouette': row['Best_Silhouette'],
                'K3_Silhouette': row['K3_Silhouette'],
                'K4_Silhouette': row['K4_Silhouette'],
                'K5_Silhouette': row['K5_Silhouette'],
                'Quality_Score': 0  # Not available for OPERA-CT
            })
        
        # Create comparison DataFrame
        df_comparison = pd.DataFrame(comparison_data)
        
        # Save comparison
        df_comparison.to_csv('Handcrafted_vs_OPERA_CT_Comparison.csv', index=False)
        
        # Create summary statistics
        handcrafted_best = df_handcrafted['Best_Silhouette'].max()
        opera_ct_best = df_opera_ct['Best_Silhouette'].max()
        handcrafted_mean = df_handcrafted['Best_Silhouette'].mean()
        opera_ct_mean = df_opera_ct['Best_Silhouette'].mean()
        
        summary = {
            'Handcrafted_Best': float(handcrafted_best),
            'OPERA_CT_Best': float(opera_ct_best),
            'Handcrafted_Mean': float(handcrafted_mean),
            'OPERA_CT_Mean': float(opera_ct_mean),
            'Performance_Gap': float(handcrafted_best - opera_ct_best),
            'Performance_Gap_Percent': float(((handcrafted_best - opera_ct_best) / opera_ct_best) * 100),
            'Handcrafted_Advantage': bool(handcrafted_best > opera_ct_best)
        }
        
        # Save summary
        with open('Comparison_Summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print results
        print(f"\n🏆 COMPARISON RESULTS:")
        print(f"   Handcrafted Best: {handcrafted_best:.3f}")
        print(f"   OPERA-CT Best: {opera_ct_best:.3f}")
        print(f"   Performance Gap: {summary['Performance_Gap']:.3f}")
        print(f"   Performance Gap: {summary['Performance_Gap_Percent']:.1f}%")
        print(f"   Handcrafted Advantage: {summary['Handcrafted_Advantage']}")
        
        print(f"\n📊 TOP 5 HANDCRAFTED METHODS:")
        for _, row in df_handcrafted.head(5).iterrows():
            print(f"   {row['Rank']}. {row['Method']} ({row['Name']}): {row['Best_Silhouette']:.3f}")
        
        print(f"\n📊 TOP 5 OPERA-CT METHODS:")
        for _, row in df_opera_ct.head(5).iterrows():
            print(f"   {row['Rank']}. {row['Method']} ({row['Name']}): {row['Best_Silhouette']:.3f}")
        
        print(f"\n✅ Comparison saved to: Handcrafted_vs_OPERA_CT_Comparison.csv")
        print(f"✅ Summary saved to: Comparison_Summary.json")
        
    else:
        print("❌ OPERA-CT results file not found!")

if __name__ == "__main__":
    create_comparison()
