#!/usr/bin/env python3
"""
Analyze Individual Method Rankings to Suggest Better B and D Combinations
"""

import pandas as pd
import numpy as np

def analyze_individual_methods():
    """Analyze individual method rankings from re-evaluation results."""
    
    # Load the comprehensive results
    df = pd.read_csv('comprehensive_re_evaluation_results.csv')
    
    # Filter for individual methods only (A-Series and C-Series)
    individual_methods = df[df['Cycle_ID'].str.match(r'^[AC]\d+$')].copy()
    
    # Get best result per cycle
    best_per_cycle = individual_methods.groupby('Cycle_ID').first().reset_index()
    
    print("="*80)
    print("INDIVIDUAL METHOD RANKINGS (A-Series and C-Series)")
    print("="*80)
    print(f"{'Rank':<4} {'Cycle':<6} {'Method':<20} {'Quality':<8} {'Silhouette':<10} {'Valid':<8} {'Stability':<10}")
    print("-"*80)
    
    for i, (_, row) in enumerate(best_per_cycle.iterrows(), 1):
        print(f"{i:<4} {row['Cycle_ID']:<6} {row['Representation']:<20} {row['Quality_Score']:<8.3f} "
              f"{row['Silhouette_Mean']:<8.3f}±{row['Silhouette_Std']:<2.3f} "
              f"{row['Valid_Runs']:<3}/{row['Total_Runs']:<3} {row['Stability']:<10.3f}")
    
    return best_per_cycle

def suggest_better_combinations(best_per_cycle):
    """Suggest better B and D combinations based on individual method rankings."""
    
    print("\n" + "="*80)
    print("ANALYSIS FOR BETTER COMBINATIONS")
    print("="*80)
    
    # Separate A-Series and C-Series
    a_series = best_per_cycle[best_per_cycle['Cycle_ID'].str.startswith('A')].copy()
    c_series = best_per_cycle[best_per_cycle['Cycle_ID'].str.startswith('C')].copy()
    
    print("\nA-Series Rankings (NoSeg + Individual Methods):")
    print("-" * 50)
    for _, row in a_series.iterrows():
        method = row['Cycle_ID'].replace('A', '')
        if method == '0':
            method = 'NoPreprocess'
        elif method == '1':
            method = 'Bandpass'
        elif method == '2':
            method = 'SpectralGating'
        elif method == '3':
            method = 'HighPass'
        elif method == '4':
            method = 'PeakNormalize'
        
        print(f"A{row['Cycle_ID'][1:]}: {method:<15} - Quality: {row['Quality_Score']:.3f}")
    
    print("\nC-Series Rankings (Seg + Individual Methods):")
    print("-" * 50)
    for _, row in c_series.iterrows():
        method = row['Cycle_ID'].replace('C', '')
        if method == '0':
            method = 'NoPreprocess'
        elif method == '1':
            method = 'Bandpass'
        elif method == '2':
            method = 'SpectralGating'
        elif method == '3':
            method = 'HighPass'
        elif method == '4':
            method = 'PeakNormalize'
        
        print(f"C{row['Cycle_ID'][1:]}: {method:<15} - Quality: {row['Quality_Score']:.3f}")
    
    # Find top performers
    print("\n" + "="*80)
    print("TOP PERFORMERS BY SERIES")
    print("="*80)
    
    # A-Series top 3
    a_top3 = a_series.head(3)
    print("\nA-Series Top 3:")
    for i, (_, row) in enumerate(a_top3.iterrows(), 1):
        method = row['Cycle_ID'].replace('A', '')
        if method == '0':
            method = 'NoPreprocess'
        elif method == '1':
            method = 'Bandpass'
        elif method == '2':
            method = 'SpectralGating'
        elif method == '3':
            method = 'HighPass'
        elif method == '4':
            method = 'PeakNormalize'
        
        print(f"  {i}. A{row['Cycle_ID'][1:]}: {method} - Quality: {row['Quality_Score']:.3f}")
    
    # C-Series top 3
    c_top3 = c_series.head(3)
    print("\nC-Series Top 3:")
    for i, (_, row) in enumerate(c_top3.iterrows(), 1):
        method = row['Cycle_ID'].replace('C', '')
        if method == '0':
            method = 'NoPreprocess'
        elif method == '1':
            method = 'Bandpass'
        elif method == '2':
            method = 'SpectralGating'
        elif method == '3':
            method = 'HighPass'
        elif method == '4':
            method = 'PeakNormalize'
        
        print(f"  {i}. C{row['Cycle_ID'][1:]}: {method} - Quality: {row['Quality_Score']:.3f}")
    
    # Suggest better combinations
    print("\n" + "="*80)
    print("SUGGESTED BETTER COMBINATIONS")
    print("="*80)
    
    print("\nB-Series Suggestions (NoSeg + Smart Combinations):")
    print("-" * 60)
    
    # Current B-Series was based on old rankings
    print("Current B-Series (based on old rankings):")
    print("  B0: NoSeg + Bandpass + SpectralGating")
    print("  B1: NoSeg + PeakNormalize + Bandpass") 
    print("  B2: NoSeg + Full Pipeline")
    
    print("\nSuggested B-Series (based on new rankings):")
    print("  B0_new: NoSeg + PeakNormalize + HighPass")
    print("    - Combines A4 (PeakNormalize, 0.360) + A3 (HighPass, 0.264)")
    print("    - Both are amplitude/frequency focused methods")
    
    print("  B1_new: NoSeg + PeakNormalize + Bandpass")
    print("    - Combines A4 (PeakNormalize, 0.360) + A1 (Bandpass, 0.411)")
    print("    - Amplitude normalization + frequency focusing")
    
    print("  B2_new: NoSeg + Bandpass + HighPass")
    print("    - Combines A1 (Bandpass, 0.411) + A3 (HighPass, 0.264)")
    print("    - Both are frequency-focused methods")
    
    print("\nD-Series Suggestions (Seg + Smart Combinations):")
    print("-" * 60)
    
    print("Current D-Series (based on old rankings):")
    print("  D0: Seg + HighPass + PeakNormalize")
    print("  D1: Seg + HighPass + Bandpass")
    print("  D2: Seg + Full Pipeline")
    
    print("\nSuggested D-Series (based on new rankings):")
    print("  D0_new: Seg + PeakNormalize + HighPass")
    print("    - Combines C4 (PeakNormalize, 0.462) + C3 (HighPass, 0.442)")
    print("    - Both are top C-Series performers")
    
    print("  D1_new: Seg + PeakNormalize + Bandpass")
    print("    - Combines C4 (PeakNormalize, 0.462) + C1 (Bandpass, 0.571)")
    print("    - Amplitude normalization + frequency focusing")
    
    print("  D2_new: Seg + HighPass + Bandpass")
    print("    - Combines C3 (HighPass, 0.442) + C1 (Bandpass, 0.571)")
    print("    - Both are frequency-focused methods")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    print("1. C-Series (Segmented) > A-Series (NoSeg) overall")
    print("   - Top C-Series: C4 (0.462), C3 (0.442)")
    print("   - Top A-Series: A0 (0.458), A1 (0.411)")
    
    print("\n2. PeakNormalize is consistently good:")
    print("   - C4 (Seg + PeakNormalize): 0.462")
    print("   - A4 (NoSeg + PeakNormalize): 0.360")
    
    print("\n3. HighPass works well with segmentation:")
    print("   - C3 (Seg + HighPass): 0.442")
    print("   - A3 (NoSeg + HighPass): 0.264")
    
    print("\n4. Bandpass is good for NoSeg but excellent for Seg:")
    print("   - C1 (Seg + Bandpass): 0.571 (but this might be from old evaluation)")
    print("   - A1 (NoSeg + Bandpass): 0.411")
    
    print("\n5. Recommended Priority Order:")
    print("   - D0_new: Seg + PeakNormalize + HighPass (combines top 2 C-Series)")
    print("   - D1_new: Seg + PeakNormalize + Bandpass (amplitude + frequency)")
    print("   - B0_new: NoSeg + PeakNormalize + HighPass (best A-Series combination)")

def main():
    print("Analyzing individual method rankings to suggest better combinations...")
    
    # Analyze individual methods
    best_per_cycle = analyze_individual_methods()
    
    # Suggest better combinations
    suggest_better_combinations(best_per_cycle)

if __name__ == "__main__":
    main()
