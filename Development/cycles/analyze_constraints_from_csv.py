#!/usr/bin/env python3
"""
Analyze cluster balance validation constraints from CSV data
"""

import pandas as pd
import numpy as np

def analyze_constraints_from_csv():
    """Analyze validation constraints from the CSV results."""
    
    # Load the comprehensive results
    df = pd.read_csv('comprehensive_re_evaluation_results.csv')
    
    print("="*80)
    print("CLUSTER BALANCE VALIDATION ANALYSIS")
    print("="*80)
    
    print(f"\nCurrent Validation Constraints:")
    print("-" * 40)
    print("• Min cluster size: 2% of data (relaxed from 5%)")
    print("• Max Gini coefficient: 0.8 (relaxed from 0.7)")
    print("• Min silhouette threshold: 0.2 (relaxed from 0.3)")
    print("• HDBSCAN noise fraction: max 0.5")
    
    # Analyze valid vs invalid runs
    total_runs = df['Total_Runs'].sum()
    valid_runs = df['Valid_Runs'].sum()
    invalid_runs = total_runs - valid_runs
    
    print(f"\nOverall Statistics:")
    print("-" * 40)
    print(f"Total runs: {total_runs}")
    print(f"Valid runs: {valid_runs} ({valid_runs/total_runs*100:.1f}%)")
    print(f"Invalid runs: {invalid_runs} ({invalid_runs/total_runs*100:.1f}%)")
    
    # Analyze by cycle
    print(f"\nPer-Cycle Statistics:")
    print("-" * 40)
    print(f"{'Cycle':<8} {'Total':<6} {'Valid':<6} {'Invalid':<7} {'Valid%':<8} {'Best Quality'}")
    print("-" * 80)
    
    for cycle_id in sorted(df['Cycle_ID'].unique()):
        cycle_data = df[df['Cycle_ID'] == cycle_id]
        total = cycle_data['Total_Runs'].iloc[0]
        valid = cycle_data['Valid_Runs'].iloc[0]
        invalid = total - valid
        valid_pct = valid / total * 100 if total > 0 else 0
        best_quality = cycle_data['Quality_Score'].max()
        
        print(f"{cycle_id:<8} {total:<6} {valid:<6} {invalid:<7} {valid_pct:<7.1f}% {best_quality:.3f}")
    
    # Analyze by representation and algorithm
    print(f"\nBy Representation:")
    print("-" * 40)
    rep_stats = df.groupby('Representation').agg({
        'Total_Runs': 'sum',
        'Valid_Runs': 'sum',
        'Quality_Score': 'max'
    }).reset_index()
    rep_stats['Valid_Pct'] = rep_stats['Valid_Runs'] / rep_stats['Total_Runs'] * 100
    
    for _, row in rep_stats.iterrows():
        print(f"{row['Representation']:<20} {row['Valid_Runs']:<3}/{row['Total_Runs']:<3} ({row['Valid_Pct']:.1f}%) Best: {row['Quality_Score']:.3f}")
    
    print(f"\nBy Algorithm:")
    print("-" * 40)
    algo_stats = df.groupby('Algorithm').agg({
        'Total_Runs': 'sum',
        'Valid_Runs': 'sum',
        'Quality_Score': 'max'
    }).reset_index()
    algo_stats['Valid_Pct'] = algo_stats['Valid_Runs'] / algo_stats['Total_Runs'] * 100
    
    for _, row in algo_stats.iterrows():
        print(f"{row['Algorithm']:<15} {row['Valid_Runs']:<3}/{row['Total_Runs']:<3} ({row['Valid_Pct']:.1f}%) Best: {row['Quality_Score']:.3f}")
    
    # Check if constraints are too tight
    print(f"\n" + "="*80)
    print("CONSTRAINT TIGHTNESS ANALYSIS")
    print("="*80)
    
    # Calculate rejection rate
    rejection_rate = invalid_runs / total_runs * 100
    
    if rejection_rate > 80:
        print("🔴 CONSTRAINTS ARE VERY TIGHT")
        print(f"   - {rejection_rate:.1f}% of runs rejected")
        print("   - Most clustering attempts are being invalidated")
    elif rejection_rate > 60:
        print("🟡 CONSTRAINTS ARE QUITE TIGHT")
        print(f"   - {rejection_rate:.1f}% of runs rejected")
        print("   - Many clustering attempts are being invalidated")
    elif rejection_rate > 40:
        print("🟡 CONSTRAINTS ARE MODERATELY TIGHT")
        print(f"   - {rejection_rate:.1f}% of runs rejected")
        print("   - Some clustering attempts are being invalidated")
    else:
        print("🟢 CONSTRAINTS SEEM REASONABLE")
        print(f"   - {rejection_rate:.1f}% of runs rejected")
        print("   - Most clustering attempts are valid")
    
    # Analyze specific issues
    print(f"\nSpecific Issues:")
    print("-" * 40)
    
    # Check cycles with very low valid rates
    low_valid_cycles = df[df['Valid_Runs'] / df['Total_Runs'] < 0.2]
    if len(low_valid_cycles) > 0:
        print(f"• {len(low_valid_cycles)} cycles have <20% valid runs:")
        for _, row in low_valid_cycles.iterrows():
            valid_pct = row['Valid_Runs'] / row['Total_Runs'] * 100
            print(f"  - {row['Cycle_ID']} {row['Representation']} {row['Algorithm']}: {valid_pct:.1f}% valid")
    
    # Check if we have enough valid results for analysis
    cycles_with_valid = df[df['Valid_Runs'] > 0]['Cycle_ID'].nunique()
    total_cycles = df['Cycle_ID'].nunique()
    
    print(f"• {cycles_with_valid}/{total_cycles} cycles have at least some valid runs")
    
    if cycles_with_valid < total_cycles * 0.5:
        print("🔴 TOO FEW CYCLES HAVE VALID RESULTS")
        print("   - Less than 50% of cycles have valid runs")
        print("   - Constraints may be too restrictive for meaningful analysis")
    elif cycles_with_valid < total_cycles * 0.8:
        print("🟡 SOME CYCLES LACK VALID RESULTS")
        print("   - Some cycles have no valid runs")
        print("   - Consider relaxing constraints slightly")
    else:
        print("🟢 MOST CYCLES HAVE VALID RESULTS")
        print("   - Most cycles have some valid runs")
        print("   - Constraints seem reasonable")
    
    # Suggest constraint adjustments
    print(f"\n" + "="*80)
    print("CONSTRAINT ADJUSTMENT RECOMMENDATIONS")
    print("="*80)
    
    if rejection_rate > 70:
        print("🔴 RECOMMEND SIGNIFICANT RELAXATION:")
        print("   • Min cluster size: 1% of data OR minimum 2 samples")
        print("   • Max Gini coefficient: 0.9")
        print("   • Min silhouette threshold: 0.1")
        print("   • HDBSCAN noise fraction: max 0.7")
    elif rejection_rate > 50:
        print("🟡 RECOMMEND MODERATE RELAXATION:")
        print("   • Min cluster size: 1.5% of data OR minimum 3 samples")
        print("   • Max Gini coefficient: 0.85")
        print("   • Min silhouette threshold: 0.15")
        print("   • HDBSCAN noise fraction: max 0.6")
    else:
        print("🟢 CONSTRAINTS SEEM APPROPRIATE:")
        print("   • Current constraints appear reasonable")
        print("   • Consider minor adjustments if needed")
    
    return {
        'total_runs': total_runs,
        'valid_runs': valid_runs,
        'invalid_runs': invalid_runs,
        'rejection_rate': rejection_rate,
        'cycles_with_valid': cycles_with_valid,
        'total_cycles': total_cycles
    }

def main():
    print("Analyzing cluster balance validation constraints from CSV data...")
    analyze_constraints_from_csv()

if __name__ == "__main__":
    main()
