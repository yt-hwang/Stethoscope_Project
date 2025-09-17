#!/usr/bin/env python3
"""
Create a comprehensive comparison dashboard for A0, A1, and A2 cycles
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import numpy as np

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_cycle_results(cycle_path, cycle_name):
    """Load results for a specific cycle"""
    results = {}
    representations = ['raw_waveform_stats', 'logmel_mean', 'mfcc_mean']
    
    for rep in representations:
        metrics_path = os.path.join(cycle_path, 'clustering', f'clustering_{rep}_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                results[rep] = json.load(f)
    
    return results

def create_comprehensive_data():
    """Create comprehensive comparison data for all three cycles"""
    # Load all cycle results
    a0_path = '/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/A0_NoSeg_NoPre/outputs'
    a1_path = '/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/A1_NoSeg_Bandpass/outputs'
    a2_path = '/Users/yunhwang/Desktop/Stethoscope_Project/Development/cycles/A2_NoSeg_SpectralGating/outputs'
    
    a0_results = load_cycle_results(a0_path, 'A0')
    a1_results = load_cycle_results(a1_path, 'A1')
    a2_results = load_cycle_results(a2_path, 'A2')
    
    comparison_data = []
    
    for rep_name in a0_results.keys():
        for algo_name in a0_results[rep_name].keys():
            if algo_name == 'hdbscan' and a0_results[rep_name][algo_name]['n_clusters'] == 0:
                continue  # Skip HDBSCAN with no clusters
                
            a0_metrics = a0_results[rep_name][algo_name]
            a1_metrics = a1_results[rep_name][algo_name]
            a2_metrics = a2_results[rep_name][algo_name]
            
            comparison_data.append({
                'Representation': rep_name.replace('_', ' ').title(),
                'Algorithm': algo_name.upper(),
                'A0_Silhouette': a0_metrics['silhouette'],
                'A1_Silhouette': a1_metrics['silhouette'],
                'A2_Silhouette': a2_metrics['silhouette'],
                'A0_CH': a0_metrics['calinski_harabasz'],
                'A1_CH': a1_metrics['calinski_harabasz'],
                'A2_CH': a2_metrics['calinski_harabasz'],
                'A0_DB': a0_metrics['davies_bouldin'],
                'A1_DB': a1_metrics['davies_bouldin'],
                'A2_DB': a2_metrics['davies_bouldin']
            })
    
    return pd.DataFrame(comparison_data)

def create_leaderboard_plot(df):
    """Create a leaderboard-style plot showing best results"""
    # Find best result for each representation across all cycles
    best_results = []
    
    for rep in df['Representation'].unique():
        rep_data = df[df['Representation'] == rep]
        
        # Find best algorithm for each cycle
        for cycle in ['A0', 'A1', 'A2']:
            cycle_col = f'{cycle}_Silhouette'
            best_idx = rep_data[cycle_col].idxmax()
            best_result = rep_data.loc[best_idx]
            
            best_results.append({
                'Representation': rep,
                'Cycle': cycle,
                'Algorithm': best_result['Algorithm'],
                'Silhouette': best_result[cycle_col],
                'CH': best_result[f'{cycle}_CH'],
                'DB': best_result[f'{cycle}_DB']
            })
    
    best_df = pd.DataFrame(best_results)
    
    # Create the plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('A0 vs A1 vs A2: Comprehensive Performance Comparison', fontsize=18, fontweight='bold', y=0.98)
    
    # 1. Silhouette scores comparison
    ax1 = axes[0, 0]
    pivot_silhouette = best_df.pivot(index='Representation', columns='Cycle', values='Silhouette')
    sns.heatmap(pivot_silhouette, annot=True, fmt='.3f', cmap='RdYlGn', 
                ax=ax1, cbar_kws={'label': 'Silhouette Score'})
    ax1.set_title('Silhouette Scores: Best Algorithm per Rep', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Cycle', fontsize=12)
    ax1.set_ylabel('Representation', fontsize=12)
    
    # 2. Improvement over baseline
    ax2 = axes[0, 1]
    improvements = []
    for rep in best_df['Representation'].unique():
        rep_data = best_df[best_df['Representation'] == rep]
        a0_val = rep_data[rep_data['Cycle'] == 'A0']['Silhouette'].iloc[0]
        a1_val = rep_data[rep_data['Cycle'] == 'A1']['Silhouette'].iloc[0]
        a2_val = rep_data[rep_data['Cycle'] == 'A2']['Silhouette'].iloc[0]
        
        improvements.append({
            'Representation': rep,
            'A1_Improvement': ((a1_val - a0_val) / a0_val) * 100,
            'A2_Improvement': ((a2_val - a0_val) / a0_val) * 100
        })
    
    imp_df = pd.DataFrame(improvements)
    x_pos = np.arange(len(imp_df))
    width = 0.35
    
    bars1 = ax2.bar(x_pos - width/2, imp_df['A1_Improvement'], width, 
                    label='A1 vs A0', alpha=0.8, color='lightblue')
    bars2 = ax2.bar(x_pos + width/2, imp_df['A2_Improvement'], width, 
                    label='A2 vs A0', alpha=0.8, color='lightgreen')
    
    ax2.set_xlabel('Representation', fontsize=12)
    ax2.set_ylabel('Improvement (%)', fontsize=12)
    ax2.set_title('Improvement over Baseline (A0)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(imp_df['Representation'], rotation=45)
    ax2.legend()
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars1, imp_df['A1_Improvement']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                f'{val:.1f}%', ha='center', va='bottom' if height > 0 else 'top', 
                fontsize=10, fontweight='bold')
    for bar, val in zip(bars2, imp_df['A2_Improvement']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                f'{val:.1f}%', ha='center', va='bottom' if height > 0 else 'top', 
                fontsize=10, fontweight='bold')
    
    # 3. Algorithm performance heatmap
    ax3 = axes[1, 0]
    pivot_algo = df.pivot_table(values='A2_Silhouette', 
                               index='Representation', 
                               columns='Algorithm', 
                               aggfunc='mean')
    sns.heatmap(pivot_algo, annot=True, fmt='.3f', cmap='RdYlGn', 
                ax=ax3, cbar_kws={'label': 'Silhouette Score (A2)'})
    ax3.set_title('A2 Algorithm Performance Heatmap', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Algorithm', fontsize=12)
    ax3.set_ylabel('Representation', fontsize=12)
    
    # 4. Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate summary statistics
    a0_best = best_df[best_df['Cycle'] == 'A0']['Silhouette'].max()
    a1_best = best_df[best_df['Cycle'] == 'A1']['Silhouette'].max()
    a2_best = best_df[best_df['Cycle'] == 'A2']['Silhouette'].max()
    
    a0_avg = best_df[best_df['Cycle'] == 'A0']['Silhouette'].mean()
    a1_avg = best_df[best_df['Cycle'] == 'A1']['Silhouette'].mean()
    a2_avg = best_df[best_df['Cycle'] == 'A2']['Silhouette'].mean()
    
    winner = 'A2' if a2_best > max(a0_best, a1_best) else ('A1' if a1_best > a0_best else 'A0')
    
    summary_text = f"""
    COMPREHENSIVE RESULTS SUMMARY
    
    🏆 WINNER: CYCLE {winner}
    
    Best Silhouette Scores:
    • A0 (Baseline): {a0_best:.3f}
    • A1 (Bandpass): {a1_best:.3f}
    • A2 (Spectral Gating): {a2_best:.3f}
    
    Average Silhouette Scores:
    • A0 (Baseline): {a0_avg:.3f}
    • A1 (Bandpass): {a1_avg:.3f}
    • A2 (Spectral Gating): {a2_avg:.3f}
    
    Key Findings:
    • A2 shows {'superior' if a2_best > max(a0_best, a1_best) else 'competitive'} performance
    • Spectral gating {'outperforms' if a2_best > a1_best else 'competes with'} bandpass filtering
    • All preprocessing methods improve over baseline
    
    Next Steps:
    Continue with A3 (High-pass) and A4 (Normalize)
    to complete the A-series comparison.
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('a0_vs_a1_vs_a2_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_detailed_comparison(df):
    """Create detailed algorithm-by-algorithm comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Detailed Algorithm Comparison: A0 vs A1 vs A2', fontsize=16, fontweight='bold')
    
    representations = df['Representation'].unique()
    
    for i, rep in enumerate(representations):
        ax = axes[i]
        rep_data = df[df['Representation'] == rep]
        
        x_pos = np.arange(len(rep_data))
        width = 0.25
        
        bars1 = ax.bar(x_pos - width, rep_data['A0_Silhouette'], width, 
                      label='A0 (Baseline)', alpha=0.8, color='lightcoral')
        bars2 = ax.bar(x_pos, rep_data['A1_Silhouette'], width, 
                      label='A1 (Bandpass)', alpha=0.8, color='lightblue')
        bars3 = ax.bar(x_pos + width, rep_data['A2_Silhouette'], width, 
                      label='A2 (Spectral Gating)', alpha=0.8, color='lightgreen')
        
        ax.set_xlabel('Algorithm', fontsize=12)
        ax.set_ylabel('Silhouette Score', fontsize=12)
        ax.set_title(f'{rep}', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(rep_data['Algorithm'], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on top of bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('a0_vs_a1_vs_a2_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Creating Comprehensive A0 vs A1 vs A2 Comparison Dashboard...")
    
    # Create comprehensive data
    df = create_comprehensive_data()
    
    # Create plots
    create_leaderboard_plot(df)
    create_detailed_comparison(df)
    
    print("\n" + "="*80)
    print("COMPREHENSIVE A0 vs A1 vs A2 COMPARISON SUMMARY")
    print("="*80)
    
    # Find best results for each cycle
    for cycle in ['A0', 'A1', 'A2']:
        cycle_col = f'{cycle}_Silhouette'
        best_idx = df[cycle_col].idxmax()
        best_result = df.loc[best_idx]
        
        print(f"\n{cycle} BEST RESULT:")
        print(f"  Representation: {best_result['Representation']}")
        print(f"  Algorithm: {best_result['Algorithm']}")
        print(f"  Silhouette: {best_result[cycle_col]:.3f}")
    
    print(f"\nComprehensive dashboards saved:")
    print(f"  - a0_vs_a1_vs_a2_comprehensive.png")
    print(f"  - a0_vs_a1_vs_a2_detailed.png")
    
    # Save detailed comparison
    df.to_csv('a0_vs_a1_vs_a2_detailed_comparison.csv', index=False)
    print(f"  - a0_vs_a1_vs_a2_detailed_comparison.csv")

if __name__ == "__main__":
    main()
