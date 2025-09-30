#!/usr/bin/env python3
"""
Create comprehensive Excel comparison between Handcrafted and OPERA-CT features.
"""

import pandas as pd
import json
from pathlib import Path

def create_comparison_excel():
    """Create comprehensive Excel comparison."""
    
    # Read comparison data
    df_comparison = pd.read_csv('Handcrafted_vs_OPERA_CT_Comparison.csv')
    
    # Read summary statistics
    with open('Comparison_Summary.json', 'r') as f:
        summary = json.load(f)
    
    # Separate handcrafted and OPERA-CT results
    df_handcrafted = df_comparison[df_comparison['Feature_Type'] == 'Handcrafted'].copy()
    df_opera_ct = df_comparison[df_comparison['Feature_Type'] == 'OPERA-CT'].copy()
    
    # Create Excel file
    excel_path = 'Handcrafted_vs_OPERA_CT_Complete_Comparison.xlsx'
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        
        # Sheet 1: Complete Comparison
        df_comparison.to_excel(writer, sheet_name='Complete_Comparison', index=False)
        
        # Sheet 2: Handcrafted Results
        df_handcrafted.to_excel(writer, sheet_name='Handcrafted_Results', index=False)
        
        # Sheet 3: OPERA-CT Results
        df_opera_ct.to_excel(writer, sheet_name='OPERA_CT_Results', index=False)
        
        # Sheet 4: Summary Statistics
        summary_data = [
            ['Metric', 'Value'],
            ['Handcrafted Best Score', f"{summary['Handcrafted_Best']:.3f}"],
            ['OPERA-CT Best Score', f"{summary['OPERA_CT_Best']:.3f}"],
            ['Performance Gap', f"{summary['Performance_Gap']:.3f}"],
            ['Performance Gap (%)', f"{summary['Performance_Gap_Percent']:.1f}%"],
            ['Handcrafted Advantage', summary['Handcrafted_Advantage']],
            ['Handcrafted Mean Score', f"{summary['Handcrafted_Mean']:.3f}"],
            ['OPERA-CT Mean Score', f"{summary['OPERA_CT_Mean']:.3f}"]
        ]
        
        summary_df = pd.DataFrame(summary_data[1:], columns=summary_data[0])
        summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
        
        # Sheet 5: Top 10 Comparison
        top_10_handcrafted = df_handcrafted.head(10)
        top_10_opera_ct = df_opera_ct.head(10)
        
        # Create side-by-side comparison
        comparison_data = []
        max_len = max(len(top_10_handcrafted), len(top_10_opera_ct))
        
        for i in range(max_len):
            h_row = top_10_handcrafted.iloc[i] if i < len(top_10_handcrafted) else None
            o_row = top_10_opera_ct.iloc[i] if i < len(top_10_opera_ct) else None
            
            comparison_data.append({
                'Rank': i + 1,
                'Handcrafted_Method': h_row['Method'] if h_row is not None else '',
                'Handcrafted_Score': h_row['Best_Silhouette'] if h_row is not None else '',
                'OPERA_CT_Method': o_row['Method'] if o_row is not None else '',
                'OPERA_CT_Score': o_row['Best_Silhouette'] if o_row is not None else ''
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_excel(writer, sheet_name='Top_10_Comparison', index=False)
        
        # Sheet 6: Method-by-Method Comparison
        method_comparison = []
        
        # Get all unique methods
        all_methods = set(df_handcrafted['Method'].unique()) | set(df_opera_ct['Method'].unique())
        
        for method in sorted(all_methods):
            h_method = df_handcrafted[df_handcrafted['Method'] == method]
            o_method = df_opera_ct[df_opera_ct['Method'] == method]
            
            h_score = h_method['Best_Silhouette'].iloc[0] if len(h_method) > 0 else None
            o_score = o_method['Best_Silhouette'].iloc[0] if len(o_method) > 0 else None
            
            method_comparison.append({
                'Method': method,
                'Handcrafted_Score': h_score,
                'OPERA_CT_Score': o_score,
                'Difference': h_score - o_score if h_score is not None and o_score is not None else None,
                'Handcrafted_Wins': h_score > o_score if h_score is not None and o_score is not None else None
            })
        
        method_df = pd.DataFrame(method_comparison)
        method_df.to_excel(writer, sheet_name='Method_by_Method', index=False)
        
        # Sheet 7: Key Insights
        insights_data = [
            ['Key Insight', 'Value'],
            ['Best Overall Method', f"B2 (Handcrafted) - {summary['Handcrafted_Best']:.3f}"],
            ['Best OPERA-CT Method', f"D1 - {summary['OPERA_CT_Best']:.3f}"],
            ['Performance Advantage', f"Handcrafted features outperform OPERA-CT by {summary['Performance_Gap_Percent']:.1f}%"],
            ['Segmentation Impact (Handcrafted)', 'Mixed - some methods benefit, others don\'t'],
            ['Segmentation Impact (OPERA-CT)', 'Consistent improvement with segmentation'],
            ['Preprocessing Impact (Handcrafted)', 'Full pipeline (B2) works best'],
            ['Preprocessing Impact (OPERA-CT)', 'HighPass + Bandpass (D1) works best'],
            ['Feature Complexity', 'Handcrafted: 13 features, OPERA-CT: 768 features'],
            ['Computational Cost', 'Handcrafted: Lower, OPERA-CT: Higher'],
            ['Domain Knowledge', 'Handcrafted: High, OPERA-CT: Learned from data']
        ]
        
        insights_df = pd.DataFrame(insights_data[1:], columns=insights_data[0])
        insights_df.to_excel(writer, sheet_name='Key_Insights', index=False)
    
    print(f"📊 Comprehensive comparison Excel created: {excel_path}")
    print(f"   - Complete Comparison: Side-by-side results")
    print(f"   - Handcrafted Results: All handcrafted methods")
    print(f"   - OPERA-CT Results: All OPERA-CT methods")
    print(f"   - Summary Statistics: Key metrics")
    print(f"   - Top 10 Comparison: Best performers")
    print(f"   - Method-by-Method: Direct comparisons")
    print(f"   - Key Insights: Analysis and conclusions")

if __name__ == "__main__":
    create_comparison_excel()

