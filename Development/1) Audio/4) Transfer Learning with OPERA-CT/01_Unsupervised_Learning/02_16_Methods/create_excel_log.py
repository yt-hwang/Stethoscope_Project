#!/usr/bin/env python3
"""
Create Excel logging for OPERA-CT 16 methods experiment.
"""

import pandas as pd
import json
from pathlib import Path

def create_excel_log():
    """Create comprehensive Excel log for OPERA-CT 16 methods."""
    
    # Read the CSV data
    df = pd.read_csv('OPERA_CT_16_Methods_Complete_Results.csv')
    
    # Read summary statistics
    with open('OPERA_CT_16_Methods_Summary_Stats.json', 'r') as f:
        summary_stats = json.load(f)
    
    # Create Excel file
    excel_path = 'OPERA_CT_16_Methods_Complete_Log.xlsx'
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        
        # Sheet 1: Complete Results
        df.to_excel(writer, sheet_name='Complete_Results', index=False)
        
        # Sheet 2: Top Performers
        top_10 = df.head(10)
        top_10.to_excel(writer, sheet_name='Top_10_Methods', index=False)
        
        # Sheet 3: Summary Statistics
        summary_df = pd.DataFrame([
            ['Total Methods Tested', summary_stats['Total_Methods']],
            ['Baseline Score (A0)', f"{summary_stats['Baseline_Score']:.3f}"],
            ['Best Score', f"{summary_stats['Best_Score']:.3f}"],
            ['Improvement Over Baseline', f"+{summary_stats['Improvement_Percent']:.1f}%"],
            ['Best Method', summary_stats['Best_Method']],
            ['Best Method Name', summary_stats['Best_Method_Name']],
            ['Segmentation Impact', f"+{summary_stats['Segmentation_Impact']:.3f}"],
            ['Bandpass Impact', f"+{summary_stats['Bandpass_Impact']:.3f}"]
        ], columns=['Metric', 'Value'])
        summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
        
        # Sheet 4: Method Categories
        category_analysis = []
        
        # A-Series (Individual NoSeg)
        a_series = df[df['Method'].str.startswith('A')]
        category_analysis.append(['A-Series (Individual NoSeg)', len(a_series), f"{a_series['Best_Silhouette'].mean():.3f}", f"{a_series['Best_Silhouette'].max():.3f}"])
        
        # B-Series (Combination NoSeg)
        b_series = df[df['Method'].str.startswith('B')]
        category_analysis.append(['B-Series (Combination NoSeg)', len(b_series), f"{b_series['Best_Silhouette'].mean():.3f}", f"{b_series['Best_Silhouette'].max():.3f}"])
        
        # C-Series (Individual Seg)
        c_series = df[df['Method'].str.startswith('C')]
        category_analysis.append(['C-Series (Individual Seg)', len(c_series), f"{c_series['Best_Silhouette'].mean():.3f}", f"{c_series['Best_Silhouette'].max():.3f}"])
        
        # D-Series (Combination Seg)
        d_series = df[df['Method'].str.startswith('D')]
        category_analysis.append(['D-Series (Combination Seg)', len(d_series), f"{d_series['Best_Silhouette'].mean():.3f}", f"{d_series['Best_Silhouette'].max():.3f}"])
        
        category_df = pd.DataFrame(category_analysis, columns=['Category', 'Count', 'Mean_Score', 'Best_Score'])
        category_df.to_excel(writer, sheet_name='Category_Analysis', index=False)
        
        # Sheet 5: Preprocessing Impact
        preprocessing_analysis = []
        
        # Segmentation impact
        segmented = df[df['Name'].str.contains('Seg')]
        non_segmented = df[~df['Name'].str.contains('Seg')]
        preprocessing_analysis.append(['Segmentation', len(segmented), f"{segmented['Best_Silhouette'].mean():.3f}", f"{non_segmented['Best_Silhouette'].mean():.3f}", f"{(segmented['Best_Silhouette'].mean() - non_segmented['Best_Silhouette'].mean()):.3f}"])
        
        # Bandpass impact
        bandpass = df[df['Name'].str.contains('Bandpass')]
        no_bandpass = df[~df['Name'].str.contains('Bandpass')]
        preprocessing_analysis.append(['Bandpass Filtering', len(bandpass), f"{bandpass['Best_Silhouette'].mean():.3f}", f"{no_bandpass['Best_Silhouette'].mean():.3f}", f"{(bandpass['Best_Silhouette'].mean() - no_bandpass['Best_Silhouette'].mean()):.3f}"])
        
        # HighPass impact
        highpass = df[df['Name'].str.contains('HighPass')]
        no_highpass = df[~df['Name'].str.contains('HighPass')]
        preprocessing_analysis.append(['HighPass Filtering', len(highpass), f"{highpass['Best_Silhouette'].mean():.3f}", f"{no_highpass['Best_Silhouette'].mean():.3f}", f"{(highpass['Best_Silhouette'].mean() - no_highpass['Best_Silhouette'].mean()):.3f}"])
        
        # PeakNormalize impact
        peaknorm = df[df['Name'].str.contains('PeakNormalize')]
        no_peaknorm = df[~df['Name'].str.contains('PeakNormalize')]
        preprocessing_analysis.append(['Peak Normalization', len(peaknorm), f"{peaknorm['Best_Silhouette'].mean():.3f}", f"{no_peaknorm['Best_Silhouette'].mean():.3f}", f"{(peaknorm['Best_Silhouette'].mean() - no_peaknorm['Best_Silhouette'].mean()):.3f}"])
        
        preprocessing_df = pd.DataFrame(preprocessing_analysis, columns=['Preprocessing_Step', 'Count', 'With_Step_Mean', 'Without_Step_Mean', 'Impact'])
        preprocessing_df.to_excel(writer, sheet_name='Preprocessing_Impact', index=False)
    
    print(f"📊 Excel log created: {excel_path}")
    print(f"   - Complete Results: {len(df)} methods")
    print(f"   - Top 10 Methods: Best performers")
    print(f"   - Summary Statistics: Key metrics")
    print(f"   - Category Analysis: A/B/C/D series comparison")
    print(f"   - Preprocessing Impact: Individual step analysis")

if __name__ == "__main__":
    create_excel_log()
