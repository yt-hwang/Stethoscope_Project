#!/usr/bin/env python3
"""
Update Excel tracking system with Step 1 results
"""

import pandas as pd
import json
from pathlib import Path

def update_excel_tracking():
    """
    Update the Excel tracking system with Step 1 data loading results
    """
    
    # Load summary data
    with open('/Users/yunhwang/Desktop/Stethoscope_Project/outputs/step1_data_summary/summary.json', 'r') as f:
        summary = json.load(f)
    
    # Create Step1_Data sheet data
    step1_data = {
        'Metric': [
            'Total Files',
            'Total Duration (minutes)',
            'Average Duration (seconds)',
            'Duration Range (seconds)',
            'Sample Rate (Hz)',
            'Average RMS',
            'Average Zero-Crossing Rate',
            'WEBSS002 Files',
            'WEBSS003 Files', 
            'WEBSS004 Files',
            'WEBSS005 Files',
            'WEBSS006 Files',
            'WEBSS007 Files'
        ],
        'Value': [
            summary['total_files'],
            f"{summary['total_duration_min']:.2f}",
            f"{summary['avg_duration_sec']:.2f}",
            f"{summary['min_duration_sec']:.2f} - {summary['max_duration_sec']:.2f}",
            summary['sample_rate'],
            f"{summary['avg_rms']:.4f}",
            f"{summary['avg_zero_crossing_rate']:.4f}",
            summary['file_distribution']['WEBSS002'],
            summary['file_distribution']['WEBSS003'],
            summary['file_distribution']['WEBSS004'],
            summary['file_distribution']['WEBSS005'],
            summary['file_distribution']['WEBSS006'],
            summary['file_distribution']['WEBSS007']
        ]
    }
    
    # Create DataFrame
    df_step1 = pd.DataFrame(step1_data)
    
    # Try to read existing Excel file or create new one
    excel_path = '/Users/yunhwang/Desktop/Stethoscope_Project/Experiment_Tracking_System_Final.xlsx'
    
    try:
        # Try to read existing Excel file
        with pd.ExcelWriter(excel_path, mode='a', if_sheet_exists='replace', engine='openpyxl') as writer:
            df_step1.to_excel(writer, sheet_name='Step1_Data', index=False)
        print(f"Updated existing Excel file: {excel_path}")
    except FileNotFoundError:
        # Create new Excel file
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df_step1.to_excel(writer, sheet_name='Step1_Data', index=False)
        print(f"Created new Excel file: {excel_path}")
    except Exception as e:
        print(f"Error updating Excel file: {e}")
        # Create backup approach - save as CSV
        csv_path = '/Users/yunhwang/Desktop/Stethoscope_Project/Step1_Data_Backup.csv'
        df_step1.to_csv(csv_path, index=False)
        print(f"Saved backup as CSV: {csv_path}")
    
    # Development column entry
    development_entry = {
        'Code_Used': 'step1_data_loading.py - librosa for audio loading, resampling to 16kHz mono',
        'Output_Files': [
            'outputs/step1_data_summary/summary.json',
            'outputs/step1_data_summary/file_details.csv', 
            'outputs/step1_data_summary/waveform_examples.png'
        ],
        'Visualizations': 'waveform_examples.png - 6 diverse waveform examples showing different durations',
        'Excel_Logging': 'Step1_Data sheet created with file summary metrics',
        'Notes': f'Successfully processed {summary["total_files"]} audio files from 6 patients. All files resampled to 16kHz mono. Average duration ~60 seconds.'
    }
    
    print("\nDevelopment Column Entry:")
    print("="*50)
    for key, value in development_entry.items():
        print(f"{key}: {value}")
    
    return development_entry

if __name__ == "__main__":
    development_entry = update_excel_tracking()
