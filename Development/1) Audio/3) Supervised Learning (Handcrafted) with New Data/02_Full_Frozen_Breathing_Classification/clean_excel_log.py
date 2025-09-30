#!/usr/bin/env python3
"""
Clean Excel Log - Remove Ensemble Entries
==========================================
Removes all ensemble-related entries from the master Excel log
"""

import pandas as pd
import os

def clean_excel_log():
    """Remove ensemble entries from master Excel log"""
    
    excel_file = "Classification_Experiments_Master_Log.xlsx"
    
    if not os.path.exists(excel_file):
        print("❌ Master Excel log not found")
        return
    
    print("🧹 Cleaning Master Excel Log")
    print("=" * 30)
    
    # Read all sheets
    excel_data = pd.read_excel(excel_file, sheet_name=None)
    
    # Clean each sheet
    for sheet_name, df in excel_data.items():
        print(f"📋 Cleaning sheet: {sheet_name}")
        
        if 'Experiment_ID' in df.columns:
            # Remove rows with ensemble experiment IDs
            original_count = len(df)
            df_cleaned = df[~df['Experiment_ID'].str.contains('ENS_', na=False)]
            removed_count = original_count - len(df_cleaned)
            
            if removed_count > 0:
                print(f"  🗑️ Removed {removed_count} ensemble entries")
                excel_data[sheet_name] = df_cleaned
            else:
                print(f"  ✅ No ensemble entries found")
        else:
            print(f"  ℹ️ No Experiment_ID column, skipping")
    
    # Write cleaned data back to Excel
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        for sheet_name, df in excel_data.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"\n✅ Master Excel log cleaned")
    print(f"📁 File: {excel_file}")

if __name__ == "__main__":
    clean_excel_log()
