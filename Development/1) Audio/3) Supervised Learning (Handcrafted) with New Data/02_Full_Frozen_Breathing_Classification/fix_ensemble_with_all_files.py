#!/usr/bin/env python3
"""
Fix Ensemble Models to Process ALL Files
========================================
Uses the correct Excel parsing method to process all 13 available files
"""

import pandas as pd
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

def parse_excel_correctly():
    """Parse Excel with the CORRECT method that finds all 14 files."""
    
    excel_file = "/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/ML test sound list breathing info.xlsx"
    
    print("📋 Parsing Excel with CORRECT method (finds all files)...")
    
    sheet1 = pd.read_excel(excel_file, sheet_name='Sheet1', header=None)
    healthy = pd.read_excel(excel_file, sheet_name='Healthy', header=None)
    
    all_files_data = {}
    
    for sheet_name, df in [('Sheet1', sheet1), ('Healthy', healthy)]:
        print(f"   Processing {sheet_name} sheet...")
        
        for idx in range(df.shape[0]):
            row = df.iloc[idx]
            
            if pd.notna(row.iloc[1]) and isinstance(row.iloc[1], str):
                excel_filename = str(row.iloc[1])
                
                # Check if this looks like a filename
                if any(pattern in excel_filename for pattern in ['KP', 'H0', 'WEBSS']):
                    if idx + 1 < df.shape[0]:
                        timestamp_row = df.iloc[idx + 1]
                        header_row = row
                        
                        all_events = []
                        
                        # Extract ALL events
                        for col_idx in range(2, df.shape[1]):
                            header_val = header_row.iloc[col_idx]
                            timestamp_val = timestamp_row.iloc[col_idx]
                            
                            if pd.notna(timestamp_val) and isinstance(timestamp_val, (int, float)):
                                timestamp = float(timestamp_val)
                                if 0 <= timestamp <= 60:
                                    event_type = 'non_breathing'
                                    if pd.notna(header_val) and isinstance(header_val, str):
                                        if 'Inhale' in header_val:
                                            event_type = 'inhale'
                                        elif 'Exhale' in header_val:
                                            event_type = 'exhale'
                                    
                                    all_events.append({'time': timestamp, 'type': event_type})
                        
                        if all_events:
                            all_events.sort(key=lambda x: x['time'])
                            
                            # Create breathing periods
                            breathing_periods = []
                            for i in range(len(all_events) - 1):
                                current = all_events[i]
                                next_event = all_events[i + 1]
                                
                                period = {
                                    'start': current['time'],
                                    'end': next_event['time'],
                                    'type': 'breathing' if current['type'] in ['inhale', 'exhale'] else 'non_breathing'
                                }
                                breathing_periods.append(period)
                            
                            all_files_data[excel_filename] = {
                                'breathing_events': all_events,
                                'complete_timeline': breathing_periods,
                                'condition': 'Healthy' if sheet_name == 'Healthy' else 'Pathological'
                            }
                            
                            print(f"     {excel_filename}: {len(all_events)} breathing events")
    
    print(f"✅ Correctly parsed {len(all_files_data)} files")
    return all_files_data

def fix_ensemble_models():
    """Fix ensemble models to process all available files."""
    
    print("🔧 FIXING ENSEMBLE MODELS TO PROCESS ALL FILES")
    print("=" * 50)
    
    # Parse Excel correctly
    excel_data = parse_excel_correctly()
    
    # Check audio file availability
    audio_dir = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/RAW sound_ML test sound list")
    
    available_files = []
    for excel_filename in excel_data.keys():
        # Find matching audio file
        audio_file = None
        for audio_path in audio_dir.glob("*.wav"):
            if excel_filename in audio_path.name or audio_path.stem in excel_filename:
                audio_file = audio_path
                break
        
        if audio_file:
            available_files.append(excel_filename)
            print(f"✅ {excel_filename} → {audio_file.name}")
        else:
            print(f"❌ {excel_filename} → NO AUDIO FILE")
    
    print(f"\n📊 SUMMARY:")
    print(f"Excel entries: {len(excel_data)}")
    print(f"Available audio: {len(available_files)}")
    print(f"Should process: {len(available_files)} files")
    
    return len(available_files)

def main():
    """Main function to check file processing."""
    
    print("🔍 INVESTIGATING FILE PROCESSING ISSUE")
    print("=" * 40)
    
    expected_files = fix_ensemble_models()
    
    print(f"\n🎯 CONCLUSION:")
    print(f"• Should process: {expected_files} files")
    print(f"• Latest ensemble processed: 9 files")
    print(f"• Missing: {expected_files - 9} files")
    
    if expected_files > 9:
        print(f"\n🚨 CONFIRMED: Ensemble models are missing {expected_files - 9} files!")
        print("✅ Need to rerun ensemble with corrected Excel parsing")
    else:
        print(f"\n✅ File count is correct")

if __name__ == "__main__":
    main()
