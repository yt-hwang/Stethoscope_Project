#!/usr/bin/env python3
"""
Analyze ML Test Sound List Breathing Info
=========================================

This script reads and analyzes the breathing information Excel file
to understand the metadata and labels for our audio dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def analyze_breathing_info():
    """Analyze the breathing info Excel file."""
    
    # File paths
    ml_breathing_file = "/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/ML test sound list breathing info.xlsx"
    main_breathing_file = "/Users/yunhwang/Desktop/Stethoscope_Project/Breathing Info.xlsx"
    
    print("🔍 ANALYZING BREATHING INFO FILES")
    print("=" * 40)
    
    # Try to read the ML test breathing info file
    try:
        print(f"\n📊 Reading: {Path(ml_breathing_file).name}")
        
        # Try to read all sheets
        excel_file = pd.ExcelFile(ml_breathing_file)
        print(f"📋 Found {len(excel_file.sheet_names)} sheet(s): {excel_file.sheet_names}")
        
        all_data = {}
        
        for sheet_name in excel_file.sheet_names:
            print(f"\n📄 Analyzing sheet: '{sheet_name}'")
            df = pd.read_excel(ml_breathing_file, sheet_name=sheet_name)
            
            print(f"   📏 Shape: {df.shape}")
            print(f"   📋 Columns: {list(df.columns)}")
            
            # Show first few rows
            print(f"   📊 First 5 rows:")
            print(df.head().to_string(max_cols=10))
            
            # Store data
            all_data[sheet_name] = {
                'shape': df.shape,
                'columns': list(df.columns),
                'data': df
            }
            
            # Check for filename matches with our audio files
            audio_files = [
                "H001.wav", "H002.wav", "H003.wav", "H004.wav",
                "KP001_WWS.wav", "KP002_WWS.wav", "KP003_WWS_1.wav", "KP003_WWS_2.wav",
                "KP004_WWS.wav", "KP005_WWS.wav", "KP006_WWS.wav", "KP007_WWS.wav",
                "KP008_WWS.wav", "KP009_WWS.wav", "KP010_WWS.wav", "KP011_WWS.wav",
                "KP012_WWS_1.wav", "KP012_WWS_2.wav",
                "WEBSS-002 TP 3_seg-1.wav", "WEBSS-002 TP 4_seg-1.wav",
                "WEBSS-003 TP1_seg-1.wav", "WEBSS-003 TP3 _seg-1.wav",
                "WEBSS-005 TP1_seg-1.wav", "WEBSS-005 TP6_seg-1.wav",
                "WEBSS-006 TP3_seg-1.wav", "WEBSS-006 TP8_seg-1.wav",
                "WEBSS-007 TP1_seg-1.wav", "WEBSS-007 TP4_seg-1.wav", "WEBSS-007 TP7_seg-1.wav"
            ]
            
            # Look for filename-like columns
            filename_cols = []
            for col in df.columns:
                if any(keyword in str(col).lower() for keyword in ['file', 'name', 'id', 'sound']):
                    filename_cols.append(col)
            
            if filename_cols:
                print(f"   📁 Potential filename columns: {filename_cols}")
                for col in filename_cols:
                    unique_values = df[col].unique()[:10]  # First 10 unique values
                    print(f"      {col}: {unique_values}")
            
            # Look for label-like columns
            label_cols = []
            for col in df.columns:
                if any(keyword in str(col).lower() for keyword in ['label', 'class', 'category', 'type', 'diagnosis']):
                    label_cols.append(col)
            
            if label_cols:
                print(f"   🏷️ Potential label columns: {label_cols}")
                for col in label_cols:
                    unique_values = df[col].unique()
                    print(f"      {col}: {unique_values}")
        
        return all_data
        
    except Exception as e:
        print(f"❌ Error reading ML breathing info: {e}")
    
    # Try the main breathing info file
    try:
        print(f"\n📊 Reading: {Path(main_breathing_file).name}")
        
        excel_file = pd.ExcelFile(main_breathing_file)
        print(f"📋 Found {len(excel_file.sheet_names)} sheet(s): {excel_file.sheet_names}")
        
        for sheet_name in excel_file.sheet_names:
            print(f"\n📄 Analyzing sheet: '{sheet_name}'")
            df = pd.read_excel(main_breathing_file, sheet_name=sheet_name)
            
            print(f"   📏 Shape: {df.shape}")
            print(f"   📋 Columns: {list(df.columns)}")
            print(f"   📊 First 5 rows:")
            print(df.head().to_string(max_cols=10))
        
    except Exception as e:
        print(f"❌ Error reading main breathing info: {e}")

def create_breathing_info_summary():
    """Create a summary of breathing info findings."""
    
    print("\n🎯 BREATHING INFO ANALYSIS SUMMARY")
    print("=" * 40)
    
    # Get our current audio files
    audio_dir = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/RAW sound_ML test sound list")
    audio_files = list(audio_dir.glob("*.wav"))
    audio_files.sort()
    
    print(f"\n📁 Our Audio Dataset ({len(audio_files)} files):")
    file_patterns = {}
    for audio_file in audio_files:
        if audio_file.name.startswith('H'):
            file_patterns.setdefault('H-series', []).append(audio_file.name)
        elif audio_file.name.startswith('KP'):
            file_patterns.setdefault('KP-series', []).append(audio_file.name)
        elif audio_file.name.startswith('WEBSS'):
            file_patterns.setdefault('WEBSS-series', []).append(audio_file.name)
    
    for pattern, files in file_patterns.items():
        print(f"   {pattern}: {len(files)} files")
        print(f"      Examples: {files[:3]}{'...' if len(files) > 3 else ''}")
    
    print(f"\n💡 NEXT STEPS:")
    print("1. Extract breathing info labels from Excel file")
    print("2. Match labels with our 29 audio files")
    print("3. Use labels for supervised learning with OPERA-CT")
    print("4. Compare supervised vs unsupervised performance")

if __name__ == "__main__":
    breathing_data = analyze_breathing_info()
    create_breathing_info_summary()
