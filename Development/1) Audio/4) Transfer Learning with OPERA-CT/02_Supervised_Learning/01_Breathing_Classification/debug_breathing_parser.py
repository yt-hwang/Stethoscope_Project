#!/usr/bin/env python3
"""
Debug Breathing Info Parser
==========================

Let's debug and understand the Excel structure better to fix the parsing.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def debug_excel_structure():
    """Debug the Excel file structure to understand the format."""
    
    excel_file = "/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/ML test sound list breathing info.xlsx"
    
    print("🔍 DEBUGGING EXCEL STRUCTURE")
    print("=" * 40)
    
    # Read both sheets
    sheet1 = pd.read_excel(excel_file, sheet_name='Sheet1')
    healthy = pd.read_excel(excel_file, sheet_name='Healthy')
    
    print(f"\nSheet1 shape: {sheet1.shape}")
    print("First 10 rows, first 10 columns:")
    print(sheet1.iloc[:10, :10].to_string())
    
    print(f"\n" + "="*50)
    print(f"\nHealthy sheet shape: {healthy.shape}")
    print("First 10 rows, first 10 columns:")
    print(healthy.iloc[:10, :10].to_string())
    
    # Look for patterns in the data
    print(f"\n" + "="*50)
    print("ANALYZING PATTERNS:")
    
    # Check for file names in column 1
    print("\nUnique values in column 1 (Sheet1):")
    unique_vals = sheet1.iloc[:, 1].dropna().unique()[:20]
    for val in unique_vals:
        print(f"  {val}")
    
    print("\nUnique values in column 1 (Healthy):")
    unique_vals = healthy.iloc[:, 1].dropna().unique()[:20]
    for val in unique_vals:
        print(f"  {val}")
    
    # Look for numeric patterns (timestamps)
    print(f"\n" + "="*50)
    print("NUMERIC PATTERNS (potential timestamps):")
    
    for sheet_name, df in [('Sheet1', sheet1), ('Healthy', healthy)]:
        print(f"\n{sheet_name} - Numeric values in first few columns:")
        for col in range(min(6, df.shape[1])):
            numeric_vals = pd.to_numeric(df.iloc[:, col], errors='coerce').dropna()
            if len(numeric_vals) > 0:
                print(f"  Column {col}: {len(numeric_vals)} numeric values, range: {numeric_vals.min():.3f} - {numeric_vals.max():.3f}")

def create_simple_breathing_parser():
    """Create a simpler, more robust breathing parser."""
    
    excel_file = "/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/ML test sound list breathing info.xlsx"
    
    print("\n🔧 CREATING SIMPLE PARSER")
    print("=" * 30)
    
    # Read both sheets with proper header handling
    sheet1 = pd.read_excel(excel_file, sheet_name='Sheet1', header=None)
    healthy = pd.read_excel(excel_file, sheet_name='Healthy', header=None)
    
    breathing_data = {}
    
    for sheet_name, df in [('Sheet1', sheet1), ('Healthy', healthy)]:
        print(f"\nProcessing {sheet_name}...")
        
        current_file = None
        
        for idx in range(df.shape[0]):
            row = df.iloc[idx]
            
            # Look for filename in second column (index 1)
            if pd.notna(row.iloc[1]) and isinstance(row.iloc[1], str):
                filename = row.iloc[1]
                
                # Check if this looks like a filename
                if any(pattern in filename for pattern in ['KP', 'H0', 'WEBSS']):
                    current_file = filename
                    breathing_data[current_file] = {'timestamps': [], 'type': 'healthy' if sheet_name == 'Healthy' else 'pathological'}
                    print(f"  Found file: {current_file}")
                    continue
            
            # If we have a current file, look for timestamps in this row
            if current_file:
                for col_idx in range(2, min(df.shape[1], 20)):  # Check first 20 columns for timestamps
                    cell_val = row.iloc[col_idx]
                    if pd.notna(cell_val) and isinstance(cell_val, (int, float)):
                        # This might be a timestamp
                        timestamp = float(cell_val)
                        if 0 <= timestamp <= 60:  # Reasonable timestamp range
                            breathing_data[current_file]['timestamps'].append(timestamp)
    
    # Clean up and sort timestamps
    for filename, data in breathing_data.items():
        if data['timestamps']:
            data['timestamps'] = sorted(list(set(data['timestamps'])))  # Remove duplicates and sort
            print(f"  {filename}: {len(data['timestamps'])} timestamps")
        else:
            print(f"  {filename}: No timestamps found")
    
    return breathing_data

def create_breathing_segments_simple(audio_file, timestamps, segment_length=2.0):
    """Create simple breathing segments using just timestamps."""
    
    import librosa
    
    # Load audio
    audio, sr = librosa.load(audio_file, sr=16000, mono=True)
    duration = len(audio) / sr
    
    # Create breathing intervals (simple approach)
    breathing_intervals = []
    for i in range(0, len(timestamps), 2):  # Assume pairs of timestamps are breathing periods
        start = timestamps[i]
        end = timestamps[i+1] if i+1 < len(timestamps) else start + 1.5
        breathing_intervals.append((start, end))
    
    print(f"    Created {len(breathing_intervals)} breathing intervals")
    
    # Create segments
    segments = []
    labels = []
    
    current_time = 0.0
    while current_time + segment_length <= duration:
        segment_start = int(current_time * sr)
        segment_end = int((current_time + segment_length) * sr)
        segment_audio = audio[segment_start:segment_end]
        
        # Check if segment overlaps with breathing
        segment_mid = current_time + segment_length / 2
        is_breathing = any(start <= segment_mid <= end for start, end in breathing_intervals)
        
        segments.append(segment_audio)
        labels.append(1 if is_breathing else 0)
        
        current_time += 1.0  # 1 second hop
    
    return segments, labels

def test_simple_approach():
    """Test the simple approach on a few files."""
    
    print("\n🧪 TESTING SIMPLE APPROACH")
    print("=" * 30)
    
    breathing_data = create_simple_breathing_parser()
    
    if not breathing_data:
        print("❌ No breathing data parsed!")
        return
    
    audio_dir = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/RAW sound_ML test sound list")
    
    total_segments = 0
    total_breathing = 0
    
    # Test on first 3 files
    for i, (filename, data) in enumerate(list(breathing_data.items())[:3]):
        if not data['timestamps']:
            continue
            
        # Find matching audio file
        audio_file = None
        for audio_path in audio_dir.glob("*.wav"):
            if filename in audio_path.name or audio_path.stem in filename:
                audio_file = audio_path
                break
        
        if not audio_file:
            print(f"⚠️ No audio file found for {filename}")
            continue
        
        print(f"\n🎵 Testing {audio_file.name}...")
        print(f"    Timestamps: {data['timestamps'][:10]}{'...' if len(data['timestamps']) > 10 else ''}")
        
        try:
            segments, labels = create_breathing_segments_simple(audio_file, data['timestamps'])
            print(f"    Segments: {len(segments)} total, {sum(labels)} breathing ({sum(labels)/len(labels)*100:.1f}%)")
            
            total_segments += len(segments)
            total_breathing += sum(labels)
            
        except Exception as e:
            print(f"    Error: {e}")
    
    if total_segments > 0:
        print(f"\n📊 SUMMARY:")
        print(f"    Total segments: {total_segments}")
        print(f"    Breathing segments: {total_breathing} ({total_breathing/total_segments*100:.1f}%)")
        print(f"    Non-breathing segments: {total_segments - total_breathing} ({(total_segments - total_breathing)/total_segments*100:.1f}%)")
        
        print(f"\n✅ Simple approach works! Ready to build full classifier.")
    else:
        print(f"\n❌ No segments created. Need to debug further.")

if __name__ == "__main__":
    debug_excel_structure()
    test_simple_approach()
