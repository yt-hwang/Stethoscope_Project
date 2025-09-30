#!/usr/bin/env python3
"""
Verify Excel vs Timeline Accuracy
=================================
Checks if the breathing periods displayed in timeline graphs match the actual Excel data
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def parse_excel_correctly():
    """Parse Excel data using the correct structure you described."""
    
    excel_file = "/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/ML test sound list breathing info.xlsx"
    
    print("📋 Parsing Excel with CORRECT structure...")
    print("Structure: filename row, disease row, blank row, repeat...")
    print("Data format: inhale1(start,end), exhale1(start,end), inhale2(start,end)...")
    
    # Load both sheets
    sheet1_df = pd.read_excel(excel_file, sheet_name='Sheet1', header=None)
    healthy_df = pd.read_excel(excel_file, sheet_name='Healthy', header=None)
    
    all_files_data = {}
    
    for sheet_name, df in [('Sheet1', sheet1_df), ('Healthy', healthy_df)]:
        print(f"\n📄 Processing {sheet_name} sheet...")
        
        # Process every 3 rows (filename, disease, blank)
        for idx in range(1, len(df), 3):  # Start from row 1, skip by 3
            if idx >= len(df):
                break
                
            # Get filename from column B
            filename = df.iloc[idx, 1]
            if pd.isna(filename):
                continue
                
            filename = str(filename).strip()
            print(f"  📁 {filename}")
            
            # Extract breathing events from the same row (filename row)
            breathing_events = []
            col = 2  # Start from column C
            
            while col < len(df.columns) - 1:
                start_time = df.iloc[idx, col]
                end_time = df.iloc[idx, col + 1]
                
                if pd.isna(start_time) or pd.isna(end_time):
                    break
                
                try:
                    start_time = float(start_time)
                    end_time = float(end_time)
                    
                    # Determine if this is inhale or exhale based on position
                    event_num = (col - 2) // 2
                    event_type = 'inhale' if event_num % 2 == 0 else 'exhale'
                    
                    breathing_events.append({
                        'start': start_time,
                        'end': end_time,
                        'type': event_type,
                        'event_num': event_num // 2 + 1  # inhale1, exhale1, inhale2, etc.
                    })
                    
                    print(f"    {event_type}{breathing_events[-1]['event_num']}: {start_time:.3f} - {end_time:.3f}")
                    
                except (ValueError, TypeError):
                    break
                
                col += 2
            
            if breathing_events:
                # Create complete timeline with non-breathing periods
                complete_timeline = []
                
                # Sort events by start time
                breathing_events.sort(key=lambda x: x['start'])
                
                # Add non-breathing period before first event (if not starting at 0)
                if breathing_events[0]['start'] > 0:
                    complete_timeline.append({
                        'start': 0.0,
                        'end': breathing_events[0]['start'],
                        'type': 'non-breathing'
                    })
                
                # Add all breathing events and non-breathing gaps
                for i, event in enumerate(breathing_events):
                    # Add the breathing event
                    complete_timeline.append({
                        'start': event['start'],
                        'end': event['end'],
                        'type': 'breathing'
                    })
                    
                    # Add non-breathing gap to next event (if exists)
                    if i < len(breathing_events) - 1:
                        next_event = breathing_events[i + 1]
                        if event['end'] < next_event['start']:
                            complete_timeline.append({
                                'start': event['end'],
                                'end': next_event['start'],
                                'type': 'non-breathing'
                            })
                
                # Add final non-breathing period (assuming file is ~30 seconds)
                last_end = breathing_events[-1]['end']
                if last_end < 30:  # Assuming files are around 30 seconds
                    complete_timeline.append({
                        'start': last_end,
                        'end': 30.0,
                        'type': 'non-breathing'
                    })
                
                all_files_data[filename] = {
                    'breathing_events': breathing_events,
                    'complete_timeline': complete_timeline,
                    'condition': 'Healthy' if sheet_name == 'Healthy' else 'Pathological',
                    'total_breathing_events': len(breathing_events)
                }
                
                print(f"    ✅ {len(breathing_events)} breathing events, {len(complete_timeline)} total periods")
    
    print(f"\n✅ Parsed {len(all_files_data)} files correctly")
    return all_files_data

def check_individual_model_parsing():
    """Check what the individual models actually parsed."""
    
    print("\n🔍 Checking Individual Model Excel Parsing...")
    
    # Check what parsing method was used in individual models
    model_folders = [
        "Individual_Models/1.0s_Individual_Models",
        "Individual_Models/0.5s_Individual_Models", 
        "Individual_Models/0.25s_Individual_Models"
    ]
    
    for folder in model_folders:
        results_dir = Path(folder) / "Center_Point_Labeling_Results"
        if not results_dir.exists():
            print(f"❌ {folder} results not found")
            continue
            
        print(f"\n📊 Checking {folder}...")
        
        # Check if there are any Python files that show the parsing logic
        python_files = list(Path(folder).glob("*.py"))
        if python_files:
            print(f"  📝 Found Python files: {[f.name for f in python_files]}")
        
        # Check the final predictions to see what data was used
        predictions_file = results_dir / "final_predictions.json"
        if predictions_file.exists():
            with open(predictions_file, 'r') as f:
                predictions = json.load(f)
            
            # Check a sample file
            sample_filename = list(predictions.keys())[0]
            sample_data = predictions[sample_filename]
            
            print(f"  📋 Sample file: {sample_filename}")
            print(f"  📊 Total predictions: {len(sample_data)}")
            print(f"  📈 Breathing predictions: {sum(1 for p in sample_data if p['prediction'] == 1)}")
            print(f"  📉 Non-breathing predictions: {sum(1 for p in sample_data if p['prediction'] == 0)}")
            
            # Show first few predictions
            print(f"  🔍 First 3 predictions:")
            for i, pred in enumerate(sample_data[:3]):
                print(f"    {pred['start_time']:.1f}-{pred['end_time']:.1f}s: GT={pred['ground_truth']}, Pred={pred['prediction']}")

def compare_excel_vs_models():
    """Compare correct Excel parsing with what models used."""
    
    print("\n🔍 COMPARISON: Correct Excel vs Model Parsing")
    print("=" * 50)
    
    # Get correct Excel data
    correct_excel = parse_excel_correctly()
    
    # Check individual models
    check_individual_model_parsing()
    
    # Sample comparison for one file
    sample_filename = "KP001_WWS"  # Choose a file that should exist
    
    if sample_filename in correct_excel:
        print(f"\n📊 DETAILED COMPARISON FOR {sample_filename}:")
        print("=" * 40)
        
        correct_data = correct_excel[sample_filename]
        
        print("🎯 CORRECT Excel Data:")
        print("  Breathing Events:")
        for event in correct_data['breathing_events']:
            print(f"    {event['type']}{event['event_num']}: {event['start']:.3f} - {event['end']:.3f}s")
        
        print("\n  Complete Timeline:")
        for period in correct_data['complete_timeline']:
            print(f"    {period['start']:.3f} - {period['end']:.3f}s: {period['type']}")
        
        # Check if individual models have this file
        for model_folder in ["Individual_Models/1.0s_Individual_Models", 
                           "Individual_Models/0.5s_Individual_Models",
                           "Individual_Models/0.25s_Individual_Models"]:
            
            predictions_file = Path(model_folder) / "Center_Point_Labeling_Results" / "final_predictions.json"
            if predictions_file.exists():
                with open(predictions_file, 'r') as f:
                    model_predictions = json.load(f)
                
                if sample_filename in model_predictions:
                    model_data = model_predictions[sample_filename]
                    
                    print(f"\n🤖 {model_folder.split('/')[-1]} Model Data:")
                    print(f"  Total segments: {len(model_data)}")
                    
                    # Show ground truth pattern
                    gt_pattern = [p['ground_truth'] for p in model_data]
                    breathing_segments = sum(gt_pattern)
                    print(f"  Ground truth: {breathing_segments}/{len(gt_pattern)} breathing segments")
                    
                    # Show pattern
                    pattern_str = ''.join(['B' if gt == 1 else 'N' for gt in gt_pattern])
                    print(f"  Pattern: {pattern_str}")

def main():
    """Main verification function."""
    
    print("🚨 SEVERE PROBLEM INVESTIGATION")
    print("=" * 35)
    print("🎯 Checking if timeline graphs match actual Excel data")
    print("📋 Using your correct Excel structure explanation")
    print()
    
    try:
        compare_excel_vs_models()
        
        print(f"\n🔍 INVESTIGATION COMPLETE")
        print("=" * 25)
        print("📊 Check the output above to identify the discrepancy")
        print("🚨 If breathing periods don't match, we found the severe problem!")
        
    except Exception as e:
        print(f"❌ Error during investigation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
