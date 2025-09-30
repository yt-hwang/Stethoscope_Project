#!/usr/bin/env python3
"""
Correct Excel Parser Investigation
==================================
Parse Excel correctly and identify the severe problem in timeline displays
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def parse_excel_correctly():
    """Parse Excel with the correct structure based on actual file examination."""
    
    excel_file = "/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/ML test sound list breathing info.xlsx"
    
    print("📋 Parsing Excel with CORRECT structure...")
    print("Structure: Row 1=filename+headers, Row 2=disease+timestamps, Row 3=blank")
    
    all_files_data = {}
    
    for sheet_name in ['Sheet1', 'Healthy']:
        print(f"\n📄 Processing {sheet_name} sheet...")
        df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
        
        # Process every 3 rows starting from row 1 (0-indexed)
        for idx in range(1, len(df), 3):
            if idx + 1 >= len(df):  # Need both filename row and data row
                break
                
            filename_row = df.iloc[idx]      # Row with filename and headers
            data_row = df.iloc[idx + 1]      # Row with disease and timestamps
            
            # Get filename from column 1 (B column)
            filename = filename_row.iloc[1]
            if pd.isna(filename):
                continue
                
            filename = str(filename).strip()
            print(f"  📁 {filename}")
            
            # Extract breathing events from data row
            breathing_events = []
            col = 2  # Start from column C
            
            while col < len(data_row) - 1:
                start_time = data_row.iloc[col]
                end_time = data_row.iloc[col + 1]
                
                if pd.isna(start_time) or pd.isna(end_time):
                    break
                
                try:
                    start_time = float(start_time)
                    end_time = float(end_time)
                    
                    # Determine event type from filename row headers
                    header = filename_row.iloc[col]
                    event_type = 'breathing'  # Default
                    if pd.notna(header) and isinstance(header, str):
                        if 'Inhale' in header:
                            event_type = 'inhale'
                        elif 'Exhale' in header:
                            event_type = 'exhale'
                    
                    breathing_events.append({
                        'start': start_time,
                        'end': end_time,
                        'type': event_type,
                        'header': str(header) if pd.notna(header) else 'Unknown'
                    })
                    
                    print(f"    {breathing_events[-1]['header']}: {start_time:.3f} - {end_time:.3f}s")
                    
                except (ValueError, TypeError):
                    break
                
                col += 2
            
            if breathing_events:
                # Create complete timeline
                complete_timeline = []
                
                # Sort by start time
                breathing_events.sort(key=lambda x: x['start'])
                
                # Add initial non-breathing period (if file doesn't start with breathing)
                if breathing_events[0]['start'] > 0:
                    complete_timeline.append({
                        'start': 0.0,
                        'end': breathing_events[0]['start'],
                        'type': 'non-breathing'
                    })
                
                # Process all breathing events and gaps
                for i, event in enumerate(breathing_events):
                    # Add breathing period
                    complete_timeline.append({
                        'start': event['start'],
                        'end': event['end'],
                        'type': 'breathing'
                    })
                    
                    # Add gap to next breathing event (if exists)
                    if i < len(breathing_events) - 1:
                        next_event = breathing_events[i + 1]
                        if event['end'] < next_event['start']:
                            complete_timeline.append({
                                'start': event['end'],
                                'end': next_event['start'],
                                'type': 'non-breathing'
                            })
                
                # Add final non-breathing period (estimate file duration as 30s)
                last_end = breathing_events[-1]['end']
                if last_end < 30:
                    complete_timeline.append({
                        'start': last_end,
                        'end': 30.0,
                        'type': 'non-breathing'
                    })
                
                all_files_data[filename] = {
                    'breathing_events': breathing_events,
                    'complete_timeline': complete_timeline,
                    'condition': 'Healthy' if sheet_name == 'Healthy' else 'Pathological'
                }
                
                print(f"    ✅ {len(breathing_events)} breathing events")
    
    print(f"\n✅ Correctly parsed {len(all_files_data)} files")
    return all_files_data

def analyze_individual_model_data():
    """Analyze what individual models actually used vs correct Excel data."""
    
    print("\n🔍 ANALYZING INDIVIDUAL MODEL vs CORRECT EXCEL")
    print("=" * 50)
    
    # Get correct Excel data
    correct_excel = parse_excel_correctly()
    
    # Check individual models for a specific file
    test_filename = "KP001_WWS"
    
    if test_filename in correct_excel:
        print(f"\n📊 ANALYSIS FOR {test_filename}:")
        print("=" * 30)
        
        correct_data = correct_excel[test_filename]
        
        print("🎯 CORRECT Excel Timeline:")
        for i, period in enumerate(correct_data['complete_timeline']):
            print(f"  {i+1:2d}. {period['start']:5.1f} - {period['end']:5.1f}s: {period['type']}")
        
        # Check what individual models used
        model_sizes = ['1.0s', '0.5s', '0.25s']
        
        for size in model_sizes:
            model_dir = f"Individual_Models/{size}_Individual_Models/Center_Point_Labeling_Results"
            predictions_file = Path(model_dir) / "final_predictions.json"
            
            if predictions_file.exists():
                with open(predictions_file, 'r') as f:
                    model_data = json.load(f)
                
                # Find matching filename (might have .wav extension)
                model_filename = None
                for fname in model_data.keys():
                    if test_filename in fname or fname.replace('.wav', '') == test_filename:
                        model_filename = fname
                        break
                
                if model_filename:
                    predictions = model_data[model_filename]
                    
                    print(f"\n🤖 {size} Individual Model Timeline:")
                    
                    # Show ground truth pattern from model
                    print("  Model's Ground Truth Pattern:")
                    for pred in predictions:
                        gt_label = "breathing" if pred['ground_truth'] == 1 else "non-breathing"
                        print(f"    {pred['start_time']:5.1f} - {pred['end_time']:5.1f}s: {gt_label}")
                    
                    # Compare patterns
                    print(f"\n  📊 Model Stats:")
                    total_segs = len(predictions)
                    breathing_segs = sum(1 for p in predictions if p['ground_truth'] == 1)
                    print(f"    Total segments: {total_segs}")
                    print(f"    Breathing: {breathing_segs} ({breathing_segs/total_segs*100:.1f}%)")
                    print(f"    Non-breathing: {total_segs-breathing_segs} ({(total_segs-breathing_segs)/total_segs*100:.1f}%)")
    
    return correct_excel

def identify_severe_problem(correct_excel):
    """Identify the severe problem by comparing correct vs model data."""
    
    print(f"\n🚨 IDENTIFYING SEVERE PROBLEM")
    print("=" * 30)
    
    # Check if the models are using the wrong Excel parsing
    print("🔍 Checking if models used incorrect Excel parsing...")
    
    # Compare correct Excel data with what we know models should have
    sample_file = "KP001_WWS"
    if sample_file in correct_excel:
        correct_timeline = correct_excel[sample_file]['complete_timeline']
        
        print(f"\n📋 CORRECT Excel Timeline for {sample_file}:")
        total_breathing_time = 0
        total_time = 30.0  # Assuming 30s files
        
        for period in correct_timeline:
            if period['type'] == 'breathing':
                duration = period['end'] - period['start']
                total_breathing_time += duration
                print(f"  BREATHING: {period['start']:.3f} - {period['end']:.3f}s ({duration:.3f}s)")
            else:
                duration = period['end'] - period['start']
                print(f"  NON-BREATHING: {period['start']:.3f} - {period['end']:.3f}s ({duration:.3f}s)")
        
        breathing_percentage = (total_breathing_time / total_time) * 100
        print(f"\n📊 CORRECT Statistics:")
        print(f"  Total breathing time: {total_breathing_time:.3f}s")
        print(f"  Breathing percentage: {breathing_percentage:.1f}%")
        
        # This should help identify if the severe problem is:
        # 1. Wrong Excel parsing leading to wrong ground truth
        # 2. Timeline display not matching Excel data
        # 3. Inconsistent labeling methodology
        
        print(f"\n🚨 SEVERE PROBLEM LIKELY:")
        print("  If individual model timelines don't match this correct Excel data,")
        print("  then the Excel parsing in individual models is WRONG!")
        print("  This would mean all ground truth labels are incorrect!")

def main():
    """Main investigation function."""
    
    print("🚨 SEVERE PROBLEM INVESTIGATION")
    print("=" * 35)
    print("🎯 Finding discrepancy between Excel data and timeline displays")
    print()
    
    try:
        # Parse Excel correctly
        correct_excel = analyze_individual_model_data()
        
        # Identify the severe problem
        identify_severe_problem(correct_excel)
        
        print(f"\n🔍 INVESTIGATION COMPLETE")
        print("=" * 25)
        print("📊 Review the comparison above to confirm the severe problem")
        print("🚨 If timelines don't match correct Excel data, we found the issue!")
        
    except Exception as e:
        print(f"❌ Error during investigation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
