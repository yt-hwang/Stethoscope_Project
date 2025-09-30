#!/usr/bin/env python3
"""
My Interpretation of Excel File Structure
Based on user's explanation
"""

import pandas as pd
from pathlib import Path

def parse_excel_correctly():
    """Parse Excel file according to user's structure explanation."""
    excel_file = Path('/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/ML test sound list breathing info.xlsx')
    
    # Read both sheets
    sheet1 = pd.read_excel(excel_file, sheet_name='Sheet1', header=None)  # Abnormal
    healthy = pd.read_excel(excel_file, sheet_name='Healthy', header=None)  # Healthy
    
    all_files_data = {}
    
    for sheet_name, df in [('Sheet1 (Abnormal)', sheet1), ('Healthy', healthy)]:
        print(f"\n🔍 PARSING {sheet_name}:")
        print("=" * 50)
        
        # Start from row 2 (index 1) since row 1 is empty
        row_idx = 1
        
        while row_idx < df.shape[0]:
            # Check if this is a filename row
            filename_row = df.iloc[row_idx]
            
            if pd.notna(filename_row.iloc[1]) and isinstance(filename_row.iloc[1], str):
                filename = str(filename_row.iloc[1])
                
                # Check if this looks like an audio filename
                if any(pattern in filename for pattern in ['KP', 'H0', 'WEBSS']):
                    print(f"\n📁 Processing: {filename}")
                    
                    # This is a filename row - extract breathing events
                    breathing_events = []
                    
                    # Parse columns starting from C (index 2)
                    col_idx = 2
                    inhale_count = 1
                    exhale_count = 1
                    
                    while col_idx < df.shape[1] - 1:  # Need pairs
                        start_time = filename_row.iloc[col_idx]
                        end_time = filename_row.iloc[col_idx + 1]
                        
                        if pd.notna(start_time) and pd.notna(end_time):
                            start_val = float(start_time)
                            end_val = float(end_time)
                            
                            # Determine if this is inhale or exhale based on position
                            event_type = 'inhale' if (col_idx - 2) % 4 == 0 else 'exhale'
                            event_number = inhale_count if event_type == 'inhale' else exhale_count
                            
                            breathing_events.append({
                                'type': event_type,
                                'number': event_number,
                                'start': start_val,
                                'end': end_val
                            })
                            
                            if event_type == 'inhale':
                                inhale_count += 1
                            else:
                                exhale_count += 1
                            
                            print(f"   {event_type}{event_number}: {start_val:.3f}s - {end_val:.3f}s")
                        else:
                            # Empty start/end times - exclude this file as per user instruction
                            if pd.isna(start_time) and pd.isna(end_time):
                                print(f"   ⚠️  Empty start/end times found - EXCLUDING {filename}")
                                breathing_events = []
                                break
                        
                        col_idx += 2  # Move to next pair
                    
                    # Get disease row (next row)
                    if row_idx + 1 < df.shape[0]:
                        disease_row = df.iloc[row_idx + 1]
                        disease_label = str(disease_row.iloc[1]) if pd.notna(disease_row.iloc[1]) else 'Unknown'
                        print(f"   Disease: {disease_label}")
                    else:
                        disease_label = 'Unknown'
                    
                    # Only add if we have breathing events (not excluded)
                    if breathing_events:
                        all_files_data[filename] = {
                            'breathing_events': breathing_events,
                            'disease': disease_label,
                            'condition': 'Pathological' if sheet_name.startswith('Sheet1') else 'Healthy'
                        }
                        print(f"   ✅ Added {len(breathing_events)} breathing events")
                    
                    # Skip to next file (filename row + disease row + empty row = +3)
                    row_idx += 3
                else:
                    row_idx += 1
            else:
                row_idx += 1
    
    return all_files_data

# Parse with my understanding
my_interpretation = parse_excel_correctly()

print(f"\n📊 MY INTERPRETATION SUMMARY:")
print("=" * 50)
print(f"Total files parsed: {len(my_interpretation)}")

for filename, data in my_interpretation.items():
    events = data['breathing_events']
    condition = data['condition']
    disease = data['disease']
    
    inhales = [e for e in events if e['type'] == 'inhale']
    exhales = [e for e in events if e['type'] == 'exhale']
    
    print(f"{filename} ({condition}):")
    print(f"  Disease: {disease}")
    print(f"  Inhales: {len(inhales)}, Exhales: {len(exhales)}")
    print(f"  Total events: {len(events)}")
"
