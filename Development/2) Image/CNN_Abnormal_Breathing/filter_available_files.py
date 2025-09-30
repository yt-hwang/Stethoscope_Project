#!/usr/bin/env python3
"""
Filter the JSON data to only include files that actually exist in the audio directory.
"""

import json
from pathlib import Path

# Paths
JSON_FILE = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/breathing_nonbreathing_intervals.json")
AUDIO_DIR = Path('/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/RAW sound_ML test sound list')
FILTERED_JSON = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Development/2) Image/CNN_Abnormal_Breathing/breathing_intervals_filtered.json")

def filter_available_files():
    """Filter JSON to only include files that exist."""
    
    # Load original JSON
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filter for existing files
    filtered_data = {}
    missing_files = []
    
    for filename, entry in data.items():
        audio_path = AUDIO_DIR / f"{filename}.wav"
        if audio_path.exists():
            filtered_data[filename] = entry
        else:
            missing_files.append(filename)
    
    # Save filtered data
    with open(FILTERED_JSON, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Filtered JSON saved to: {FILTERED_JSON}")
    print(f"📊 Original files: {len(data)}")
    print(f"📊 Available files: {len(filtered_data)}")
    print(f"📊 Missing files: {len(missing_files)}")
    
    if missing_files:
        print(f"\n❌ Missing files:")
        for f in missing_files:
            print(f"   - {f}")
    
    # Show class distribution
    from collections import Counter
    diagnoses = [entry.get('diagnosis', 'Unknown') for entry in filtered_data.values()]
    
    # Map to classes
    def map_diagnosis_to_class(diagnosis: str) -> str:
        diagnosis_lower = diagnosis.lower()
        
        # Check for Bronchi first (most specific)
        if 'bronchi' in diagnosis_lower or 'brhonchi' in diagnosis_lower:
            return 'Bronchi'
        elif 'wheezing' in diagnosis_lower:
            return 'Wheezing'
        elif 'crackle' in diagnosis_lower:
            return 'Crackle'
        elif 'rhonchi' in diagnosis_lower:
            return 'Rhonchi'
        elif 'healthy' in diagnosis_lower:
            return 'Healthy'
        else:
            return 'Healthy'
    
    mapped_classes = [map_diagnosis_to_class(d) for d in diagnoses]
    class_counts = Counter(mapped_classes)
    
    print(f"\n📈 Class distribution after filtering:")
    for class_name, count in class_counts.items():
        print(f"   {class_name}: {count} samples")

if __name__ == "__main__":
    filter_available_files()
