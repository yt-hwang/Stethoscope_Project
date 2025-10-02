# analyze_data.py - COMPLETE FIXED VERSION
import pickle
import numpy as np
import pandas as pd
from collections import Counter
import os
import json

FEATURES_DIR = 'D:\\Stethoscope_Project\\Development\\1) Audio\\5) Conventional Feature Extraction + XXX\\features'
FEATURE_FILE = 'mfcc_features.pkl'

def main():
    print("=== Dataset Analysis ===")
    
    # Check if features directory exists
    if not os.path.exists(FEATURES_DIR):
        print(f"❌ ERROR: Features directory not found: {FEATURES_DIR}")
        print("Make sure you ran extract_features.py first!")
        return
    
    feature_path = os.path.join(FEATURES_DIR, FEATURE_FILE)
    
    # Check if feature file exists
    if not os.path.exists(feature_path):
        print(f"❌ ERROR: Feature file not found: {feature_path}")
        print("Available files in features directory:")
        if os.path.exists(FEATURES_DIR):
            for f in os.listdir(FEATURES_DIR):
                print(f"  - {f}")
        print("Make sure you ran extract_features.py first!")
        return
    
    # Load features
    try:
        with open(feature_path, 'rb') as f:
            data = pickle.load(f)
        print(f"✅ Successfully loaded {feature_path}")
    except Exception as e:
        print(f"❌ ERROR loading feature file: {e}")
        return
    
    # Print what's in the data file
    print(f"Keys in data file: {list(data.keys())}")
    
    # Validate data structure
    required_keys = ['features', 'labels', 'filenames', 'feature_type']
    missing_keys = []
    for key in required_keys:
        if key not in data:
            missing_keys.append(key)
    
    if missing_keys:
        print(f"❌ ERROR: Missing keys in feature file: {missing_keys}")
        return
    
    features = data['features']
    labels = data['labels']
    filenames = data['filenames']
    
    print(f"Raw features type: {type(features)}")
    print(f"Raw features shape/length: {features.shape if hasattr(features, 'shape') else len(features)}")
    print(f"Labels length: {len(labels)}")
    print(f"Filenames length: {len(filenames)}")
    
    # Check if data is empty
    if len(features) == 0:
        print("❌ ERROR: No features found in file!")
        print("This means feature extraction failed silently.")
        print("Debugging info:")
        print(f"  Feature type: {data.get('feature_type', 'unknown')}")
        print(f"  Data keys: {list(data.keys())}")
        
        # Check if there are any segments to extract from
        segments_dir = 'segments_2000ms'
        if os.path.exists(segments_dir):
            wav_files = [f for f in os.listdir(segments_dir) if f.endswith('.wav')]
            lab_files = [f for f in os.listdir(segments_dir) if f.endswith('.lab')]
            print(f"  Available WAV files: {len(wav_files)}")
            print(f"  Available LAB files: {len(lab_files)}")
            if len(wav_files) > 0:
                print("  Sample WAV files:")
                for f in wav_files[:5]:
                    print(f"    - {f}")
        else:
            print(f"  Segments directory '{segments_dir}' not found!")
        
        print("\n🔧 SOLUTION: Re-run extract_features.py with verbose output")
        return
    
    print(f"✅ Found {len(features)} feature vectors")
    print(f"Feature type: {data['feature_type'].upper()}")
    print("=" * 50)
    
    # Handle different feature array formats
    if isinstance(features, list):
        features = np.array(features)
    
    # Handle 1D features (convert to 2D)
    if features.ndim == 1:
        print("Converting 1D features to 2D...")
        features = features.reshape(-1, 1)
    
    print(f"Total segments: {len(features)}")
    print(f"Feature dimensionality: {features.shape[1]}")
    print()
    
    # Class distribution
    class_counts = Counter(labels)
    print("Class Distribution:")
    total_samples = len(labels)
    for cls, count in sorted(class_counts.items()):
        percentage = (count / total_samples) * 100
        print(f"  {cls}: {count} ({percentage:.1f}%)")
    print()
    
    # Feature statistics
    print("Feature Statistics:")
    print(f"  Mean: {np.mean(features):.4f}")
    print(f"  Std:  {np.std(features):.4f}")
    print(f"  Min:  {np.min(features):.4f}")
    print(f"  Max:  {np.max(features):.4f}")
    print()
    
    # Check for missing values
    missing = np.isnan(features).sum()
    print(f"Missing values: {missing}")
    
    if missing > 0:
        print("⚠️  Warning: Found missing values in features!")
    
    # Check for infinite values
    infinite = np.isinf(features).sum()
    print(f"Infinite values: {infinite}")
    
    if infinite > 0:
        print("⚠️  Warning: Found infinite values in features!")
    
    print()
    
    # Save summary
    summary = {
        'total_segments': len(features),
        'feature_dim': int(features.shape[1]),
        'feature_type': data['feature_type'],
        'class_distribution': dict(class_counts),
        'feature_stats': {
            'mean': float(np.mean(features)),
            'std': float(np.std(features)),
            'min': float(np.min(features)),
            'max': float(np.max(features)),
            'missing_values': int(missing),
            'infinite_values': int(infinite)
        }
    }
    
    summary_path = os.path.join(FEATURES_DIR, 'dataset_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Dataset summary saved to {summary_path}")
    print("🎉 Analysis complete!")

if __name__ == "__main__":
    main()
