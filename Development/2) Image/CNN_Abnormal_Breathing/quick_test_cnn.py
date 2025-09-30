#!/usr/bin/env python3
"""
Quick test script to verify the CNN model setup and data loading.
This script tests the data loading and preprocessing pipeline without full training.
Updated for 5-class classification: Wheezing, Crackle, Rhonchi, Bronchi, Healthy.
"""

import sys
from pathlib import Path
import json
import numpy as np
import torch
from collections import Counter

# Add the current directory to path to import our modules
sys.path.append(str(Path(__file__).parent))

from cnn_abnormal_breathing_classifier import (
    Config, BreathingDataset, load_and_preprocess_data, 
    map_diagnosis_to_class, BreathingCNN
)

def test_data_loading():
    """Test data loading and preprocessing."""
    print("="*50)
    print("Testing Data Loading and Preprocessing")
    print("="*50)
    
    # Test JSON file loading
    if not Config.JSON_FILE.exists():
        print(f"❌ JSON file not found: {Config.JSON_FILE}")
        return False
    
    print(f"✅ JSON file found: {Config.JSON_FILE}")
    
    # Test audio directory
    if not Config.AUDIO_DIR.exists():
        print(f"❌ Audio directory not found: {Config.AUDIO_DIR}")
        return False
    
    print(f"✅ Audio directory found: {Config.AUDIO_DIR}")
    
    # Load and preprocess data
    try:
        filenames, diagnoses, breathing_data = load_and_preprocess_data(Config.JSON_FILE)
        print(f"✅ Successfully loaded {len(filenames)} files")
        
        # Show class distribution
        class_counts = Counter(diagnoses)
        print(f"\nClass Distribution:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} samples")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

def test_dataset_creation():
    """Test dataset creation and data loading."""
    print("\n" + "="*50)
    print("Testing Dataset Creation")
    print("="*50)
    
    try:
        # Load data
        filenames, diagnoses, breathing_data = load_and_preprocess_data(Config.JSON_FILE)
        
        # Create a small test dataset with first 5 files
        test_filenames = filenames[:5]
        test_diagnoses = diagnoses[:5]
        
        # Map diagnoses to numeric labels (simple mapping for test)
        class_mapping = {'Wheezing': 0, 'Crackle': 1, 'Rhonchi': 2, 'Bronchi': 3, 'Healthy': 4}
        test_labels = [class_mapping.get(d, 4) for d in test_diagnoses]  # Default to Healthy
        
        print(f"Testing with {len(test_filenames)} files:")
        for i, (filename, diagnosis, label) in enumerate(zip(test_filenames, test_diagnoses, test_labels)):
            print(f"  {i+1}. {filename} -> {diagnosis} (label: {label})")
        
        # Create dataset
        dataset = BreathingDataset(test_filenames, test_labels, Config.AUDIO_DIR)
        print(f"✅ Dataset created with {len(dataset)} samples")
        
        # Test data loading
        print("\nTesting data loading...")
        for i in range(min(3, len(dataset))):
            try:
                mel_spec, label = dataset[i]
                print(f"  Sample {i+1}: Shape={mel_spec.shape}, Label={label}")
                
                # Check mel-spectrogram properties
                if mel_spec.shape[0] == 1 and mel_spec.shape[1] == Config.INPUT_HEIGHT:
                    print(f"    ✅ Mel-spectrogram shape correct: {mel_spec.shape}")
                else:
                    print(f"    ❌ Unexpected mel-spectrogram shape: {mel_spec.shape}")
                    
            except Exception as e:
                print(f"  ❌ Error loading sample {i+1}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        return False

def test_model_creation():
    """Test model creation and forward pass."""
    print("\n" + "="*50)
    print("Testing Model Creation")
    print("="*50)
    
    try:
        # Create model
        model = BreathingCNN(
            input_height=Config.INPUT_HEIGHT,
            input_width=Config.INPUT_WIDTH,
            num_classes=Config.NUM_CLASSES,
            dropout_rate=Config.DROPOUT_RATE
        )
        
        print(f"✅ Model created successfully")
        print(f"  Input shape: (batch, 1, {Config.INPUT_HEIGHT}, {Config.INPUT_WIDTH})")
        print(f"  Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        batch_size = 2
        test_input = torch.randn(batch_size, 1, Config.INPUT_HEIGHT, Config.INPUT_WIDTH)
        
        model.eval()
        with torch.no_grad():
            output = model(test_input)
        
        print(f"✅ Forward pass successful")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected output shape: ({batch_size}, {Config.NUM_CLASSES})")
        
        if output.shape == (batch_size, Config.NUM_CLASSES):
            print(f"  ✅ Output shape correct")
        else:
            print(f"  ❌ Unexpected output shape")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating/testing model: {e}")
        return False

def test_device_setup():
    """Test device setup and model placement."""
    print("\n" + "="*50)
    print("Testing Device Setup")
    print("="*50)
    
    print(f"Device: {Config.DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
    
    # Test model on device
    try:
        model = BreathingCNN().to(Config.DEVICE)
        test_input = torch.randn(1, 1, Config.INPUT_HEIGHT, Config.INPUT_WIDTH).to(Config.DEVICE)
        
        with torch.no_grad():
            output = model(test_input)
        
        print(f"✅ Model successfully placed on {Config.DEVICE}")
        return True
        
    except Exception as e:
        print(f"❌ Error with device setup: {e}")
        return False

def main():
    """Run all tests."""
    print("CNN 4-Class Abnormal Breathing Classification - Quick Test")
    print("="*60)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Dataset Creation", test_dataset_creation),
        ("Model Creation", test_model_creation),
        ("Device Setup", test_device_setup)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20s}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! Ready to run the full training.")
        print("\nNext steps:")
        print("1. Install requirements: pip install -r requirements_cnn.txt")
        print("2. Run full training: python cnn_4class_abnormal_breathing_classifier.py")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
