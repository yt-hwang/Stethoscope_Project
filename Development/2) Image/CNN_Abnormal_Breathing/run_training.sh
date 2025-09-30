#!/bin/bash

# CNN 5-Class Abnormal Breathing Classification Training Script
# This script sets up the environment and runs the CNN training

echo "=========================================="
echo "CNN 5-Class Abnormal Breathing Classification"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "cnn_abnormal_breathing_classifier.py" ]; then
    echo "❌ Error: Please run this script from the CNN_5Class_Abnormal_Breathing directory"
    echo "   Current directory: $(pwd)"
    echo "   Expected files: cnn_abnormal_breathing_classifier.py"
    exit 1
fi

echo "✅ Running from correct directory: $(pwd)"

# Create Results directory if it doesn't exist
mkdir -p Results
echo "✅ Results directory ready"

# Check Python installation
if ! command -v python &> /dev/null; then
    echo "❌ Error: Python not found. Please install Python 3.7+"
    exit 1
fi

echo "✅ Python found: $(python --version)"

# Check if requirements are installed
echo ""
echo "🔍 Checking dependencies..."

if ! python -c "import torch" &> /dev/null; then
    echo "⚠️  PyTorch not found. Installing requirements..."
    pip install -r requirements_cnn.txt
else
    echo "✅ PyTorch found"
fi

if ! python -c "import librosa" &> /dev/null; then
    echo "⚠️  Librosa not found. Installing requirements..."
    pip install -r requirements_cnn.txt
else
    echo "✅ Librosa found"
fi

if ! python -c "import sklearn" &> /dev/null; then
    echo "⚠️  Scikit-learn not found. Installing requirements..."
    pip install -r requirements_cnn.txt
else
    echo "✅ Scikit-learn found"
fi

echo ""
echo "🚀 Starting CNN training..."

# Run quick test first
echo "📋 Running quick test to verify setup..."
python quick_test_cnn.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Quick test passed! Starting full training..."
    echo "=========================================="
    
    # Run the main training script
    python3 cnn_abnormal_breathing_classifier.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 Training completed successfully!"
        echo "📁 Results saved in: Results/"
        echo ""
        echo "📊 Check the following files:"
        echo "   - Results/training_history.png (training curves)"
        echo "   - Results/confusion_matrix.png (performance visualization)"
        echo "   - Results/classification_report.csv (detailed metrics)"
        echo "   - Results/best_model.pth (trained model)"
    else
        echo "❌ Training failed. Check the error messages above."
        exit 1
    fi
else
    echo "❌ Quick test failed. Please fix the issues before running training."
    exit 1
fi
