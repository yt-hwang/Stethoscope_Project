#!/bin/bash

# OPERA-CT Installation Script
# This script sets up the OPERA-CT model and dependencies

echo "🚀 Setting up OPERA-CT Transfer Learning Environment"
echo "=================================================="

# Create and activate virtual environment (optional)
# python3 -m venv opera_env
# source opera_env/bin/activate

# Install basic requirements
echo "📦 Installing basic requirements..."
pip install -r requirements.txt

# Clone OPERA repository
echo "📥 Cloning OPERA repository..."
if [ ! -d "OPERA" ]; then
    git clone https://github.com/evelyn0414/OPERA.git
    cd OPERA
else
    echo "OPERA repository already exists"
    cd OPERA
    git pull origin main  # Update to latest version
fi

# Install OPERA-specific dependencies
echo "🔧 Installing OPERA dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

# Install OPERA package
echo "📦 Installing OPERA package..."
pip install -e .

# Download pretrained models (if available)
echo "⬇️ Downloading pretrained models..."
if [ -f "download_models.py" ]; then
    python download_models.py
elif [ -f "scripts/download_models.sh" ]; then
    bash scripts/download_models.sh
else
    echo "⚠️ No automatic model download script found"
    echo "Please check the OPERA repository for model download instructions"
fi

cd ..

# Verify installation
echo "✅ Verifying installation..."
python -c "
try:
    import torch
    import librosa
    import transformers
    print('✅ Core dependencies installed successfully')
    
    # Try to import OPERA (if it has a Python package)
    try:
        import opera  # This might not exist - depends on OPERA structure
        print('✅ OPERA package imported successfully')
    except ImportError:
        print('⚠️ OPERA package not found - may need manual setup')
        
except ImportError as e:
    print(f'❌ Installation issue: {e}')
"

echo ""
echo "🎉 Setup complete!"
echo "Next steps:"
echo "1. Check OPERA/README.md for specific usage instructions"
echo "2. Download pretrained models if not done automatically"
echo "3. Test with a sample audio file"
echo "4. Begin transfer learning experiments"
