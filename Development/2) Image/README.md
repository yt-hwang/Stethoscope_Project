# Development/2) Image - CNN and Image Processing

This directory contains CNN models and image processing implementations for the stethoscope project.

## 📁 **Current Implementations**

### **CNN 5-Class Abnormal Breathing Classification**
- **Location**: `CNN_5Class_Abnormal_Breathing/`
- **Purpose**: Deep learning classification of abnormal breathing patterns
- **Classes**: Wheezing, Crackle, Rhonchi, Bronchi, Healthy
- **Input**: Mel-spectrograms (64 × 128)
- **Architecture**: Custom CNN with 4 ConvBlocks + Global Average Pooling

## 🎯 **CNN Model Overview**

The CNN implementation provides a complete deep learning solution for breathing classification:

### **Key Features**
- **5-Class Classification**: Wheezing, Crackle, Rhonchi, Bronchi, Healthy
- **Mel-Spectrogram Input**: Optimized for audio classification
- **Advanced Architecture**: ConvBlocks with batch normalization and dropout
- **Robust Training**: Early stopping, learning rate scheduling, model checkpointing
- **Comprehensive Evaluation**: Classification reports, confusion matrices, visualizations

### **Model Architecture**
```
Input: (batch, 1, 64, 128) mel-spectrograms
├── ConvBlock(1→32) + MaxPool2d
├── ConvBlock(32→64) + MaxPool2d  
├── ConvBlock(64→128) + MaxPool2d
├── ConvBlock(128→256) + MaxPool2d
├── Global Average Pooling
├── FC(256→512) + Dropout
├── FC(512→256) + Dropout
└── FC(256→5) → 5-class output
```

## 🚀 **Quick Start**

### **1. Navigate to CNN Directory**
```bash
cd "Development/2) Image/CNN_5Class_Abnormal_Breathing"
```

### **2. Run Quick Test**
```bash
python quick_test_cnn.py
```

### **3. Install Dependencies**
```bash
pip install -r requirements_cnn.txt
```

### **4. Run Training**
```bash
# Option 1: Direct execution
python cnn_5class_abnormal_breathing_classifier.py

# Option 2: Using the shell script (recommended)
./run_training.sh
```

## 📊 **Expected Performance**

Based on existing project results:
- **Binary Classification**: 77.8% accuracy (handcrafted features)
- **5-Class CNN**: Expected 50-75% accuracy (more challenging with additional class)
- **Training Time**: ~30-60 minutes on GPU, 2-4 hours on CPU

## 🔄 **Integration with Project**

The CNN model integrates seamlessly with existing work:

### **Data Compatibility**
- Uses same JSON format: `breathing_nonbreathing_intervals.json`
- Compatible with existing audio files
- Consistent mel-spectrogram preprocessing

### **Complementary Approaches**
- **CNN**: Deep learning, automatic feature learning
- **Handcrafted Features**: Domain knowledge, interpretability
- **OPERA-CT**: Transfer learning, pre-trained embeddings

### **Comparison Framework**
All approaches can be compared on the same dataset:
- Same train/test splits
- Consistent evaluation metrics
- Performance benchmarking

## 📈 **Future Extensions**

### **Planned Enhancements**
- **Data Augmentation**: Time stretching, pitch shifting, noise addition
- **Transfer Learning**: Pre-trained audio models, fine-tuning strategies
- **Advanced Architectures**: ResNet, DenseNet, attention mechanisms
- **Ensemble Methods**: Combining CNN with existing approaches
- **Real-time Inference**: Optimized models for deployment

### **Research Directions**
- **Multi-modal Fusion**: Audio + clinical metadata
- **Temporal Modeling**: RNN/LSTM integration
- **Unsupervised Learning**: Self-supervised pre-training
- **Model Interpretability**: Grad-CAM, attention visualization

## 📁 **Directory Structure**

```
Development/2) Image/
├── CNN_5Class_Abnormal_Breathing/          # Main CNN implementation
│   ├── cnn_5class_abnormal_breathing_classifier.py
│   ├── quick_test_cnn.py
│   ├── requirements_cnn.txt
│   ├── README_CNN_Model.md
│   ├── FOLDER_STRUCTURE.md
│   ├── run_training.sh
│   ├── Results/                            # Training outputs
│   └── Examples/                           # Future extensions
└── README.md                               # This file
```

## 🎯 **Best Practices**

### **Before Training**
1. Run `quick_test_cnn.py` to verify setup
2. Check data availability and paths
3. Ensure sufficient disk space for results
4. Verify GPU availability if using CUDA

### **During Training**
1. Monitor training curves for overfitting
2. Use early stopping to prevent overfitting
3. Save model checkpoints regularly
4. Document hyperparameter choices

### **After Training**
1. Evaluate on test set with proper metrics
2. Visualize confusion matrix and training history
3. Compare with baseline approaches
4. Document findings and insights

## 🔧 **Troubleshooting**

### **Common Issues**
- **CUDA out of memory**: Reduce batch size or input dimensions
- **File not found**: Check audio file paths and naming
- **Poor performance**: Try data augmentation or architecture changes
- **Training instability**: Adjust learning rate or add regularization

### **Performance Optimization**
- Use GPU if available (automatically detected)
- Adjust batch size based on available memory
- Implement mixed precision training for speed
- Use data augmentation for better generalization

---

*This directory serves as the central hub for CNN and image processing implementations in the stethoscope project. The CNN model provides a state-of-the-art deep learning approach to complement existing handcrafted feature and transfer learning methods.*
