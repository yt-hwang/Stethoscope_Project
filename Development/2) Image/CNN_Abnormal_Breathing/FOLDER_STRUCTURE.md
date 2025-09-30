# CNN 5-Class Abnormal Breathing Classification - Folder Structure

## 📁 **Directory Organization**

```
Development/2) Image/CNN_5Class_Abnormal_Breathing/
├── cnn_5class_abnormal_breathing_classifier.py    # Main CNN model implementation
├── quick_test_cnn.py                              # Quick test script for setup verification
├── requirements_cnn.txt                           # Python dependencies
├── README_CNN_Model.md                            # Comprehensive documentation
├── FOLDER_STRUCTURE.md                            # This file
├── Results/                                       # Training results and outputs
│   ├── best_model.pth                            # Best model weights (after training)
│   ├── complete_model.pth                        # Complete model with config
│   ├── training_history.png                      # Training curves visualization
│   ├── confusion_matrix.png                      # Confusion matrix plot
│   ├── classification_report.csv                 # Detailed performance metrics
│   └── evaluation_results.pkl                    # Complete evaluation results
└── Examples/                                      # Example usage and notebooks (future)
```

## 🎯 **File Descriptions**

### **Core Implementation Files**

- **`cnn_5class_abnormal_breathing_classifier.py`**
  - Complete CNN implementation for 5-class breathing classification
  - Includes data loading, model architecture, training, and evaluation
  - Classes: Wheezing, Crackle, Rhonchi, Bronchi, Healthy

- **`quick_test_cnn.py`**
  - Test script to verify setup before full training
  - Tests data loading, model creation, and device setup
  - Run this first to ensure everything works

### **Configuration and Documentation**

- **`requirements_cnn.txt`**
  - All Python dependencies needed for the CNN model
  - Includes PyTorch, librosa, scikit-learn, matplotlib, etc.

- **`README_CNN_Model.md`**
  - Comprehensive documentation and usage guide
  - Architecture details, configuration options, troubleshooting

### **Results Directory**
- **`Results/`** - Created automatically during training
  - Contains all training outputs, model weights, and evaluation results
  - Generated when running the main training script

## 🚀 **Usage Instructions**

### **1. Quick Test (Recommended First)**
```bash
cd "Development/2) Image/CNN_4Class_Abnormal_Breathing"
python quick_test_cnn.py
```

### **2. Install Dependencies**
```bash
pip install -r requirements_cnn.txt
```

### **3. Full Training**
```bash
python cnn_5class_abnormal_breathing_classifier.py
```

## 📊 **Expected Outputs**

After running the training, the `Results/` directory will contain:

1. **Model Files**
   - `best_model.pth` - Best performing model weights
   - `complete_model.pth` - Full model with configuration and label encoder

2. **Visualizations**
   - `training_history.png` - Training/validation loss and accuracy curves
   - `confusion_matrix.png` - Classification performance visualization

3. **Evaluation Data**
   - `classification_report.csv` - Per-class precision, recall, F1-score
   - `evaluation_results.pkl` - Complete evaluation results for further analysis

## 🔧 **Integration with Existing Project**

This CNN implementation integrates seamlessly with your existing stethoscope project:

- **Uses existing data**: Reads from `Audio shared/breathing_nonbreathing_intervals.json`
- **Compatible audio**: Works with your RAW sound files
- **Consistent preprocessing**: Uses same mel-spectrogram approach as your other experiments
- **Complementary**: Can be compared with your OPERA-CT and handcrafted feature approaches

## 📈 **Performance Expectations**

Based on your existing work:
- **Binary classification**: 77.8% accuracy (from your handcrafted features)
- **4-class CNN**: Expected 60-80% accuracy depending on class balance
- **Improvement potential**: Data augmentation, architecture tuning, ensemble methods

## 🔄 **Future Extensions**

Potential enhancements to add in the `Examples/` directory:
- Data augmentation techniques
- Transfer learning with pre-trained models
- Ensemble methods combining CNN with your existing approaches
- Hyperparameter optimization notebooks
- Model interpretation and visualization tools
