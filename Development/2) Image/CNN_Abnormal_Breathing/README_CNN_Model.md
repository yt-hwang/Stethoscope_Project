# CNN 5-Class Abnormal Breathing Classification

This project implements a Convolutional Neural Network (CNN) for classifying abnormal breathing patterns into 5 classes using mel-spectrogram features.

## Classes

1. **Wheezing** (0) - Including variations like "Wheezing (잘들림)"
2. **Crackle** (1) - Including variations like "Crackle (잘들림)", "Crackle (Exhale이 잘 안들림)"  
3. **Rhonchi** (2)
4. **Bronchi** (3) - Including variations like "Wheezing, Bronchi"
5. **Healthy** (4) - Normal breathing

## Model Architecture

The CNN model consists of:

- **Input**: Mel-spectrograms (64 mel bins × 128 time frames)
- **Convolutional Layers**: 4 ConvBlocks with increasing filters (32→64→128→256)
- **Pooling**: MaxPool2d after each conv layer
- **Global Average Pooling**: Reduces spatial dimensions
- **Fully Connected**: 3 FC layers (256→512→256→5)
- **Regularization**: Dropout and BatchNorm

## Key Features

- **Data Preprocessing**: Automatic audio loading and mel-spectrogram generation
- **Smart Data Splitting**: Stratified train/validation/test splits (70%/15%/15%)
- **Robust Training**: Early stopping, learning rate scheduling, model checkpointing
- **Comprehensive Evaluation**: Classification report, confusion matrix, training history
- **Visualization**: Training curves and confusion matrix plots

## Usage

### 1. Install Requirements
```bash
pip install -r requirements_cnn.txt
```

### 2. Run the Model
```bash
python cnn_4class_abnormal_breathing_classifier.py
```

### 3. Output Files
The model will create a `CNN_4Class_Results/` directory containing:
- `best_model.pth` - Best model weights during training
- `complete_model.pth` - Complete model with configuration and label encoder
- `training_history.png` - Training/validation loss and accuracy curves
- `confusion_matrix.png` - Confusion matrix visualization
- `classification_report.csv` - Detailed per-class performance metrics
- `evaluation_results.pkl` - Complete evaluation results

## Configuration

Key parameters in the `Config` class:

```python
# Audio parameters
SAMPLE_RATE = 4000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 64
FMAX = 4000

# Model parameters
INPUT_HEIGHT = 64    # Mel bins
INPUT_WIDTH = 128    # Time frames
NUM_CLASSES = 5

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
```

## Data Requirements

The model expects:
- **Audio files**: `.wav` files in the RAW sound directory
- **Metadata**: `breathing_nonbreathing_intervals.json` with diagnosis labels
- **File naming**: Audio files should match the keys in the JSON file

## Model Performance

The model includes comprehensive evaluation:
- **Accuracy**: Overall classification accuracy
- **Per-class metrics**: Precision, Recall, F1-score for each class
- **Confusion matrix**: Detailed class-wise predictions
- **Training history**: Loss and accuracy curves

## Customization

### Adding New Classes
1. Update the `map_diagnosis_to_class()` function
2. Change `NUM_CLASSES` in the Config
3. Update the final FC layer output size
4. Retrain the model

### Adjusting Architecture
Modify the `BreathingCNN` class:
- Add/remove convolutional layers
- Change filter sizes and numbers
- Adjust fully connected layers

### Data Augmentation
Add augmentation in the `BreathingDataset.__getitem__()` method:
- Time stretching
- Pitch shifting
- Noise addition
- Mixup/CutMix

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch size or input dimensions
2. **File not found**: Check audio file paths and naming
3. **Poor performance**: Try data augmentation or architecture changes

### Performance Tips
1. Use GPU if available (automatically detected)
2. Adjust batch size based on GPU memory
3. Use mixed precision training for faster training
4. Implement data augmentation for better generalization

## Integration with Existing Pipeline

This CNN model complements your existing work:
- Uses the same data format as your breathing interval analysis
- Compatible with your mel-spectrogram generation approach
- Can be integrated with your OPERA-CT transfer learning experiments
- Results can be compared with your handcrafted feature approaches
