#!/usr/bin/env python3
"""
CNN Model for 5-Class Abnormal Breathing Classification

Classes:
1. Wheezing (0)
2. Crackle (1) 
3. Rhonchi (2)
4. Bronchi (3)
5. Healthy (4)

This model uses mel-spectrograms as input features and implements:
- Data loading and preprocessing pipeline
- CNN architecture optimized for audio classification
- Training with validation and metrics
- Model evaluation and visualization
"""

import os
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import librosa

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# =========================
# Configuration
# =========================
class Config:
    # Paths
    JSON_FILE = Path("/Users/yunhwang/Desktop/Stethoscope_Project/Development/2) Image/CNN_Abnormal_Breathing/breathing_intervals_filtered.json")
    AUDIO_DIR = Path('/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/ML test sound list/RAW sound_ML test sound list')
    OUTPUT_DIR = Path('/Users/yunhwang/Desktop/Stethoscope_Project/Development/2) Image/CNN_Abnormal_Breathing/Results')
    
    # Audio parameters
    SAMPLE_RATE = 4000
    N_FFT = 1024
    HOP_LENGTH = 256
    N_MELS = 64
    FMAX = 4000
    
    # CNN parameters
    INPUT_HEIGHT = N_MELS  # Mel bins
    INPUT_WIDTH = 128      # Time frames (adjustable)
    NUM_CLASSES = 5
    
    # Training parameters (adjusted for small dataset)
    BATCH_SIZE = 8  # Smaller batch size for small dataset
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50  # Fewer epochs for small dataset
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Model architecture
    DROPOUT_RATE = 0.3
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create output directory
Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Data Loading and Preprocessing
# =========================

class BreathingDataset(Dataset):
    """PyTorch Dataset for breathing audio classification."""
    
    def __init__(self, file_list: List[str], labels: List[int], audio_dir: Path, 
                 target_sr: int = 4000, input_width: int = 128, n_mels: int = 64):
        self.file_list = file_list
        self.labels = labels
        self.audio_dir = audio_dir
        self.target_sr = target_sr
        self.input_width = input_width
        self.n_mels = n_mels
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        label = self.labels[idx]
        
        # Load audio file
        audio_path = self.audio_dir / f"{filename}.wav"
        try:
            audio, sr = librosa.load(str(audio_path), sr=self.target_sr, mono=True)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return zero audio if file not found
            audio = np.zeros(self.target_sr * 30)  # 30 seconds of silence
        
        # Create mel-spectrogram
        mel_spec = self._create_mel_spectrogram(audio, self.target_sr)
        
        # Convert to tensor and add channel dimension
        mel_spec_tensor = torch.FloatTensor(mel_spec).unsqueeze(0)  # Shape: (1, n_mels, time_frames)
        
        return mel_spec_tensor, label
    
    def _create_mel_spectrogram(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Create mel-spectrogram from audio."""
        # Create mel-spectrogram
        S = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH,
            n_mels=self.n_mels, fmax=Config.FMAX, power=2.0, center=True
        )
        
        # Convert to dB
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Normalize to [0, 1]
        if S_db.max() != S_db.min():
            S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min())
        
        # Resize to fixed width (time frames)
        if S_db.shape[1] > self.input_width:
            # Truncate if too long
            S_db = S_db[:, :self.input_width]
        elif S_db.shape[1] < self.input_width:
            # Pad if too short
            pad_width = self.input_width - S_db.shape[1]
            S_db = np.pad(S_db, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        
        return S_db

def load_and_preprocess_data(json_file: Path) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """Load and preprocess data from JSON file."""
    print("Loading breathing intervals data...")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    filenames = []
    diagnoses = []
    breathing_data = {}
    
    for filename, entry in data.items():
        diagnosis = entry.get('diagnosis', 'Unknown')
        
        # Map diagnoses to 4 main classes
        mapped_diagnosis = map_diagnosis_to_class(diagnosis)
        
        filenames.append(filename)
        diagnoses.append(mapped_diagnosis)
        breathing_data[filename] = entry
    
    return filenames, diagnoses, breathing_data

def map_diagnosis_to_class(diagnosis: str) -> str:
    """Map various diagnosis strings to 5 main classes."""
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
        # Default to Healthy for unknown cases
        return 'Healthy'

def create_data_splits(filenames: List[str], labels: List[str]) -> Tuple[List, List, List, List, List, List]:
    """Create train/validation/test splits."""
    print("Creating data splits...")
    
    from collections import Counter
    label_counts = Counter(labels)
    min_class_count = min(label_counts.values())
    
    print(f"Class distribution: {dict(label_counts)}")
    
    if min_class_count < 3:  # Need at least 3 samples per class for train/val/test
        print(f"⚠️  Warning: Some classes have < 3 samples. Using simple random split.")
        
        # Simple random split without stratification
        X_train, X_temp, y_train, y_temp = train_test_split(
            filenames, labels, test_size=(Config.VAL_RATIO + Config.TEST_RATIO), 
            random_state=42
        )
        
        val_size = Config.VAL_RATIO / (Config.VAL_RATIO + Config.TEST_RATIO)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1-val_size), 
            random_state=42
        )
    else:
        # Use stratified splitting for datasets with sufficient samples per class
        X_train, X_temp, y_train, y_temp = train_test_split(
            filenames, labels, test_size=(Config.VAL_RATIO + Config.TEST_RATIO), 
            random_state=42, stratify=labels
        )
        
        val_size = Config.VAL_RATIO / (Config.VAL_RATIO + Config.TEST_RATIO)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1-val_size), 
            random_state=42, stratify=y_temp
        )
    
    # Print final splits
    print(f"Train: {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples") 
    print(f"Test: {len(X_test)} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# =========================
# CNN Model Architecture
# =========================

class ConvBlock(nn.Module):
    """Convolutional block with batch normalization and dropout."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, dropout_rate: float = 0.3):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class BreathingCNN(nn.Module):
    """CNN model for 5-class abnormal breathing classification."""
    
    def __init__(self, input_height: int = 64, input_width: int = 128, num_classes: int = 5, dropout_rate: float = 0.3):
        super(BreathingCNN, self).__init__()
        
        self.input_height = input_height
        self.input_width = input_width
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv1 = ConvBlock(1, 32, kernel_size=3, dropout_rate=dropout_rate)
        self.conv2 = ConvBlock(32, 64, kernel_size=3, dropout_rate=dropout_rate)
        self.conv3 = ConvBlock(64, 128, kernel_size=3, dropout_rate=dropout_rate)
        self.conv4 = ConvBlock(128, 256, kernel_size=3, dropout_rate=dropout_rate)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Dropout for fully connected layers
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # Convolutional layers with pooling
        x = self.conv1(x)  # (batch, 32, height, width)
        x = self.pool(x)   # (batch, 32, height/2, width/2)
        
        x = self.conv2(x)  # (batch, 64, height/2, width/2)
        x = self.pool(x)   # (batch, 64, height/4, width/4)
        
        x = self.conv3(x)  # (batch, 128, height/4, width/4)
        x = self.pool(x)   # (batch, 128, height/8, width/8)
        
        x = self.conv4(x)  # (batch, 256, height/8, width/8)
        x = self.pool(x)   # (batch, 256, height/16, width/16)
        
        # Global average pooling
        x = self.global_avg_pool(x)  # (batch, 256, 1, 1)
        x = x.view(x.size(0), -1)    # (batch, 256)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x

# =========================
# Training Functions
# =========================

def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                optimizer: optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    """Train model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                  device: torch.device) -> Tuple[float, float]:
    """Validate model for one epoch."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                num_epochs: int, learning_rate: float, device: torch.device) -> Dict[str, List[float]]:
    """Train the CNN model."""
    print(f"Training on device: {device}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    best_val_acc = 0.0
    patience = 10  # Reduced patience for small dataset
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), Config.OUTPUT_DIR / 'best_model.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs
    }

# =========================
# Evaluation Functions
# =========================

def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device, 
                  class_names: List[str]) -> Dict[str, Any]:
    """Evaluate the trained model on test set."""
    print("Evaluating model on test set...")
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    
    # Get unique classes in the test set
    unique_classes = sorted(set(all_targets + all_predictions))
    available_class_names = [class_names[i] for i in unique_classes]
    
    report = classification_report(all_targets, all_predictions, 
                                 labels=unique_classes,
                                 target_names=available_class_names, 
                                 output_dict=True, zero_division=0)
    cm = confusion_matrix(all_targets, all_predictions, labels=unique_classes)
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'targets': all_targets
    }

def plot_training_history(history: Dict[str, List[float]], output_dir: Path):
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(history['train_losses'], label='Train Loss', color='blue')
    ax1.plot(history['val_losses'], label='Validation Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(history['train_accs'], label='Train Accuracy', color='blue')
    ax2.plot(history['val_accs'], label='Validation Accuracy', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], output_dir: Path):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - 4-Class Abnormal Breathing Classification')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_results(results: Dict[str, Any], output_dir: Path):
    """Save evaluation results to files."""
    # Save classification report
    report_df = pd.DataFrame(results['classification_report']).transpose()
    report_df.to_csv(output_dir / 'classification_report.csv')
    
    # Save detailed results
    with open(output_dir / 'evaluation_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS SUMMARY")
    print("="*60)
    print(f"Test Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print("\nPer-class Performance:")
    for class_name, metrics in results['classification_report'].items():
        if isinstance(metrics, dict) and 'precision' in metrics:
            print(f"{class_name:12s}: Precision={metrics['precision']:.3f}, "
                  f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")

# =========================
# Main Function
# =========================

def main():
    """Main function to run the complete pipeline."""
    print("="*60)
    print("CNN 4-Class Abnormal Breathing Classification")
    print("="*60)
    
    # Load and preprocess data
    filenames, diagnoses, breathing_data = load_and_preprocess_data(Config.JSON_FILE)
    
    # Print class distribution
    from collections import Counter
    class_counts = Counter(diagnoses)
    print(f"\nClass Distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} samples")
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(diagnoses)
    class_names = label_encoder.classes_.tolist()
    
    # Create data splits
    X_train, X_val, X_test, y_train, y_val, y_test = create_data_splits(filenames, encoded_labels)
    
    print(f"\nData Splits:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Create datasets
    train_dataset = BreathingDataset(X_train, y_train, Config.AUDIO_DIR)
    val_dataset = BreathingDataset(X_val, y_val, Config.AUDIO_DIR)
    test_dataset = BreathingDataset(X_test, y_test, Config.AUDIO_DIR)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = BreathingCNN(
        input_height=Config.INPUT_HEIGHT,
        input_width=Config.INPUT_WIDTH,
        num_classes=Config.NUM_CLASSES,
        dropout_rate=Config.DROPOUT_RATE
    ).to(Config.DEVICE)
    
    print(f"\nModel Architecture:")
    print(f"  Input shape: (batch, 1, {Config.INPUT_HEIGHT}, {Config.INPUT_WIDTH})")
    print(f"  Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print(f"\nStarting training...")
    history = train_model(model, train_loader, val_loader, Config.NUM_EPOCHS, 
                         Config.LEARNING_RATE, Config.DEVICE)
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(Config.OUTPUT_DIR / 'best_model.pth'))
    
    # Evaluate model
    results = evaluate_model(model, test_loader, Config.DEVICE, class_names)
    
    # Save results and plots
    plot_training_history(history, Config.OUTPUT_DIR)
    plot_confusion_matrix(results['confusion_matrix'], class_names, Config.OUTPUT_DIR)
    save_results(results, Config.OUTPUT_DIR)
    
    # Save model and label encoder
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_height': Config.INPUT_HEIGHT,
            'input_width': Config.INPUT_WIDTH,
            'num_classes': Config.NUM_CLASSES,
            'dropout_rate': Config.DROPOUT_RATE
        },
        'label_encoder': label_encoder,
        'class_names': class_names
    }, Config.OUTPUT_DIR / 'complete_model.pth')
    
    print(f"\nResults saved to: {Config.OUTPUT_DIR}")
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
