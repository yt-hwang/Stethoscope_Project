#!/usr/bin/env python3
"""
Phase 2 - Training Script
Trains classification head on extracted features.
"""

import argparse
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from run_manager import RunManager

class FeatureDataset(Dataset):
    """Dataset for features and labels."""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class LinearHead(nn.Module):
    """Linear classification head."""
    
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.classifier(x)

class MLPHead(nn.Module):
    """MLP classification head."""
    
    def __init__(self, input_dim, num_classes, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)

class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience=10, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif (self.mode == 'max' and score < self.best_score + self.min_delta) or \
             (self.mode == 'min' and score > self.best_score - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop

def load_features_and_labels(features_path):
    """Load features and labels from parquet file."""
    df = pd.read_parquet(features_path)
    
    # Extract features (columns starting with 'feature_')
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    features = df[feature_cols].values
    
    # Extract labels
    labels = df['label'].values
    filenames = df['filename'].values
    
    return features, labels, filenames

def create_model(head_type, input_dim, num_classes):
    """Create classification model."""
    if head_type == 'linear':
        return LinearHead(input_dim, num_classes)
    elif head_type == 'mlp':
        return MLPHead(input_dim, num_classes)
    else:
        raise ValueError(f"Unknown head type: {head_type}")

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for features, labels in tqdm(dataloader, desc="Training", leave=False):
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, accuracy, f1

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Validation", leave=False):
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, accuracy, f1

def plot_training_curves(history, save_dir):
    """Plot and save training curves."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # F1 curves
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_f1'], 'b-', label='Train F1')
    plt.plot(epochs, history['val_f1'], 'r-', label='Val F1')
    plt.title('Training and Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'loss.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Separate F1 plot
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history['train_f1'], 'b-', label='Train F1', linewidth=2)
    plt.plot(epochs, history['val_f1'], 'r-', label='Val F1', linewidth=2)
    plt.title('F1 Score Over Training', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('F1 Score (Macro)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'f1_macro.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train classification head")
    parser.add_argument("--run_id", required=True, help="Run ID")
    parser.add_argument("--config", help="Config file (optional)")
    
    args = parser.parse_args()
    
    # Load configuration
    manager = RunManager()
    run_paths = manager.get_run_paths(args.run_id)
    
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config_path = run_paths['artifacts'] / "config_dump.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    print(f"🚀 Starting training for run: {args.run_id}")
    
    # Set device and seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(config.get('seed', 42))
    np.random.seed(config.get('seed', 42))
    
    print(f"🔧 Device: {device}")
    
    # Load training and validation data
    train_features, train_labels, train_filenames = load_features_and_labels(
        run_paths['features'] / 'train.parquet'
    )
    val_features, val_labels, val_filenames = load_features_and_labels(
        run_paths['features'] / 'val.parquet'
    )
    
    print(f"📊 Train samples: {len(train_features)}")
    print(f"📊 Val samples: {len(val_features)}")
    print(f"📊 Feature dimension: {train_features.shape[1]}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    all_labels = np.concatenate([train_labels, val_labels])
    label_encoder.fit(all_labels)
    
    train_labels_encoded = label_encoder.transform(train_labels)
    val_labels_encoded = label_encoder.transform(val_labels)
    
    num_classes = len(label_encoder.classes_)
    print(f"📊 Classes: {list(label_encoder.classes_)}")
    print(f"📊 Number of classes: {num_classes}")
    
    # Compute class weights for imbalanced data
    class_weights = compute_class_weight(
        'balanced', classes=np.unique(train_labels_encoded), y=train_labels_encoded
    )
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"⚖️ Class weights: {class_weights.cpu().numpy()}")
    
    # Create datasets and dataloaders
    train_config = config.get('train', {})
    batch_size = train_config.get('batch_size', 32)
    
    train_dataset = FeatureDataset(train_features, train_labels_encoded)
    val_dataset = FeatureDataset(val_features, val_labels_encoded)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    head_type = train_config.get('head', 'linear')
    model = create_model(head_type, train_features.shape[1], num_classes)
    model = model.to(device)
    
    print(f"🧠 Model: {head_type} head")
    print(f"🧠 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=train_config.get('lr', 1e-3))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    early_stopping = EarlyStopping(patience=10, mode='max')
    
    # Training loop
    max_epochs = train_config.get('epochs', 50)
    history = {
        'train_loss': [], 'val_loss': [],
        'train_f1': [], 'val_f1': [],
        'train_acc': [], 'val_acc': []
    }
    
    best_val_f1 = 0
    best_epoch = 0
    
    print(f"🎯 Starting training for {max_epochs} epochs...")
    
    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch+1}/{max_epochs}")
        
        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, val_f1 = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Print metrics
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'label_encoder': label_encoder
            }, run_paths['train'] / 'ckpts' / 'best.ckpt')
        
        # Save last checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_f1': val_f1,
            'label_encoder': label_encoder,
            'history': history
        }, run_paths['train'] / 'ckpts' / 'last.ckpt')
        
        # Learning rate scheduling
        scheduler.step(val_f1)
        
        # Early stopping
        if early_stopping(val_f1):
            print(f"🛑 Early stopping at epoch {epoch+1}")
            break
    
    print(f"\n✅ Training completed!")
    print(f"🏆 Best validation F1: {best_val_f1:.4f} at epoch {best_epoch+1}")
    
    # Save training metrics
    metrics = {
        'best_val_f1': float(best_val_f1),
        'best_epoch': int(best_epoch),
        'final_train_f1': float(history['train_f1'][-1]),
        'final_val_f1': float(history['val_f1'][-1]),
        'num_epochs': len(history['train_loss']),
        'num_classes': int(num_classes),
        'classes': list(label_encoder.classes_),
        'class_weights': class_weights.cpu().numpy().tolist()
    }
    
    metrics_path = run_paths['train'] / 'metrics_train_val.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Plot training curves
    plot_training_curves(history, run_paths['train'] / 'curves')
    
    print(f"💾 Metrics saved: {metrics_path}")
    print(f"📊 Curves saved: {run_paths['train'] / 'curves'}")
    print(f"🎯 Best checkpoint: {run_paths['train'] / 'ckpts' / 'best.ckpt'}")

if __name__ == "__main__":
    main()
