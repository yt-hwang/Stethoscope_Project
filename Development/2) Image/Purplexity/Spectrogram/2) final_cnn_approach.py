

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ENHANCED CNN WITH VISUALIZATIONS - Saves confusion matrix and training curves

Adds comprehensive visualization saving to your successful CNN approach:
- Confusion matrix heatmap
- Training/validation curves
- Per-class performance chart
- Classification report as text file
"""

import os, sys, json, time, random
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ===== PATHS TO YOUR IMPROVED SPECTROGRAMS =====
IMG_ROOT = Path("D:\\Stethoscope_Project\\Development\\2) Image\\Purplexity\\Spectrogram\\Processed Data")
OUT_ROOT = Path("D:\\Stethoscope_Project\\Development\\2) Image\\Purplexity\\Spectrogram\\CNN_Fixed_Error")

# Parameters optimized for your balanced dataset
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS_LP = 5
EPOCHS_FT = 20
EARLY_STOP = 5
LR_HEAD = 2e-4
LR_BACKBONE = 1e-5
WEIGHT_DECAY = 5e-3
LABEL_SMOOTHING = 0.1
DROPOUT_RATE = 0.4
NUM_WORKERS = 0

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def list_images_from_dirs(root: Path):
    """List images from directory structure"""
    items = []
    for label_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        label = label_dir.name
        for p in label_dir.rglob("*"):
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                pid = p.stem.split("_")[0] if "_" in p.stem else p.stem
                items.append((str(p), label, pid))
    return pd.DataFrame(items, columns=["path","label","patient_id"])

class ImgDataset(Dataset):
    def __init__(self, df, classes, img_size=224, train=True):
        self.df = df.reset_index(drop=True)
        self.classes = classes
        self.cls2idx = {c:i for i,c in enumerate(classes)}
        
        if train:
            self.tx = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.03, contrast=0.03, 
                                         saturation=0.02, hue=0.01)
                ], p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])
        else:
            self.tx = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])
    
    def __len__(self): 
        return len(self.df)
    
    def __getitem__(self, i):
        r = self.df.iloc[i]
        x = Image.open(r.path).convert("RGB")
        x = self.tx(x)
        y = self.cls2idx[r.label]
        return x, y, r.path

def load_backbone_optimized(num_classes: int):
    """Load EfficientNet optimized for your dataset"""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(DROPOUT_RATE),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(DROPOUT_RATE),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(DROPOUT_RATE),
        nn.Linear(128, num_classes)
    )
    
    return model

def compute_class_weights(labels, classes):
    """Compute balanced class weights"""
    counts = Counter(labels)
    total = sum(counts[c] for c in classes)
    weights = []
    for c in classes:
        cw = total / max(1, counts[c])
        weights.append(cw)
    
    w = torch.tensor(weights, dtype=torch.float32)
    return w

@torch.no_grad()
def eval_epoch(model, dl, device, criterion):
    model.eval()
    total_loss, n = 0.0, 0
    ys, ps = [], []
    
    for x, y, _ in dl:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        
        ys.append(y.cpu().numpy())
        ps.append(logits.softmax(1).cpu().numpy())
    
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    pred = p.argmax(1)
    
    macro_f1 = f1_score(y, pred, average="macro", zero_division=0)
    accuracy = (y == pred).mean()
    
    return total_loss/n, macro_f1, accuracy, y, pred

def train_epoch(model, dl, device, criterion, optimizer):
    model.train()
    total_loss, n = 0.0, 0
    ys, ps = [], []
    
    for x, y, _ in dl:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        
        ys.append(y.detach().cpu().numpy())
        ps.append(logits.detach().softmax(1).cpu().numpy())
    
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    pred = p.argmax(1)
    macro_f1 = f1_score(y, pred, average="macro", zero_division=0)
    accuracy = (y == pred).mean()
    
    return total_loss/n, macro_f1, accuracy

def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    """Create and save confusion matrix heatmap"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix - Respiratory Sound Classification', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    
    # Add accuracy annotations
    accuracy = (y_true == y_pred).mean()
    plt.figtext(0.02, 0.02, f'Overall Accuracy: {accuracy:.3f}', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   📊 Confusion matrix saved: {save_path}")

def plot_training_curves(history, save_path):
    """Create and save training/validation curves"""
    epochs = range(1, len(history['train_acc']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy curves
    ax1.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax1.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add best accuracy annotation
    best_val_idx = np.argmax(history['val_acc'])
    best_val_acc = history['val_acc'][best_val_idx]
    ax1.annotate(f'Best: {best_val_acc:.3f}', 
                xy=(best_val_idx + 1, best_val_acc),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Loss curves
    ax2.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax2.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   📈 Training curves saved: {save_path}")

def plot_class_performance(y_true, y_pred, classes, save_path):
    """Create and save per-class performance chart"""
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    
    # Create DataFrame for easy plotting
    metrics_df = pd.DataFrame({
        'Class': classes,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar chart of precision, recall, F1
    x = np.arange(len(classes))
    width = 0.25
    
    ax1.bar(x - width, precision, width, label='Precision', alpha=0.8, color='skyblue')
    ax1.bar(x, recall, width, label='Recall', alpha=0.8, color='lightcoral')
    ax1.bar(x + width, f1, width, label='F1-Score', alpha=0.8, color='lightgreen')
    
    ax1.set_xlabel('Classes')
    ax1.set_ylabel('Score')
    ax1.set_title('Per-Class Performance Metrics', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.0)
    
    # Add value labels on bars
    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        ax1.text(i - width, p + 0.01, f'{p:.2f}', ha='center', va='bottom', fontsize=10)
        ax1.text(i, r + 0.01, f'{r:.2f}', ha='center', va='bottom', fontsize=10)
        ax1.text(i + width, f + 0.01, f'{f:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Support (sample count) chart
    bars = ax2.bar(classes, support, color='gold', alpha=0.7)
    ax2.set_xlabel('Classes')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('Sample Count per Class (Test Set)', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars, support):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(count)}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   📊 Class performance chart saved: {save_path}")

def save_classification_report(y_true, y_pred, classes, save_path):
    """Save classification report as text file"""
    report = classification_report(y_true, y_pred, target_names=classes, zero_division=0)
    
    with open(save_path, 'w') as f:
        f.write("RESPIRATORY SOUND CLASSIFICATION REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(report)
        f.write(f"\n\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"   📝 Classification report saved: {save_path}")

def main():
    set_seed(42)
    
    print("🎯 CNN WITH COMPREHENSIVE VISUALIZATIONS")
    print("="*60)
    print("Saves: Confusion matrix, training curves, class performance")
    
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    # Load improved data
    if not IMG_ROOT.exists():
        print(f"❌ ERROR: Improved spectrogram directory not found!")
        print(f"   Looking for: {IMG_ROOT}")
        return
    
    df = list_images_from_dirs(IMG_ROOT)
    
    print(f"\n📊 Dataset loaded:")
    print(f"   Total samples: {len(df)}")
    class_counts = df['label'].value_counts()
    print(f"   Class distribution:")
    for cls, count in class_counts.items():
        print(f"     {cls}: {count}")
    
    if len(df) < 50:
        print(f"   ❌ Not enough samples")
        return
    
    # Filter classes
    min_per_class = 15
    keep_classes = class_counts[class_counts >= min_per_class].index.tolist()
    df_final = df[df['label'].isin(keep_classes)].copy()
    
    classes = sorted(df_final['label'].unique())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n📊 Training dataset:")
    print(f"   Total samples: {len(df_final)}")
    print(f"   Classes: {classes}")
    
    # Train/validation split
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        df_final, test_size=0.2, stratify=df_final['label'], random_state=42
    )
    
    print(f"\n🔄 Data split:")
    print(f"   Train: {len(train_df)}")
    print(f"   Validation: {len(val_df)}")
    
    # Datasets and loaders
    ds_tr = ImgDataset(train_df, classes, img_size=IMG_SIZE, train=True)
    ds_va = ImgDataset(val_df, classes, img_size=IMG_SIZE, train=False)
    
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # Model
    model = load_backbone_optimized(num_classes=len(classes)).to(device)
    
    # Freeze backbone initially
    for p in model.features.parameters():
        p.requires_grad_(False)
    
    # Loss with class weights
    class_weights = compute_class_weights(train_df['label'].tolist(), classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    
    # Training history for curves
    history = {
        'train_acc': [], 'val_acc': [],
        'train_loss': [], 'val_loss': [],
        'train_f1': [], 'val_f1': []
    }
    
    # Phase 1: Linear probe
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=WEIGHT_DECAY
    )
    
    best_acc = 0.0
    stamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    best_path = OUT_ROOT / f"best_model_{stamp}.pt"
    
    print(f"\n🔧 Phase 1: Linear Probe")
    for epoch in range(1, EPOCHS_LP + 1):
        tr_loss, tr_f1, tr_acc = train_epoch(model, dl_tr, device, criterion, optimizer)
        va_loss, va_f1, va_acc, _, _ = eval_epoch(model, dl_va, device, criterion)
        
        # Record history
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(va_acc)
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_f1'].append(tr_f1)
        history['val_f1'].append(va_f1)
        
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), best_path)
        
        print(f"  [LP {epoch}] Train: {tr_acc:.3f} | Val: {va_acc:.3f}")
    
    # Phase 2: Fine-tune
    model.load_state_dict(torch.load(best_path, map_location=device))
    
    # Unfreeze last blocks
    for block in list(model.features.children())[-2:]:
        for p in block.parameters():
            p.requires_grad_(True)
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_BACKBONE, weight_decay=WEIGHT_DECAY
    )
    
    print(f"\n🎯 Phase 2: Fine-tuning")
    no_improve = 0
    for epoch in range(1, EPOCHS_FT + 1):
        tr_loss, tr_f1, tr_acc = train_epoch(model, dl_tr, device, criterion, optimizer)
        va_loss, va_f1, va_acc, _, _ = eval_epoch(model, dl_va, device, criterion)
        
        # Record history
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(va_acc)
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_f1'].append(tr_f1)
        history['val_f1'].append(va_f1)
        
        improved = va_acc > best_acc
        if improved:
            best_acc = va_acc
            torch.save(model.state_dict(), best_path)
            no_improve = 0
        else:
            no_improve += 1
        
        print(f"  [FT {epoch}] Train: {tr_acc:.3f} | Val: {va_acc:.3f} {'(*)' if improved else ''}")
        
        if no_improve >= EARLY_STOP:
            print(f"  Early stopping at epoch {epoch}")
            break
    
    # Final evaluation with predictions
    model.load_state_dict(torch.load(best_path, map_location=device))
    va_loss, va_f1, va_acc, y_true, y_pred = eval_epoch(model, dl_va, device, criterion)
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Best Accuracy: {best_acc:.1%}")
    print(f"   Final F1: {va_f1:.3f}")
    
    # SAVE ALL VISUALIZATIONS
    print(f"\n💾 Saving visualizations...")
    
    # 1. Confusion Matrix
    cm_path = OUT_ROOT / f"confusion_matrix_{stamp}.png"
    plot_confusion_matrix(y_true, y_pred, classes, cm_path)
    
    # 2. Training Curves
    curves_path = OUT_ROOT / f"training_curves_{stamp}.png"
    plot_training_curves(history, curves_path)
    
    # 3. Class Performance
    perf_path = OUT_ROOT / f"class_performance_{stamp}.png"
    plot_class_performance(y_true, y_pred, classes, perf_path)
    
    # 4. Classification Report
    report_path = OUT_ROOT / f"classification_report_{stamp}.txt"
    save_classification_report(y_true, y_pred, classes, report_path)
    
    # 5. Results JSON
    results = {
        'best_accuracy': float(best_acc),
        'final_f1': float(va_f1),
        'approach': 'CNN with Comprehensive Visualizations',
        'dataset_size': int(len(df_final)),
        'train_size': int(len(train_df)),
        'val_size': int(len(val_df)),
        'classes': [str(c) for c in classes],
        'class_distribution': {str(k): int(v) for k, v in class_counts.items()},
        'training_history': {
            'epochs_total': len(history['val_acc']),
            'best_epoch': int(np.argmax(history['val_acc']) + 1),
            'final_train_acc': float(history['train_acc'][-1]),
            'final_val_acc': float(history['val_acc'][-1])
        }
    }
    
    with open(OUT_ROOT / f'complete_results_{stamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 All files saved to: {OUT_ROOT}")
    print(f"   • Model: {best_path.name}")
    print(f"   • Confusion Matrix: {cm_path.name}")
    print(f"   • Training Curves: {curves_path.name}")
    print(f"   • Class Performance: {perf_path.name}")
    print(f"   • Classification Report: {report_path.name}")
    print(f"   • Complete Results: complete_results_{stamp}.json")
    
    print(f"\n🎉 Training complete with full visualization suite!")

if __name__ == "__main__":
    main()
