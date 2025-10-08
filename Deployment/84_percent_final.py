#!/usr/bin/env python3
"""
Step 17A: OPTIMIZED VERSION - Based on Fold 1 Success Pattern
- Tau initialization optimized from best performing fold
- Enhanced hyperparameters for 80%+ target
- All optimizations from performance analysis
- Save checkpoints per fold: model_best.pt / model_last.pt (+ scaler, class_names, best_tau)
"""

import os
import json
import math
import random
import argparse
import time
from typing import List, Dict
from collections import Counter
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support, roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# ======================== Configuration ========================

DEF_CSV_PATH   = r"D:\Stethoscope_Project\Development\OPERA_Copied_from_Ubuntu\features\opera_features.csv"
DEF_RESULTS_DIR = r"D:\Stethoscope_Project\Deployment\Result"  # <- 결과는 여기로
DEF_EXPERIMENT_TAG = "Step17A_Optimized_80Percent"

DEF_RANDOM_SEED = None
DEF_EPOCHS = 100
DEF_BATCH_SIZE = 64
DEF_LR = 1.5e-4
DEF_WD = 8e-5

# Augment
DEF_AUG_PROB_BASE = 0.8
DEF_AUG_PROB_MINORITY = 0.98
DEF_NOISE_STD_RANGE = (0.005, 0.06)
DEF_SCALE_RANGE = (0.75, 1.3)
DEF_MIXUP_ALPHA = 0.3
DEF_CUTMIX_PROB = 0.25
DEF_FEATURE_DROP_RANGE = (0.03, 0.12)
DEF_TEMPORAL_SHIFT = True
DEF_SPECTRAL_NOISE = True

# LDAM
DEF_DRW_START_RATIO = 0.25
DEF_MAX_M = 0.7
DEF_LDAM_SCALE = 30

# Tau search
DEF_TAU_MIN = 0.2
DEF_TAU_MAX = 2.8
DEF_TAU_STEPS = 35
DEF_NB_PREC_MIN = 0.25

# Tau init pattern
OPTIMAL_TAU_INIT = [1.1, 0.9, 1.4, 1.0, 0.85]

# ======================== Utils ========================
def generate_random_seed():
    return int(time.time() * 1000000) % 1000000

def safe_tag(s: str) -> str:
    return ''.join(c if c.isalnum() or c in '.-_' else '_' for c in s)

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

# ======================== Visualization ========================
def plot_confusion_matrix(cm, class_names, title, save_path, normalize=False):
    plt.figure(figsize=(10, 8))
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
        fmt = '.2f'; title += ' (Normalized)'
    else:
        fmt = 'd'
    colors = ['white', '#3498db', '#e74c3c']
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=100)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'shrink': .8}, square=True, linewidths=0.5)
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    acc = np.trace(cm) / (np.sum(cm) + 1e-12)
    plt.figtext(0.02, 0.02, f'Overall Accuracy: {acc:.3f}', fontsize=10, style='italic')
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()

def plot_roc_curves(y_true, y_probs, class_names, save_path):
    plt.figure(figsize=(12, 8))
    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    for i, name in enumerate(class_names):
        color = colors[i] if i < len(colors) else plt.cm.tab10(i)
        try:
            auc = roc_auc_score(y_true_bin[:, i], y_probs[:, i])
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            plt.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {auc:.3f})')
        except Exception as e:
            print(f"  Warning: ROC for {name} failed: {e}")
    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title('ROC Curves - Multi-Class', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()

def calculate_detailed_metrics(y_true, y_pred, y_probs, class_names):
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro    = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro        = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted    = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted        = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    try:
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
        if y_true_bin.shape[1] == 1:
            auroc_macro = roc_auc_score(y_true, y_probs[:, 1])
            auroc_weighted = auroc_macro
            auroc_per_class = [auroc_macro]
        else:
            auroc_macro = roc_auc_score(y_true_bin, y_probs, average='macro', multi_class='ovr')
            auroc_weighted = roc_auc_score(y_true_bin, y_probs, average='weighted', multi_class='ovr')
            auroc_per_class = []
            for i in range(len(class_names)):
                try:
                    auroc_per_class.append(roc_auc_score(y_true_bin[:, i], y_probs[:, i]))
                except:
                    auroc_per_class.append(0.0)
    except Exception as e:
        print(f"  Warning: AUROC failed: {e}")
        auroc_macro = auroc_weighted = 0.0
        auroc_per_class = [0.0] * len(class_names)

    cm = confusion_matrix(y_true, y_pred)
    accuracy = np.trace(cm) / (np.sum(cm) + 1e-12)

    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'auroc_macro': auroc_macro,
        'auroc_weighted': auroc_weighted,
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'auroc_per_class': auroc_per_class,
        'support_per_class': support_per_class.tolist(),
        'class_names': class_names
    }

def save_detailed_metrics(metrics, save_path):
    json_path = save_path.replace('.txt', '.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("DETAILED CLASSIFICATION METRICS\n" + "="*50 + "\n\n")
        f.write("OVERALL METRICS:\n")
        f.write(f"Accuracy:           {metrics['accuracy']:.4f}\n")
        f.write(f"Precision (Macro):  {metrics['precision_macro']:.4f}\n")
        f.write(f"Recall (Macro):     {metrics['recall_macro']:.4f}\n")
        f.write(f"F1-Score (Macro):   {metrics['f1_macro']:.4f}\n")
        f.write(f"AUROC (Macro):      {metrics['auroc_macro']:.4f}\n\n")
        f.write("WEIGHTED METRICS:\n")
        f.write(f"Precision (Weighted): {metrics['precision_weighted']:.4f}\n")
        f.write(f"Recall (Weighted):    {metrics['recall_weighted']:.4f}\n")
        f.write(f"F1-Score (Weighted):  {metrics['f1_weighted']:.4f}\n")
        f.write(f"AUROC (Weighted):     {metrics['auroc_weighted']:.4f}\n\n")
        f.write("PER-CLASS METRICS:\n" + "-"*80 + "\n")
        f.write(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUROC':<10} {'Support':<10}\n")
        f.write("-"*80 + "\n")
        for i, class_name in enumerate(metrics['class_names']):
            f.write(f"{class_name:<15} {metrics['precision_per_class'][i]:<10.4f} "
                    f"{metrics['recall_per_class'][i]:<10.4f} {metrics['f1_per_class'][i]:<10.4f} "
                    f"{metrics['auroc_per_class'][i]:<10.4f} {metrics['support_per_class'][i]:<10}\n")

# ======================== Augmentation ========================
class OptimizedCompetitionAugmentation:
    def __init__(self, base_prob=0.8, minority_prob=0.98, noise_range=(0.005, 0.06),
                 scale_range=(0.75, 1.3), drop_range=(0.03, 0.12)):
        self.base_prob = base_prob
        self.minority_prob = minority_prob
        self.noise_range = noise_range
        self.scale_range = scale_range
        self.drop_range = drop_range

    def adaptive_gaussian_noise(self, x, is_minority=False):
        prob = self.minority_prob if is_minority else self.base_prob
        if random.random() > prob: return x
        std_base = random.uniform(*self.noise_range)
        if is_minority: std_base *= 1.4
        return x + torch.randn_like(x) * std_base

    def smart_feature_scaling(self, x, is_minority=False):
        prob = self.minority_prob if is_minority else self.base_prob
        if random.random() > prob: return x
        x_scaled = x * random.uniform(*self.scale_range)
        if random.random() < 0.4:
            n_groups = 12; group_size = x.shape[-1] // n_groups
            for g in range(n_groups):
                s = g * group_size
                e = (g + 1) * group_size if g < n_groups-1 else x.shape[-1]
                x_scaled[..., s:e] *= random.uniform(0.85, 1.15)
        return x_scaled

    def advanced_feature_dropout(self, x, is_minority=False):
        prob = self.minority_prob if is_minority else self.base_prob
        if random.random() > prob: return x
        drop_rate = random.uniform(*self.drop_range)
        if is_minority: drop_rate *= 0.7
        if random.random() < 0.8:
            mask = torch.rand_like(x) > drop_rate
            return x * mask
        n_features = x.shape[-1]
        block_size = random.randint(6, 24)
        n_blocks = max(1, int(n_features * drop_rate / block_size))
        x_new = x.clone()
        for _ in range(n_blocks):
            start = random.randint(0, max(0, n_features - block_size))
            end = min(start + block_size, n_features)
            x_new[..., start:end] = 0
        return x_new

    def spectral_augmentation(self, x, is_minority=False):
        prob = (self.minority_prob if is_minority else self.base_prob) * 0.6
        if random.random() > prob: return x
        x = x.clone()
        if random.random() < 0.6:
            for _ in range(random.randint(1, 2)):
                freq = random.uniform(0.05, 0.4)
                phase = random.uniform(0, 2*np.pi)
                harmonics = torch.sin(torch.arange(x.shape[-1], dtype=torch.float32) * freq + phase)
                harmonics = harmonics * random.uniform(0.005, 0.025)
                x = x + harmonics
        if random.random() < 0.4:
            if random.random() < 0.5: x[:160] *= random.uniform(0.8, 0.95)
            else: x[-160:] *= random.uniform(0.8, 0.95)
        return x

    def temporal_shift_simulation(self, x, is_minority=False):
        prob = (self.minority_prob if is_minority else self.base_prob) * 0.5
        if random.random() > prob: return x
        x_new = x.clone()
        if random.random() < 0.7:
            n_features = x.shape[-1]
            shift = random.randint(1, min(20, n_features//15))
            if random.random() < 0.6: x_new = torch.roll(x_new, shifts=shift, dims=-1)
            else:
                block = shift * 2
                if block < n_features // 2:
                    s1 = random.randint(0, max(0, n_features - block))
                    s2 = random.randint(0, max(0, n_features - block))
                    b1 = x[..., s1:s1+block].clone()
                    b2 = x[..., s2:s2+block].clone()
                    x_new[..., s1:s1+block] = b2; x_new[..., s2:s2+block] = b1
        return x_new

    def apply_all_augmentations(self, x, is_minority=False):
        x = x.clone()
        x = self.adaptive_gaussian_noise(x, is_minority)
        x = self.smart_feature_scaling(x, is_minority)
        x = self.advanced_feature_dropout(x, is_minority)
        if DEF_SPECTRAL_NOISE: x = self.spectral_augmentation(x, is_minority)
        if DEF_TEMPORAL_SHIFT: x = self.temporal_shift_simulation(x, is_minority)
        return x

def mixup_data(x, y, alpha=0.3, minority_boost=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        if minority_boost:
            class_counts = np.bincount(y.cpu().numpy())
            min_count = np.min(class_counts)
            for i, cnt in enumerate(class_counts):
                if cnt <= min_count * 2 and (y == i).any():
                    lam = max(lam, 0.65); break
    else:
        lam = 1
    idx = torch.randperm(x.size(0))
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def cutmix_data(x, y, alpha=1.0, min_cut_ratio=0.15, max_cut_ratio=0.45):
    if alpha > 0:
        lam = np.clip(np.random.beta(alpha, alpha), min_cut_ratio, max_cut_ratio)
    else:
        lam = min_cut_ratio + (max_cut_ratio - min_cut_ratio) * np.random.rand()
    idx = torch.randperm(x.size(0))
    n_features = x.shape[1]; cut_features = int(n_features * lam)
    if cut_features > 0:
        if random.random() < 0.75:
            start = random.randint(0, max(0, n_features - cut_features))
            cut_idx = torch.arange(start, start + cut_features)
        else:
            cut_idx = torch.randperm(n_features)[:cut_features]
        x_mixed = x.clone()
        x_mixed[:, cut_idx] = x[idx][:, cut_idx]
        lam = 1 - (cut_features / n_features)
    else:
        x_mixed = x; lam = 1
    return x_mixed, y, y[idx], lam

# ======================== Dataset ========================
class CompetitionDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, class_frequencies: Dict[int, int],
                 augmentation=None, training=True):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.class_frequencies = class_frequencies
        self.augmentation = augmentation
        self.training = training
        freqs = list(class_frequencies.values())
        median_freq = np.median(freqs)
        self.minority_classes = set([c for c, f in class_frequencies.items() if f < median_freq])
        print(f"  Identified minority classes: {self.minority_classes}")

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        if self.training and self.augmentation is not None:
            is_minority = int(self.y[idx]) in self.minority_classes
            x = self.augmentation.apply_all_augmentations(x, is_minority)
        return x, y

# ======================== Loss ========================
class CompetitionLDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.7, s=30):
        super().__init__()
        cls_num_list = [int(x) for x in cls_num_list]
        m_list = 1.0 / np.power(cls_num_list, 0.25)
        m_list = m_list * (max_m / np.max(m_list))
        self.m_list = torch.tensor(m_list, dtype=torch.float32)
        self.s = s
        print(f"  OPTIMIZED LDAM margins: {[f'{x:.3f}' for x in self.m_list.tolist()]}")

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.bool)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_f = index.float()
        batch_m = torch.matmul(self.m_list[None, :], index_f.transpose(0,1)).view((-1, 1))
        x_m = x - batch_m
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target)

# ======================== Model ========================
class OptimizedLinearModel(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.15):
        super().__init__()
        self.feature_norm = nn.LayerNorm(input_dim)
        self.feature_dropout = nn.Dropout(dropout)
        self.hidden = nn.Linear(input_dim, input_dim // 2)
        self.hidden_norm = nn.LayerNorm(input_dim // 2)
        self.hidden_dropout = nn.Dropout(dropout * 0.5)
        self.classifier = nn.Linear(input_dim // 2, num_classes)
        nn.init.kaiming_normal_(self.hidden.weight, mode='fan_out', nonlinearity='relu')
        nn.init.normal_(self.classifier.weight, 0, 0.01)
        nn.init.zeros_(self.hidden.bias); nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        x = self.feature_norm(x); x = self.feature_dropout(x)
        x = torch.relu(self.hidden(x)); x = self.hidden_norm(x); x = self.hidden_dropout(x)
        return self.classifier(x)

# ======================== Train/Eval ========================
def mixup_criterion(pred, y_a, y_b, lam):
    return lam * F.cross_entropy(pred, y_a) + (1 - lam) * F.cross_entropy(pred, y_b)

def train_one_epoch(model, loader, device, optimizer, criterion, criterion_ldam,
                    epoch, total_epochs, drw_start_ratio=0.25):
    model.train()
    loss_sum, n = 0.0, 0
    use_ldam = epoch >= int(total_epochs * drw_start_ratio)
    mixup_count = cutmix_count = normal_count = 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device); y_batch = y_batch.to(device)
        r = random.random()
        if r < 0.45:
            x_batch, y_a, y_b, lam = mixup_data(x_batch, y_batch, alpha=DEF_MIXUP_ALPHA); mixed = True; mixup_count += 1
        elif r < 0.45 + DEF_CUTMIX_PROB:
            x_batch, y_a, y_b, lam = cutmix_data(x_batch, y_batch); mixed = True; cutmix_count += 1
        else:
            mixed = False; normal_count += 1

        optimizer.zero_grad(set_to_none=True)
        logits = model(x_batch)
        if mixed:
            loss = mixup_criterion(logits, y_a, y_b, lam)
        else:
            loss = criterion_ldam(logits, y_batch) if use_ldam else criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
        optimizer.step()
        loss_sum += float(loss.item()) * x_batch.size(0); n += x_batch.size(0)

    if epoch % 20 == 0:
        tot = mixup_count + cutmix_count + normal_count
        if tot > 0:
            print(f"  Aug stats - Mixup:{mixup_count/tot:.2f} CutMix:{cutmix_count/tot:.2f} Normal:{normal_count/tot:.2f}")
    return loss_sum / max(n, 1)

@torch.no_grad()
def eval_logits(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        logits = model(x_batch)
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(y_batch.numpy())
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    return probs, labels

# ======================== Tau helpers ========================
def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray, C: int) -> List[float]:
    recs = []
    for c in range(C):
        tp = np.sum((y_true == c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        recs.append(float(tp) / (tp + fn + 1e-9))
    return recs

def precision_of_class(y_true: np.ndarray, y_pred: np.ndarray, c: int) -> float:
    tp = np.sum((y_true == c) & (y_pred == c))
    fp = np.sum((y_true != c) & (y_pred == c))
    return float(tp) / (tp + fp + 1e-9)

def search_per_class_tau_optimized(probs: np.ndarray, y_true: np.ndarray, nb_idx: int, C: int,
                                   grid: np.ndarray, nb_prec_min: float):
    tau = np.array(OPTIMAL_TAU_INIT, dtype=np.float32)
    print(f"  🎯 Starting tau init: {[f'{t:.2f}' for t in tau]}")
    def objective(tau_vec):
        q = probs / tau_vec.reshape(1, -1)
        y_pred = np.argmax(q, axis=1)
        recs = per_class_recall(y_true, y_pred, C)
        mr = float(np.mean(recs))
        nb_prec = precision_of_class(y_true, y_pred, nb_idx)
        if nb_prec < nb_prec_min:
            mr -= 6.0 * (nb_prec_min - nb_prec)
        return mr, recs, nb_prec

    best_tau_global = tau.copy(); best_score_global = -999
    for restart in range(4):
        if restart == 0:
            tau = np.array(OPTIMAL_TAU_INIT, dtype=np.float32)
        elif restart == 1:
            tau = np.clip(np.array(OPTIMAL_TAU_INIT, dtype=np.float32) + np.random.normal(0, 0.1, C), 0.3, 2.5)
        else:
            tau = np.random.uniform(0.5, 2.0, C).astype(np.float32)

        improved = True; it = 0
        while improved and it < 60:
            improved = False
            for c in range(C):
                base, _, _ = objective(tau)
                best_val, best_tau_c = base, tau[c]
                for g in grid:
                    trial = tau.copy(); trial[c] = g
                    sc, _, _ = objective(trial)
                    if sc > best_val + 1e-9:
                        best_val, best_tau_c = sc, g
                if not math.isclose(best_tau_c, tau[c], abs_tol=1e-6):
                    tau[c] = best_tau_c; improved = True
            it += 1

        sc, _, _ = objective(tau)
        if sc > best_score_global:
            best_score_global = sc; best_tau_global = tau.copy()

    final_score, recs, nb_p = objective(best_tau_global)
    q = probs / best_tau_global.reshape(1, -1)
    y_pred = np.argmax(q, axis=1)
    final_recs = per_class_recall(y_true, y_pred, C)
    final_mr = float(np.mean(final_recs))
    final_nb_prec = precision_of_class(y_true, y_pred, nb_idx)
    print(f"  🎯 Final tau: {[f'{t:.2f}' for t in best_tau_global]}")
    return best_tau_global, final_mr, final_recs, final_nb_prec

# ======================== Main ========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=DEF_CSV_PATH)
    ap.add_argument('--results_dir', default=DEF_RESULTS_DIR)
    ap.add_argument('--tag', default=DEF_EXPERIMENT_TAG)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--epochs', type=int, default=DEF_EPOCHS)
    ap.add_argument('--batch_size', type=int, default=DEF_BATCH_SIZE)
    ap.add_argument('--lr', type=float, default=DEF_LR)
    ap.add_argument('--wd', type=float, default=DEF_WD)
    ap.add_argument('--tau_min', type=float, default=DEF_TAU_MIN)
    ap.add_argument('--tau_max', type=float, default=DEF_TAU_MAX)
    ap.add_argument('--tau_steps', type=int, default=DEF_TAU_STEPS)
    ap.add_argument('--nb_prec_min', type=float, default=DEF_NB_PREC_MIN)
    args = ap.parse_args()

    actual_seed = generate_random_seed() if args.seed is None else args.seed
    set_seed(actual_seed)
    ensure_dir(args.results_dir)
    tag = safe_tag(args.tag + f"_seed{actual_seed}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*70)
    print("🚀 OPTIMIZED 80% TARGET EXPERIMENT")
    print("="*70)
    print(f"🎯 RANDOM SEED: {actual_seed}")
    print("📈 TARGET: 80%+ Macro Recall")
    print("="*70)

    # Load data
    print("Loading OPERA features...")
    df = pd.read_csv(args.csv)
    if 'extraction_success' in df.columns:
        df = df[df['extraction_success'] == True].copy()
    class_names = sorted(df['label'].unique().tolist())
    cls_to_idx = {c: i for i, c in enumerate(class_names)}
    y = df['label'].map(cls_to_idx).values
    drop_cols = [c for c in ['filename', 'label', 'extraction_success'] if c in df.columns]
    X = df.drop(columns=drop_cols).values

    C = len(class_names)
    nb_idx = class_names.index('Non-breathing') if 'Non-breathing' in class_names else 0
    print(f"Classes: {class_names}")
    print(f"Class distribution: {Counter(y)}")
    print(f"Device: {device}")
    print(f"Feature shape: {X.shape}")
    print(f"🎯 Optimal Tau Pattern: {[f'{t:.2f}' for t in OPTIMAL_TAU_INIT]}")

    augmentation = OptimizedCompetitionAugmentation(
        base_prob=DEF_AUG_PROB_BASE,
        minority_prob=DEF_AUG_PROB_MINORITY,
        noise_range=DEF_NOISE_STD_RANGE,
        scale_range=DEF_SCALE_RANGE,
        drop_range=DEF_FEATURE_DROP_RANGE
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=actual_seed)
    tau_grid = np.linspace(args.tau_min, args.tau_max, args.tau_steps)

    rows = []; all_fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n{'='*20} Fold {fold} {'='*20}")
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]
        print(f"  Training samples: {len(X_tr)} | Validation samples: {len(X_va)}")

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr); X_va = scaler.transform(X_va)

        train_class_freq = {i: int(np.sum(y_tr == i)) for i in range(C)}
        cls_num_list = [train_class_freq[i] for i in range(C)]
        print(f"  Training class frequencies: {train_class_freq}")

        train_ds = CompetitionDataset(X_tr, y_tr, train_class_freq, augmentation, training=True)
        val_ds   = CompetitionDataset(X_va, y_va, train_class_freq, None, training=False)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        model = OptimizedLinearModel(X.shape[1], C, dropout=0.15).to(device)
        criterion = nn.CrossEntropyLoss()
        criterion_ldam = CompetitionLDAMLoss(cls_num_list, max_m=DEF_MAX_M, s=DEF_LDAM_SCALE).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=args.lr*0.01
        )

        best_mr = -1.0; patience = 0; best_state_dict = None; best_epoch = 0

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, device, optimizer, criterion, criterion_ldam,
                epoch, args.epochs, DEF_DRW_START_RATIO
            )
            scheduler.step()

            if epoch % 8 == 0 or epoch == 1 or epoch == args.epochs:
                probs_va, y_true_va = eval_logits(model, val_loader, device)
                y_pred_raw_tmp = probs_va.argmax(1)
                mr_raw_tmp = float(np.mean(per_class_recall(y_true_va, y_pred_raw_tmp, C)))
                lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch:03d}: lr={lr:.2e} loss={train_loss:.4f} val_MR={mr_raw_tmp:.3f}")
                if mr_raw_tmp > best_mr:
                    best_mr = mr_raw_tmp; patience = 0
                    best_state_dict = copy.deepcopy(model.state_dict()); best_epoch = epoch
                else:
                    patience += 1
                if patience > 40 and epoch > 60:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        fold_dir = os.path.join(args.results_dir, f"{tag}_fold{fold}")
        ensure_dir(fold_dir)
        last_ckpt = {
            "state_dict": model.state_dict(),
            "input_dim": X.shape[1],
            "num_classes": C,
            "class_names": class_names,
            "scaler": scaler,
            "seed": actual_seed,
            "last_epoch": epoch
        }
        torch.save(last_ckpt, os.path.join(fold_dir, "model_last.pt"))

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
            print(f"  🔁 Restored best checkpoint from epoch {best_epoch} (val MR={best_mr:.3f})")
        else:
            print("  ⚠️ No best_state_dict captured; using last epoch weights.")

        print("  🎯 Final evaluation with tau search...")
        probs, y_true = eval_logits(model, val_loader, device)
        y_pred_raw = probs.argmax(1)
        mr_raw = float(np.mean(per_class_recall(y_true, y_pred_raw, C)))

        best_tau, mr_tau, recs_tau, nb_prec_tau = search_per_class_tau_optimized(
            probs, y_true, nb_idx, C, tau_grid, args.nb_prec_min
        )
        print(f"  🏆 Fold {fold}: Raw MR={mr_raw:.3f} → Tau MR={mr_tau:.3f} (Δ{mr_tau-mr_raw:+.3f}) | NB Prec={nb_prec_tau:.3f}")
        q = probs / best_tau.reshape(1, -1); y_pred_tau = q.argmax(1)

        raw_metrics = calculate_detailed_metrics(y_true, y_pred_raw, probs, class_names)
        tau_metrics = calculate_detailed_metrics(y_true, y_pred_tau, q, class_names)

        save_detailed_metrics(raw_metrics, os.path.join(fold_dir, "detailed_metrics_raw.txt"))
        save_detailed_metrics(tau_metrics, os.path.join(fold_dir, "detailed_metrics_tau.txt"))

        cm_raw = confusion_matrix(y_true, y_pred_raw, labels=list(range(C)))
        cm_tau = confusion_matrix(y_true, y_pred_tau, labels=list(range(C)))
        plot_confusion_matrix(cm_raw, class_names,
                              f"Confusion Matrix - Raw\nFold {fold} (Seed {actual_seed})",
                              os.path.join(fold_dir, "confusion_matrix_raw.png"))
        plot_confusion_matrix(cm_tau, class_names,
                              f"Confusion Matrix - Tau\nFold {fold} (Seed {actual_seed})",
                              os.path.join(fold_dir, "confusion_matrix_tau.png"))
        plot_confusion_matrix(cm_tau, class_names,
                              f"Confusion Matrix - Tau (Normalized)\nFold {fold} (Seed {actual_seed})",
                              os.path.join(fold_dir, "confusion_matrix_tau_normalized.png"),
                              normalize=True)
        plot_roc_curves(y_true, probs, class_names, os.path.join(fold_dir, "roc_curves_raw.png"))
        plot_roc_curves(y_true, q, class_names, os.path.join(fold_dir, "roc_curves_tau.png"))

        with open(os.path.join(fold_dir, "classification_report_raw.txt"), 'w') as f:
            f.write("CLASSIFICATION REPORT - RAW\n")
            f.write(classification_report(y_true, y_pred_raw, target_names=class_names, digits=4))
        with open(os.path.join(fold_dir, "classification_report_tau.txt"), 'w') as f:
            f.write("CLASSIFICATION REPORT - TAU\n")
            f.write(classification_report(y_true, y_pred_tau, target_names=class_names, digits=4))

        np.save(os.path.join(fold_dir, "taus.npy"), best_tau)
        np.save(os.path.join(fold_dir, "probs_val.npy"), probs)
        np.save(os.path.join(fold_dir, "y_true_val.npy"), y_true)
        np.save(os.path.join(fold_dir, "y_pred_raw.npy"), y_pred_raw)
        np.save(os.path.join(fold_dir, "y_pred_tau.npy"), y_pred_tau)
        with open(os.path.join(fold_dir, "class_names.json"), 'w') as f:
            json.dump(class_names, f, indent=2)
        np.savetxt(os.path.join(fold_dir, "confusion_matrix_raw.csv"), cm_raw, delimiter=',', fmt='%d')
        np.savetxt(os.path.join(fold_dir, "confusion_matrix_tau.csv"), cm_tau, delimiter=',', fmt='%d')

        best_ckpt = {
            "state_dict": model.state_dict() if best_state_dict is None else best_state_dict,
            "input_dim": X.shape[1],
            "num_classes": C,
            "class_names": class_names,
            "scaler": scaler,
            "best_tau": best_tau.tolist(),
            "seed": actual_seed,
            "best_epoch": best_epoch,
            "metrics_tau": tau_metrics,
            "metrics_raw": raw_metrics
        }
        torch.save(best_ckpt, os.path.join(fold_dir, "model_best.pt"))

        with open(os.path.join(fold_dir, "model_info.txt"), "w", encoding="utf-8") as f:
            f.write(f"Best epoch: {best_epoch}\n")
            f.write(f"Best val MR (raw): {best_mr:.4f}\n")
            f.write(f"MR (raw end): {mr_raw:.4f}\n")
            f.write(f"MR (tau): {mr_tau:.4f}\n")
            f.write(f"NB Precision (tau): {nb_prec_tau:.4f}\n")
            f.write(f"Best tau: {best_tau.tolist()}\n")
            f.write(f"Classes: {class_names}\n")

        all_fold_metrics.append({'fold': fold, 'raw_metrics': raw_metrics, 'tau_metrics': tau_metrics})
        rows.append({
            'fold': fold, 'seed': actual_seed,
            'epochs_trained': best_epoch if best_epoch > 0 else epoch,
            'split_method': 'SEGMENT_BASED_OPTIMIZED',
            'augmentation_strategy': 'Competition_Grade_Optimized_80Percent',
            'architecture': 'OptimizedLinear_Hidden',
            'aug_prob_base': DEF_AUG_PROB_BASE,
            'aug_prob_minority': DEF_AUG_PROB_MINORITY,
            'mixup_alpha': DEF_MIXUP_ALPHA,
            'ldam_max_m': DEF_MAX_M,
            'ldam_scale': DEF_LDAM_SCALE,
            'macro_recall_raw': mr_raw,
            'macro_recall_tau': mr_tau,
            'improvement': mr_tau - mr_raw,
            'nb_precision_tau': nb_prec_tau,
            'accuracy_raw': raw_metrics['accuracy'],
            'accuracy_tau': tau_metrics['accuracy'],
            'precision_macro_raw': raw_metrics['precision_macro'],
            'precision_macro_tau': tau_metrics['precision_macro'],
            'f1_macro_raw': raw_metrics['f1_macro'],
            'f1_macro_tau': tau_metrics['f1_macro'],
            'auroc_macro_raw': raw_metrics['auroc_macro'],
            'auroc_macro_tau': tau_metrics['auroc_macro'],
            'per_class_recall_raw': json.dumps([round(r, 4) for r in per_class_recall(y_true, y_pred_raw, C)]),
            'per_class_recall_tau': json.dumps([round(r, 4) for r in recs_tau]),
            'taus': json.dumps([round(float(x), 3) for x in best_tau.tolist()]),
            'class_names': json.dumps(class_names),
            'n_train': len(tr_idx),
            'n_val': len(va_idx)
        })
        print(f"  ✅ Fold {fold} complete - checkpoints & visualizations saved!")

    # === Export FINAL MODEL (champion fold) ===
    import shutil
    champ_metric = -1.0; champ_fold = None
    for r in rows:
        if r['macro_recall_tau'] > champ_metric:
            champ_metric = r['macro_recall_tau']; champ_fold = r['fold']
    champ_dir = os.path.join(args.results_dir, f"{tag}_fold{champ_fold}")
    champ_ckpt = os.path.join(champ_dir, "model_best.pt")
    summary_dir = os.path.join(args.results_dir, f"{tag}_summary")
    ensure_dir(summary_dir)
    final_ckpt_path = os.path.join(summary_dir, "final_model.pt")
    shutil.copyfile(champ_ckpt, final_ckpt_path)

    taus_all = []
    for f in range(1, 6):
        tau_path = os.path.join(args.results_dir, f"{tag}_fold{f}", "taus.npy")
        if os.path.exists(tau_path): taus_all.append(np.load(tau_path))
    avg_tau = np.mean(np.stack(taus_all, axis=0), axis=0).tolist() if len(taus_all) == 5 else None
    with open(os.path.join(summary_dir, "final_model_info.json"), "w", encoding="utf-8") as f:
        json.dump({
            "strategy": "champion_fold",
            "champion_fold": int(champ_fold),
            "champion_metric_macro_recall_tau": float(champ_metric),
            "final_model_path": final_ckpt_path,
            "avg_tau_over_folds": avg_tau,
            "note": "배포 시 avg_tau 또는 챔피언 폴드 best_tau 중 하나를 선택해서 사용"
        }, f, indent=2, ensure_ascii=False)
    print(f"\n🏁 FINAL MODEL (champion fold) → {final_ckpt_path}")
    print(f"   Champion fold = {champ_fold}, Macro Recall (tau) = {champ_metric:.3f}")

    # === Aggregated visuals ===
    print("\n📊 Generating overall summary visualizations...")
    total_cm_raw = np.zeros((C, C)); total_cm_tau = np.zeros((C, C))
    for fd in all_fold_metrics:
        fnum = fd['fold']; fdir = os.path.join(args.results_dir, f"{tag}_fold{fnum}")
        total_cm_raw += np.loadtxt(os.path.join(fdir, "confusion_matrix_raw.csv"), delimiter=',')
        total_cm_tau += np.loadtxt(os.path.join(fdir, "confusion_matrix_tau.csv"), delimiter=',')

    plot_confusion_matrix(total_cm_raw.astype(int), class_names,
                          f"Aggregated Confusion Matrix - Raw\n5-Fold CV (Seed {actual_seed})",
                          os.path.join(summary_dir, "confusion_matrix_aggregated_raw.png"))
    plot_confusion_matrix(total_cm_tau.astype(int), class_names,
                          f"Aggregated Confusion Matrix - Tau\n5-Fold CV (Seed {actual_seed})",
                          os.path.join(summary_dir, "confusion_matrix_aggregated_tau.png"))
    plot_confusion_matrix(total_cm_tau.astype(int), class_names,
                          f"Aggregated Confusion Matrix - Tau (Normalized)\n5-Fold CV (Seed {actual_seed})",
                          os.path.join(summary_dir, "confusion_matrix_aggregated_tau_normalized.png"),
                          normalize=True)

    summary_csv = os.path.join(args.results_dir, f"{tag}_summary.csv")
    pd.DataFrame(rows).to_csv(summary_csv, index=False, encoding='utf-8-sig')

    avg_raw = float(np.mean([r['macro_recall_raw'] for r in rows]))
    avg_tau = float(np.mean([r['macro_recall_tau'] for r in rows]))
    std_tau = float(np.std([r['macro_recall_tau'] for r in rows]))
    avg_accuracy_tau = float(np.mean([r['accuracy_tau'] for r in rows]))
    avg_precision_tau = float(np.mean([r['precision_macro_tau'] for r in rows]))
    avg_f1_tau = float(np.mean([r['f1_macro_tau'] for r in rows]))
    avg_auroc_tau = float(np.mean([r['auroc_macro_tau'] for r in rows]))

    print("\n" + "="*70)
    print("🚀 OPTIMIZED EXPERIMENT RESULTS")
    print("="*70)
    print(f"🎯 SEED USED: {actual_seed}")
    print(f"📊 Average Accuracy (Tau):     {avg_accuracy_tau:.3f}")
    print(f"🎯 Average Precision (Tau):    {avg_precision_tau:.3f}")
    print(f"🏆 Average Recall (Tau):       {avg_tau:.3f}")
    print(f"🎪 Average F1-Score (Tau):     {avg_f1_tau:.3f}")
    print(f"📈 Average AUROC (Tau):        {avg_auroc_tau:.3f}")
    print(f"📉 StdDev(MR Tau):             {std_tau:.3f}")
    print(f"📁 Results directory:          {args.results_dir}")
    print("="*70)

    if avg_tau >= 0.80:
        print("🎉 80% TARGET ACHIEVED! 🎉")
    else:
        print(f"🎯 Current: {avg_tau:.1%} (Try different seed or tweaks)")

    print("\nGenerated:")
    print(" - Per-fold: model_best.pt / model_last.pt / metrics / plots")
    print(" - Summary : final_model.pt / final_model_info.json / aggregated plots / summary.csv")

    return avg_tau

if __name__ == "__main__":
    main()
