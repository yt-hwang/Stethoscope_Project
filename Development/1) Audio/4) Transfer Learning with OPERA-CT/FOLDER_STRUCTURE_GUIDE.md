# OPERA-CT Transfer Learning - Folder Structure Guide

## 🎯 Overview

This folder contains all experiments using OPERA-CT (Contrastive Learning-based Transformer for Respiratory Acoustic Analysis) for transfer learning on respiratory audio data.

## 📁 Main Structure

```
4) Transfer Learning with OPERA-CT/
├── 01_Unsupervised_Learning/          # Clustering experiments
├── 02_Supervised_Learning/            # Classification experiments  
├── 03_Hybrid_Approaches/              # Combined methods
└── FOLDER_STRUCTURE_GUIDE.md          # This file
```

## 🔍 Detailed Structure

### 01_Unsupervised_Learning/
**Purpose**: Clustering respiratory audio segments using OPERA-CT features

```
01_Unsupervised_Learning/
├── 01_Baseline/                       # Single method, default preprocessing
│   ├── results/                       # UMAP plots, clustering results
│   ├── scripts/                       # OPERA-CT pipeline scripts
│   └── README.md                      # Baseline experiment details
├── 02_16_Methods/                     # 16 preprocessing combinations
│   ├── A0_NoSeg_NoPre/               # Individual methods (A0-A4)
│   ├── B0_NoSeg_Bandpass_SpectralGating/ # Combination methods (B0-B2)
│   ├── C0_Seg_NoPre/                 # Segmentation methods (C0-C4)
│   ├── D0_Seg_HighPass_PeakNormalize/ # Seg + combination (D0-D2)
│   ├── FINAL_ANALYSIS.md             # Comprehensive results
│   └── results_summary.csv           # Performance comparison
└── 03_Layer_Analysis/                # Different OPERA-CT layers
    ├── Layer_0_Early/                # Early layer features
    ├── Layer_1_Middle/               # Middle layer features
    ├── Layer_2_Late/                 # Late layer features
    ├── Layer_3_Final/                # Final layer features
    └── Layer_Comparison/             # Layer comparison results
```

### 02_Supervised_Learning/
**Purpose**: Classification tasks using OPERA-CT features

```
02_Supervised_Learning/
├── 01_Breathing_Classification/       # Breathing vs non-breathing
│   ├── Individual_Models/            # RF, SVM, LR results
│   ├── Ensemble_Models/              # Ensemble method results
│   ├── scripts/                      # Classification scripts
│   └── Classification_Experiments_Master_Log.xlsx
├── 02_Fine_Tuning/                   # Fine-tuning experiments
└── 03_Custom_Heads/                  # Custom head architectures
```

### 03_Hybrid_Approaches/
**Purpose**: Combine OPERA-CT with other methods

```
03_Hybrid_Approaches/
├── 01_OPERA_CT_Plus_Handcrafted/     # Feature fusion
└── 02_Ensemble_Methods/              # Model ensembles
```

## 🏆 Key Results Summary

### Unsupervised Learning
- **Best Method**: D1/C1 (Seg + HighPass + Bandpass) = 0.267 silhouette
- **Improvement**: +36.2% over baseline
- **Key Insight**: Segmentation is critical for OPERA-CT

### Supervised Learning
- **Best Model**: Random Forest (varies by segment size)
- **Best Segment Size**: 0.5s often performs best
- **Visualization**: Consistent timeline format across all methods

## 🔬 Scientific Insights

1. **Domain Expertise Matters**: Our preprocessing improves OPERA-CT by 36.2%
2. **Segmentation is Critical**: +7.0% improvement with segmentation
3. **Simple > Complex**: Fewer preprocessing steps often perform better
4. **Foundation Models Benefit**: From domain-specific preprocessing

## 📊 Performance Comparison

| Method | Silhouette Score | Type |
|--------|------------------|------|
| D1 (OPERA-CT + Our Preprocessing) | 0.267 | Unsupervised |
| C1 (OPERA-CT + Seg + Bandpass) | 0.267 | Unsupervised |
| A0 (OPERA-CT Baseline) | 0.196 | Unsupervised |
| Handcrafted Features (Best) | 0.406 | Unsupervised |

## 🎯 Next Steps

1. **Fine-tuning**: Test if adaptation can close the performance gap
2. **Hybrid Approaches**: Combine OPERA-CT with handcrafted features
3. **Custom Architectures**: Test different head designs
4. **Multi-Modal**: Combine with metadata and other features

---

*Last Updated: September 2025*
*Total Experiments: 20+ (16 unsupervised + 4+ supervised)*
*Total UMAP Plots: 50+ (16 methods × 3 k-values + additional)*