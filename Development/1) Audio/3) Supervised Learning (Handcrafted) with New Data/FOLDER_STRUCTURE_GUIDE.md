# Transfer Learning with OPERA-CT - Folder Structure Guide

## 📁 ORGANIZED FOLDER STRUCTURE

```
Transfer Learning with OPERA-CT/
├── 01_Full_Frozen_Unsupervised/           # Clustering experiments (A0-D2)
├── 02_Full_Frozen_Breathing_Classification/  # Breathing vs non-breathing classification
├── 03_Layer_Analysis/                     # Different layer feature extraction
├── 04_Fine_Tuning/                        # (Future) Fine-tuned OPERA-CT
├── 05_Custom_Heads/                       # (Future) Custom classification heads
├── 06_Hybrid_Features/                    # (Future) OPERA-CT + handcrafted features
└── FOLDER_STRUCTURE_GUIDE.md             # This file
```

## 🎯 WHAT EACH FOLDER CONTAINS

### **01_Full_Frozen_Unsupervised/**
- **Purpose**: Unsupervised clustering with frozen OPERA-CT
- **Methods**: 16 preprocessing variations (A0-D2)
- **Task**: Audio clustering without labels
- **Best Result**: 0.267 silhouette (C1/D1)
- **Key Files**: OPERA_16_Methods/, results summaries

### **02_Full_Frozen_Breathing_Classification/** ⭐ CURRENT
- **Purpose**: Supervised breathing detection with frozen OPERA-CT
- **Methods**: OPERA-CT + Random Forest classifier
- **Task**: Breathing vs non-breathing classification
- **Best Result**: 68.8% accuracy
- **Key Files**: 
  - `breathing_classification_results/` - Main results
  - `improved_overlays/` - Fixed visualization issues
  - `README.md` - Experiment description

### **03_Layer_Analysis/**
- **Purpose**: Test different OPERA-CT layers for feature extraction
- **Methods**: Layer 0, 1, 2, 3 feature extraction
- **Task**: Find optimal layer for clustering
- **Best Result**: Layer 3 (final) = 0.265 silhouette
- **Status**: Completed (intermediate layers failed)

### **04_Fine_Tuning/** (Future)
- **Purpose**: Adapt OPERA-CT to respiratory data
- **Methods**: Unfreeze layers, train on our dataset
- **Task**: Improve foundation model for our domain
- **Expected**: Close the 34.2% performance gap

### **05_Custom_Heads/** (Future)
- **Purpose**: Add trainable layers on frozen OPERA-CT
- **Methods**: OPERA-CT → MLP/Linear → Classification
- **Task**: Task-specific adaptation
- **Expected**: Better than full frozen approach

### **06_Hybrid_Features/** (Future)
- **Purpose**: Combine OPERA-CT + handcrafted features
- **Methods**: Concatenate 768-dim + our engineered features
- **Task**: Best of both worlds
- **Expected**: Highest performance combining approaches

## 🔍 HOW TO IDENTIFY EXPERIMENT TYPE

### **Folder Naming Convention:**
```
[Approach]_[Model_State]_[Task_Type]/
```

**Examples:**
- `02_Full_Frozen_Breathing_Classification` = Full Frozen OPERA-CT for Breathing Classification
- `03_Layer_Analysis` = Different Layer Feature Extraction
- `04_Fine_Tuning` = Fine-tuned OPERA-CT (future)

### **README Files:**
Each folder contains a `README.md` explaining:
- Experiment type and approach
- Model configuration (frozen/fine-tuned/etc.)
- Task description
- Key results and performance
- File organization

## 📊 PERFORMANCE SUMMARY

| Folder | Approach | Task | Best Result | Status |
|--------|----------|------|-------------|---------|
| 01 | Full Frozen | Unsupervised Clustering | 0.267 silhouette | ✅ Complete |
| 02 | Full Frozen | Breathing Classification | 68.8% accuracy | ✅ Complete |
| 03 | Layer Analysis | Feature Extraction | 0.265 silhouette | ✅ Complete |
| 04 | Fine-Tuning | Adaptation | TBD | ❌ Future |
| 05 | Custom Heads | Task-Specific | TBD | ❌ Future |
| 06 | Hybrid | Feature Combination | TBD | ❌ Future |

---

*This structure makes it crystal clear what type of experiment each folder contains and how OPERA-CT was used in each case.*
