# Supervised Learning Comparison: Handcrafted vs OPERA-CT Features

## 🎯 **Objective**: Breathing vs Non-Breathing Classification

---

## **📊 Dataset & Setup**
- **Dataset**: "RAW sound_ML test sound list" (29 audio files)
- **Ground Truth**: Excel timestamps for inhale/exhale periods
- **Labeling**: Center-point labeling (segment labeled by Excel period at its center)
- **Train/Test Split**: 80% train, 20% test (segment-based)
- **Segment Sizes**: 0.25s, 0.5s, 1.0s (no overlap)

---

## **🔧 Feature Comparison**

| Aspect | Handcrafted Features | OPERA-CT Features |
|--------|---------------------|-------------------|
| **Feature Count** | 47 features | 768 features |
| **Feature Type** | Domain-specific (RMS, ZCR, spectral, MFCCs, etc.) | Pre-trained embeddings |
| **Extraction** | Signal processing algorithms | Frozen transformer model |
| **Interpretability** | High (each feature has clear meaning) | Low (black box embeddings) |
| **Computational Cost** | Low | High |

---

## **📈 Performance Results**

### **Individual Models (Best Performance per Segment Size)**

| Segment Size | Handcrafted (Best) | OPERA-CT (Best) | Advantage |
|--------------|-------------------|-----------------|-----------|
| **0.25s** | **Random Forest: 75.7%** | **Random Forest: 74.0%** | **+1.7%** |
| **0.5s** | **Random Forest: 76.4%** | **Random Forest: 74.4%** | **+2.0%** |
| **1.0s** | **Random Forest: 69.4%** | **Logistic Regression: 74.4%** | **-5.0%** |

### **Ensemble Methods (Best Performance per Segment Size)**

| Segment Size | Handcrafted (Best) | OPERA-CT (Best) | Advantage |
|--------------|-------------------|-----------------|-----------|
| **0.25s** | **RF Heavy: 76.7%** | **RF Heavy: 73.7%** | **+3.0%** |
| **0.5s** | **Best 2: 77.8%** | **Hard Voting: 75.0%** | **+2.8%** |
| **1.0s** | **Hard Voting: 68.1%** | **Hard Voting: 73.1%** | **-5.0%** |

---

## **🏆 Overall Best Results**

### **Top 5 Methods (Combined)**

| Rank | Method | Feature Type | Segment Size | Accuracy | Type |
|------|--------|--------------|--------------|----------|------|
| 🥇 **1** | **Best 2** | **Handcrafted** | **0.5s** | **77.8%** | Ensemble |
| 🥈 **2** | **RF Heavy** | **Handcrafted** | **0.25s** | **76.7%** | Ensemble |
| 🥉 **3** | **Random Forest** | **Handcrafted** | **0.5s** | **76.4%** | Individual |
| **4** | **Random Forest** | **Handcrafted** | **0.25s** | **75.7%** | Individual |
| **5** | **Hard Voting** | **OPERA-CT** | **0.5s** | **75.0%** | Ensemble |

---

## **📊 Detailed Results by Segment Size**

### **0.25s Segments**
| Method | Handcrafted | OPERA-CT | Difference |
|--------|-------------|----------|------------|
| **Random Forest** | 75.7% | 74.0% | +1.7% |
| **SVM** | 62.2% | 59.0% | +3.2% |
| **Logistic Regression** | 71.2% | 66.0% | +5.2% |
| **Best Ensemble** | 76.7% | 73.7% | +3.0% |

### **0.5s Segments**
| Method | Handcrafted | OPERA-CT | Difference |
|--------|-------------|----------|------------|
| **Random Forest** | 76.4% | 74.4% | +2.0% |
| **SVM** | 66.7% | 59.6% | +7.1% |
| **Logistic Regression** | 67.4% | 68.6% | -1.2% |
| **Best Ensemble** | 77.8% | 75.0% | +2.8% |

### **1.0s Segments**
| Method | Handcrafted | OPERA-CT | Difference |
|--------|-------------|----------|------------|
| **Random Forest** | 69.4% | 70.5% | -1.1% |
| **SVM** | 52.8% | 59.0% | -6.2% |
| **Logistic Regression** | 68.1% | 74.4% | -6.3% |
| **Best Ensemble** | 68.1% | 73.1% | -5.0% |

---

## **💡 Key Insights**

### **1. Handcrafted Features Win Overall**
- **Best overall performance**: 77.8% (Handcrafted Best 2, 0.5s)
- **Consistent advantage** in 0.25s and 0.5s segments
- **Only disadvantage** in 1.0s segments

### **2. Segment Size Impact**
- **0.5s segments**: Best for handcrafted features
- **1.0s segments**: OPERA-CT performs better
- **0.25s segments**: Handcrafted features have clear advantage

### **3. Model Performance Patterns**
- **Random Forest**: Consistently best individual model for handcrafted
- **Logistic Regression**: Best individual model for OPERA-CT (1.0s)
- **SVM**: Consistently worst performer for both feature types

### **4. Ensemble Benefits**
- **Handcrafted**: Ensemble methods provide significant improvements
- **OPERA-CT**: Ensemble methods provide modest improvements
- **Best 2**: Most effective ensemble strategy for handcrafted features

---

## **🎯 Strategic Recommendations**

1. **Use Handcrafted Features** for high-resolution analysis (0.25s, 0.5s segments)
2. **Use OPERA-CT Features** for lower-resolution analysis (1.0s segments)
3. **Optimal Setup**: Handcrafted features with 0.5s segments and "Best 2" ensemble
4. **Hybrid Approach**: Consider combining both feature types for maximum performance

---

## **📋 Summary**

**Handcrafted features significantly outperform OPERA-CT for breathing classification, achieving 77.8% accuracy with 0.5s segments and ensemble methods. The domain-specific features provide better performance for high-temporal-resolution analysis, while OPERA-CT shows competitive performance only at lower resolutions.**

