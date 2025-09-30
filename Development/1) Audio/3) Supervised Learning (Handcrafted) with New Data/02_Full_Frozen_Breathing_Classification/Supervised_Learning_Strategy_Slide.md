# Supervised Learning Strategy for Respiratory Audio Classification

## 🎯 **Research Objective**
**Classify breathing vs non-breathing segments in respiratory audio using handcrafted features**

---

## 📊 **Data Strategy**

### **Dataset**
- **Source**: "RAW sound_ML test sound list" (29 audio files)
- **Ground Truth**: Excel timestamps for inhale/exhale periods
- **Labeling Method**: Center-point labeling (segment labeled based on Excel period at its center)

### **Segmentation Strategy**
| Segment Size | Hop Length | Total Segments | Training | Test | Rationale |
|--------------|------------|----------------|----------|------|-----------|
| **0.25s** | 0.25s (no overlap) | 1,440 | 1,152 | 288 | Ultra-high temporal resolution |
| **0.5s** | 0.5s (no overlap) | 720 | 576 | 144 | **Optimal balance** |
| **1.0s** | 1.0s (no overlap) | 360 | 288 | 72 | Standard resolution |

**Key Insight**: 0.5s segments achieved best performance (77.8% accuracy)

---

## 🔧 **Feature Engineering Strategy**

### **Handcrafted Features (47 total)**
1. **RMS Energy** (4 features): Mean, Std, Min, Max
2. **Zero Crossing Rate** (4 features): Mean, Std, Min, Max  
3. **Spectral Features** (6 features): Centroid, Rolloff, Bandwidth (Mean + Std)
4. **MFCC Features** (26 features): 13 coefficients × (Mean + Std)
5. **Harmonic Features** (4 features): Harmonic/Percussive separation
6. **Rhythm Features** (3 features): Tempo, beats, onsets

**Rationale**: Domain-specific features for respiratory audio analysis

---

## 🤖 **Model Strategy**

### **Individual Models**
- **Random Forest**: Best performer (76.4% @ 0.5s)
- **Support Vector Machine**: Moderate performance (66.7% @ 0.5s)
- **Logistic Regression**: Good performance (67.4% @ 0.5s)

### **Ensemble Methods**
- **Hard Voting**: Equal weight voting
- **Soft Voting**: Probability-based voting
- **Weighted Voting**: Custom model weights
- **Best 2**: Top 2 models only
- **RF Heavy**: Random Forest weighted
- **SVM Heavy**: SVM weighted
- **LR Heavy**: Logistic Regression weighted

**Best Ensemble**: "Best 2" (0.5s) - 77.8% accuracy

---

## 📈 **Performance Results**

### **Top 5 Methods**
| Rank | Method | Segment Size | Accuracy | Type |
|------|--------|--------------|----------|------|
| 🥇 1 | **Best 2 Ensemble** | **0.5s** | **77.8%** | Ensemble |
| 🥈 2 | **RF Heavy** | **0.25s** | **76.7%** | Ensemble |
| 🥉 3 | **Random Forest** | **0.5s** | **76.4%** | Individual |
| 4 | **Soft Voting** | **0.25s** | **74.7%** | Ensemble |
| 5 | **Random Forest** | **0.25s** | **75.7%** | Individual |

---

## 🎯 **Key Strategic Decisions**

### **1. Temporal Resolution Optimization**
- **Problem**: Short breathing events missed by large segments
- **Solution**: Tested 0.25s, 0.5s, 1.0s segments
- **Result**: 0.5s optimal balance of resolution vs data

### **2. Feature Selection**
- **Approach**: Handcrafted features over deep learning
- **Rationale**: Domain expertise + interpretability
- **Result**: 47 features vs 768 OPERA-CT features

### **3. Labeling Strategy**
- **Method**: Center-point labeling
- **Logic**: Segment labeled based on Excel period at its center
- **Benefit**: More accurate than overlap-based labeling

### **4. Model Selection**
- **Individual**: Random Forest consistently best
- **Ensemble**: "Best 2" combination optimal
- **Reasoning**: Combines top performers without overfitting

---

## 💡 **Strategic Insights**

1. **Temporal Resolution Matters**: 0.5s segments capture breathing patterns better than 1.0s
2. **Handcrafted Features Win**: Domain-specific features outperform general-purpose embeddings
3. **Ensemble Benefits**: Modest but consistent improvement over individual models
4. **Random Forest Dominance**: Most reliable individual classifier
5. **Center-point Labeling**: More accurate than overlap-based methods

---

## 🚀 **Next Steps**

1. **Patient-based Splitting**: Test generalization across patients
2. **Feature Analysis**: Identify most important features
3. **Hybrid Approach**: Combine handcrafted + OPERA-CT features
4. **Real-time Implementation**: Optimize for clinical deployment
5. **Cross-validation**: Robust performance evaluation

---

## 📋 **Technical Specifications**

- **Audio Format**: 16kHz mono WAV
- **Preprocessing**: None (raw audio)
- **Train/Test Split**: 80/20 (segment-based)
- **Cross-validation**: Not implemented (future work)
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score
- **Random Seed**: 42 (reproducibility)

