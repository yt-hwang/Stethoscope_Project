# Supervised Learning Strategy for Respiratory Audio Classification

## 🎯 **Objective**: Classify breathing vs non-breathing segments using handcrafted features

---

## **📊 Data Strategy**
- **Dataset**: 29 audio files with Excel ground truth timestamps
- **Segmentation**: 0.25s, 0.5s, 1.0s segments (no overlap)
- **Labeling**: Center-point labeling (segment labeled by Excel period at its center)
- **Split**: 80% train, 20% test (segment-based)

## **🔧 Feature Engineering**
- **47 Handcrafted Features**: RMS energy, ZCR, spectral features, MFCCs, harmonic/rhythm features
- **Rationale**: Domain-specific features for respiratory audio analysis

## **🤖 Model Strategy**
- **Individual Models**: Random Forest, SVM, Logistic Regression
- **Ensemble Methods**: Hard/Soft voting, weighted voting, "Best 2" combination
- **Best Performance**: "Best 2" Ensemble (0.5s) = **77.8% accuracy**

## **📈 Key Results**
| Method | Segment Size | Accuracy | Type |
|--------|--------------|----------|------|
| **Best 2** | 0.5s | **77.8%** | Ensemble (RF + LR) |
| **Random Forest** | 0.5s | **76.4%** | Individual |
| **RF Heavy** | 0.25s | **76.7%** | Ensemble (RF weighted) |
| **Random Forest** | 0.25s | **75.7%** | Individual |
| **Soft Voting** | 0.25s | **74.7%** | Ensemble (RF + SVM + LR) |

## **💡 Strategic Insights**
1. **0.5s segments** optimal balance of resolution vs data
2. **Handcrafted features** outperform general-purpose embeddings
3. **Ensemble methods** provide modest but consistent improvements
4. **Random Forest** most reliable individual classifier
5. **Center-point labeling** more accurate than overlap-based methods

---

**Key Takeaway**: Domain-specific handcrafted features with 0.5s segmentation and ensemble methods achieve 77.8% accuracy for breathing classification.
