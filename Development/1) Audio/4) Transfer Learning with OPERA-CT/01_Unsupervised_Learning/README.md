# OPERA-CT Unsupervised Learning Experiments

This folder contains all unsupervised learning experiments using OPERA-CT as a frozen feature extractor for clustering respiratory audio data.

## 📁 Structure

### 01_Baseline/
- **Purpose**: Single experiment using OPERA-CT with default preprocessing
- **Content**: OPERA-CT pipeline with minimal preprocessing
- **Results**: One UMAP plot, one clustering result
- **Key Question**: "How good is OPERA-CT out-of-the-box?"

### 02_16_Methods/
- **Purpose**: Comprehensive testing of 16 different preprocessing methods
- **Content**: A0-D2 series (NoSeg, Seg, Bandpass, SpectralGating, etc.)
- **Results**: 16 UMAP plots, 16 clustering results, comprehensive comparison
- **Key Question**: "Can domain expertise improve OPERA-CT performance?"
- **Key Finding**: +36.2% improvement with our preprocessing methods

### 03_Layer_Analysis/
- **Purpose**: Test different OPERA-CT layers for feature extraction
- **Content**: Layer 0 (Early), Layer 1 (Middle), Layer 2 (Late), Layer 3 (Final)
- **Results**: Layer comparison, optimal layer identification
- **Key Question**: "Which layer provides the best features?"

## 🏆 Key Results

- **Best Performance**: D1/C1 (Seg + HighPass + Bandpass) = 0.267 silhouette
- **Baseline**: A0 (No preprocessing) = 0.196 silhouette
- **Improvement**: +36.2% with our preprocessing methods
- **Critical Finding**: Segmentation dramatically improves OPERA-CT (+7.0%)

## 📊 Performance Comparison

| Method | Silhouette Score | Improvement |
|--------|------------------|-------------|
| D1 (Seg_HighPass_Bandpass) | 0.267 | +36.2% |
| C1 (Seg_Bandpass) | 0.267 | +36.2% |
| A0 (NoSeg_NoPre) | 0.196 | Baseline |

## 🔬 Scientific Insights

1. **Segmentation is Critical**: +7.0% improvement with segmentation
2. **Bandpass Filtering is MVP**: Appears in top 2 methods
3. **Simple > Complex**: C1 (2 operations) ties D1 (3 operations)
4. **Domain Expertise Matters**: 36.2% improvement over baseline
