# Cycle B0: NoSeg + Bandpass + SpectralGating

## Overview
This cycle combines the top 2 performers from A-Series: A1 (Bandpass) + A2 (SpectralGating). This represents the first B-Series experiment testing smart preprocessing combinations.

## Parameters
- **Segmentation**: None (whole file)
- **Preprocessing**: Bandpass(100-2000 Hz) + SpectralGating
- **Representations**: 
  - Raw waveform stats (RMS, ZCR, spectral flatness, kurtosis, skewness)
  - Log-mel spectrogram mean (64 bins)
  - MFCC mean (13 coefficients)
- **Clustering**: KMeans (k=3,4,5) and HDBSCAN (min_cluster_size=25)

## Preprocessing Pipeline
1. **Bandpass Filter**: 100-2000 Hz (focuses on relevant frequency range)
2. **Spectral Gating**: Noise reduction using spectral thresholding

## Files
- `code/extract_features.py` - Feature extraction with combined preprocessing
- `code/run_clustering.py` - Clustering and evaluation
- `code/make_visuals.py` - UMAP visualization generation
- `code/log_to_excel.py` - Excel logging
- `code/run_cycle.py` - Main runner script

## Usage
```bash
cd code
python run_cycle.py
```

## Output Structure
```
outputs/
├── features/
│   ├── features_raw_waveform_stats.csv
│   ├── features_logmel_mean.csv
│   └── features_mfcc_mean.csv
├── clustering/
│   ├── clustering_raw_waveform_stats_metrics.json
│   ├── clustering_logmel_mean_metrics.json
│   └── clustering_mfcc_mean_metrics.json
└── visualizations/
    ├── umap_raw_waveform_stats_kmeans_k3.png
    ├── umap_raw_waveform_stats_kmeans_k4.png
    ├── umap_raw_waveform_stats_kmeans_k5.png
    ├── umap_raw_waveform_stats_hdbscan.png
    └── ... (similar for logmel_mean and mfcc_mean)
```

## Expected Performance
Based on A-Series results:
- **A1 (Bandpass)**: Quality Score 0.6828 (best overall)
- **A2 (SpectralGating)**: Quality Score 0.6677 (second best)
- **Expected B0**: Should perform better than individual methods due to complementary effects

## Notes
- This is the first B-Series experiment testing smart preprocessing combinations
- Results will be compared against A-Series individual methods
- Bandpass targets frequency range, spectral gating targets noise - complementary approaches
