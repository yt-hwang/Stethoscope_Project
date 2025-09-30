# Cycle B1: NoSeg + PeakNormalize + Bandpass

## Overview
This cycle combines A4 (PeakNormalize) + A1 (Bandpass) - focusing on consistency + performance. This represents the second B-Series experiment testing smart preprocessing combinations.

## Parameters
- **Segmentation**: None (whole file)
- **Preprocessing**: PeakNormalize + Bandpass(100-2000 Hz)
- **Representations**: 
  - Raw waveform stats (RMS, ZCR, spectral flatness, kurtosis, skewness)
  - Log-mel spectrogram mean (64 bins)
  - MFCC mean (13 coefficients)
- **Clustering**: KMeans (k=3,4,5) and HDBSCAN (min_cluster_size=25)

## Preprocessing Pipeline
1. **Peak Normalization**: Target -20 dB for consistent loudness levels
2. **Bandpass Filter**: 100-2000 Hz (focuses on relevant frequency range)

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
- **A4 (PeakNormalize)**: Quality Score 0.6500+ (good consistency)
- **Expected B1**: Should combine consistency of A4 with performance of A1

## Notes
- This is the second B-Series experiment testing smart preprocessing combinations
- Peak normalization ensures consistent loudness levels across files
- Bandpass filtering focuses on relevant frequency range
- Results will be compared against A-Series individual methods and B0
