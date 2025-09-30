# Cycle B2: NoSeg + PeakNormalize + Bandpass + SpectralGating

## Overview
This cycle combines all the best-performing A-Series methods: A4 (PeakNormalize) + A1 (Bandpass) + A2 (SpectralGating). This represents the complete preprocessing pipeline testing the full combination of all top performers.

## Parameters
- **Segmentation**: None (whole file)
- **Preprocessing**: PeakNormalize + Bandpass(100-2000 Hz) + SpectralGating
- **Representations**: 
  - Raw waveform stats (RMS, ZCR, spectral flatness, kurtosis, skewness)
  - Log-mel spectrogram mean (64 bins)
  - MFCC mean (13 coefficients)
- **Clustering**: KMeans (k=3,4,5) and HDBSCAN (min_cluster_size=25)

## Preprocessing Pipeline
1. **Peak Normalization**: Target -20 dB for consistent loudness levels
2. **Bandpass Filter**: 100-2000 Hz (focuses on relevant frequency range)
3. **Spectral Gating**: Noise reduction using spectral thresholding

## Files
- `code/extract_features.py` - Feature extraction with complete preprocessing pipeline
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
Based on A-Series and B-Series results:
- **B1 (PeakNormalize + Bandpass)**: Quality Score 0.6908 (current best)
- **B0 (Bandpass + SpectralGating)**: Quality Score 0.6803 (second best)
- **Expected B2**: Should potentially exceed B1 by adding spectral gating for noise reduction

## Notes
- This is the final B-Series experiment testing the complete preprocessing pipeline
- Combines all three best-performing A-Series methods
- Peak normalization ensures consistency, bandpass focuses frequency range, spectral gating reduces noise
- Results will be compared against all A-Series and B-Series cycles
- This represents the most comprehensive preprocessing approach tested
