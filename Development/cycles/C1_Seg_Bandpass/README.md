# Cycle C1: Seg + Bandpass

## Overview
This cycle uses segmented audio files (10-second segments) with bandpass filtering (100-2000 Hz). This tests how segmentation affects the best-performing A-Series method (A1: NoSeg + Bandpass).

## Parameters
- **Segmentation**: 10-second segments
- **Preprocessing**: Bandpass filter (100-2000 Hz)
- **Representations**: 
  - Raw waveform stats (RMS, ZCR, spectral flatness, kurtosis, skewness)
  - Log-mel spectrogram mean (64 bins)
  - MFCC mean (13 coefficients)
- **Clustering**: KMeans (k=3,4,5) and HDBSCAN (min_cluster_size=25)

## Data Source
- **Audio Directory**: `/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/Hospital sound_raw segmented into 10 sec`
- **File Format**: WAV files, 10-second segments
- **Expected Count**: ~290+ segmented files (vs ~39 original files)

## Preprocessing Pipeline
1. **Load segmented audio**: 10-second WAV files
2. **Bandpass Filter**: 100-2000 Hz (focuses on relevant frequency range)
3. **Feature extraction**: Raw stats, log-mel, MFCC

## Files
- `code/extract_features.py` - Feature extraction from segmented audio with bandpass filtering
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
Based on A-Series and C-Series results:
- **A1 (NoSeg + Bandpass)**: Quality Score ~0.68 (A-Series best)
- **C0 (Seg + NoPreprocess)**: Quality Score ~0.65 (segmentation baseline)
- **Expected C1**: Should show how segmentation affects the best A-Series method
- **Key Question**: Does segmentation + bandpass outperform A1 or C0?

## Notes
- This tests how segmentation affects the best A-Series method
- Combines segmentation benefits (more samples) with bandpass filtering benefits
- Results will be compared against A1 (NoSeg + Bandpass) and C0 (Seg + NoPreprocess)
- Should help determine if segmentation enhances or degrades preprocessing effectiveness
