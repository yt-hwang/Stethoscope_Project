# Cycle C3: Seg + HighPass(20 Hz)

## Overview
This cycle uses segmented audio files (10-second segments) with high-pass filtering (20 Hz cutoff). This tests how segmentation affects the A3 method (NoSeg + HighPass(20 Hz)).

## Parameters
- **Segmentation**: 10-second segments
- **Preprocessing**: High-pass filter (20 Hz cutoff)
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
2. **High-pass filter**: Remove frequencies below 20 Hz (5th order Butterworth)
3. **Feature extraction**: Raw stats, log-mel, MFCC

## Files
- `code/extract_features.py` - Feature extraction from segmented audio with high-pass filtering
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
- **A3 (NoSeg + HighPass(20 Hz))**: Quality Score ~0.65 (A-Series method)
- **C0 (Seg + NoPreprocess)**: Quality Score ~0.65 (segmentation baseline)
- **C1 (Seg + Bandpass)**: Quality Score ~0.69 (current C-Series best)
- **Expected C3**: Should show how segmentation affects high-pass filtering performance
- **Key Question**: Does segmentation + high-pass filtering outperform A3 or match C1?

## Notes
- This tests how segmentation affects the A3 method (high-pass filtering)
- Combines segmentation benefits (more samples) with low-frequency noise removal
- Results will be compared against A3 (NoSeg + HighPass(20 Hz)) and C0 (Seg + NoPreprocess)
- Should help determine if segmentation enhances high-pass filtering effectiveness
