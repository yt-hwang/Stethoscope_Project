# Cycle D0: Seg + HighPass + PeakNormalize

## Overview
This cycle combines the two best-performing C-Series methods: C3 (Seg + HighPass) and C4 (Seg + PeakNormalize). It applies both high-pass filtering (20 Hz cutoff) and peak normalization to segmented audio files to test if combining these preprocessing methods yields even better clustering performance.

## Parameters
- **Segmentation**: 10-second segments
- **Preprocessing**: High-pass filter (20 Hz) + Peak normalization
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
3. **Peak normalization**: Normalize amplitude to peak of 1.0
4. **Feature extraction**: Raw stats, log-mel, MFCC

## Expected Performance
Based on C-Series results:
- **C3 (Seg + HighPass)**: 0.659 silhouette, 0.7423 quality score
- **C4 (Seg + PeakNormalize)**: 0.663 silhouette, 0.7264 quality score
- **Expected D0**: Should combine the benefits of both methods
- **Key Question**: Does combining HighPass + PeakNormalize outperform individual methods?

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

## Notes
- This tests if combining the two best C-Series methods yields synergistic benefits
- Combines noise reduction (high-pass) with amplitude normalization (peak normalize)
- Results will be compared against C3 and C4 to see if combination > individual methods
- Should help determine optimal preprocessing pipeline for segmented audio
