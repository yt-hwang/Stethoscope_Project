# Cycle D1: Seg + HighPass + Bandpass

## Overview
This cycle combines C3 (Seg + HighPass) and C1 (Seg + Bandpass) to test if combining high-pass filtering with bandpass filtering yields synergistic benefits for segmented audio clustering. This tests whether frequency-focused preprocessing methods work better together.

## Parameters
- **Segmentation**: 10-second segments
- **Preprocessing**: High-pass filter (20 Hz) + Bandpass filter (100-2000 Hz)
- **Representations**: 
  - Raw waveform stats (RMS, ZCR, spectral flatness, kurtosis, skewness)
  - Log-mel spectrogram mean (64 bins)
  - MFCC mean (13 coefficients)
- **Clustering**: KMeans (k=3,4,5) and HDBSCAN (min_cluster_size=25)

## Data Source
- **Audio Directory**: `/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/Hospital sound_raw segmented into 10 sec`
- **File Format**: WAV files, 10-second segments
- **Expected Count**: ~290+ segmented files

## Preprocessing Pipeline
1. **Load segmented audio**: 10-second WAV files
2. **High-pass filter**: Remove frequencies below 20 Hz (5th order Butterworth)
3. **Bandpass filter**: Focus on 100-2000 Hz range (5th order Butterworth)
4. **Feature extraction**: Raw stats, log-mel, MFCC

## Expected Performance
Based on C-Series results:
- **C1 (Seg + Bandpass)**: 0.571 silhouette (raw_waveform_stats + hdbscan)
- **C3 (Seg + HighPass)**: 0.671 silhouette (raw_waveform_stats + kmeans_k3)
- **Expected D1**: Should combine benefits of both frequency-focused methods
- **Key Question**: Does HighPass + Bandpass outperform individual methods?

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
- This tests if combining two frequency-focused preprocessing methods yields synergistic benefits
- Combines noise reduction (high-pass) with frequency focusing (bandpass)
- Results will be compared against C1 and C3 to see if combination > individual methods
- Should help determine optimal frequency preprocessing pipeline for segmented audio
