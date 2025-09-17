# Cycle D2: Seg + Full Pipeline

## Overview
This cycle applies the complete preprocessing pipeline to segmented audio files. It combines all four preprocessing methods: peak normalization, bandpass filtering, spectral gating, and high-pass filtering. This tests if the full preprocessing pipeline provides the best clustering performance on segmented audio.

## Parameters
- **Segmentation**: 10-second segments
- **Preprocessing**: Complete pipeline (PeakNormalize + Bandpass + SpectralGating + HighPass)
- **Representations**: 
  - Raw waveform stats (RMS, ZCR, spectral flatness, kurtosis, skewness)
  - Log-mel spectrogram mean (64 bins)
  - MFCC mean (13 coefficients)
- **Clustering**: KMeans (k=3,4,5) and HDBSCAN (min_cluster_size=3)

## Data Source
- **Audio Directory**: `/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/Hospital sound_raw segmented into 10 sec`
- **File Format**: WAV files, 10-second segments
- **Expected Count**: ~290+ segmented files

## Preprocessing Pipeline
1. **Load segmented audio**: 10-second WAV files
2. **Peak normalization**: Normalize amplitude to peak of 1.0
3. **Bandpass filter**: Focus on 100-2000 Hz range (5th order Butterworth)
4. **Spectral gating**: Noise reduction using spectral masking
5. **High-pass filter**: Remove frequencies below 20 Hz (5th order Butterworth)
6. **Feature extraction**: Raw stats, log-mel, MFCC

## Expected Performance
Based on previous results:
- **D1 (Seg + HighPass + Bandpass)**: 0.488 quality score (current winner)
- **C4 (Seg + PeakNormalize)**: 0.462 quality score
- **D0 (Seg + HighPass + PeakNormalize)**: 0.451 quality score
- **Expected D2**: Should test if complete pipeline > smart combinations

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

## Notes
- This tests if the complete preprocessing pipeline outperforms smart combinations
- Combines all four preprocessing methods in sequence
- Results will be compared against D1, C4, and D0 to see if more preprocessing = better performance
- Should help determine the optimal preprocessing pipeline for segmented audio clustering
