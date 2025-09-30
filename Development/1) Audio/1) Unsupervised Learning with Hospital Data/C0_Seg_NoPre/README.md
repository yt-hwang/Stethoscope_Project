# Cycle C0: Seg + NoPreprocess

## Overview
This cycle uses segmented audio files (10-second segments) with no preprocessing. This represents the C-Series baseline to test how segmentation affects clustering performance compared to the A-Series (NoSeg + individual methods).

## Parameters
- **Segmentation**: 10-second segments
- **Preprocessing**: None (raw audio)
- **Representations**: 
  - Raw waveform stats (RMS, ZCR, spectral flatness, kurtosis, skewness)
  - Log-mel spectrogram mean (64 bins)
  - MFCC mean (13 coefficients)
- **Clustering**: KMeans (k=3,4,5) and HDBSCAN (min_cluster_size=25)

## Data Source
- **Audio Directory**: `/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/Hospital sound_raw segmented into 10 sec`
- **File Format**: WAV files, 10-second segments
- **Expected Count**: ~290+ segmented files (vs ~39 original files)

## Files
- `code/extract_features.py` - Feature extraction from segmented audio
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
- **A0 (NoSeg + NoPreprocess)**: Quality Score ~0.65 (baseline)
- **Expected C0**: Should show how segmentation affects the baseline performance
- **Key Question**: Does segmentation improve or degrade clustering quality?

## Notes
- This is the first C-Series experiment testing segmentation effects
- Uses 10-second segments instead of full 60-second files
- No preprocessing to isolate segmentation effects
- Results will be compared against A0 (NoSeg + NoPreprocess)
- Higher sample count due to segmentation may affect clustering behavior
