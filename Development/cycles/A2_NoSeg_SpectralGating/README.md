# Cycle A2: NoSeg + SpectralGating

## Overview
This cycle applies spectral gating to remove background noise before feature extraction, comparing performance against A0 baseline and A1 bandpass results.

## Configuration
- **Segmentation**: None (uses whole 60-second files)
- **Preprocessing**: Spectral gating (threshold=-20dB, ratio=4.0)
- **Representations**: 
  - `raw_waveform_stats`: RMS, ZCR, spectral flatness, kurtosis, skewness
  - `logmel_mean`: 64-bin log-mel spectrogram averaged over time
  - `mfcc_mean`: 13 MFCCs averaged over time
- **Clustering Algorithms**:
  - KMeans with k ∈ {3, 4, 5}
  - HDBSCAN with min_cluster_size=25

## Data
- **Source**: `/Users/yunhwang/Desktop/Stethoscope_Project/Audio shared/Hospital sound_raw 60sec`
- **Files**: 39 WAV files (60-second recordings)
- **Processing**: Resampled to 16kHz mono + spectral gating

## Spectral Gating Parameters
- **Threshold**: -20 dB (signals below this level are gated)
- **Ratio**: 4.0 (compression ratio for gated signals)
- **Attack Time**: 0.01 seconds
- **Release Time**: 0.1 seconds

## Scripts
- `extract_features.py`: Extract features with spectral gating
- `run_clustering.py`: Run KMeans and HDBSCAN clustering
- `make_visuals.py`: Create UMAP visualizations
- `log_to_excel.py`: Log results to Excel tracking system
- `run_cycle.py`: Main script to run the entire cycle

## Usage
```bash
# Run the complete cycle
python3 run_cycle.py

# Run with custom gating parameters
python3 extract_features.py --threshold_db -25 --ratio 6.0

# Skip specific steps
python3 run_cycle.py --skip_extraction --skip_visualization
```

## Output Structure
```
outputs/
├── features/
│   ├── features_all.csv
│   ├── features_raw_waveform_stats.csv
│   ├── features_logmel_mean.csv
│   └── features_mfcc_mean.csv
├── clustering/
│   ├── clustering_raw_waveform_stats_metrics.json
│   ├── clustering_logmel_mean_metrics.json
│   └── clustering_mfcc_mean_metrics.json
├── visualizations/
│   └── umap_*.png (12 files)
└── summaries/
    └── cycle_a2_summary.md
```

## Expected Impact
Spectral gating should:
- Remove background noise and artifacts
- Enhance signal-to-noise ratio
- Improve clustering performance by focusing on relevant audio content
- Potentially outperform bandpass filtering for noise reduction

## Comparison Targets
This cycle will be compared against:
- **A0 Baseline**: No preprocessing
- **A1 Bandpass**: Bandpass filtering (100-2000 Hz)

Current leader: A1 with 0.4109 Silhouette score (Log-Mel Mean + K3)
