# Cycle A1: NoSeg + Bandpass(100-2000 Hz)

## Overview
This cycle applies bandpass filtering (100-2000 Hz) to the audio files before feature extraction, comparing performance against the A0 baseline.

## Configuration
- **Segmentation**: None (uses whole 60-second files)
- **Preprocessing**: Bandpass filter (100-2000 Hz, 4th order Butterworth)
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
- **Processing**: Resampled to 16kHz mono + bandpass filter (100-2000 Hz)

## Scripts
- `extract_features.py`: Extract features with bandpass filtering
- `run_clustering.py`: Run KMeans and HDBSCAN clustering
- `make_visuals.py`: Create UMAP visualizations
- `log_to_excel.py`: Log results to Excel tracking system
- `run_cycle.py`: Main script to run the entire cycle

## Usage
```bash
# Run the complete cycle
python3 run_cycle.py

# Run with custom filter parameters
python3 extract_features.py --lowcut 100 --highcut 2000

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
    └── cycle_a1_summary.md
```

## Expected Impact
Bandpass filtering should:
- Remove low-frequency noise (< 100 Hz)
- Remove high-frequency artifacts (> 2000 Hz)
- Focus on the most relevant frequency range for lung sounds
- Potentially improve clustering performance compared to A0 baseline

## Comparison with A0
This cycle will be compared against A0 baseline to evaluate:
- ΔSilhouette score
- ΔCalinski-Harabasz index  
- ΔDavies-Bouldin index
- Overall clustering quality improvement
