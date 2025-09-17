# Cycle A0: NoSeg + NoPreprocess Baseline

## Overview
This cycle establishes the baseline for unsupervised audio clustering with no segmentation and no preprocessing applied to the audio files.

## Configuration
- **Segmentation**: None (uses whole 60-second files)
- **Preprocessing**: None (raw audio, resampled to 16kHz mono only)
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
- **Processing**: Resampled to 16kHz mono for consistency

## Scripts
- `extract_features.py`: Extract all three representations from audio files
- `run_clustering.py`: Run KMeans and HDBSCAN clustering
- `make_visuals.py`: Create UMAP visualizations
- `log_to_excel.py`: Log results to Excel tracking system
- `run_cycle.py`: Main script to run the entire cycle

## Usage
```bash
# Run the complete cycle
python run_cycle.py

# Run with custom paths
python run_cycle.py --audio_dir /path/to/audio --output_dir /path/to/output

# Skip specific steps
python run_cycle.py --skip_extraction --skip_visualization
```

## Outputs
- `features_raw_waveform_stats.csv`: Raw waveform statistical features
- `features_logmel_mean.csv`: Log-mel spectrogram features
- `features_mfcc_mean.csv`: MFCC features
- `clustering_*_metrics.json`: Clustering results and metrics
- `umap_*.png`: UMAP visualizations for each representation/algorithm combination

## Metrics
- **Silhouette Score**: Higher is better (range: -1 to 1)
- **Calinski-Harabasz Index**: Higher is better
- **Davies-Bouldin Index**: Lower is better

## Notes
- All random seeds set to 42 for reproducibility
- Features are standardized before clustering
- UMAP parameters: n_neighbors=15, min_dist=0.1
- This serves as the baseline for comparison with preprocessing variants (A1-A4)
