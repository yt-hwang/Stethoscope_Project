# D1_new: Segmentation + Peak Normalization + Bandpass Filtering (100-2000 Hz)

## Description
This cycle implements a better combination based on the re-evaluation results. It combines:
- **Peak Normalization**: Amplitude normalization for consistent signal levels
- **Bandpass Filtering**: Frequency focusing on the 100-2000 Hz range

This represents an "amplitude + frequency" combination approach, testing if these complementary preprocessing methods work synergistically.

## Preprocessing Steps
1. **Peak Normalization**: Audio is normalized to a peak amplitude of 1.0, ensuring consistent signal levels.
2. **Bandpass Filter**: Audio is filtered to retain frequencies between 100 Hz and 2000 Hz, focusing on a specific range.

## Feature Representations
The following feature representations are extracted from the preprocessed audio:
- `raw_waveform_stats`: RMS, Zero Crossing Rate (ZCR), Spectral Flatness, Kurtosis, Skewness
- `logmel_mean`: 64-bin log-mel spectrogram, averaged over time
- `mfcc_mean`: 13 Mel-frequency Cepstral Coefficients (MFCCs), averaged over time

## Clustering Algorithms
The following unsupervised clustering algorithms are applied to each feature representation:
- **K-Means**: With `k` values of 3, 4, and 5
- **HDBSCAN**: With `min_cluster_size` set to 3

## Evaluation
Clustering performance is evaluated using intrinsic metrics:
- **Silhouette Score** (higher is better)
- **Calinski-Harabasz Index** (higher is better)
- **Davies-Bouldin Index** (lower is better)

## Expected Performance
This combination tests the hypothesis that:
- **Amplitude normalization** (PeakNormalize) provides consistent signal levels
- **Frequency focusing** (Bandpass) removes irrelevant frequency content
- **Together** they may provide better clustering than either alone

## Outputs
All outputs are stored under `Development/cycles/D1_new_Seg_PeakNormalize_Bandpass/outputs/`:
- `features/`: CSV files containing extracted features
- `clustering/`: JSON files with clustering metrics and labels
- `visualizations/`: PNG images of 2D UMAP projections

## Usage
```bash
cd D1_new_Seg_PeakNormalize_Bandpass/code
python3 run_cycle.py
```

## Excel Logging
Results are logged to `Experiment_Tracking_System_Final.xlsx` in the `CycleSummary` and `MetricsByRun` sheets.
