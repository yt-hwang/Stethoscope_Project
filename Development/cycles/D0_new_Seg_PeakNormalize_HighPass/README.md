# D0_new: Segmentation + Peak Normalization + High-pass Filtering (20 Hz)

## Description
This cycle implements a better combination based on the re-evaluation results. It combines the top 2 C-Series performers:
- **C4 (Seg + PeakNormalize)**: Quality score 0.462
- **C3 (Seg + HighPass)**: Quality score 0.442

The goal is to evaluate if this "smart combination" leads to synergistic improvements in clustering performance.

## Preprocessing Steps
1. **Peak Normalization**: Audio is normalized to a peak amplitude of 1.0, ensuring consistent signal levels.
2. **High-pass Filter**: Audio is filtered to retain frequencies above 20 Hz, removing low-frequency rumble.

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
Based on the individual method rankings:
- **C4 (Seg + PeakNormalize)**: 0.462 quality score
- **C3 (Seg + HighPass)**: 0.442 quality score
- **Expected D0_new**: Should outperform both individual methods if there's synergy

## Outputs
All outputs are stored under `Development/cycles/D0_new_Seg_PeakNormalize_HighPass/outputs/`:
- `features/`: CSV files containing extracted features
- `clustering/`: JSON files with clustering metrics and labels
- `visualizations/`: PNG images of 2D UMAP projections

## Usage
```bash
cd D0_new_Seg_PeakNormalize_HighPass/code
python3 run_cycle.py
```

## Excel Logging
Results are logged to `Experiment_Tracking_System_Final.xlsx` in the `CycleSummary` and `MetricsByRun` sheets.
