# Cycle A0 Results Summary: NoSeg + NoPreprocess Baseline

## Overview
- **Cycle ID**: A0
- **Phase**: A_NoSeg  
- **Segmentation**: None (whole 60-second files)
- **Preprocessing**: None (raw audio, resampled to 16kHz mono only)
- **Files Processed**: 39 WAV files
- **Date**: $(date)

## Best Results by Representation

### Raw Waveform Stats (5 features)
| Algorithm | Silhouette | Calinski-Harabasz | Davies-Bouldin | Clusters |
|-----------|------------|-------------------|----------------|----------|
| **KMeans k=3** | **0.3400** | 23.41 | 0.8051 | 3 |
| KMeans k=4 | 0.2721 | 27.18 | 0.7216 | 4 |
| KMeans k=5 | 0.3095 | 28.97 | 0.6876 | 5 |
| HDBSCAN | -1.0000 | 0.00 | inf | 0 (all noise) |

### Log-Mel Mean (64 features)
| Algorithm | Silhouette | Calinski-Harabasz | Davies-Bouldin | Clusters |
|-----------|------------|-------------------|----------------|----------|
| KMeans k=3 | 0.3172 | 23.88 | 1.0850 | 3 |
| **KMeans k=4** | **0.3402** | 24.69 | 0.9546 | 4 |
| KMeans k=5 | 0.3369 | 24.49 | 0.9795 | 5 |
| HDBSCAN | -1.0000 | 0.00 | inf | 0 (all noise) |

### MFCC Mean (13 features)
| Algorithm | Silhouette | Calinski-Harabasz | Davies-Bouldin | Clusters |
|-----------|------------|-------------------|----------------|----------|
| **KMeans k=3** | **0.3174** | 18.05 | 1.2596 | 3 |
| KMeans k=4 | 0.3074 | 16.51 | 1.1511 | 4 |
| KMeans k=5 | 0.3030 | 16.13 | 0.9947 | 5 |
| HDBSCAN | -1.0000 | 0.00 | inf | 0 (all noise) |

## Key Observations

1. **HDBSCAN Performance**: HDBSCAN with min_cluster_size=25 classified all 39 samples as noise, suggesting the dataset may be too small or the clusters too sparse for this algorithm.

2. **Best Overall Performance**: 
   - **Raw Waveform Stats + KMeans k=3**: Highest Silhouette score (0.3400)
   - **Log-Mel Mean + KMeans k=4**: Best balance of Silhouette (0.3402) and Calinski-Harabasz (24.69)

3. **Representation Comparison**:
   - Raw waveform stats show the most consistent clustering performance
   - Log-mel features show good performance with k=4
   - MFCC features show moderate performance, best with k=3

4. **Cluster Count**: KMeans with k=3 or k=4 appears optimal for this dataset size

## Files Generated
- `features_raw_waveform_stats.csv`: 39 samples × 7 features
- `features_logmel_mean.csv`: 39 samples × 66 features  
- `features_mfcc_mean.csv`: 39 samples × 15 features
- `clustering_*_metrics.json`: Detailed clustering results
- `umap_*.png`: UMAP visualizations for all combinations

## Next Steps
This baseline will be compared against preprocessing variants (A1-A4) to evaluate the impact of:
- Bandpass filtering (100-2000 Hz)
- Spectral gating
- High-pass filtering (20 Hz)
- Peak normalization
