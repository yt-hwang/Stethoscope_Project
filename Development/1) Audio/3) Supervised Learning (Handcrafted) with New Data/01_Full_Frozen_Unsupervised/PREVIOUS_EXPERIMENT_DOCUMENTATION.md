# Comprehensive Unsupervised Audio Clustering Experiment Documentation

## Executive Summary

We conducted a systematic unsupervised audio clustering experiment to establish a pretrained-free baseline for respiratory audio analysis and identify optimal preprocessing methods. The experiment tested 16 different preprocessing configurations across two datasets, generating 144+ quality-scored results with comprehensive statistical validation.

## Experimental Objective

**Primary Goal**: Establish an unsupervised baseline for respiratory audio clustering and identify which preprocessing methods produce the best natural groupings in audio data, without using any pretrained models or labeled data.

**Secondary Goals**:
- Compare segmentation vs non-segmentation approaches
- Identify optimal preprocessing method combinations
- Validate findings across different datasets
- Establish preprocessing method rankings for future supervised learning

## Datasets Used

### Dataset 1: Original Large Hospital Audio Collection
- **Source**: Hospital sound recordings from multiple sources
- **Size**: Large collection (exact count varies by cycle)
- **Characteristics**: Complex, diverse respiratory audio patterns
- **Evaluation**: 5 cycles completed with HDBSCAN + K-Means
- **Best Performance**: 0.488 quality score (D1 with HDBSCAN)

### Dataset 2: RAW sound_ML Test Sound List (New Dataset)
- **Source**: Curated test set for ML validation
- **Size**: 29 audio files → 87 segments (with segmentation)
- **Characteristics**: More homogeneous, controlled audio patterns
- **Evaluation**: 16 cycles completed with K-Means only
- **Best Performance**: 0.406 quality score (B2 with K-Means)

## Preprocessing Methods Tested (16 Total)

### A-Series: Individual NoSeg Methods (5 methods)
- **A0**: No preprocessing (baseline)
- **A1**: Bandpass filter (100-2000 Hz)
- **A2**: Spectral gating (noise reduction)
- **A3**: High-pass filter (20 Hz)
- **A4**: Peak normalization

### B-Series: Combination NoSeg Methods (3 methods)
- **B0**: Bandpass + Spectral gating
- **B1**: Peak normalize + Bandpass
- **B2**: Full pipeline (Peak normalize + Bandpass + Spectral gating + High-pass)

### C-Series: Individual Seg Methods (5 methods)
- **C0**: Segmentation only (10-second windows)
- **C1**: Seg + Bandpass
- **C2**: Seg + Spectral gating
- **C3**: Seg + High-pass
- **C4**: Seg + Peak normalize

### D-Series: Combination Seg Methods (3 methods)
- **D0**: Seg + High-pass + Peak normalize
- **D1**: Seg + High-pass + Bandpass
- **D2**: Seg + Full pipeline

## Technical Implementation

### Audio Processing Pipeline
```python
1. Load audio (16kHz, mono) using librosa
2. Apply preprocessing method (A0-D2)
3. Segment audio (if C/D series): 10-second windows, no overlap
4. Extract features: 3 representations per segment/file
5. Cluster features: K-Means (k=3,4,5) with N=7 seeds for robustness
6. Evaluate clustering: Intrinsic metrics + stability analysis
7. Generate visualizations: UMAP plots for each result
8. Log results: Excel + CSV with comprehensive metadata
```

### Feature Representations (3 types)
1. **Raw Waveform Stats**: RMS, ZCR, spectral flatness, kurtosis, skewness (5 features)
2. **Log-Mel Spectrograms**: Mean-pooled over time (64 features)
3. **MFCCs**: Mean-pooled over time (13 features)

### Clustering Algorithms
- **K-Means**: k=3, 4, 5 clusters (primary algorithm for new dataset)
- **HDBSCAN**: Automatic cluster detection (used in original dataset, later excluded)

### Preprocessing Method Details

#### Individual Methods:
```python
# A1/C1: Bandpass Filter
def apply_bandpass_filter(audio, sr, low_freq=100, high_freq=2000):
    # 4th-order Butterworth bandpass filter
    
# A2/C2: Spectral Gating  
def spectral_gating(audio, sr, alpha=0.1):
    # STFT → noise floor estimation → soft masking → ISTFT
    
# A3/C3: High-pass Filter
def apply_highpass_filter(audio, sr, cutoff_freq=20):
    # 4th-order Butterworth high-pass filter
    
# A4/C4: Peak Normalization
def peak_normalize(audio):
    # Normalize to peak amplitude of 1.0 using librosa
```

#### Combination Methods:
```python
# B2/D2: Full Pipeline (order matters)
1. Peak normalization (amplitude domain)
2. Bandpass filter (100-2000 Hz)
3. Spectral gating (noise reduction)
4. High-pass filter (20 Hz, final cleanup)
```

### Segmentation Implementation
```python
def segment_audio(audio, sr, segment_length=10, overlap=0):
    # Fixed 10-second windows, no overlap
    # Converts 29 files → 87 segments (3 segments per file average)
```

## Evaluation Methodology

### Robust Statistical Evaluation
- **N=7 seeds** for each clustering run to ensure statistical robustness
- **Stability analysis**: `1.0 - min(silhouette_std, 1.0)`
- **Quality scoring**: `Silhouette Score × Stability`
- **Cluster validation**: Minimum size, balance (Gini coefficient), separation thresholds

### Clustering Quality Metrics
1. **Silhouette Score**: Measures cluster separation and cohesion (-1 to 1, higher better)
2. **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster variance
3. **Davies-Bouldin Index**: Average similarity ratio of clusters (lower better)
4. **Stability**: Consistency across multiple random seeds (0 to 1, higher better)

### Validation Constraints (Improved Evaluation System)
```python
MIN_CLUSTER_SIZE_RATIO = 0.02    # Clusters must be ≥2% of dataset
MAX_GINI_COEFFICIENT = 0.8       # Prevent extremely imbalanced clusters
MIN_SILHOUETTE_THRESHOLD = 0.2   # Minimum separation quality
```

### Quality Score Calculation
```python
# Primary metric combining separation and consistency
Quality_Score = Silhouette_Score × Stability

# Where:
Silhouette_Score = mean(silhouette_scores_across_7_seeds)
Stability = 1.0 - min(std(silhouette_scores), 1.0)
```

## Key Findings

### Dataset 1 (Original): Top Results
1. **D1** (Seg + HighPass + Bandpass): 0.488 (HDBSCAN)
2. **C4** (Seg + PeakNormalize): 0.462 (K-Means)
3. **A0** (NoSeg + NoPreprocess): 0.458 (HDBSCAN)
4. **D0** (Seg + HighPass + Peak): 0.451 (HDBSCAN)
5. **C3** (Seg + HighPass): 0.442 (HDBSCAN)

### Dataset 2 (New): Top Results
1. **B2** (NoSeg + FullPipeline): 0.406 (K-Means k=3, logmel_mean)
2. **A4** (NoSeg + PeakNormalize): 0.360 (K-Means k=3, logmel_mean)
3. **A0** (NoSeg + NoPreprocess): 0.360 (K-Means k=3, logmel_mean)
4. **D0** (Seg + HighPass + Peak): 0.357 (K-Means k=3, logmel_mean)
5. **D1** (Seg + HighPass + Bandpass): 0.352 (K-Means k=4, logmel_mean)

### Cross-Dataset Consistency
- **60% ranking consistency** (A0, D0, D1 in both top 5)
- **-27% average performance drop** in new dataset
- **Core patterns generalize** but absolute performance is dataset-dependent

## Scientific Insights Discovered

### 1. Segmentation Effects
```
Method           | NoSeg → Seg | Penalty | Compatibility Rank
HighPass20       | 0.306 → 0.295 | -3.6%  | #1 (Best)
SpectralGating   | 0.350 → 0.323 | -7.7%  | #2
Bandpass         | 0.349 → 0.317 | -9.2%  | #3
PeakNormalize    | 0.360 → 0.323 | -10.1% | #4
NoPreprocess     | 0.360 → 0.323 | -10.3% | #5 (Worst)
```

**Key Insight**: Segmentation generally hurts performance (-8% average) but improves robustness (+50% more valid results)

### 2. Combination Effects
- **Smart 2-method combinations** (D0, D1) beat individual methods
- **Over-processing penalty**: 4-method combo (D2) performs worse than 2-method combos
- **"Less is more" principle**: Simpler combinations work better with segmentation

### 3. Algorithm Preferences
- **Original dataset**: HDBSCAN dominant (80% of top results)
- **New dataset**: K-Means only (HDBSCAN excluded for consistency)
- **Representation preference**: Dataset-dependent (mfcc_mean vs logmel_mean)

### 4. Visual vs Metric Discrepancy
- **B2 paradox**: Highest metric score (0.406) but poor visual clustering
- **Importance of visual inspection**: Metrics alone insufficient for quality assessment

## Experimental Design Philosophy

### "Smart Combination" Criteria
1. **Performance-based selection**: Combine top individual performers
2. **Functional complementarity**: Mix amplitude (peak normalize) + frequency (filters) methods
3. **Segmentation context**: Use segmentation compatibility rankings for C/D series

### Why K-Means as "Gold Standard"
1. **Interpretability**: Clear cluster assignments, no noise classification
2. **Consistency**: Deterministic results with fixed seeds
3. **Scalability**: Works reliably across different dataset sizes
4. **Comparability**: Standard baseline for clustering evaluation

### Evaluation System Evolution
- **Initial**: Basic silhouette scoring
- **Improved**: Added stability analysis (N=7 seeds)
- **Final**: Cluster balance validation, robust statistical evaluation

## Technical Infrastructure

### File Organization
```
Development/
├── cycles/                           # Original dataset experiments
│   ├── A0_NoSeg_NoPre/              # Individual cycle directories
│   │   ├── code/                    # Processing scripts
│   │   ├── outputs/                 # Results, features, visualizations
│   │   └── README.md               # Cycle description
│   ├── comprehensive_re_evaluation_results.csv
│   └── Experiment_Tracking_System_Final.xlsx
└── Unsupervised Learning with New Data/  # New dataset experiments
    ├── A0_NoSeg_NoPre/ ... D2_Seg_FullPipeline/  # All 16 cycles
    ├── new_data_comprehensive_results.csv
    └── New_Data_Experiment_Tracking.xlsx
```

### Code Architecture
```python
# Core processing scripts per cycle:
extract_features.py     # Audio → features (3 representations)
run_clustering.py      # Features → clustering results  
make_visuals.py        # Results → UMAP visualizations
log_to_excel.py        # Results → Excel/CSV logging
run_cycle.py           # Orchestrate full cycle execution

# Shared utilities:
kmeans_only_clustering.py  # Standardized K-Means evaluation
log_results.py            # Global results aggregation
```

### Reproducibility Features
- **Fixed random seeds**: `np.random.seed(42)`, `RANDOM_SEED = 42`
- **Consistent parameters**: Same clustering parameters across all runs
- **Version control**: All scripts saved per cycle for reproducibility
- **Comprehensive logging**: Every result logged with metadata

## Statistical Rigor

### Robust Evaluation Protocol
```python
# For each (representation, algorithm, k-value) combination:
for seed in range(N_SEEDS):  # N_SEEDS = 7
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED + seed)
    labels = kmeans.fit_predict(features)
    
    # Validate cluster quality
    if validate_cluster_quality(labels):
        metrics = compute_metrics(features, labels)
        valid_results.append(metrics)

# Aggregate across seeds
mean_silhouette = np.mean([r.silhouette for r in valid_results])
std_silhouette = np.std([r.silhouette for r in valid_results])
stability = 1.0 - min(std_silhouette, 1.0)
quality_score = mean_silhouette * stability
```

### Cluster Quality Validation
```python
def validate_cluster_quality(labels, n_samples):
    # Check cluster size constraints
    min_cluster_ratio = min_cluster_size / n_samples
    if min_cluster_ratio < MIN_CLUSTER_SIZE_RATIO:
        return False
    
    # Check cluster balance (Gini coefficient)
    gini_coeff = calculate_gini_coefficient(cluster_sizes)
    if gini_coeff > MAX_GINI_COEFFICIENT:
        return False
    
    # Check separation quality
    if silhouette_score < MIN_SILHOUETTE_THRESHOLD:
        return False
    
    return True
```

## Results Summary

### New Dataset (144 results from 16 cycles)

#### Top 10 Results:
1. **B2** (NoSeg + FullPipeline): 0.406
2. **A4** (NoSeg + PeakNormalize): 0.360  
3. **A0** (NoSeg + NoPreprocess): 0.360
4. **D0** (Seg + HighPass + Peak): 0.357
5. **D1** (Seg + HighPass + Bandpass): 0.352
6. **A2** (NoSeg + SpectralGating): 0.350
7. **A1** (NoSeg + Bandpass): 0.349
8. **B1** (NoSeg + Peak + Bandpass): 0.349
9. **C4** (Seg + PeakNormalize): 0.323
10. **C0** (Seg + NoPreprocess): 0.323

#### Series Performance Summary:
- **A-Series**: Individual NoSeg methods, A0/A4 tied winners (0.360)
- **B-Series**: NoSeg combinations, B2 overall winner (0.406) but poor visuals
- **C-Series**: Individual Seg methods, consistent ~10% penalty vs A-series
- **D-Series**: Seg combinations, D0 best segmented method (0.357)

### Cross-Dataset Comparison
- **Consistent performers**: A0, D0, D1 (appear in both top 5s)
- **Performance drop**: -27% average in new dataset
- **Algorithm preference**: HDBSCAN (original) vs K-Means (new)
- **Generalizability**: Core patterns hold, absolute performance dataset-dependent

## Key Scientific Discoveries

### 1. The "Less is More" Principle
**Finding**: Simpler combinations outperform complex ones with segmentation
- **D0** (2 methods): 0.357
- **D1** (2 methods): 0.352  
- **D2** (4 methods): 0.308
- **Insight**: Over-processing hurts segmented audio clustering

### 2. Segmentation Compatibility Ranking
**Finding**: Methods have different compatibility with segmentation
1. **HighPass20**: -3.6% penalty (most compatible)
2. **SpectralGating**: -7.7% penalty
3. **Bandpass**: -9.2% penalty
4. **PeakNormalize**: -10.1% penalty
5. **NoPreprocess**: -10.3% penalty (least compatible)

### 3. Amplitude + Frequency Synergy
**Finding**: Combining amplitude and frequency domain methods is optimal
- **D0** (HighPass + PeakNormalize): Best segmented performance
- **Rationale**: HighPass (frequency) + PeakNormalize (amplitude) = complementary benefits

### 4. Visual vs Metric Discrepancy
**Finding**: High metric scores don't guarantee good visual clustering
- **B2**: 0.406 quality score but poor visual separation
- **Implication**: Visual inspection remains critical for clustering validation

### 5. Robustness vs Performance Trade-off
**Finding**: Segmentation increases robustness but decreases peak performance
- **Robustness**: +50% more valid clustering results
- **Performance**: -8% average quality score penalty
- **Insight**: Segmentation provides more stable but lower-scoring clusters

## Methodological Innovations

### 1. Stability-Based Quality Scoring
```python
# Novel metric combining separation and consistency
Quality_Score = Silhouette_Score × Stability
Stability = 1.0 - min(silhouette_std, 1.0)
```

### 2. Comprehensive Cluster Validation
- **Size constraints**: Prevent tiny clusters (<2% of data)
- **Balance constraints**: Prevent extremely imbalanced clusters (Gini > 0.8)
- **Separation constraints**: Ensure meaningful cluster separation (silhouette > 0.2)

### 3. Multi-Seed Robust Evaluation
- **N=7 seeds** per clustering run for statistical reliability
- **Failure handling**: Invalid runs marked as 0.000 quality score
- **Comprehensive logging**: All 144 results logged, including failures

## Limitations and Considerations

### 1. K-Means Limitations
- **"K Problem"**: Must specify number of clusters in advance
- **"Round Cluster" assumption**: Assumes spherical cluster shapes
- **"Distance Assumption"**: Uses Euclidean distance metric
- **"Noise Sensitivity"**: Outliers affect cluster centroids

### 2. Dataset Dependency
- **Performance levels**: Highly dependent on dataset complexity
- **Method rankings**: Partially consistent but not absolute
- **Algorithm preference**: HDBSCAN vs K-Means effectiveness varies

### 3. Evaluation Constraints
- **Intrinsic metrics only**: No ground truth labels used
- **Visual subjectivity**: Human visual assessment not quantified
- **Limited clustering algorithms**: Focused on K-Means for consistency

## Implications for Transfer Learning

### Validated Insights (Ready for Transfer Learning Test)
1. **A0, A4**: Strong individual baselines worth testing
2. **D0**: Best amplitude+frequency combination for segmented audio
3. **D1**: Strong frequency-domain combination
4. **Segmentation effects**: Expect ~10% performance penalty but better robustness

### Open Questions (Need Transfer Learning to Answer)
1. **Do clustering insights translate to classification performance?**
2. **How do foundation models change preprocessing effectiveness?**
3. **Is visual quality more important for supervised learning?**
4. **Do segmentation effects hold in supervised settings?**

### Preprocessing Method Priorities for Transfer Learning
1. **High Priority**: A0 (baseline), A4 (winner), D0 (best segmented)
2. **Medium Priority**: D1 (frequency combo), B2 (investigate metric vs visual discrepancy)
3. **Low Priority**: Remaining methods for comprehensive validation

## Technical Artifacts Generated

### Data Files
- **Original dataset**: 5 cycles, ~33 results, Excel tracking
- **New dataset**: 16 cycles, 144 results, comprehensive CSV + Excel
- **Cross-dataset**: Comparison analysis and consistency metrics

### Visualizations
- **144 UMAP plots** (new dataset): All clustering results visualized
- **Comprehensive plots** (original dataset): Selected high-performing results
- **Error visualizations**: Failed clustering attempts also plotted

### Code Base
- **16 cycle directories**: Complete code for each preprocessing method
- **Standardized scripts**: Consistent evaluation across all methods
- **Utility functions**: Reusable clustering, visualization, and logging tools

## Reproducibility Package

### Complete Experimental Record
1. **All source code**: Saved per cycle for exact reproduction
2. **Configuration files**: Parameters documented for each run
3. **Random seeds**: Fixed for deterministic reproduction
4. **Environment**: Library versions and system information logged
5. **Results**: Comprehensive CSV with all metrics and metadata

### Validation Protocol
1. **Statistical robustness**: N=7 seeds, stability analysis
2. **Quality constraints**: Cluster size, balance, separation validation
3. **Comprehensive logging**: Success and failure cases documented
4. **Visual validation**: UMAP plots for all results

This experiment established a comprehensive unsupervised baseline and identified optimal preprocessing methods for respiratory audio clustering. The findings provide a solid foundation for transfer learning validation using pretrained models like OPERA-CT.

## Next Steps: Transfer Learning Validation

The comprehensive unsupervised analysis provides clear hypotheses for transfer learning validation:

1. **Test if A0, A4, D0 remain top performers** with OPERA-CT features
2. **Validate segmentation effects** in supervised learning context  
3. **Investigate B2 metric vs visual discrepancy** using classification performance
4. **Compare preprocessing effectiveness** with foundation model features vs handcrafted features

This systematic approach ensures that transfer learning experiments build meaningfully on established unsupervised insights while respecting the different requirements of supervised learning.
