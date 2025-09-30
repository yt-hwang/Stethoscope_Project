# Comprehensive OPERA-CT Validation Guide

## Overview
This guide shows how to systematically validate all 16 preprocessing methods from our unsupervised clustering analysis using OPERA-CT transfer learning.

## All 16 Methods Defined

### A-Series: Individual NoSeg Methods
- **A0**: No preprocessing (baseline)
- **A1**: Bandpass filter (100-2000 Hz)
- **A2**: Spectral gating (noise reduction)
- **A3**: High-pass filter (20 Hz)
- **A4**: Peak normalization

### B-Series: Combination NoSeg Methods
- **B0**: Bandpass + Spectral gating
- **B1**: Peak normalize + Bandpass
- **B2**: Full pipeline (all methods)

### C-Series: Individual Seg Methods
- **C0**: Segmentation only (10s windows)
- **C1**: Seg + Bandpass
- **C2**: Seg + Spectral gating
- **C3**: Seg + High-pass
- **C4**: Seg + Peak normalize

### D-Series: Combination Seg Methods
- **D0**: Seg + High-pass + Peak normalize
- **D1**: Seg + High-pass + Bandpass
- **D2**: Seg + Full pipeline

## Quick Start: Run All Methods

### Option 1: Complete Validation (All 16 Methods)
```bash
# Run all methods with default settings
python scripts/run_all_methods.py

# Run with specific configuration
python scripts/run_all_methods.py \
    --dataset_name hospital_audio \
    --features_backend opera_ct \
    --train_head mlp \
    --seed 42
```

### Option 2: Selective Methods
```bash
# Run only winners from unsupervised analysis
python scripts/run_all_methods.py \
    --methods A0 A4 D0 D1 B2 \
    --dataset_name respiratory

# Run A-series for baseline comparison
python scripts/run_all_methods.py \
    --methods A0 A1 A2 A3 A4

# Run segmentation comparison (A vs C series)
python scripts/run_all_methods.py \
    --methods A0 C0 A1 C1 A4 C4
```

## Manual Method-by-Method Execution

### Individual Method Examples

#### A0 - Baseline (No Preprocessing)
```bash
python scripts/new_run.py --preprocess.method A0
python scripts/extract_features.py --run_id <run_id>
python scripts/train.py --run_id <run_id>
```

#### A4 - Peak Normalization (Tied Winner)
```bash
python scripts/new_run.py --preprocess.method A4
python scripts/extract_features.py --run_id <run_id>
python scripts/train.py --run_id <run_id>
```

#### D0 - Best Segmented Method (Amplitude + Frequency)
```bash
python scripts/new_run.py --preprocess.method D0
python scripts/extract_features.py --run_id <run_id>
python scripts/train.py --run_id <run_id>
```

#### B2 - Full Pipeline (Highest Score but Poor Visuals)
```bash
python scripts/new_run.py --preprocess.method B2
python scripts/extract_features.py --run_id <run_id>
python scripts/train.py --run_id <run_id>
```

## Scientific Validation Questions

### 1. Ranking Consistency Validation
**Question**: Do the same methods that performed well in unsupervised clustering also perform well in supervised classification?

**Expected from Unsupervised Analysis**:
- A0, A4: Top individual methods (0.360)
- D0: Best segmented method (0.357)
- D1: Second best segmented (0.352)
- B2: Highest score overall (0.406) but poor visuals

**Test**: Compare supervised F1 scores with unsupervised quality scores

### 2. Segmentation Effect Validation
**Question**: Does segmentation have the same impact in supervised learning?

**Expected from Unsupervised Analysis**:
- Segmentation penalty: -3.6% to -10.3%
- HighPass20 most compatible with segmentation
- PeakNormalize least compatible

**Test**: Compare A-series vs C-series performance

### 3. Combination vs Individual Validation
**Question**: Do smart combinations outperform individual methods in supervised setting?

**Expected from Unsupervised Analysis**:
- D0 (combo) > C3, C4 (individuals)
- D1 (combo) > C1, C3 (individuals)
- But D2 (full) < D0, D1 (simpler combos)

**Test**: Compare B/D-series vs A/C-series performance

### 4. Visual Quality Investigation
**Question**: Does B2's high metric score but poor visuals translate to supervised performance?

**Expected**: B2 might have high supervised accuracy but poor generalization or interpretability

**Test**: Examine B2's confusion matrix, error analysis, and transfer performance

## Expected Outcomes

### If Our Unsupervised Insights Are Valid:
1. **A0, A4** should rank high in supervised F1 scores
2. **D0** should be the best segmented method
3. **Segmentation penalty** should be consistent (-5% to -15%)
4. **B2** might have high training accuracy but issues in evaluation
5. **Method ranking correlation** should be significant (r > 0.6)

### If Transfer Learning Reveals New Insights:
1. **OPERA-CT features** might change method rankings
2. **Foundation model** might be more robust to preprocessing
3. **Some methods** might work better with learned vs handcrafted features
4. **Cross-dataset transfer** might favor different preprocessing

## Analysis Scripts

### After Running Experiments
```bash
# Update global results index
python scripts/summarize_runs.py

# Compare with unsupervised results
python scripts/compare_unsupervised_supervised.py  # (to be created)

# Generate validation report
python scripts/generate_validation_report.py  # (to be created)
```

### Key Metrics to Track
- **Classification F1 Score** (macro-averaged)
- **Cross-dataset transfer performance**
- **Few-shot learning performance**
- **Clustering quality of learned features**
- **Correlation with unsupervised quality scores**

## Timeline Estimation

### Complete Validation (All 16 Methods)
- **Feature extraction**: ~2-4 hours (depends on dataset size)
- **Training**: ~30 minutes per method = 8 hours total
- **Evaluation**: ~15 minutes per method = 4 hours total
- **Total**: ~12-16 hours for complete validation

### Strategic Subset (8 Key Methods)
- **Methods**: A0, A4, C0, C4, D0, D1, B2, plus one comparison
- **Total time**: ~6-8 hours

## Computational Requirements

### Minimum Requirements
- **RAM**: 8GB (for feature extraction and model training)
- **Storage**: 5GB (for all experiment outputs)
- **GPU**: Optional but recommended for faster training

### Recommended Setup
- **RAM**: 16GB+ 
- **GPU**: Any CUDA-compatible GPU
- **Storage**: 10GB+ (for comprehensive logging and visualizations)

## Success Criteria

### Scientific Validation Success
1. **Ranking correlation** r > 0.6 between unsupervised and supervised
2. **Segmentation effects** consistent direction and magnitude
3. **Top methods** (A0, A4, D0) remain in top 25% of supervised results
4. **Statistical significance** of method differences (p < 0.05)

### Technical Pipeline Success
1. **All methods execute** without critical failures
2. **Reproducible results** across runs with same seed
3. **Comprehensive logging** for all experiments
4. **Global results index** updated correctly

This comprehensive validation will definitively answer whether our unsupervised clustering insights generalize to supervised learning with state-of-the-art respiratory audio models.
