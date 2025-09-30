# OPERA-CT Transfer Learning Pipeline

## Overview
Structured, stepwise execution pipeline for OPERA-CT transfer learning experiments with comprehensive output management and reproducibility.

## Pipeline Structure

### Results Organization
```
results/
└─ experiments/
   └─ <run_id>/
      ├─ 00_setup/          # Installation logs
      ├─ 01_features/       # Feature extraction outputs
      ├─ 02_train/          # Training outputs  
      ├─ 03_eval/           # Evaluation outputs
      ├─ 04_cluster/        # Clustering analysis
      ├─ 05_transfer/       # Transfer learning results
      └─ artifacts/         # Reproducibility artifacts
```

### Run ID Convention
`YYYYMMDD-HHMMSS__{dataset}__{prepTag}__{featLayer}-{pool}__{head}__seed{seed}`

Example: `20250919-2130__asthmaA__D0__last-mean__mlp__seed1337`

## Phase-by-Phase Execution

### Phase 0 - Setup & Environment
```bash
# Install OPERA-CT and dependencies
cd setup/
./install_opera.sh
```

**Outputs:**
- `00_setup/install_log.txt`

### Phase 1 - Feature Extraction
```bash
# Create new run
python scripts/new_run.py \
    --data.dataset_name respiratory \
    --preprocess.tag D0 \
    --preprocess.d0_normalize \
    --preprocess.d0_highpass \
    --features.backend opera_ct \
    --train.head mlp

# Extract features
python scripts/extract_features.py --run_id <run_id>
```

**Outputs:**
- `01_features/train.parquet` - Training features & labels
- `01_features/val.parquet` - Validation features & labels  
- `01_features/test.parquet` - Test features & labels
- `01_features/stats.json` - Feature extraction statistics
- `01_features/preview/` - Sample waveform & spectrogram plots

### Phase 2 - Training
```bash
python scripts/train.py --run_id <run_id>
```

**Outputs:**
- `02_train/ckpts/best.ckpt` - Best model checkpoint
- `02_train/ckpts/last.ckpt` - Last model checkpoint
- `02_train/curves/loss.png` - Training loss curves
- `02_train/curves/f1_macro.png` - F1 score curves
- `02_train/metrics_train_val.json` - Training & validation metrics

### Phase 3 - Evaluation
```bash
python scripts/eval.py --run_id <run_id>
```

**Outputs:**
- `03_eval/metrics_test.json` - Test set metrics
- `03_eval/per_class.csv` - Per-class performance
- `03_eval/confusion_matrix.png` - Confusion matrix plot
- `03_eval/roc_curves.png` - ROC curves
- `03_eval/errors/false_positives.tsv` - Misclassified samples
- `03_eval/errors/false_negatives.tsv`

### Phase 4 - Clustering Diagnostics
```bash
python scripts/cluster_eval.py --run_id <run_id>
```

**Outputs:**
- `04_cluster/umap_train.png` - UMAP visualization of training features
- `04_cluster/umap_val.png` - UMAP visualization of validation features
- `04_cluster/clustering_report.json` - Clustering quality metrics

### Phase 5 - Cross-Dataset Transfer
```bash
python scripts/cross_transfer.py --run_id <run_id> \
    --data.transfer_train_dir data/other/train \
    --data.transfer_test_dir data/other/test \
    --shots_per_class 5
```

**Outputs:**
- `05_transfer/shots_{N}/metrics.json` - Transfer learning metrics
- `05_transfer/shots_{N}/cm.png` - Confusion matrix

## Global Management

### Summarize All Runs
```bash
python scripts/summarize_runs.py
```
Updates `results/index.csv` with metrics from all runs.

### Quick Access
- `results/latest` - Symlink to most recent run
- `results/index.csv` - Global summary of all experiments

## Configuration

### Example Usage
```bash
# D0 preprocessing experiment
python scripts/new_run.py \
    --data.dataset_name hospital_audio \
    --preprocess.tag D0 \
    --preprocess.d0_normalize \
    --preprocess.d0_highpass \
    --features.backend opera_ct \
    --train.head mlp \
    --seed 42

# D1 preprocessing experiment  
python scripts/new_run.py \
    --data.dataset_name hospital_audio \
    --preprocess.tag D1 \
    --preprocess.d1_bandpass \
    --preprocess.d0_highpass \
    --features.backend opera_ct \
    --train.head mlp \
    --seed 42

# B2 full pipeline experiment
python scripts/new_run.py \
    --data.dataset_name hospital_audio \
    --preprocess.tag B2 \
    --preprocess.b2_full_pipeline \
    --features.backend opera_ct \
    --train.head mlp \
    --seed 42
```

### Configuration Files
- `configs/example_config.yaml` - Template configuration
- `configs/<run_id>.yaml` - Per-run configuration backup

## Reproducibility Features

Every run automatically creates:
- `artifacts/config_dump.yaml` - Complete configuration
- `artifacts/manifest.json` - Git commit, system info, environment
- `artifacts/versions.txt` - Library versions, Python, CUDA

## Key Features

### Automatic Run Management
- Unique run IDs prevent overwrites
- Automatic versioning (\_v2, \_v3) if conflicts
- Structured output organization
- Global experiment tracking

### Preprocessing Integration
- D0: HighPass + PeakNormalize (amplitude + frequency)
- D1: HighPass + Bandpass (frequency domain)
- B2: Full pipeline (all methods)
- Custom combinations supported

### OPERA-CT Integration
- Automatic fallback to mel spectrograms if OPERA-CT unavailable
- Flexible feature extraction layers and pooling
- Compatible with our existing preprocessing pipeline

### Comprehensive Evaluation
- Classification metrics (accuracy, F1, ROC-AUC)
- Clustering analysis (UMAP, silhouette scores)
- Cross-dataset transfer learning
- Few-shot learning evaluation
- Visual diagnostics and error analysis

## Next Steps

1. **Phase 0**: Install OPERA-CT using `setup/install_opera.sh`
2. **Phase 1**: Create first run and extract features
3. **Validation**: Compare with our unsupervised clustering insights
4. **Scale**: Run systematic experiments across preprocessing methods
5. **Analysis**: Validate D0/D1 superiority in supervised setting

This pipeline directly builds on our comprehensive unsupervised clustering findings and provides a systematic way to validate them through transfer learning with state-of-the-art respiratory audio models.
