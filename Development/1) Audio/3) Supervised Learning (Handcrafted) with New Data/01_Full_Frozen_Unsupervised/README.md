# Transfer Learning with OPERA-CT

## Project Overview
This project explores transfer learning using the OPERA-CT (OPEn Respiratory Acoustic foundation models - Contrastive Transformer) model for respiratory audio analysis. We aim to validate our unsupervised clustering insights using a state-of-the-art pretrained respiratory audio encoder.

## OPERA-CT Model
- **Paper**: [OPERA: Alleviating Hallucination in Multi-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation](https://arxiv.org/html/2406.16148)
- **GitHub**: https://github.com/evelyn0414/OPERA
- **Architecture**: Hierarchical Token-Semantic Audio Transformer
- **Training**: Contrastive learning-based foundation model
- **Domain**: Respiratory acoustic analysis

## Experimental Goals

### Phase 1: Integration & Baseline
- [ ] Set up OPERA-CT from GitHub repository
- [ ] Extract features from our audio datasets using pretrained encoder
- [ ] Test preprocessing compatibility (D0, D1, B2 methods)
- [ ] Baseline clustering using OPERA-CT features

### Phase 2: Transfer Learning Validation
- [ ] Fine-tune OPERA-CT on our labeled respiratory data
- [ ] Compare preprocessing methods in supervised setting
- [ ] Validate unsupervised insights (D0 amplitude+frequency combo)
- [ ] Cross-dataset transfer (original → new dataset)

### Phase 3: Hybrid Analysis
- [ ] Combine our preprocessing + OPERA-CT features
- [ ] Compare vs raw audio → OPERA-CT pipeline
- [ ] Analyze contribution of each component

## Directory Structure

```
Transfer Learning with OPERA-CT/
├── setup/                  # Installation scripts and environment setup
├── experiments/           # Experiment configurations and scripts
├── data/                 # Data preparation and preprocessing
├── models/               # Model weights and configurations
├── results/              # Experimental results and analysis
├── notebooks/            # Jupyter notebooks for exploration
├── utils/                # Utility functions and helpers
└── README.md            # This file
```

## Key Research Questions

1. **Preprocessing Validation**: Do our winning preprocessing methods (D0, D1) still perform best with OPERA-CT?
2. **Unsupervised → Supervised Transfer**: Do good clustering results translate to good classification performance?
3. **Cross-Dataset Generalization**: How well do OPERA-CT + our methods generalize across datasets?
4. **Feature Analysis**: What respiratory patterns does OPERA-CT capture vs our handcrafted features?

## Expected Outcomes

- Validation of unsupervised clustering insights using supervised learning
- Optimal preprocessing pipeline for OPERA-CT-based respiratory analysis
- Cross-dataset generalization study
- Comparative analysis: handcrafted features vs foundation model features

## Connection to Previous Work

This builds directly on our comprehensive unsupervised clustering experiment:
- **Best methods identified**: D0 (Seg + HighPass + PeakNormalize), D1 (Seg + HighPass + Bandpass)
- **Key insights**: Amplitude+frequency combinations, segmentation effects, over-processing penalties
- **Cross-dataset findings**: 60% ranking consistency, -27% performance drop but pattern preservation

## Getting Started

1. Clone OPERA-CT repository
2. Set up environment and dependencies
3. Prepare our audio datasets
4. Run baseline feature extraction
5. Execute transfer learning experiments

---

*Created: December 2024*
*Last Updated: December 2024*
