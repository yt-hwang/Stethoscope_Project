# Respiratory Sound Project — Starter Kit

This scaffold helps you build a full pipeline for **wheezing / breathing / noise** segmentation and classification.

## What’s inside
- `src/config.py` — central configuration (sample rate, frame/hop, mel settings).
- `src/audio_io.py` — robust audio loading (mono, resample, pre-emphasis).
- `src/features.py` — STFT, log-mel, MFCC, and lightweight wheeze indicators.
- `src/segmentation.py` — simple energy-based and envelope-based segmentation.
- `src/dataset.py` — PyTorch `Dataset` for (path, start, end, label) metadata.
- `src/models/cnn_small.py` — small CNN (optional CRNN hook) for spectrograms.
- `src/train.py` — training loop skeleton with metrics and checkpoints.
- `src/realtime.py` — streaming-friendly inference skeleton (causal windowing).
- `requirements.txt` — suggested Python deps.
- `data/metadata_example.csv` — example of how to structure labels.
- `notebooks/` — your experiments go here (optional).

## Quick start
1) Create and activate a virtualenv or conda env (Python ≥ 3.10).
2) Install deps:
```bash
pip install -r requirements.txt
```
3) Put your `.wav` files under `data/` and prepare a CSV like `data/metadata_example.csv`.
4) Try a quick feature extraction:
```bash
python -m src.features --wav data/example.wav --out out.npy
```
5) Train a simple classifier once you have segments:
```bash
python -m src.train --metadata data/metadata_example.csv --out_dir runs/exp01
```

## Notes
- Real-time: use `src/realtime.py` and adjust `HOP_MS`, model stride, and lookahead.
- Labels: start with frame-level labels (`breathing`, `wheezing`, `noise`) and later add event-level (onset/offset) annotations.
