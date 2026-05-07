# PloidyVision 🔬

Advanced cell-cycle analysis and ploidy determination from microscopic images. This project implements a robust pipeline for predicting cell cycle stages using DAPI intensity and nuclear area features.

## 🌟 Overview

PloidyVision is designed to automate the classification of cell cycle phases (ploidy levels) in microscopy datasets. The core strategy focuses on a **DAPI-only prediction** pipeline, ensuring that final Stage predictions are produced without reading ground-truth labels, which are reserved strictly for evaluation.

### Key Methodology
- **Feature Space**: 2D classification using `log(Intensity_NoBg)` and `log(Area)`.
- **Primary Predictor**: A weighted kNN atlas implementation.
- **Benchmarks**: Performance is validated against a baseline k-means approach, showing significant improvements in accuracy through the reference-fit model.

## 📁 Repository Structure

- **`python/`**: Core implementation of the cell cycle pipeline.
  - `cell_cycle_pipeline.py`: Main processing logic.
  - `downloader.py`: Data acquisition scripts.
  - `gerar_projecoes.py`: Image projection utilities.
  - `results/`: Benchmarks and performance figures.
- **`matlab/`**: Ported implementation of the cell-cycle scripts for MATLAB environments.
  - Includes benchmarks and application scripts (`cell_cycle_v12_app_script.m`).
- **`docs/`**: Scientific documentation, including:
  - MSc Thesis by Hemaxi Narotamo.
  - Project guidelines and research articles.

## 📊 Performance Summary

Based on the latest benchmarks across multiple datasets (`sub1` to `sub4`):

- **Mean Reference-Fit Accuracy**: ~100.0%
- **Mean Leave-One-Dataset-Out Accuracy**: ~66.5%
- **Mean GT Coverage**: ~97.3%

The results indicate that while the DAPI+Area model reaches very high performance on annotated nuclei, generalization across different datasets remains the primary focus for optimization.

## 🚀 Getting Started

1. Ensure the DAPI segmentation pipeline is focused on the blue channel.
2. The final predictor should not use red/green signal channels (these are for GT evaluation only).
3. Use the weighted kNN atlas on the standardized feature space: `[log(int_nobg), log(area)]`.

---
*Developed as part of the FBIB Research Project.*
