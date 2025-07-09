# PAI-meets-AI

Open-source codebase for training, inference, and explainability of deep learning models for Periapical Index (PAI) [1] scoring from intraoral radiograph clips.

---

## Overview

This repository contains the full code and utilities to train, validate, and run inference with deep learning models that classify periapical lesions on dental radiographs according to the PAI scoring system (scores 1–5). The focus is on 300×300 pixel image clips centered on tooth apices, with human-annotated PAI labels.

The codebase also integrates explainable AI (XAI) methods, primarily multi-layer Grad-CAM variants, to visualize and interpret model decisions.

---

## Contents

- **Training pipeline**: Data loading, augmentation, model definition, training loops, checkpointing, and evaluation.
- **Inference pipeline**: Batch inference on new datasets, probability outputs, and Grad-CAM visualizations.
- **Data preparation utilities**: Scripts and notebooks for dataset metadata analysis, image statistics calculation, and data splitting.
- **Visualization tools**: Plotting training curves, Grad-CAM heatmaps, and summarizing results.
- **Configuration management**: Centralized hyperparameter and path configuration via `config.py`.
- **Requirements**: `requirements.txt` specifying all Python dependencies for reproducibility.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/geraldoslo/pai-meets-ai.git
   cd pai-meets-ai
   ```
2. Create and activate a Python environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate # Linux/macOS
   venv\Scripts\activate # Windows
   ```
3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. **Download model checkpoints:**
   - Model checkpoints must be downloaded from: https://huggingface.co/geraldOslo/pai-meets-ai
   - After downloading, copy the checkpoint files into the folder: `code/inference/model`
---

## Configuration

- Copy `code/training/configtemplate.py` to `code/training/config.py`.
- Edit `config.py` to specify:
- Paths to your dataset CSV files and image root directories.
- Training hyperparameters (batch size, learning rates, epochs, etc.).
- Data augmentation settings.
- Model architecture and fine-tuning parameters.
- Normalization statistics (mean and std) for your image data.
- The configuration file centralizes all adjustable parameters for easy experimentation.

---

## Data Preparation

- Use the provided notebook `code/tools/AnalyzeDataset.ipynb` to:
- Load and analyze dataset metadata CSVs.
- Compute class distributions and identify potential data issues.
- Calculate image pixel mean and standard deviation for normalization.
- These statistics should be transferred into `config.py` for consistent preprocessing during training.

**Note:**

If you are generating new datasets, refer to the method described by Jordal et al.[2] and use the [EndodonticMeasurements ImageJ plugin](https://github.com/geraldOslo/EndodonticMeasurements) [3] for standardized ROI extraction and measurement.

---

## Training

- The main training workflow is implemented in the Jupyter notebook `code/training/trainingpaimeetsai.ipynb`.
- It imports modular utilities from:
- `datautils.py` for dataset loading and augmentation.
- `modelutils.py` for model creation and modification.
- `trainutils.py` for training loops, loss functions, optimizers, schedulers, and checkpointing.
- `visualizationutils.py` for plotting metrics and GPU usage.
- Training supports:
- Stratified train/validation split.
- Oversampling to address class imbalance.
- Mixed precision training (AMP).
- Early stopping based on validation F1 score.
- Checkpoints, training history, and summary YAML files are saved automatically.

---

## Inference & Explainability

There are two main approaches for inference and explainability:

### 1. Batch Inference Notebook

- **Notebook:** `code/inference/Inference-MultiLayerXAI.ipynb`
- **Purpose:** Batch processing of all images in a folder. This notebook was used for testing the model on the test set.
- **Features:**
- Runs inference on multiple images in a directory.
- Outputs PAI predictions and class probabilities for each image.
- Generates Grad-CAM and related heatmaps for explainability.
- Supports population-level and per-class attention map visualization.

### 2. Interactive GUI for Single Images

- **Script:** `code/inference/PAIinference.py`
- **Purpose:** Interactive Python program with GUI for single-image analysis.
- **Features:**
- Load a single image file.
- Click on one of the apexes to select the region of interest for analysis.
- Run PAI prediction on the selected region.
- Optionally visualize CAM (Class Activation Map) for the prediction.
- Displays the probability distribution of PAI scores graphically.
- Supports TIFF and standard image formats, automatic device detection (CPU/GPU), and adjustable pixel size for ROI extraction.

---

## Project Structure
```text
pai-meets-ai/
├── code/
│ ├── training/
│ │ ├── configtemplate.py # Configuration template
│ │ ├── datautils.py # Dataset and augmentation utilities
│ │ ├── modelutils.py # Model loading and modification
│ │ ├── trainutils.py # Training loop, loss, optimizer
│ │ ├── visualizationutils.py # Plotting and visualization
│ │ └── trainingpaimeetsai.ipynb # Training notebook
│ ├── inference/
│ │ ├── configinference.py # Inference-specific config
│ │ ├── PAIinference.py # Inference script
│ │ ├── inferenceutils.py # Inference helper functions
│ │ └── Inference-MultiLayerXAI.ipynb # Inference + Grad-CAM notebook
│ ├── tools/
│ │ ├── analyzeruns.py # Run summary and analysis utilities
│ │ └── AnalyzeDataset.ipynb # Dataset analysis notebook
| ├── dataset_extraction/ # Scripts to convert scored images to anonymized training clips
├── requirements.txt # Python dependencies
├── LICENSE # MIT License
└── README.md # This file
```
---

## Reproducibility

- The repository follows best practices for reproducibility in medical AI research:
  - All code and pre-trained weights are openly available.
  - Configuration files separate code from environment-specific settings.
  - Data preprocessing and augmentation pipelines are fully documented.
  - Training and inference scripts include logging and checkpointing.
  - Environment dependencies are captured in `requirements.txt`.

---

## Contact & Contribution

- For questions, issues, or contributions, please open an issue or pull request on the GitHub repository.
- Contributions are welcome to improve code, add features, or enhance documentation.

---

## References
[1] 1. Ørstavik D, Kerekes K, Eriksen HM. The periapical index: A scoring system for radiographic assessment of apical periodontitis. Dental Traumatology. 1986 Feb;2(1):20–34.

[2] 2. Jordal, K., Skudutyte-Rysstad, R., Sen, A., Torgersen, G., Ørstavik, D., & Sunde, P. T. (2021). Effects of an individualized training course on technical quality and periapical status of teeth treated endodontically by dentists in the Public Dental Service in Norway. An observational intervention study. *International Endodontic Journal*. https://doi.org/10.1111/iej.13669

[3] 3. EndodonticMeasurements ImageJ plugin: https://github.com/geraldOslo/EndodonticMeasurements

---
