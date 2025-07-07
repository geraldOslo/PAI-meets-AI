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

- Use `code/inference/PAIinference.py` or the notebook `code/inference/Inference-MultiLayerXAI.ipynb` to:
- Run batch inference on new image clips.
- Generate per-sample PAI predictions and class probabilities.
- Produce Grad-CAM and related heatmaps for explainability.
- The inference pipeline supports:
- Automatic device detection (CPU/GPU).
- TIFF and standard image formats.
- Adjustable pixel size parameters for consistent ROI extraction.
- Multi-threaded processing for responsiveness.
- Visualizations include individual sample heatmaps and population-average attention maps.

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


---
