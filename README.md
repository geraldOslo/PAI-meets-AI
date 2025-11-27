# PAI-meets-AI: Deep Learning for Periapical Index Classification

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Deep learning system for automated classification of dental radiographs using the Periapical Index (PAI) scale. This project implements a systematic hyperparameter search framework comparing CNN architectures (ResNet50, EfficientNet-B3, ConvNeXt-Tiny) for assessing periapical periodontitis severity.

**Institution**: University of Oslo, Faculty of Dentistry
**Author**: Gerald Torgersen
**Year**: 2025

---
## Overview

### Clinical Background

The **Periapical Index (PAI)** is a 5-point ordinal scale for assessing periapical bone changes in dental radiographs:

| Score | Description | Clinical Status |
|-------|-------------|-----------------|
| PAI 1 | Normal apical periodontium | Healthy |
| PAI 2 | Bone structural changes indicating, but not pathognomonic for, apical periodontitis | Healthy |
| PAI 3 | Bone structural changes with some mineral loss characteristic of apical periodontitis | **Pathological** |
| PAI 4 | Well defined radiolucency | **Pathological** |
| PAI 5 | Radiolucency with radiating expansions of bone structural changes| **Pathological** |
> **Reference:** Ørstavik, D. (1996) Time‐course and risk analyses of the development and healing of chronic apical periodontitis in man. *International Endodontic Journal*. [Online] 29 (3), 150–155. Available from: [Wiley Online Library](https://onlinelibrary.wiley.com/doi/10.1111/j.1365-2591.1996.tb01361.x) (Accessed 25 November 2025).


**Primary Metric**: Quadratic Weighted Kappa (QWK) - accounts for ordinal nature and severity of misclassifications.

### Model Architectures

| Model | Parameters | Input Size | Batch Size | Strength |
|-------|------------|------------|------------|----------|
| **ResNet50** | ~25M | 224×224 | 128 | Medical imaging standard, fast training |
| **EfficientNet-B3** | ~12M | 300×300 | 64 | Parameter efficient, captures fine details |
| **ConvNeXt-Tiny** | ~28M | 224×224 | 96 | State-of-the-art, stable training |

All models use ImageNet pretraining and are optimized for A100 40GB GPUs.

---

## Key Features

- ✅ **Three CNN architectures** with systematic comparison
- ✅ **Three-phase hyperparameter search protocol** for research
- ✅ **Comprehensive result tracking** (CSV with all metrics + hyperparameters)
- ✅ **Class imbalance handling** (weighted sampling, focal loss)
- ✅ **Model interpretability** (GradCAM++ visualization)
- ✅ **HPC ready** (Slurm scripts for cluster training)
- ✅ **Automated normalization** and data preprocessing
- ✅ **Early stopping** with configurable patience
- ✅ **Multiple metrics** (QWK, Accuracy, F1, MAE, per-class sensitivity)

---

## Installation

### Requirements

- Python 3.11+
- CUDA 12.1+
- GPU with 20GB+ VRAM
- 16GB+ system RAM

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/PAI-meets-AI.git
cd PAI-meets-AI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm pandas scikit-learn matplotlib tqdm pillow tifffile grad-cam

# Create config from template
cp code/config_example.py code/config.py
# Edit code/config.py with your paths (data_csv, data_root, output_dir)
```

### Calculate Dataset Statistics

**IMPORTANT**: Must run before training!

```bash
cd code/data_utils
python calculate_dataset_statistics.py
# Update mean/std values in code/config.py with the output
```

---

## Quick Start

### 1. Single Model Training

```bash
# Train one model
python code/training/train_simple.py --models resnet50 --epochs 50 --use_oversampling

# Train all three models
python code/training/train_simple.py \
    --models resnet50 efficientnet_b3 convnext_tiny \
    --epochs 50 --patience 15 --use_oversampling

# View configuration
python code/training/train_simple.py --print_config
```

### 2. Research Workflow (Three-Phase Protocol)

For **systematic research** with publication-quality results:

```bash
# Phase 1: Baseline Exploration (15 experiments: 5 configs × 3 models)
python code/hyperparam_search_phase1.py --submit
python code/summarize_and_infer.py  # After completion

# Phase 2: Focused Optimization (8 experiments on best model)
# Update hyperparam_search_phase2.py based on Phase 1 results
python code/hyperparam_search_phase2.py --submit
python code/summarize_and_infer.py

# Phase 3: Final Validation (5-8 experiments, statistical replicates)
# Update hyperparam_search_phase3.py with Phase 2 best config
python code/hyperparam_search_phase3.py --submit
python code/summarize_and_infer.py
```

**Timeline**: 2 weeks (~30-40 GPU hours, parallelizable)


### 3. Results Analysis

```bash
# Generate comprehensive CSV summary
python code/summarize_and_infer.py --skip-inference --print-table

# Output: experiments/hyperparam_search/hyperparameter_summary_TIMESTAMP.csv
```

The CSV contains:
- Test metrics (QWK, accuracy, F1, per-class sensitivities)
- Validation metrics (best val QWK)
- All hyperparameters (LR, dropout, focal loss params, etc.)
- Training metadata (epochs, duration)

---


## Project Structure

```
PAI-meets-AI/
├── code/
│   ├── config.py                    # User config (create from config_example.py)
│   ├── summarize_and_infer.py       # Result aggregation & CSV generation
│   │
│   ├── hyperparam_search_phase*.py  # Phase 1/2/3/4/5/6 experiment definitions
│   │
│   ├── training/
│   │   ├── train_simple.py          # Main training script
│   │   ├── data_utils.py            # Dataset, transforms
│   │   ├── train_utils.py           # Loss functions (FocalLoss, etc.)
│   │   ├── model_utils.py           # Model creation
│   │   └── visualization_utils.py   # Plotting
│   │
│   ├── data_utils/
│   │   ├── calculate_dataset_statistics.py  # ⚠️ Run this first!
│   │
│   ├── test_inference/
│   │   ├── inference_gradcam.py              # Batch inference + GradCAM
│   │   ├── multilayer_xai_analysis.py        # Multi-layer XAI
│   │   └── PAI_inference.py                  # Interactive GUI
│   │
│   ├── utils/
│   │   ├── diagnose_logs.py                  # Debug log parsing
│   │   └── run_single_inference.py
│   │
│   └── slurm_scripts/
│       ├── run_summary.sh                    # Slurm job for summarization
│       ├── slurm_inference_*.sh              # Inference jobs
│       └── logs/                             # Auto-generated job logs
│
├── experiments/
│   ├── exp*_YYYYMMDD_HHMMSS/                # Phase 1 experiments (15)
│   ├── *_exp*_YYYYMMDD_HHMMSS/              # Phase 2+ experiments
│   │   ├── training.log                      # Complete log with structured hyperparams
│   │   ├── MODEL_best.pth                    # Best checkpoint (by val QWK)
│   │   ├── MODEL_history.json                # Training curves
│   │   └── statistics_report.txt             # Test metrics
│   │
│   └── hyperparam_search/
│       └── hyperparameter_summary_*.csv      # Aggregated results from all phases
└── readme.md                                 # This file
```

### Key Design Patterns

1. **Dataclass-Based Configuration**: All configs use Python dataclasses for type safety
2. **Structured Logging**: Hyperparameters logged in parseable format for automated extraction
3. **Flexible Dataset**: Supports multiple CSV formats with `filename` or `image_path` columns
4. **Class Imbalance Handling**: Weighted oversampling + focal loss + class weights
5. **Model Configs**: Pre-optimized settings per architecture (batch size, input size, dropout)

---

## Command Reference

### Training

```bash
# Single model with defaults
python code/training/train_simple.py --models resnet50 --epochs 50

# Multiple models with oversampling
python code/training/train_simple.py \
    --models resnet50 efficientnet_b3 convnext_tiny \
    --epochs 50 --patience 15 --use_oversampling

# Custom hyperparameters
python code/training/train_simple.py \
    --models efficientnet_b3 \
    --epochs 100 \
    --base_lr 1e-4 --max_lr 1e-3 \
    --weight_decay 1e-2 --dropout 0.6 \
    --focal_gamma 3.0 --use_oversampling
```

### Hyperparameter Search (Three-Phase)

```bash
# Phase 1: Generate scripts (dry run)
python code/hyperparam_search_phase1.py

# Phase 1: Generate and submit jobs
python code/hyperparam_search_phase1.py --submit

# Monitor jobs
squeue -u $USER
tail -f code/slurm_scripts/logs/p1_*.out

# Summarize results after completion
python code/summarize_and_infer.py --skip-inference --print-table

# Proceed to Phase 2 (update config based on Phase 1 first!)
python code/hyperparam_search_phase2.py --submit
```

### Result Analysis

```bash
# Generate summary CSV
python code/summarize_and_infer.py

# Skip inference (faster, uses existing reports)
python code/summarize_and_infer.py --skip-inference

# Print to console
python code/summarize_and_infer.py --print-table

# Debug mode
python code/summarize_and_infer.py --debug

# Run via Slurm
sbatch code/slurm_scripts/run_summary.sh
```

### Diagnostics

```bash
# Diagnose log parsing issues
python code/utils/diagnose_logs.py experiments/

# Test single image inference
python code/utils/run_single_inference.py
```

---

## Evaluation Metrics

### Primary: Quadratic Weighted Kappa (QWK)
- Accounts for ordinal nature of PAI scale
- Weights disagreements by severity (PAI 1→5 is worse than PAI 2→3)
- Range: -1 (complete disagreement) to 1 (perfect agreement)
- Interpretation: >0.80 excellent, 0.70-0.80 good, 0.60-0.70 moderate

### Secondary Metrics
- **Accuracy**: Overall correctness
- **F1-Score**: Weighted F1 for class imbalance
- **MAE**: Mean absolute error (average class distance)
- **Sensitivity/Specificity**: Per-class performance (PAI 1-5)
- **Confusion Matrix**: Detailed misclassification patterns

---

## Citation

```bibtex
@misc{torgersen2025pai,
  author = {Torgersen, Gerald},
  title = {PAI-meets-AI: Deep Learning for Periapical Index Classification},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/PAI-meets-AI}},
  institution = {University of Oslo, Faculty of Dentistry}
}
```

### Key References

1. **PAI System**: Ørstavik et al. (1986). "The periapical index: a scoring system for radiographic assessment."
2. **ResNet**: He et al. (2016). "Deep residual learning for image recognition."
3. **EfficientNet**: Tan & Le (2019). "EfficientNet: Rethinking model scaling."
4. **ConvNeXt**: Liu et al. (2022). "A ConvNet for the 2020s."
5. **Focal Loss**: Lin et al. (2017). "Focal loss for dense object detection."
6. **GradCAM++**: Chattopadhay et al. (2018). "Grad-CAM++: Improved visual explanations."

---
## ⚠️ Disclaimer & Limitations

### 1. Research Use Only
This software and the associated model weights are intended for **academic research and educational purposes only**. 
*   It is **not** a certified Software as a Medical Device (SaMD) (e.g., FDA, MDR).
*   It must **not** be used for primary clinical diagnosis, treatment planning, or decision-making on human patients.
*   Any clinical application is the sole responsibility of the user and requires appropriate regulatory approval.

### 2. Technical "As-Is" Provision
This repository represents a research codebase that has evolved through multiple iterations of experimentation. While we aim for reproducibility:
*   The code is provided **"as is"** without warranty of any kind.
*   Users may encounter environment-specific issues or require minor modifications ("tweaks") to file paths, library versions, or hardware configurations to get the pipeline running on their local systems.
*   We cannot guarantee that the code will run "out of the box" on every operating system or hardware setup.

### 3. Liability
The authors and the University of Oslo assume **no liability** for errors, misuse, or damages resulting from the use of this software or the pre-trained models. Use strictly at your own risk.

---
## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Contact

**Gerald Torgersen**
University of Oslo, Faculty of Dentistry
GitHub: [@geraldOslo](https://github.com/geraldOslo)

---

## Acknowledgments

- **University of Oslo, Faculty of Dentistry** - Institutional support
- **Norwegian Research Computing** - Computational resources (Fox HPC cluster)
- **timm library** (Ross Wightman) - Pretrained models
- **pytorch-grad-cam** (Jacob Gildenblat) - Visualization tools
