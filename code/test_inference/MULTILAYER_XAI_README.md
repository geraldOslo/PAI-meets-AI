# Multi-Layer XAI Analysis for PAI Classification

## Overview

This script performs comprehensive multi-layer explainability analysis on trained PAI classification models using GradCAM and GradCAM++ methods. It extends the basic inference script to provide deep insights into how different layers of the neural network contribute to the final prediction.

## Features

### Multi-Layer Analysis
- **Layer-wise heatmap generation**: Visualizes attention patterns at multiple depths
  - ResNet50: 4 major blocks (layer1-layer4)
  - EfficientNet-B3: 6 blocks (blocks 1-6)
  - ConvNeXt-Tiny: 4 stages (stages 0-3)

- **Weighted fusion**: Combines layer heatmaps with progressive weighting (0.5→1.0)
  - Deeper layers receive higher weights
  - Provides comprehensive view of model's decision process

- **Difference mapping**: Highlights divergence between final layer and combined heatmap
  - Reveals unique information in last layer vs. accumulated features

### Visualization Outputs

Each test image generates a **4-row comprehensive visualization**:

#### Row 1: Overview
- Original image with true/predicted labels
- Probability bar chart for all 5 PAI classes

#### Row 2: Layer-wise Heatmaps
- Individual CAM overlays for each targeted layer
- Shows progression from low-level to high-level features

#### Row 3: Fusion Analysis
- Combined (weighted fusion) heatmap
- Difference map: |last_layer - combined|

#### Row 4: Class-Specific Analysis
- Last-layer heatmaps for all 5 PAI classes
- Shows what the model "sees" for each class

### Average Heatmaps with Quadrant Awareness

Generates **per-class average heatmaps** with intelligent orientation correction:

- **Quadrant-aware flipping** (European tooth numbering):
  - Quadrants 2, 3 (left side): Horizontal flip
  - Quadrants 1, 2 (upper jaw): Vertical flip
  - Ensures anatomically consistent averaging

- **Separate averages for**:
  - Combined (weighted fusion) heatmaps
  - Last layer heatmaps
  - Both GradCAM and GradCAM++ methods

## Installation

### Prerequisites

```bash
# Required packages
pip install pytorch-grad-cam torch torchvision timm
pip install pandas numpy matplotlib scikit-learn pillow opencv-python tqdm
```

### Verify Installation

```bash
python -c "from pytorch_grad_cam import GradCAM, GradCAMPlusPlus; print('✓ GradCAM installed')"
```

## Usage

### Quick Start

```bash
# Analyze all three models with checkpoints in a directory
python code/test_inference/multilayer_xai_analysis.py \
    --model efficientnet_b3 resnet50 convnext_tiny \
    --checkpoint-dir experiments/phase3_best_checkpoints \
    --test-csv data/test.csv \
    --test-root data/images
```

### Single Model Analysis

```bash
# Analyze EfficientNet-B3 only
python code/test_inference/multilayer_xai_analysis.py \
    --model efficientnet_b3 \
    --checkpoint experiments/best_efficientnet_b3.pth \
    --test-csv data/test.csv \
    --test-root data/images
```

### GradCAM Only (Faster)

```bash
# Skip GradCAM++ for faster processing
python code/test_inference/multilayer_xai_analysis.py \
    --model efficientnet_b3 \
    --checkpoint-dir experiments/phase3_best_checkpoints \
    --test-csv data/test.csv \
    --test-root data/images \
    --cam-methods gradcam
```

### Custom Output Location

```bash
python code/test_inference/multilayer_xai_analysis.py \
    --model efficientnet_b3 \
    --checkpoint-dir experiments/phase3_best_checkpoints \
    --test-csv data/test.csv \
    --test-root data/images \
    --output-dir experiments/xai_analysis
```

## Command-Line Arguments

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--model` | Model(s) to analyze | `efficientnet_b3 resnet50 convnext_tiny` |
| `--checkpoint` OR `--checkpoint-dir` | Single checkpoint or directory | `--checkpoint-dir experiments/checkpoints` |
| `--test-csv` | Path to test CSV file | `data/test.csv` |
| `--test-root` | Root directory for test images | `data/images` |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--output-dir` | `experiments` | Base output directory |
| `--cam-methods` | `gradcam gradcamplusplus` | CAM methods to use |
| `--batch-size` | Model default | Override batch size |
| `--num-workers` | `4` | Data loading workers |
| `--heatmap-transparency` | `0.5` | Overlay transparency (0-1) |

## Running on HPC (Slurm)

### Setup

**Good news: Your best model checkpoints are already organized!**

The `model_checkpoints/` directory contains the three best models:
```
model_checkpoints/
├── resnet50_exp1_baseline_20251024_213049/
│   └── resnet50_best.pth (QWK=0.7951, Champion)
├── effnet_exp5_lower_lr_20251025_235232/
│   └── efficientnet-b3_best.pth (QWK=0.7811)
└── convnext_exp2_moderate_settings_20251026_061225/
    └── convnext-tiny_best.pth (QWK=0.7447, Best PAI-3=0.68)
```

### Automatic Configuration

The Slurm script **automatically extracts paths from `config.py`**:
- ✅ Test data paths from `InferenceConfig.test_csv_paths`
- ✅ Test root directories from `InferenceConfig.test_root_dirs`
- ✅ Output directory from `InferenceConfig.base_experiments_dir`
- ✅ Model checkpoints from hardcoded `model_checkpoints/` paths

**No manual path editing required!**

### Submit Job

```bash
cd code/slurm_scripts
sbatch slurm_multilayer_xai.sh
```

The script will:
1. Extract paths from `config.py`
2. Process ResNet50 (Champion model, QWK=0.7951)
3. Process EfficientNet-B3 (QWK=0.7811)
4. Process ConvNeXt-Tiny (Best PAI-3=0.68)
5. Generate comprehensive XAI visualizations for all three models

### Monitor Progress

```bash
# Check job status
squeue -u $USER

# View live output
tail -f code/slurm_scripts/logs/multilayer_xai_*.out

# Check for errors
tail -f code/slurm_scripts/logs/multilayer_xai_*.err
```

### Resource Requirements

- **GPU**: 1× A100 (recommended) or RTX30
- **Memory**: 64GB (can reduce for smaller models)
- **Time**: ~6 hours for all 3 models on full test set
- **CPUs**: 8 (for data loading)

**Estimated processing time per model**:
- EfficientNet-B3: ~2 hours (300×300 input, 6 layers)
- ResNet50: ~1.5 hours (224×224 input, 4 layers)
- ConvNeXt-Tiny: ~1.5 hours (224×224 input, 4 stages)

## Output Structure

```
experiments/
└── {model_name}/
    └── multilayer_xai_{timestamp}/
        ├── xai_individual_gradcam/
        │   ├── example_0_true_1_pred_1_gradcam_image001.png
        │   ├── example_1_true_2_pred_2_gradcam_image002.png
        │   └── ... (one per test image)
        │
        ├── xai_individual_gradcamplusplus/
        │   ├── example_0_true_1_pred_1_gradcamplusplus_image001.png
        │   └── ... (one per test image)
        │
        ├── average_heatmaps/
        │   ├── average_combined_gradcam_pai_1.png
        │   ├── average_combined_gradcam_pai_2.png
        │   ├── average_combined_gradcamplusplus_pai_1.png
        │   ├── average_layer4_gradcam_pai_1.png  (ResNet50)
        │   ├── average_block_6_gradcam_pai_1.png  (EfficientNet-B3)
        │   ├── average_stage_3_gradcam_pai_1.png  (ConvNeXt-Tiny)
        │   └── ... (per class × per method × per heatmap type)
        │
        ├── prediction_results.csv
        ├── confusion_matrix.png
        └── final_metrics.txt
```

### File Descriptions

**Individual Visualizations** (`xai_individual_{method}/`):
- Comprehensive 4-row PNG for each test image
- Filename format: `example_{idx}_true_{true_class}_pred_{pred_class}_{method}_{filename}.png`
- Shows: original, probabilities, layer heatmaps, combined, difference, class-specific

**Average Heatmaps** (`average_heatmaps/`):
- Per-class average heatmaps with quadrant-aware orientation
- Filename format: `average_{heatmap_type}_{method}_pai_{class}.png`
- Includes sample count in title

**Prediction Results** (`prediction_results.csv`):
- Columns: `filename`, `true_pai`, `predicted_pai`, `prob_pai_1`, ..., `prob_pai_5`
- Same format as basic inference script

**Metrics** (`final_metrics.txt`):
- Accuracy, QWK, MAE
- Confusion matrix

**Confusion Matrix** (`confusion_matrix.png`):
- Visual confusion matrix with QWK/accuracy in title

## Understanding the Visualizations

### Layer Progression

**Early Layers** (ResNet: layer1, EfficientNet: blocks 1-2):
- Focus on low-level features: edges, textures, simple patterns
- Heatmaps often diffuse and less class-specific

**Mid Layers** (ResNet: layer2-3, EfficientNet: blocks 3-4):
- Intermediate patterns bridging textures to structures
- Beginning to show anatomical relevance

**Late Layers** (ResNet: layer4, EfficientNet: blocks 5-6):
- High-level complex structures and semantic information
- Most class-discriminative features
- Critical for PAI classification

### Weighted Fusion Interpretation

The **combined heatmap** uses linear weights from 0.5 to 1.0:
- **Purpose**: Captures multi-scale information hierarchy
- **Weighting**: Deeper layers contribute more (0.5→1.0)
- **Advantage**: More comprehensive than single-layer CAM
- **Use case**: Understanding full decision pathway

### Difference Map Interpretation

The **difference map** shows `|last_layer - combined|`:
- **High intensity**: Last layer has unique information not in earlier layers
- **Low intensity**: Decision is consistent across all layers
- **Clinical insight**: Identifies if model relies on specific high-level features vs. holistic analysis

### Class-Specific Heatmaps

Row 4 shows what the model "sees" for each PAI class:
- **Correct predictions**: Predicted class should show strong activation on relevant anatomy
- **Misclassifications**: Compare true vs. predicted heatmaps to identify confusion sources
- **Model confidence**: Check if high confidence aligns with focused heatmaps

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

**Symptoms**: CUDA out of memory error

**Solutions**:
```bash
# Reduce batch size
python multilayer_xai_analysis.py ... --batch-size 16

# Or process models separately
python multilayer_xai_analysis.py --model efficientnet_b3 ...
python multilayer_xai_analysis.py --model resnet50 ...
```

#### 2. Checkpoint Not Found

**Symptoms**: "No valid checkpoints found!" or "Checkpoint path doesn't exist"

**Solutions**:
```bash
# Verify checkpoints exist in model_checkpoints/
ls -lh model_checkpoints/*/

# Should show:
# model_checkpoints/resnet50_exp1_baseline_20251024_213049/resnet50_best.pth
# model_checkpoints/effnet_exp5_lower_lr_20251025_235232/efficientnet-b3_best.pth
# model_checkpoints/convnext_exp2_moderate_settings_20251026_061225/convnext-tiny_best.pth

# If missing, restore from experiments/ or rerun training
```

#### 3. Layer Identification Error

**Symptoms**: "Could not find {layers} in model"

**Cause**: Model architecture mismatch or custom timm version

**Solution**: Verify timm model structure:
```python
import timm
model = timm.create_model('efficientnet_b3', pretrained=False)
print([name for name, _ in model.named_children()])
# Should include 'blocks' for EfficientNet
```

#### 4. Missing Quadrant Information

**Symptoms**: Warning about missing quadrant column

**Impact**: Average heatmaps will not apply quadrant-aware flipping
**Solution**: Optional - add `quadrant` column (1-4) to test CSV if available

## Performance Optimization

### Speed vs. Quality Trade-offs

| Configuration | Speed | Quality | Use Case |
|--------------|-------|---------|----------|
| GradCAM only | Fast (50% faster) | Good | Quick analysis, method comparison |
| GradCAM + GradCAM++ | Moderate | Best | Publication, comprehensive analysis |
| Reduced batch size | Slow | Same | Limited GPU memory |
| Skip average heatmaps | N/A | Reduced | Individual image focus only |

### Batch Size Guidelines

| Model | GPU | Recommended Batch Size |
|-------|-----|----------------------|
| EfficientNet-B3 | A100 40GB | 32-64 |
| EfficientNet-B3 | RTX30 24GB | 16-32 |
| ResNet50 | A100 40GB | 64-128 |
| ResNet50 | RTX30 24GB | 32-64 |
| ConvNeXt-Tiny | A100 40GB | 48-96 |
| ConvNeXt-Tiny | RTX30 24GB | 24-48 |

## Comparison with Basic Inference Script

| Feature | `inference_gradcam.py` | `multilayer_xai_analysis.py` |
|---------|----------------------|----------------------------|
| Single-layer CAM | ✓ (last layer only) | ✓ (all layers) |
| Multi-layer analysis | ✗ | ✓ |
| Weighted fusion | ✗ | ✓ |
| Difference maps | ✗ | ✓ |
| Class-specific heatmaps | ✗ | ✓ |
| Average heatmaps | ✗ | ✓ |
| Quadrant-aware flipping | ✗ | ✓ |
| Dual (pred+true) viz | ✓ | ✗ (4-row instead) |
| Speed | Fast | Moderate |
| Use case | Quick evaluation | Deep analysis |

**Recommendation**: Use `inference_gradcam.py` for initial model evaluation, then `multilayer_xai_analysis.py` for publication-quality explainability analysis.

## Citation and Credits

This implementation is based on the multi-layer XAI methodology from the previous version of the PAI project, adapted to support multiple architectures (ResNet50, EfficientNet-B3, ConvNeXt-Tiny).

**Original notebook**: `EfficientNett_Inference-MultiLayer_XAI-old.ipynb`

**CAM library**: [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)

**Project**: PAI meets AI - Automated Periapical Index Classification

**Author**: Gerald Torgersen

**Institution**: University of Oslo, Faculty of Dentistry

**Year**: 2025

## Advanced Usage

### Custom Layer Selection

To analyze specific layers, modify the `identify_target_layers()` function in `multilayer_xai_analysis.py`:

```python
# Example: Analyze only layers 2-4 for ResNet50
if 'resnet' in model_name_lower:
    for i in range(2, 5):  # Changed from range(1, 5)
        layer_name = f'layer{i}'
        if hasattr(model, layer_name):
            layers[layer_name] = getattr(model, layer_name)
```

### Custom Fusion Weights

Modify the `get_multi_layer_heatmaps()` function:

```python
# Current: Linear weights 0.5→1.0
combine_weights = np.linspace(0.5, 1.0, num_layers)

# Alternative: Equal weights
combine_weights = np.ones(num_layers) / num_layers

# Alternative: Exponential weighting (favor later layers more)
combine_weights = np.exp(np.linspace(0, 1, num_layers))
combine_weights /= combine_weights.sum()
```

### Processing Subset of Test Data

```python
# Modify test_data after loading in main()
test_data = test_data.head(50)  # Process only first 50 images
```

## License

This code is part of the PAI-meets-AI-2 project.

**SPDX-License-Identifier**: MIT

**Copyright** (c) 2025 Gerald Torgersen
