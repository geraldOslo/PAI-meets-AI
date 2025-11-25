# Multi-Layer XAI Analysis - Quick Start Guide

## TL;DR

```bash
# 1. Best model checkpoints are already organized in model_checkpoints/
#    - resnet50_exp1_baseline_20251024_213049/resnet50_best.pth (QWK=0.7951, Champion)
#    - effnet_exp5_lower_lr_20251025_235232/efficientnet-b3_best.pth (QWK=0.7811)
#    - convnext_exp2_moderate_settings_20251026_061225/convnext-tiny_best.pth (QWK=0.7447)

# 2. Run analysis (local) - using individual checkpoints
python code/test_inference/multilayer_xai_analysis.py \
    --model resnet50 \
    --checkpoint model_checkpoints/resnet50_exp1_baseline_20251024_213049/resnet50_best.pth \
    --test-csv data/test.csv \
    --test-root data/images

# 3. Or submit to Slurm (HPC) - automatically uses model_checkpoints/
#    Paths are extracted from config.py automatically!
sbatch code/slurm_scripts/slurm_multilayer_xai.sh
```

## What You Get

### For Each Model:

**Individual Visualizations** (one per test image):
- 4-row comprehensive PNG showing:
  - Row 1: Original image + probability bar chart
  - Row 2: Layer-wise heatmaps (4-6 layers depending on model)
  - Row 3: Combined (weighted fusion) + difference map
  - Row 4: Class-specific heatmaps (all 5 PAI classes)

**Average Heatmaps** (per PAI class):
- Combined (weighted fusion) average heatmap
- Last layer average heatmap
- Both for GradCAM and GradCAM++ methods
- With quadrant-aware orientation correction

**Metrics**:
- prediction_results.csv
- confusion_matrix.png
- final_metrics.txt (QWK, accuracy, MAE)

## Output Location

```
experiments/
├── efficientnet_b3/
│   └── multilayer_xai_20250127_143022/
│       ├── xai_individual_gradcam/          (100s of images)
│       ├── xai_individual_gradcamplusplus/  (100s of images)
│       ├── average_heatmaps/                (~20 images)
│       ├── prediction_results.csv
│       ├── confusion_matrix.png
│       └── final_metrics.txt
├── resnet50/
│   └── multilayer_xai_20250127_145530/
│       └── ... (same structure)
└── convnext_tiny/
    └── multilayer_xai_20250127_152045/
        └── ... (same structure)
```

## Key Differences from Basic Inference

| Feature | Basic (`inference_gradcam.py`) | Multi-Layer (`multilayer_xai_analysis.py`) |
|---------|-------------------------------|-------------------------------------------|
| Layers analyzed | 1 (last layer) | 4-6 (all major blocks) |
| Weighted fusion | ✗ | ✓ |
| Difference maps | ✗ | ✓ |
| Class-specific CAMs | ✗ | ✓ (all 5 classes) |
| Average heatmaps | ✗ | ✓ (with quadrant correction) |
| Processing time | ~30 min/model | ~2 hours/model |

## Common Options

### Fast Mode (GradCAM Only)
```bash
--cam-methods gradcam
```
Reduces time by ~50%

### Single Model
```bash
--model efficientnet_b3
```
Analyze only one model

### Adjust Transparency
```bash
--heatmap-transparency 0.3  # Less heatmap, more original image
--heatmap-transparency 0.7  # More heatmap, less original image
```
Default: 0.5

### Custom Output
```bash
--output-dir experiments/xai_publication
```

## Slurm Job Configuration

**No manual configuration needed!** The script automatically:
- Extracts test data paths from `config.py` (`InferenceConfig`)
- Uses best model checkpoints from `model_checkpoints/` directory
- Runs all three models sequentially

**Optional customization** in `code/slurm_scripts/slurm_multilayer_xai.sh`:

```bash
# OPTIONAL: Customize analysis (defaults are good)
MODELS="efficientnet_b3 resnet50 convnext_tiny"  # All three models
CAM_METHODS="gradcam gradcamplusplus"            # Both methods
BATCH_SIZE=""                                     # Use model defaults
NUM_WORKERS=8                                     # Data loading workers
```

Then submit:
```bash
sbatch code/slurm_scripts/slurm_multilayer_xai.sh
```

**Checkpoint paths used** (best models from project):
- ResNet50: `model_checkpoints/resnet50_exp1_baseline_20251024_213049/resnet50_best.pth`
- EfficientNet-B3: `model_checkpoints/effnet_exp5_lower_lr_20251025_235232/efficientnet-b3_best.pth`
- ConvNeXt-Tiny: `model_checkpoints/convnext_exp2_moderate_settings_20251026_061225/convnext-tiny_best.pth`

## Resource Requirements

| GPU | Models | Time Estimate | Memory |
|-----|--------|--------------|--------|
| A100 40GB | All 3 | ~6 hours | 64GB |
| A100 40GB | 1 model | ~2 hours | 32GB |
| RTX30 24GB | 1 model | ~3 hours | 32GB |

## Troubleshooting

### "No valid checkpoints found"
- Check checkpoint directory exists
- Verify naming: `{model_name}_best.pth` (e.g., `efficientnet_b3_best.pth`)
- Accepts hyphenated names: `efficientnet-b3_best.pth`

### Out of Memory
```bash
--batch-size 16  # Reduce from default
```

### Job Killed on HPC
- Increase time: `#SBATCH --time=08:00:00`
- Increase memory: `#SBATCH --mem=96G`

## Understanding the Visualizations

### Row 2: Layer Progression
- **Left columns**: Early layers (edges, textures)
- **Middle columns**: Mid layers (patterns, structures)
- **Right columns**: Late layers (high-level anatomy, class-specific features)

### Row 3: Fusion Analysis
- **Left (Combined)**: Weighted average of all layers (0.5→1.0 weights)
- **Right (Difference)**: Unique information in last layer vs. combined

### Row 4: Class Attribution
- Shows what model "sees" for each PAI class (1-5)
- Predicted and true classes are highlighted
- Check if high confidence aligns with focused heatmaps

### Average Heatmaps
- Per-class averages with anatomical orientation correction
- Reveals consistent attention patterns for each PAI class
- Compare across methods (GradCAM vs. GradCAM++)

## Next Steps

1. **Run the analysis** on your best models
2. **Examine individual visualizations** for interesting cases:
   - Correct high-confidence predictions (model is certain and right)
   - Incorrect high-confidence predictions (systematic errors)
   - Low-confidence predictions (uncertain/ambiguous cases)
3. **Compare average heatmaps** across PAI classes:
   - Do PAI 1 (healthy) and PAI 5 (severe) show distinct patterns?
   - Are intermediate classes (PAI 2-4) progressively different?
4. **Analyze difference maps**:
   - High difference: Model relies on specific high-level features
   - Low difference: Decision is consistent across abstraction levels

## Publication-Ready Figures

For publications, use these outputs:

1. **Figure 1**: Example individual visualizations showing correct predictions for each PAI class
2. **Figure 2**: Average heatmaps for all PAI classes (both combined and last layer)
3. **Figure 3**: Difference maps for challenging cases (misclassifications or PAI 2-4 confusion)
4. **Supplementary**: Grid of individual visualizations for systematic error analysis

## Full Documentation

See [MULTILAYER_XAI_README.md](MULTILAYER_XAI_README.md) for:
- Detailed parameter descriptions
- Architecture-specific layer information
- Advanced customization options
- Performance optimization guide
- Comparison with basic inference script

## Questions?

Check the main README or contact Gerald Torgersen (project author).
