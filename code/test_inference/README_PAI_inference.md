# PAI_inference.py - Multi-Architecture Support

## Overview

The `PAI_inference.py` script is an interactive GUI application for analyzing dental radiographs using deep learning models trained for PAI (Periapical Index) classification. It now supports all three model architectures with automatic optimization.

## New Features (v2.0)

### ✅ Multi-Architecture Support
- **ResNet50**: Automatically uses 224×224 input size
- **EfficientNet-B3**: Automatically uses 300×300 input size
- **ConvNeXt-Tiny**: Automatically uses 224×224 input size

### ✅ Automatic Configuration
- Detects model architecture from checkpoint
- Sets optimal input size per architecture
- Loads model-specific normalization parameters
- Auto-configures GradCAM target layers

### ✅ Flexible Checkpoint Loading
- Command-line argument to specify checkpoint path
- Supports checkpoints in separate directories
- Falls back to `./model/` subdirectory if no path specified

## Usage

### Option 1: Specify Checkpoint via Command Line (Recommended)

```bash
# For ResNet50 (Top Performer, QWK=0.8068)
python code/test_inference/PAI_inference.py \
  --checkpoint model_checkpoints/p5_resnet_alpha_0_5_20251028_113346/resnet50_best.pth

# For EfficientNet-B3 (Runner-up, QWK=0.7811)
python code/test_inference/PAI_inference.py \
  --checkpoint model_checkpoints/effnet_exp5_lower_lr_20251025_235232/efficientnet-b3_best.pth

# For ConvNeXt-Tiny (Third Place, QWK=0.7447)
python code/test_inference/PAI_inference.py \
  --checkpoint model_checkpoints/convnext_exp2_moderate_settings_20251026_061225/convnext-tiny_best.pth
```

### Option 2: Default Model Directory

```bash
# Create model directory and copy checkpoint
mkdir -p code/test_inference/model
cp model_checkpoints/your_model/checkpoint.pth code/test_inference/model/

# Run without arguments (uses first .pth in ./model/)
python code/test_inference/PAI_inference.py
```

## Features

### Interactive Analysis
1. **Load Image**: Supports TIFF, PNG, JPEG formats
2. **Click to Analyze**: Click on tooth apex to analyze a 12×12 mm region
3. **Real-time Results**:
   - PAI classification (1-5)
   - Confidence scores
   - Probability distribution

### GradCAM Visualization
- Toggle-able GradCAM overlays
- Shows attention maps for all 5 PAI classes
- Highlights predicted class in red

### Adjustable Parameters
- **Pixel Size**: Set correct mm/pixel for your radiographs (typically 0.03-0.05mm)
- Ensures consistent physical region size (12×12 mm) across different resolutions

## Technical Details

### Architecture-Specific Settings

| Model | Input Size | Optimal For | Test QWK |
|-------|------------|-------------|----------|
| ResNet50 | 224×224 | Speed & accuracy balance | 0.8068 |
| EfficientNet-B3 | 300×300 | Highest detail capture | 0.7811 |
| ConvNeXt-Tiny | 224×224 | Modern architecture | 0.7447 |

### Checkpoint Requirements

The script expects checkpoints saved with the following structure:
```python
{
    'model_config': {
        'model_name': 'resnet50' | 'efficientnet_b3' | 'convnext_tiny',
        'num_classes': 5,
        'dropout_rate': float,
        'drop_path_rate': float
    },
    'model_state_dict': {...},
    'normalization': {
        'mean': [float, float, float],
        'std': [float, float, float]
    },
    'epoch': int,
    'best_metric_val': float,
    'timestamp': str
}
```

All checkpoints from your training pipeline (`train_simple.py`) are compatible.

## System Requirements

### Required Libraries
```bash
pip install torch torchvision timm
pip install opencv-python numpy matplotlib pillow
pip install tkinter  # Usually included with Python
```

### Hardware
- **CPU**: Works on CPU (slower)
- **GPU**: Automatically uses CUDA if available (recommended)

## Troubleshooting

### Issue: "No .pth checkpoint files found"
**Solution**: Use `--checkpoint` argument to specify full path to checkpoint file

### Issue: Model loads but predictions seem wrong
**Solution**: Verify pixel size setting matches your radiograph's actual pixel size

### Issue: GradCAM not showing
**Solution**: Ensure `Show Grad-CAM` checkbox is enabled. Some architectures may not support GradCAM if target layer detection fails.

### Issue: Out of memory on GPU
**Solution**: The script uses minimal memory (~2GB VRAM). If issues persist, close other GPU applications or use CPU mode.

## Output Interpretation

### PAI Scale
- **PAI 1**: Normal periapical structures
- **PAI 2**: Small changes in bone structure
- **PAI 3**: Changes with some mineral loss
- **PAI 4**: Periodontitis with well-defined radiolucency
- **PAI 5**: Severe periodontitis with exacerbating features

### Confidence Scores
- **>80%**: High confidence prediction
- **50-80%**: Moderate confidence (review recommended)
- **<50%**: Low confidence (clinical judgment essential)

### GradCAM Heatmaps
- **Red/Yellow**: High attention areas (model focuses here)
- **Blue/Purple**: Low attention areas
- Compare across all 5 PAI classes to understand model reasoning

## Use Cases

### 1. Clinical Demonstration
Show how AI analyzes specific regions of radiographs in real-time

### 2. Model Comparison
Load different checkpoints to compare predictions across architectures

### 3. Educational Tool
Visualize what features the model considers important for each PAI class

### 4. Quality Control
Verify model performance on new/edge case images

## Limitations

- **Single image at a time**: For batch processing, use `multilayer_xai_analysis.py`
- **GUI required**: Requires display (X11/Windows). For headless servers, use batch scripts
- **12×12 mm fixed**: Region size is fixed to training standard

## Related Scripts

- **`multilayer_xai_analysis.py`**: Batch analysis with per-class average heatmaps
- **`inference_gradcam.py`**: Batch inference without GUI
- **`slurm_multilayer_xai.sh`**: HPC batch processing for all models

## Support

For issues or questions about this script:
1. Check that checkpoint format matches requirements
2. Verify all dependencies are installed
3. Test with `--checkpoint` argument to ensure path is correct
4. Check console output for detailed error messages

## Version History

- **v2.0**: Multi-architecture support, command-line arguments, auto input-size detection
- **v1.0**: Initial version (EfficientNet-B3 only, 300×300 fixed)
