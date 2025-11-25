#!/usr/bin/env python3
"""
================================================================================
Multi-Layer XAI Analysis for PAI Classification Models
================================================================================
Purpose:
  This script performs comprehensive multi-layer explainability analysis
  using GradCAM and GradCAM++ on trained PAI classification models.

  Supports three architectures:
    - ResNet50: 4 major blocks (layer1-layer4)
    - EfficientNet-B3: 6 blocks (blocks 1-6)
    - ConvNeXt-Tiny: 4 stages (stages 0-3)

Features:
  - Multi-layer heatmap generation with weighted fusion
  - Per-class average heatmaps with quadrant-aware orientation
  - Comprehensive visualization (individual layers + combined + class-wise)
  - Comparison between late-layer and combined heatmaps
  - Batch processing with Slurm support

Outputs:
  experiments/
    └── {model_name}/
        └── multilayer_xai_{timestamp}/
            ├── xai_individual_gradcam/      # Individual image visualizations
            ├── xai_individual_gradcamplusplus/
            ├── average_heatmaps/            # Per-class average heatmaps
            ├── prediction_results.csv
            ├── confusion_matrix.png
            └── final_metrics.txt

Author: Gerald Torgersen
SPDX-License-Identifier: MIT
Copyright (c) 2025 Gerald Torgersen
================================================================================
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import timm

from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score, mean_absolute_error

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))

try:
    from config import get_model_config, DataConfig
    from training.data_utils import CustomDataset
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError as e:
    print(f"ERROR: A required module could not be imported: {e}")
    print("Please ensure that:")
    print("  1. 'pytorch_grad_cam' is installed: pip install grad-cam")
    print("  2. 'config.py' exists in the code/ directory")
    sys.exit(1)

from tqdm import tqdm


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-layer XAI analysis for PAI classification models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze EfficientNet-B3 with checkpoint
  python multilayer_xai_analysis.py --model efficientnet_b3 \\
      --checkpoint experiments/best_efficientnet_b3.pth \\
      --test-csv data/test.csv --test-root data/images

  # Analyze on MULTIPLE test sets (will be concatenated automatically)
  python multilayer_xai_analysis.py --model resnet50 \\
      --checkpoint path/to/checkpoint.pth \\
      --test-csv data/test1.csv data/test2.csv \\
      --test-root data/images1 data/images2

  # Analyze all three models
  python multilayer_xai_analysis.py --model resnet50 efficientnet_b3 convnext_tiny \\
      --checkpoint-dir experiments/phase3_best/ \\
      --test-csv data/test.csv --test-root data/images

  # Skip GradCAM++ for faster processing
  python multilayer_xai_analysis.py --model efficientnet_b3 \\
      --checkpoint path/to/checkpoint.pth \\
      --cam-methods gradcam
        """
    )

    # Model selection
    parser.add_argument(
        '--model',
        nargs='+',
        choices=['resnet50', 'efficientnet_b3', 'convnext_tiny'],
        required=True,
        help='Model(s) to analyze'
    )

    # Checkpoint options
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument(
        '--checkpoint',
        type=str,
        help='Path to single model checkpoint (use with single --model)'
    )
    checkpoint_group.add_argument(
        '--checkpoint-dir',
        type=str,
        help='Directory containing checkpoints (auto-detects based on model names)'
    )

    # Test data (required) - supports multiple test sets
    parser.add_argument(
        '--test-csv',
        nargs='+',
        type=str,
        required=True,
        help='Path(s) to test CSV file(s). Multiple CSVs will be concatenated.'
    )
    parser.add_argument(
        '--test-root',
        nargs='+',
        type=str,
        required=True,
        help='Root directory(ies) for test images. Must match number of --test-csv arguments.'
    )

    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        default='experiments',
        help='Base output directory (default: experiments/)'
    )

    # CAM methods
    parser.add_argument(
        '--cam-methods',
        nargs='+',
        choices=['gradcam', 'gradcamplusplus'],
        default=['gradcam', 'gradcamplusplus'],
        help='CAM methods to use (default: gradcam gradcamplusplus)'
    )

    # Processing options
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Override batch size (default: use model config)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loading workers (default: 4)'
    )
    parser.add_argument(
        '--heatmap-transparency',
        type=float,
        default=0.5,
        help='Heatmap overlay transparency (default: 0.5)'
    )

    return parser.parse_args()


def resolve_checkpoints(args) -> Dict[str, str]:
    """Resolve checkpoint paths for each model."""
    checkpoints = {}

    if args.checkpoint:
        # Single checkpoint mode
        if len(args.model) > 1:
            print("ERROR: --checkpoint can only be used with a single --model")
            sys.exit(1)
        checkpoints[args.model[0]] = args.checkpoint

    elif args.checkpoint_dir:
        # Auto-detect mode
        checkpoint_dir = Path(args.checkpoint_dir)

        for model_name in args.model:
            # Try common naming patterns
            possible_names = [
                f"{model_name}_best.pth",
                f"{model_name.replace('_', '-')}_best.pth",
                f"{model_name}.pth",
                f"{model_name.replace('_', '-')}.pth",
            ]

            found = False
            for name in possible_names:
                candidate = checkpoint_dir / name
                if candidate.exists():
                    checkpoints[model_name] = str(candidate)
                    print(f"✓ Found checkpoint for {model_name}: {candidate}")
                    found = True
                    break

            if not found:
                print(f"✗ No checkpoint found for {model_name} in {checkpoint_dir}")
                print(f"  Tried: {possible_names}")

    return checkpoints


# ============================================================================
# MODEL-SPECIFIC LAYER IDENTIFICATION
# ============================================================================

def identify_target_layers(model: nn.Module, model_name: str) -> Dict[str, nn.Module]:
    """
    Identify target layers for multi-layer XAI analysis.

    Returns dictionary mapping layer names to modules:
      ResNet50: {'layer1': module, 'layer2': module, ...}
      EfficientNet-B3: {'block_1': module, 'block_2': module, ...}
      ConvNeXt-Tiny: {'stage_0': module, 'stage_1': module, ...}
    """
    model_name_lower = model_name.lower()
    layers = {}

    if 'resnet' in model_name_lower:
        # ResNet: 4 major blocks (layer1-layer4)
        # Each layer contains multiple residual blocks
        for i in range(1, 5):
            layer_name = f'layer{i}'
            if hasattr(model, layer_name):
                layers[layer_name] = getattr(model, layer_name)

        if not layers:
            raise ValueError(f"Could not find ResNet layers (layer1-layer4) in model")

    elif 'efficientnet' in model_name_lower:
        # EfficientNet: blocks attribute (timm implementation)
        # Use blocks 1-6 (skip block 0 - stem layer)
        if hasattr(model, 'blocks'):
            blocks_module = model.blocks
            # EfficientNet-B3 has 7 blocks (0-6), use 1-6
            for i in range(1, min(7, len(blocks_module))):
                layers[f'block_{i}'] = blocks_module[i]
        else:
            raise ValueError(f"Could not find 'blocks' attribute in EfficientNet model")

    elif 'convnext' in model_name_lower:
        # ConvNeXt: stages attribute (timm implementation)
        # 4 stages (0-3)
        if hasattr(model, 'stages'):
            stages_module = model.stages
            for i in range(len(stages_module)):
                layers[f'stage_{i}'] = stages_module[i]
        else:
            raise ValueError(f"Could not find 'stages' attribute in ConvNeXt model")

    else:
        raise ValueError(f"Unknown model architecture: {model_name}")

    print(f"Identified {len(layers)} target layers: {list(layers.keys())}")
    return layers


# ============================================================================
# MULTI-LAYER HEATMAP GENERATION
# ============================================================================

def get_multi_layer_heatmaps(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_layers: Dict[str, nn.Module],
    cam_method: str,
    target_class: Optional[int] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], int]:
    """
    Generate heatmaps for multiple layers and create weighted fusion.

    Args:
        model: The model to analyze
        input_tensor: Input image tensor (1, C, H, W)
        target_layers: Dict of layer name -> layer module
        cam_method: 'gradcam' or 'gradcamplusplus'
        target_class: Target class index (if None, uses predicted class)

    Returns:
        individual_heatmaps: Dict of layer_name -> heatmap (H, W)
        combined_heatmaps: Dict with 'combined' and 'difference' keys
        predicted_class: The class index used for visualization
    """
    # Get prediction if target_class not specified
    if target_class is None:
        with torch.no_grad():
            outputs = model(input_tensor)
            predicted_class = outputs.argmax(dim=1).item()
    else:
        predicted_class = target_class

    targets = [ClassifierOutputTarget(predicted_class)]

    # Select CAM method
    if cam_method == 'gradcam':
        cam_class = GradCAM
    elif cam_method == 'gradcamplusplus':
        cam_class = GradCAMPlusPlus
    else:
        raise ValueError(f"Unknown CAM method: {cam_method}")

    # Generate individual layer heatmaps
    individual_heatmaps = {}
    layer_names = list(target_layers.keys())

    for layer_name, layer_module in target_layers.items():
        cam_generator = cam_class(model=model, target_layers=[layer_module])
        grayscale_cam = cam_generator(input_tensor=input_tensor, targets=targets)[0, :]
        individual_heatmaps[layer_name] = grayscale_cam

    # Weighted fusion: linear weights from 0.5 to 1.0
    num_layers = len(layer_names)
    combine_weights = np.linspace(0.5, 1.0, num_layers)

    # Combine heatmaps with weights
    combined = np.zeros_like(list(individual_heatmaps.values())[0])
    weight_sum = 0.0

    for i, layer_name in enumerate(layer_names):
        combined += combine_weights[i] * individual_heatmaps[layer_name]
        weight_sum += combine_weights[i]

    combined /= weight_sum

    # Difference map: absolute difference between last layer and combined
    last_layer_name = layer_names[-1]
    difference = np.abs(individual_heatmaps[last_layer_name] - combined)

    combined_heatmaps = {
        'combined': combined,
        'difference': difference,
        'last_layer': individual_heatmaps[last_layer_name]
    }

    return individual_heatmaps, combined_heatmaps, predicted_class


def get_class_specific_heatmaps(
    model: nn.Module,
    input_tensor: torch.Tensor,
    last_layer: nn.Module,
    cam_method: str,
    num_classes: int = 5
) -> Dict[int, np.ndarray]:
    """
    Generate heatmaps for all PAI classes using the last layer.

    Returns:
        Dict mapping class index (0-4) to heatmap
    """
    if cam_method == 'gradcam':
        cam_class = GradCAM
    else:
        cam_class = GradCAMPlusPlus

    cam_generator = cam_class(model=model, target_layers=[last_layer])

    class_heatmaps = {}
    for class_idx in range(num_classes):
        targets = [ClassifierOutputTarget(class_idx)]
        grayscale_cam = cam_generator(input_tensor=input_tensor, targets=targets)[0, :]
        class_heatmaps[class_idx] = grayscale_cam

    return class_heatmaps


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_heatmap_overlay(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    transparency: float = 0.5
) -> np.ndarray:
    """
    Create heatmap overlay on original image.

    Args:
        original_image: RGB image normalized to [0, 1]
        heatmap: Grayscale heatmap [0, 1]
        transparency: Overlay transparency (0=only image, 1=only heatmap)

    Returns:
        RGB overlay image [0, 1]
    """
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0

    # Blend
    overlay = cv2.addWeighted(original_image, 1 - transparency, heatmap_colored, transparency, 0)
    return overlay


def generate_individual_visualization(
    image: Image.Image,
    individual_heatmaps: Dict[str, np.ndarray],
    combined_heatmaps: Dict[str, np.ndarray],
    class_heatmaps: Dict[int, np.ndarray],
    probabilities: np.ndarray,
    true_class: int,
    predicted_class: int,
    filename: str,
    model_name: str,
    cam_method: str,
    output_path: Path,
    transparency: float = 0.5,
    input_size: int = 224
):
    """
    Generate comprehensive multi-layer visualization with 4 rows:
      Row 1: Original image + probability bar chart
      Row 2: Individual layer heatmaps (up to 6 layers)
      Row 3: Combined heatmap + difference map
      Row 4: Class-specific heatmaps (5 PAI classes)
    """
    # Prepare original image
    img_resized = image.resize((input_size, input_size))
    img_array = np.array(img_resized) / 255.0

    # Create figure with GridSpec
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(4, 6, figure=fig, hspace=0.3, wspace=0.2)

    # === ROW 1: Original + Probabilities ===
    # Original image
    ax_img = fig.add_subplot(gs[0, :2])
    ax_img.imshow(img_array)
    ax_img.axis('off')
    correct_marker = "✓" if predicted_class == true_class else "✗"
    ax_img.set_title(
        f"True: PAI {true_class+1} | Pred: PAI {predicted_class+1} {correct_marker}",
        fontsize=12,
        fontweight='bold',
        color='green' if predicted_class == true_class else 'red'
    )

    # Probability bar chart
    ax_prob = fig.add_subplot(gs[0, 2:])
    pai_classes = [f'PAI {i+1}' for i in range(5)]
    colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
    bars = ax_prob.barh(pai_classes, probabilities, color=colors)
    # Highlight predicted class
    bars[predicted_class].set_edgecolor('black')
    bars[predicted_class].set_linewidth(2.5)
    ax_prob.set_xlim([0, 1])
    ax_prob.set_xlabel('Probability', fontsize=10)
    ax_prob.set_title('Class Probabilities', fontsize=11, fontweight='bold')
    ax_prob.grid(axis='x', alpha=0.3)

    # === ROW 2: Individual Layer Heatmaps ===
    layer_names = list(individual_heatmaps.keys())
    for i, layer_name in enumerate(layer_names[:6]):  # Max 6 layers
        ax = fig.add_subplot(gs[1, i])
        overlay = create_heatmap_overlay(img_array, individual_heatmaps[layer_name], transparency)
        ax.imshow(overlay)
        ax.axis('off')
        ax.set_title(layer_name, fontsize=9)

    # === ROW 3: Combined + Difference ===
    # Combined (weighted fusion)
    ax_combined = fig.add_subplot(gs[2, :3])
    overlay_combined = create_heatmap_overlay(img_array, combined_heatmaps['combined'], transparency)
    ax_combined.imshow(overlay_combined)
    ax_combined.axis('off')
    ax_combined.set_title(
        f'Combined (Weighted Fusion)\nWeights: 0.5→1.0',
        fontsize=10,
        fontweight='bold'
    )

    # Difference map
    ax_diff = fig.add_subplot(gs[2, 3:])
    overlay_diff = create_heatmap_overlay(img_array, combined_heatmaps['difference'], transparency)
    ax_diff.imshow(overlay_diff)
    ax_diff.axis('off')
    ax_diff.set_title(
        f'Difference Map\n|{layer_names[-1]} - Combined|',
        fontsize=10,
        fontweight='bold'
    )

    # === ROW 4: Class-Specific Heatmaps (Last Layer) ===
    for class_idx in range(5):
        ax = fig.add_subplot(gs[3, class_idx])
        overlay = create_heatmap_overlay(img_array, class_heatmaps[class_idx], transparency)
        ax.imshow(overlay)
        ax.axis('off')
        is_true = (class_idx == true_class)
        is_pred = (class_idx == predicted_class)
        marker = ""
        if is_true and is_pred:
            marker = " ✓✓"
        elif is_true:
            marker = " (True)"
        elif is_pred:
            marker = " (Pred)"
        ax.set_title(
            f'PAI {class_idx+1}{marker}\n{probabilities[class_idx]:.1%}',
            fontsize=9,
            fontweight='bold' if is_true or is_pred else 'normal'
        )

    # Overall title
    fig.suptitle(
        f'{filename} | {model_name} | {cam_method.upper()}',
        fontsize=13,
        fontweight='bold'
    )

    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# AVERAGE HEATMAPS WITH QUADRANT-AWARE FLIPPING
# ============================================================================

def apply_quadrant_flipping(heatmap: np.ndarray, quadrant: int) -> np.ndarray:
    """
    Apply quadrant-aware flipping to orient all heatmaps consistently.

    Quadrant mapping (European tooth numbering):
      1: Upper right (teeth 11-18)
      2: Upper left (teeth 21-28)
      3: Lower left (teeth 31-38)
      4: Lower right (teeth 41-48)

    Flipping logic:
      - Quadrants 2, 3 (left side): Horizontal flip
      - Quadrants 1, 2 (upper jaw): Vertical flip
    """
    flipped = heatmap.copy()

    # Horizontal flip for left side
    if quadrant in [2, 3]:
        flipped = np.fliplr(flipped)

    # Vertical flip for upper jaw
    if quadrant in [1, 2]:
        flipped = np.flipud(flipped)

    return flipped


def calculate_average_heatmaps(
    all_heatmaps: Dict[int, List[Tuple[np.ndarray, int]]],
    output_dir: Path,
    model_name: str,
    cam_method: str,
    heatmap_type: str = 'combined'
):
    """
    Calculate and save per-class average heatmaps with quadrant-aware flipping.

    Args:
        all_heatmaps: Dict mapping true_class -> list of (heatmap, quadrant) tuples
        output_dir: Output directory for average heatmaps
        model_name: Model name for filename
        cam_method: CAM method name
        heatmap_type: 'combined', 'last_layer', or specific layer name
    """
    os.makedirs(output_dir, exist_ok=True)

    for true_class, heatmap_list in all_heatmaps.items():
        if not heatmap_list:
            continue

        # Apply quadrant flipping
        flipped_heatmaps = []
        for heatmap, quadrant in heatmap_list:
            if quadrant is not None:
                flipped = apply_quadrant_flipping(heatmap, quadrant)
            else:
                flipped = heatmap
            flipped_heatmaps.append(flipped)

        # Average
        avg_heatmap = np.mean(flipped_heatmaps, axis=0)

        # Normalize to [0, 1]
        avg_heatmap = (avg_heatmap - avg_heatmap.min()) / (avg_heatmap.max() - avg_heatmap.min() + 1e-8)

        # Save as image
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * avg_heatmap), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        filename = f"average_{heatmap_type}_{cam_method}_pai_{true_class+1}.png"
        save_path = output_dir / filename

        plt.figure(figsize=(8, 8))
        plt.imshow(heatmap_colored)
        plt.axis('off')
        plt.title(
            f'Average {heatmap_type.replace("_", " ").title()}\n'
            f'PAI {true_class+1} | {cam_method.upper()} | n={len(heatmap_list)}',
            fontsize=14,
            fontweight='bold'
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {filename} (n={len(heatmap_list)})")


# ============================================================================
# MAIN INFERENCE LOOP
# ============================================================================

def load_model(model_name: str, checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load model from checkpoint."""
    print(f"Loading model: {model_name}")
    model_config = get_model_config(model_name)

    model = timm.create_model(
        model_config.timm_name,
        pretrained=False,
        num_classes=model_config.num_classes,
        drop_rate=model_config.dropout,
        drop_path_rate=model_config.drop_path
    )

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Model loaded successfully\n")

    return model


def get_transforms(model_name: str, mean: List[float], std: List[float]) -> transforms.Compose:
    """Create inference transforms."""
    model_config = get_model_config(model_name)

    return transforms.Compose([
        transforms.Resize((model_config.input_size, model_config.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def run_multilayer_xai(
    model: nn.Module,
    model_name: str,
    test_data: pd.DataFrame,
    target_layers: Dict[str, nn.Module],
    cam_methods: List[str],
    output_dir: Path,
    device: torch.device,
    mean: List[float],
    std: List[float],
    transparency: float = 0.5
):
    """
    Run multi-layer XAI analysis on test set.
    """
    model_config = get_model_config(model_name)
    transform = get_transforms(model_name, mean, std)

    # Prepare for inverse normalization (for visualization)
    inv_mean = torch.tensor(mean).view(3, 1, 1)
    inv_std = torch.tensor(std).view(3, 1, 1)

    # Storage for results
    results = []

    # Storage for average heatmaps (per CAM method, per true class)
    avg_heatmaps = {
        cam_method: {
            'combined': defaultdict(list),
            'last_layer': defaultdict(list)
        }
        for cam_method in cam_methods
    }

    # Get last layer for class-specific heatmaps
    last_layer_name = list(target_layers.keys())[-1]
    last_layer_module = target_layers[last_layer_name]

    print(f"\nProcessing {len(test_data)} test images...")

    for idx, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Multi-layer XAI"):
        try:
            # Load image
            image_path = Path(row['root_dir']) / row['filename']
            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                continue

            original_image = Image.open(image_path).convert('RGB')
            input_tensor = transform(original_image).unsqueeze(0).to(device)

            # Get prediction and probabilities
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
                predicted_class = outputs.argmax(dim=1).item()

            true_class = int(row['PAI_0_indexed'])
            quadrant = row.get('quadrant', None)  # Optional

            # Process each CAM method
            for cam_method in cam_methods:
                # Generate multi-layer heatmaps
                individual_heatmaps, combined_heatmaps, _ = get_multi_layer_heatmaps(
                    model, input_tensor, target_layers, cam_method, target_class=predicted_class
                )

                # Generate class-specific heatmaps
                class_heatmaps = get_class_specific_heatmaps(
                    model, input_tensor, last_layer_module, cam_method, num_classes=5
                )

                # Save individual visualization
                viz_dir = output_dir / f"xai_individual_{cam_method}"
                os.makedirs(viz_dir, exist_ok=True)

                viz_filename = f"example_{idx}_true_{true_class+1}_pred_{predicted_class+1}_{cam_method}_{row['filename'].replace('.tif', '.png')}"
                viz_path = viz_dir / viz_filename

                generate_individual_visualization(
                    original_image,
                    individual_heatmaps,
                    combined_heatmaps,
                    class_heatmaps,
                    probs,
                    true_class,
                    predicted_class,
                    row['filename'],
                    model_name,
                    cam_method,
                    viz_path,
                    transparency,
                    model_config.input_size
                )

                # Store for averaging
                avg_heatmaps[cam_method]['combined'][true_class].append(
                    (combined_heatmaps['combined'], quadrant)
                )
                avg_heatmaps[cam_method]['last_layer'][true_class].append(
                    (combined_heatmaps['last_layer'], quadrant)
                )

            # Record result (once per image, not per CAM method)
            results.append({
                'filename': row['filename'],
                'test_set': row.get('source', 'unknown'),  # Which test set (test_set_1, test_set_2)
                'source_path': row.get('source_path', ''),  # Full path to CSV
                'root_dir': row['root_dir'],  # Image directory
                'true_pai': true_class + 1,
                'predicted_pai': predicted_class + 1,
                **{f'prob_pai_{i+1}': probs[i] for i in range(5)}
            })

        except Exception as e:
            print(f"Error processing {row['filename']}: {e}")
            continue

    # Save prediction results
    results_df = pd.DataFrame(results)
    results_csv = output_dir / "prediction_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\nPrediction results saved: {results_csv}")

    # Calculate metrics
    calculate_final_metrics(results_df, output_dir)

    # Calculate and save average heatmaps
    print("\nGenerating average heatmaps...")
    avg_dir = output_dir / "average_heatmaps"

    for cam_method in cam_methods:
        print(f"\n{cam_method.upper()}:")
        calculate_average_heatmaps(
            avg_heatmaps[cam_method]['combined'],
            avg_dir,
            model_name,
            cam_method,
            'combined'
        )
        calculate_average_heatmaps(
            avg_heatmaps[cam_method]['last_layer'],
            avg_dir,
            model_name,
            cam_method,
            last_layer_name
        )

    print(f"\nAverage heatmaps saved to: {avg_dir}")


def calculate_final_metrics(df: pd.DataFrame, output_dir: Path):
    """Calculate and save final metrics."""
    true_labels = df['true_pai'].values
    pred_labels = df['predicted_pai'].values

    # Metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    qwk = cohen_kappa_score(true_labels, pred_labels, weights='quadratic')
    mae = mean_absolute_error(true_labels, pred_labels)
    cm = confusion_matrix(true_labels, pred_labels, labels=[1, 2, 3, 4, 5])

    # Save metrics
    metrics_txt = output_dir / "final_metrics.txt"
    with open(metrics_txt, 'w') as f:
        f.write("="*60 + "\n")
        f.write("Multi-Layer XAI Analysis - Final Metrics\n")
        f.write("="*60 + "\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Quadratic Weighted Kappa: {qwk:.4f}\n")
        f.write(f"Mean Absolute Error: {mae:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write("-" * 40 + "\n")
        f.write("Rows: True Labels, Columns: Predicted Labels\n\n")
        header = "      " + " ".join([f"P{i}" for i in range(1, 6)])
        f.write(header + "\n")
        for i, row in enumerate(cm):
            row_str = f"True {i+1} |" + " ".join([f"{val:>3}" for val in row])
            f.write(row_str + "\n")

    print(f"Metrics saved: {metrics_txt}")

    # Save confusion matrix plot
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels([f'PAI {i}' for i in range(1, 6)])
    ax.set_yticklabels([f'PAI {i}' for i in range(1, 6)])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix\nQWK: {qwk:.4f} | Acc: {accuracy:.4f}')

    # Add text annotations
    for i in range(5):
        for j in range(5):
            text = ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Confusion matrix saved: {cm_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution."""
    args = parse_arguments()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Models to analyze: {args.model}")
    print(f"CAM methods: {args.cam_methods}")

    # Load data config for normalization
    data_config = DataConfig()
    mean = data_config.mean
    std = data_config.std

    # Load and concatenate test data from multiple sources
    print(f"\nLoading test data from {len(args.test_csv)} source(s)...")

    # Validate matching number of CSVs and root dirs
    if len(args.test_csv) != len(args.test_root):
        print(f"ERROR: Number of --test-csv ({len(args.test_csv)}) must match --test-root ({len(args.test_root)})")
        sys.exit(1)

    test_data_list = []
    for i, (csv_path, root_dir) in enumerate(zip(args.test_csv, args.test_root), 1):
        print(f"  Loading: {csv_path}")
        df = pd.read_csv(csv_path)
        df['root_dir'] = root_dir
        # Create readable source label: "test_set_1", "test_set_2", etc.
        df['source'] = f"test_set_{i}"
        df['source_path'] = csv_path  # Also keep full path for reference
        test_data_list.append(df)
        print(f"    - Loaded {len(df)} samples from {root_dir}")

    # Concatenate all test sets
    test_data = pd.concat(test_data_list, ignore_index=True)
    test_data['PAI_0_indexed'] = test_data['PAI'] - 1
    print(f"\nTotal test samples: {len(test_data)} (from {len(args.test_csv)} test set(s))\n")

    # Resolve checkpoints
    checkpoints = resolve_checkpoints(args)

    if not checkpoints:
        print("\nERROR: No valid checkpoints found!")
        sys.exit(1)

    print(f"\nFound {len(checkpoints)} checkpoint(s)")

    # Process each model
    for model_name, checkpoint_path in checkpoints.items():
        print("\n" + "="*80)
        print(f"Processing: {model_name}")
        print("="*80)

        try:
            # Load model
            model = load_model(model_name, checkpoint_path, device)

            # Identify target layers
            target_layers = identify_target_layers(model, model_name)

            # Setup output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(args.output_dir) / model_name / f"multilayer_xai_{timestamp}"
            os.makedirs(output_dir, exist_ok=True)
            print(f"Output directory: {output_dir}")

            # Run multi-layer XAI
            run_multilayer_xai(
                model,
                model_name,
                test_data,
                target_layers,
                args.cam_methods,
                output_dir,
                device,
                mean,
                std,
                args.heatmap_transparency
            )

            print(f"\n✓ Successfully completed multi-layer XAI for {model_name}")

        except Exception as e:
            print(f"\nERROR: Failed to process {model_name}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "="*80)
    print("Multi-layer XAI analysis completed!")
    print("="*80)


if __name__ == '__main__':
    main()
