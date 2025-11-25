#!/usr/bin/env python3
"""
================================================================================
Model Inference, Reporting, and GradCAM Explainer (ENHANCED VERSION)
================================================================================
Purpose:
  This script evaluates trained PAI classification models on a test set.
  It is driven by the InferenceConfig class in `config.py`.

NEW FEATURES:
  - Command-line options to override checkpoint paths
  - Auto-detect checkpoints in a directory
  - Skip heatmap generation for faster inference
  - Override test data and output paths

Outputs for each model:
  1.  A detailed CSV file with per-image predictions and class probabilities.
  2.  A comprehensive statistics report (.txt) including accuracy, QWK,
      per-class sensitivity/specificity, and a confusion matrix.
  3.  A folder of GradCAM++ heatmap visualizations for every image in the
      test set, overlaid on the original image.

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
from typing import Dict, List, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import timm

from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score, classification_report

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))

try:
    from config import get_inference_config, get_model_config, DataConfig
    from training.data_utils import CustomDataset
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: A required module could not be imported: {e}")
    print("Please ensure that:")
    print("  1. 'pytorch_grad_cam' is installed: pip install grad-cam")
    print("  2. 'code/' and 'code/training/' directories contain '__init__.py' files")
    sys.exit(1)

from tqdm import tqdm

# ============================================================================
# COMMAND LINE ARGUMENT PARSING
# ============================================================================

def parse_arguments():
    """Parse command-line arguments for checkpoint override and configuration."""
    parser = argparse.ArgumentParser(
        description="Run inference on PAI classification models with optional checkpoint override",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use checkpoints from model_checkpoints/ directory
  python code/test_inference/inference_gradcam.py --checkpoint-dir model_checkpoints
  
  # Test specific models only
  python code/test_inference/inference_gradcam.py --checkpoint-dir model_checkpoints --models efficientnet_b3 resnet50
  
  # Specify individual checkpoint files
  python code/test_inference/inference_gradcam.py \\
      --efficientnet-b3 model_checkpoints/efficientnet-b3_best.pth \\
      --resnet50 model_checkpoints/resnet50_best.pth
  
  # Skip heatmap generation (faster)
  python code/test_inference/inference_gradcam.py --checkpoint-dir model_checkpoints --no-heatmaps
  
  # Use config.py settings (default behavior)
  python code/test_inference/inference_gradcam.py
        """
    )
    
    # Checkpoint override options
    checkpoint_group = parser.add_argument_group('Checkpoint Options')
    checkpoint_group.add_argument(
        '--checkpoint-dir',
        type=str,
        help='Directory containing model checkpoints (overrides config.py paths)'
    )
    checkpoint_group.add_argument(
        '--efficientnet-b3',
        type=str,
        metavar='PATH',
        dest='efficientnet_b3_checkpoint',
        help='Path to EfficientNet-B3 checkpoint'
    )
    checkpoint_group.add_argument(
        '--resnet50',
        type=str,
        metavar='PATH',
        dest='resnet50_checkpoint',
        help='Path to ResNet50 checkpoint'
    )
    checkpoint_group.add_argument(
        '--convnext-tiny',
        type=str,
        metavar='PATH',
        dest='convnext_tiny_checkpoint',
        help='Path to ConvNeXt-Tiny checkpoint'
    )
    
    # Model selection
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['efficientnet_b3', 'resnet50', 'convnext_tiny'],
        help='Specific models to run inference on (default: all with valid checkpoints)'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Base output directory (default: from config.py)'
    )
    parser.add_argument(
        '--no-heatmaps',
        action='store_true',
        help='Skip generating GradCAM heatmaps (faster inference)'
    )
    
    # Test data options
    parser.add_argument(
        '--test-csv',
        type=str,
        help='Path to test CSV file (overrides config.py)'
    )
    parser.add_argument(
        '--test-root',
        type=str,
        help='Root directory for test images (overrides config.py)'
    )
    
    return parser.parse_args()


def resolve_checkpoints(args, inference_config) -> Dict[str, str]:
    """
    Resolve checkpoint paths based on command-line arguments and config.
    Priority: individual args > checkpoint_dir > config.py
    """
    checkpoints = {}
    
    # Get models to process
    if args.models:
        models_to_process = args.models
    else:
        models_to_process = inference_config.active_models
    
    # Try each model
    for model_name in models_to_process:
        checkpoint_path = None
        
        # Priority 1: Individual checkpoint argument
        checkpoint_arg = f"{model_name}_checkpoint"
        if hasattr(args, checkpoint_arg) and getattr(args, checkpoint_arg):
            checkpoint_path = getattr(args, checkpoint_arg)
            print(f"✓ Using CLI checkpoint for {model_name}: {checkpoint_path}")
        
        # Priority 2: Auto-detect in checkpoint_dir
        elif args.checkpoint_dir:
            checkpoint_dir = Path(args.checkpoint_dir)
            
            # Try common naming patterns
            arg_name = model_name.replace('_', '-')
            possible_names = [
                f"{arg_name}_best.pth",
                f"{model_name}_best.pth",
                f"{arg_name}.pth",
                f"{model_name}.pth",
            ]
            
            for name in possible_names:
                candidate = checkpoint_dir / name
                if candidate.exists():
                    checkpoint_path = str(candidate)
                    print(f"✓ Auto-detected checkpoint for {model_name}: {checkpoint_path}")
                    break
            
            if not checkpoint_path:
                print(f"⚠ No checkpoint found for {model_name} in {checkpoint_dir}")
                print(f"  Tried: {possible_names}")
                continue
        
        # Priority 3: Use config.py
        else:
            checkpoint_path = inference_config.get_checkpoint_path(model_name)
            if Path(checkpoint_path).exists():
                print(f"✓ Using config.py checkpoint for {model_name}")
            else:
                print(f"⚠ Config checkpoint doesn't exist: {checkpoint_path}")
                continue
        
        # Verify checkpoint exists
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoints[model_name] = checkpoint_path
        elif checkpoint_path:
            print(f"✗ Checkpoint path doesn't exist: {checkpoint_path}")
    
    return checkpoints


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """Get the target layer for GradCAM based on model architecture."""
    model_name_lower = model_name.lower()
    
    if 'resnet' in model_name_lower:
        # ResNet: last layer of layer4
        return model.layer4[-1]
    elif 'efficientnet' in model_name_lower:
        # EfficientNet: last block
        return model.blocks[-1]
    elif 'convnext' in model_name_lower:
        # ConvNeXt: last stage
        return model.stages[-1]
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")


def load_model(model_name: str, checkpoint_path: str, device: torch.device) -> nn.Module:
    """Loads a trained model from a checkpoint file."""
    print(f"Loading model configuration for: {model_name}")
    model_config = get_model_config(model_name)
    
    print(f"Creating model: {model_config.timm_name}")
    model = timm.create_model(
        model_config.timm_name,
        pretrained=False,
        num_classes=model_config.num_classes,
        drop_rate=model_config.dropout,
        drop_path_rate=model_config.drop_path
    )
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Model loaded successfully")
    
    return model


def get_transforms(model_name: str, mean: List[float], std: List[float]) -> transforms.Compose:
    """Creates standard inference transforms."""
    model_config = get_model_config(model_name)
    
    return transforms.Compose([
        transforms.Resize((model_config.input_size, model_config.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


# ============================================================================
# CORE INFERENCE AND VISUALIZATION
# ============================================================================

@torch.no_grad()
def run_batch_inference(model: nn.Module, dataloader: DataLoader, device: torch.device) -> pd.DataFrame:
    """Runs inference on the entire dataset and returns results in a DataFrame."""
    all_preds, all_labels, all_probs, all_filenames = [], [], [], []

    for inputs, labels, filenames in tqdm(dataloader, desc="Running Inference"):
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_filenames.extend(filenames)
    
    # Create probability columns
    prob_cols = {f'prob_pai_{i+1}': [p[i] for p in all_probs] for i in range(5)}
    
    results_df = pd.DataFrame({
        'filename': all_filenames,
        'true_pai': np.array(all_labels) + 1,  # Convert back to 1-indexed
        'predicted_pai': np.array(all_preds) + 1,  # Convert back to 1-indexed
        **prob_cols
    })
    
    return results_df


def generate_and_save_heatmaps(df: pd.DataFrame, model: nn.Module, model_name: str,
                               output_dir: Path, cam_method: str, transparency: float,
                               device: torch.device, mean: List[float], std: List[float]):
    """
    Generates and saves dual GradCAM heatmaps (predicted + true class) for every image.
    Creates a side-by-side comparison with captions.
    """
    print(f"Generating dual GradCAM heatmaps (predicted + true class)...")
    
    try:
        target_layer = get_target_layer(model, model_name)
        print(f"Using target layer: {target_layer.__class__.__name__}")
    except Exception as e:
        print(f"ERROR: Could not get target layer: {e}")
        return
    
    cam_generator = GradCAMPlusPlus(model=model, target_layers=[target_layer])
    model_config = get_model_config(model_name)
    transform = get_transforms(model_name, mean, std)
    
    # Create heatmaps directory
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    error_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating Dual Heatmaps"):
        try:
            image_path = Path(row['root_dir']) / row['filename']
            
            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                error_count += 1
                continue
            
            # Load and preprocess image
            original_image = Image.open(image_path).convert('RGB')
            input_tensor = transform(original_image).unsqueeze(0).to(device)
            
            # Resize original for overlay
            img_resized = np.array(original_image.resize(
                (model_config.input_size, model_config.input_size)
            )) / 255.0
            
            # Get class indices (0-indexed for model)
            predicted_class_idx = int(row['predicted_pai']) - 1
            true_class_idx = int(row['true_pai']) - 1
            
            # Get probabilities for display
            prob_pred = row[f'prob_pai_{row["predicted_pai"]}']
            prob_true = row[f'prob_pai_{row["true_pai"]}']
            
            # Generate heatmap for PREDICTED class
            targets_pred = [ClassifierOutputTarget(predicted_class_idx)]
            grayscale_cam_pred = cam_generator(input_tensor=input_tensor, targets=targets_pred)[0, :]
            
            cam_image_pred = cv2.applyColorMap(np.uint8(255 * grayscale_cam_pred), cv2.COLORMAP_JET)
            cam_image_pred = cv2.cvtColor(cam_image_pred, cv2.COLOR_BGR2RGB) / 255.0
            overlay_pred = cv2.addWeighted(img_resized, 1 - transparency, cam_image_pred, transparency, 0)
            
            # Generate heatmap for TRUE class
            targets_true = [ClassifierOutputTarget(true_class_idx)]
            grayscale_cam_true = cam_generator(input_tensor=input_tensor, targets=targets_true)[0, :]
            
            cam_image_true = cv2.applyColorMap(np.uint8(255 * grayscale_cam_true), cv2.COLORMAP_JET)
            cam_image_true = cv2.cvtColor(cam_image_true, cv2.COLOR_BGR2RGB) / 255.0
            overlay_true = cv2.addWeighted(img_resized, 1 - transparency, cam_image_true, transparency, 0)
            
            # Create side-by-side figure with captions
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Predicted class heatmap (LEFT)
            axes[0].imshow(overlay_pred)
            axes[0].axis('off')
            correct_marker = "✓" if row['predicted_pai'] == row['true_pai'] else "✗"
            axes[0].set_title(
                f"PREDICTED: PAI {row['predicted_pai']} {correct_marker}\n"
                f"Confidence: {prob_pred:.1%}",
                fontsize=12,
                fontweight='bold',
                color='green' if row['predicted_pai'] == row['true_pai'] else 'red'
            )
            
            # True class heatmap (RIGHT)
            axes[1].imshow(overlay_true)
            axes[1].axis('off')
            axes[1].set_title(
                f"TRUE CLASS: PAI {row['true_pai']}\n"
                f"Model's confidence: {prob_true:.1%}",
                fontsize=12,
                fontweight='bold',
                color='darkblue'
            )
            
            # Add overall figure title
            fig.suptitle(
                f"{row['filename']} | Model: {model_name}",
                fontsize=10,
                y=0.98
            )
            
            plt.tight_layout()
            
            # Save the combined figure
            heatmap_filename = f"PAI{row['true_pai']}_pred{row['predicted_pai']}_{row['filename'].replace('.tif', '.png')}"
            save_path = output_dir / heatmap_filename
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            success_count += 1
            
        except Exception as e:
            if error_count < 5:  # Only print first 5 errors
                print(f"Error for {row['filename']}: {e}")
            error_count += 1
    
    print(f"\nDual heatmap generation complete:")
    print(f"  - Success: {success_count}")
    print(f"  - Errors: {error_count}")
    
    # List files in output directory
    if output_dir.exists():
        files = list(output_dir.glob("*.png"))
        print(f"  - Files in output dir: {len(files)}")
        if files:
            print(f"  - Example files: {[f.name for f in files[:3]]}")


def calculate_and_save_report(df: pd.DataFrame, output_path: Path):
    """Calculates comprehensive metrics and saves them to a text file."""
    true_labels = df['true_pai'].values
    pred_labels = df['predicted_pai'].values
    labels = sorted(df['true_pai'].unique())

    # --- Core Metrics ---
    accuracy = accuracy_score(true_labels, pred_labels)
    qwk = cohen_kappa_score(true_labels, pred_labels, weights='quadratic')
    report = classification_report(true_labels, pred_labels, labels=labels, zero_division=0)
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)

    # --- Per-class Specificity ---
    specificities = []
    for i in range(len(labels)):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(specificity)

    # --- Formatting the Report ---
    with open(output_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("           PAI Classification Inference Report\n")
        f.write("="*60 + "\n\n")

        f.write("Overall Metrics:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Quadratic Weighted Kappa (QWK): {qwk:.4f}\n\n")

        f.write("Classification Report (Sensitivity is 'recall'):\n")
        f.write("-" * 50 + "\n")
        f.write(report + "\n\n")
        
        f.write("Per-Class Specificity:\n")
        f.write("-" * 25 + "\n")
        for i, label in enumerate(labels):
            f.write(f"  - Class PAI {label}: {specificities[i]:.4f}\n")
        f.write("\n")

        f.write("Confusion Matrix:\n")
        f.write("-" * 20 + "\n")
        f.write("Rows: True Labels, Columns: Predicted Labels\n\n")
        header = "      " + " ".join([f"Pred {l:<2}" for l in labels])
        f.write(header + "\n")
        f.write("      " + "-" * (len(header) - 6) + "\n")
        for i, row in enumerate(cm):
            row_str = f"True {labels[i]:<2} |"
            for val in row:
                row_str += f" {val:<5}"
            f.write(row_str + "\n")
        f.write("\n")
    
    print(f"Statistics report saved to: {output_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to drive the inference process."""
    
    # --- NEW: Parse command-line arguments ---
    args = parse_arguments()
    
    # --- 1. Load Configuration ---
    print("Loading configuration...")
    inference_config = get_inference_config()['inference']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting inference process on device: {device}")

    # --- NEW: Override output directory if specified ---
    if args.output_dir:
        inference_config.base_experiments_dir = args.output_dir
        print(f"✓ Using custom output directory: {args.output_dir}")

    # Load data config for normalization values
    data_config = DataConfig()
    mean = data_config.mean
    std = data_config.std

    # --- 2. Load and Combine Test Data ---
    print("\nLoading and combining test data...")
    
    # --- NEW: Override test data if specified ---
    if args.test_csv and args.test_root:
        test_csv_paths = [args.test_csv]
        test_root_dirs = [args.test_root]
        print(f"✓ Using custom test data: {args.test_csv}")
    else:
        test_csv_paths = inference_config.test_csv_paths
        test_root_dirs = inference_config.test_root_dirs
    
    all_dfs = []
    for csv_path, root_dir in zip(test_csv_paths, test_root_dirs):
        if not Path(csv_path).exists():
            print(f"WARNING: Test CSV not found: {csv_path}")
            continue
            
        df = pd.read_csv(csv_path)
        df['root_dir'] = root_dir  # Add root_dir for CustomDataset
        all_dfs.append(df)
    
    if not all_dfs:
        print("ERROR: No test data found!")
        return
    
    test_data = pd.concat(all_dfs, ignore_index=True)
    test_data['PAI_0_indexed'] = test_data['PAI'] - 1  # Convert to 0-indexed
    print(f"Loaded a total of {len(test_data)} test samples.")
    
    # --- 3. Get checkpoints - NEW: Use command-line overrides ---
    print("\n" + "="*80)
    print("Resolving Model Checkpoints")
    print("="*80)
    checkpoints = resolve_checkpoints(args, inference_config)
    
    if not checkpoints:
        print("\nERROR: No valid checkpoints found!")
        print("\nTo use checkpoints from model_checkpoints/ directory:")
        print("  python code/test_inference/inference_gradcam.py --checkpoint-dir model_checkpoints")
        print("\nOr specify individual checkpoints:")
        print("  python code/test_inference/inference_gradcam.py --efficientnet-b3 path/to/checkpoint.pth")
        return
    
    print(f"\nFound {len(checkpoints)} valid checkpoint(s)")
    if args.no_heatmaps:
        print("⚠ Heatmap generation disabled (--no-heatmaps)")

    # --- 4. Loop Through Each Model Checkpoint ---
    for model_name, checkpoint_path in checkpoints.items():
        print("\n" + "="*80)
        print(f"Processing Model: {model_name}")
        print("="*80)

        try:
            # --- Setup Output Directory ---
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_output_dir = Path(inference_config.base_experiments_dir) / model_name / f"inference_{timestamp}"
            heatmaps_dir = model_output_dir / "heatmaps"
            os.makedirs(heatmaps_dir, exist_ok=True)
            print(f"Results will be saved to: {model_output_dir}")

            # --- Load Model and Data ---
            model = load_model(model_name, checkpoint_path, device)
            model_config = get_model_config(model_name)
            
            test_transform = get_transforms(model_name, mean, std)
            test_dataset = CustomDataset(test_data, transform=test_transform)
            test_loader = DataLoader(
                test_dataset,
                batch_size=model_config.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )

            # --- Run Inference and Save Predictions CSV ---
            print("\nRunning inference...")
            predictions_df = run_batch_inference(model, test_loader, device)
            predictions_df = pd.merge(predictions_df, test_data[['filename', 'root_dir']], on='filename')
            
            predictions_csv_path = model_output_dir / "predictions.csv"
            predictions_df.to_csv(predictions_csv_path, index=False)
            print(f"Predictions saved to: {predictions_csv_path}")

            # --- Generate and Save Statistics Report ---
            print("\nGenerating statistics report...")
            report_path = model_output_dir / "statistics_report.txt"
            calculate_and_save_report(predictions_df, report_path)

            # --- Generate and Save Heatmaps (NEW: check flag) ---
            if not args.no_heatmaps:
                print("\nGenerating heatmaps...")
                generate_and_save_heatmaps(
                    predictions_df, model, model_name, heatmaps_dir,
                    inference_config.cam_method, inference_config.heatmap_transparency,
                    device, mean, std
                )
            else:
                print("\n⚠ Skipping heatmap generation (--no-heatmaps flag set)")
            
            print(f"\n✓ Successfully completed inference for {model_name}")

        except Exception as e:
            print(f"\nERROR: Failed to process {model_name}")
            print(f"Error details: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "="*80)
    print("Inference process completed for all models.")
    print("="*80)

if __name__ == '__main__':
    main()