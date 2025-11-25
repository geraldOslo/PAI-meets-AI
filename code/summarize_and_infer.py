#!/usr/bin/env python3
"""
================================================================================
Hyperparameter Search Summarizer and Inference Runner (CORRECTED VERSION)
================================================================================
Purpose:
  Automated evaluation of all hyperparameter search experiments with
  comprehensive CSV output containing all required metrics and hyperparameters.

CRITICAL FIX: Regex patterns now properly skip the first line of equals signs
and capture all content between the opening and closing delimiter lines.

Author: Gerald Torgersen
SPDX-License-Identifier: MIT
Copyright (c) 2025 Gerald Torgersen
================================================================================
"""

import os
import sys
import re
import argparse
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from code.config import get_inference_config, get_model_config, DataConfig
    from code.training.data_utils import CustomDataset
    from code.test_inference.inference_gradcam import (
        load_model, get_transforms, run_batch_inference, calculate_and_save_report
    )
except ImportError as e:
    print(f"ERROR: Could not import a required module: {e}", file=sys.stderr)
    sys.exit(1)


def safe_float_extract(value_str):
    """Safely extract a float from a string."""
    if not value_str:
        return None
    try:
        cleaned = str(value_str).strip().rstrip('.,;:!]')
        return float(cleaned)
    except (ValueError, AttributeError):
        return None


def safe_bool_extract(value_str):
    """Safely extract a boolean from a string."""
    if not value_str:
        return None
    try:
        return str(value_str).strip().lower() == 'true'
    except (ValueError, AttributeError):
        return None


def extract_training_duration(log_content: str, debug: bool = False) -> dict:
    """
    Extract training duration from training log.
    Looks for the structured "TRAINING DURATION (for parsing)" section.
    
    FIXED: Now properly skips the first line of equals signs and captures
    all content between the opening and closing lines.
    """
    duration_info = {
        'Training Duration (min)': None,
        'Training Duration (hours)': None
    }
    
    # FIXED: Properly skip first ===== line and capture content until second =====
    duration_match = re.search(
        r"TRAINING DURATION \(for parsing\)\s*\n={80}\s*\n(.*?)\n={80}",
        log_content,
        re.DOTALL
    )
    
    if duration_match:
        duration_section = duration_match.group(1)
        
        if debug:
            print(f"    ✓ Found duration section ({len(duration_section)} chars)")
        
        # Extract minutes
        min_match = re.search(r"Training Duration \(minutes\):\s*([\d.]+)", duration_section)
        if min_match:
            duration_info['Training Duration (min)'] = round(float(min_match.group(1)), 1)
        
        # Extract hours
        hr_match = re.search(r"Training Duration \(hours\):\s*([\d.]+)", duration_section)
        if hr_match:
            duration_info['Training Duration (hours)'] = round(float(hr_match.group(1)), 2)
        
        if debug:
            duration_min = duration_info['Training Duration (min)']
            duration_hr = duration_info['Training Duration (hours)']
            print(f"      Duration: {duration_hr} hours ({duration_min} min)")
        
        return duration_info
    
    # Fallback: look for elapsed wallclock time anywhere in log
    elapsed_match = re.search(r"Elapsed wallclock time:\s+([\d.]+)\s+(\w+)", log_content)
    if elapsed_match:
        time_value = float(elapsed_match.group(1))
        time_unit = elapsed_match.group(2).lower()
        
        if 'minute' in time_unit:
            duration_minutes = time_value
        elif 'hour' in time_unit:
            duration_minutes = time_value * 60
        elif 'second' in time_unit:
            duration_minutes = time_value / 60
        else:
            duration_minutes = time_value
        
        duration_info['Training Duration (min)'] = round(duration_minutes, 1)
        duration_info['Training Duration (hours)'] = round(duration_minutes / 60, 2)
        
        if debug:
            print(f"    ⚠ Duration (fallback): {duration_info['Training Duration (hours)']} hours")
    
    return duration_info


def extract_hyperparameters_from_log(log_content: str, debug: bool = False) -> dict:
    """
    Extract hyperparameters from the structured logging section.
    This function is designed to work with the "HYPERPARAMETERS (for parsing)" section
    that's printed by the train_simple.py script.
    
    FIXED: Now properly skips the first line of equals signs and captures
    all content between the opening and closing lines.
    """
    results = {}
    
    # FIXED: Properly skip first ===== line and capture content until second =====
    hyperparam_match = re.search(
        r"HYPERPARAMETERS \(for parsing\)\s*\n={80}\s*\n(.*?)\n={80}",
        log_content,
        re.DOTALL
    )
    
    if not hyperparam_match:
        if debug:
            print("    ⚠ No structured hyperparameter section found")
        return results
    
    hyperparam_section = hyperparam_match.group(1)
    
    if debug:
        print(f"    ✓ Found hyperparameter section ({len(hyperparam_section)} chars)")
    
    # Define extraction patterns
    patterns = {
        'LR': r"base_lr:\s*([\d.e-]+)",
        'Max LR': r"max_lr:\s*([\d.e-]+)",
        'Weight Decay': r"weight_decay:\s*([\d.e-]+)",
        'Dropout': r"dropout:\s*([\d.e-]+)",
        'Drop Path': r"drop_path:\s*([\d.e-]+)",
        'Grad Clip': r"grad_clip:\s*([\d.e-]+)",
        'Loss Type': r"loss_type:\s*(\w+)",
        'Focal Alpha': r"focal_alpha:\s*([\d.e-]+)",
        'Focal Gamma': r"focal_gamma:\s*([\d.e-]+)",
        'Label Smoothing': r"label_smoothing:\s*([\d.e-]+)",
        'Use Class Weights': r"use_class_weights:\s*(True|False)",
        'Class Weights': r"class_weights:\s*(\[[\d.,\s]+\])",
        'Use Oversampling': r"use_oversampling:\s*(True|False)",
        'Batch Size': r"batch_size:\s*(\d+)",
        'Input Size': r"input_size:\s*(\d+)",
        'Mixup': r"mixup:\s*(True|False)",
        'Mixup Alpha': r"mixup_alpha:\s*([\d.e-]+)",
        'CutMix': r"cutmix:\s*(True|False)",
        'Epochs Configured': r"epochs:\s*(\d+)",
        'Patience': r"patience:\s*(\d+)",
        'Optimizer': r"optimizer:\s*(\w+)",
        'Scheduler': r"scheduler:\s*(\w+)",
        'Min Delta': r"min_delta:\s*([\d.e-]+)",
        'PCT Start': r"pct_start:\s*([\d.e-]+)",
        'Use AMP': r"use_amp:\s*(True|False)",
        'Num Workers': r"num_workers:\s*(\d+)",
    }
    
    for param_name, pattern in patterns.items():
        match = re.search(pattern, hyperparam_section, re.IGNORECASE)
        if match:
            value = match.group(1)
            
            # Handle boolean values
            if param_name in ['Use Class Weights', 'Use Oversampling', 'Mixup', 'CutMix', 'Use AMP']:
                results[param_name] = safe_bool_extract(value)
                # Convert to Yes/No for Mixup and CutMix
                if param_name in ['Mixup', 'CutMix']:
                    results[param_name] = 'Yes' if results[param_name] else 'No'
            # Handle class weights (keep as string)
            elif param_name == 'Class Weights':
                results[param_name] = value
            # Handle numeric values
            elif param_name in ['Batch Size', 'Input Size', 'Epochs Configured', 'Patience', 'Num Workers']:
                results[param_name] = int(value)
            # Handle string values
            elif param_name in ['Loss Type', 'Optimizer', 'Scheduler']:
                results[param_name] = value
            # Handle float values
            else:
                results[param_name] = safe_float_extract(value)
            
            if debug and results[param_name] is not None:
                print(f"      {param_name}: {results[param_name]}")
    
    if debug:
        print(f"    ✓ Extracted {len(results)} hyperparameters")
    
    return results


def extract_training_metrics_from_log(log_content: str, debug: bool = False) -> dict:
    """Extract training metrics like epochs run and best validation QWK."""
    results = {}
    
    # Extract epochs run (last epoch number)
    epochs_matches = re.findall(r"Epoch\s+(\d+)/\d+", log_content)
    if epochs_matches:
        results['Epochs Run'] = int(epochs_matches[-1])
        if debug:
            print(f"    Epochs Run: {results['Epochs Run']}")
    
    # Extract best validation QWK
    best_qwk_matches = re.findall(r"New best QWK:\s*([\d.]+)", log_content)
    if best_qwk_matches:
        results['Best Val QWK'] = safe_float_extract(best_qwk_matches[-1])
        if debug:
            print(f"    Best Val QWK: {results['Best Val QWK']}")
    
    return results


def extract_test_metrics_from_report(report_path: Path, debug: bool = False) -> dict:
    """
    Extract test set metrics from the inference statistics report.
    """
    results = {}
    
    if not report_path.exists():
        if debug:
            print(f"    ⚠ Statistics report not found: {report_path.name}")
        return results
    
    try:
        content = report_path.read_text()
        
        # Extract overall metrics
        qwk_match = re.search(r"Quadratic Weighted Kappa\s*\(QWK\):\s*([\d.]+)", content)
        if qwk_match:
            results['Test QWK'] = safe_float_extract(qwk_match.group(1))
        
        acc_match = re.search(r"Accuracy:\s*([\d.]+)", content)
        if acc_match:
            results['Test Accuracy'] = safe_float_extract(acc_match.group(1))
        
        # Extract weighted F1 from classification report
        f1_match = re.search(r"weighted\s+avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", content)
        if f1_match:
            results['Test F1 (Weighted)'] = safe_float_extract(f1_match.group(3))
        
        # Extract per-class sensitivity (recall)
        for i in range(1, 6):
            # Look for pattern like: "    1    0.95    0.92    0.93"
            recall_match = re.search(
                rf"^\s*{i}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", 
                content, 
                re.MULTILINE
            )
            if recall_match:
                recall_value = safe_float_extract(recall_match.group(2))
                results[f'Test Sens PAI {i}'] = recall_value
        
        if debug and results:
            print(f"    Test QWK: {results.get('Test QWK')}")
            print(f"    Test Accuracy: {results.get('Test Accuracy')}")
            print(f"    Test F1: {results.get('Test F1 (Weighted)')}")
            for i in range(1, 6):
                sens = results.get(f'Test Sens PAI {i}')
                if sens is not None:
                    print(f"    Test Sens PAI {i}: {sens}")
        
    except Exception as e:
        if debug:
            print(f"    ✗ Error reading report: {e}")
    
    return results

def find_completed_experiments(base_dir: Path) -> list:
    """
    Find all completed experiment directories.

    Supports two patterns:
    1. Old: experiments/hyperparam_search/{model}/{experiment}/pai_training_{timestamp}/
    2. New: experiments/{experiment}_{timestamp}/ (with checkpoint files to identify completion)
    """
    print(f"Searching for completed experiments in: {base_dir}")

    exp_paths = []

    # Pattern 1: Nested pai_training_* folders (old experiments + Phase 6)
    # Finds: experiments/*/pai_training_YYYYMMDD_HHMMSS/
    # Excludes: anything in 'old/' subdirectory to avoid duplicates
    nested_pattern_paths = [p for p in base_dir.rglob("pai_training_*") if "/old/" not in str(p) and "\\old\\" not in str(p)]
    if nested_pattern_paths:
        print(f"  Found {len(nested_pattern_paths)} experiments in nested pattern (pai_training_*)")
        exp_paths.extend([p for p in nested_pattern_paths if p.is_dir()])

    # Pattern 2: New flat structure - experiments directly in base_dir
    # Look for directories matching:
    #   - exp*_YYYYMMDD_HHMMSS (Phase 1)
    #   - effnet_exp*_YYYYMMDD_HHMMSS (Phase 2 EfficientNet)
    #   - resnet_exp*_YYYYMMDD_HHMMSS (Phase 2 ResNet)
    #   - convnext_exp*_YYYYMMDD_HHMMSS (Phase 2 ConvNeXt)
    #   - p3_*_YYYYMMDD_HHMMSS (Phase 3 PAI 2/3 Boundary Optimization)
    #   - final_*_YYYYMMDD_HHMMSS (Phase 3 alternative naming)
    #   - p4_*_YYYYMMDD_HHMMSS (Phase 4 Champion Refinement)
    #   - p5_*_YYYYMMDD_HHMMSS (Phase 5 Focal Loss Deep Dive)
    #   - p6_*_YYYYMMDD_HHMMSS (Phase 6 Focal Loss Optimization)
    for item in base_dir.iterdir():
        # Skip 'old' directory
        if item.name == "old":
            continue
        if item.is_dir() and "_" in item.name:
            # Check if it matches Phase 1-6 naming patterns
            is_phase1 = item.name.startswith("exp")
            is_phase2 = (item.name.startswith("effnet_") or
                        item.name.startswith("resnet_") or
                        item.name.startswith("convnext_"))
            is_phase3 = (item.name.startswith("p3_") or
                        item.name.startswith("final_"))
            is_phase4 = item.name.startswith("p4_")
            is_phase5 = item.name.startswith("p5_")
            is_phase6 = item.name.startswith("p6_")

            if is_phase1 or is_phase2 or is_phase3 or is_phase4 or is_phase5 or is_phase6:
                # Check if it has checkpoint files (indicating successful completion)
                checkpoint_files = list(item.glob("*_best.pth"))
                if checkpoint_files:
                    exp_paths.append(item)

    if not exp_paths:
        print("Warning: No completed experiments found (checked both old and new patterns).")
        return []

    # Ensure unique and sorted
    unique_exp_paths = sorted(set(exp_paths), key=lambda p: str(p))

    print(f"Found {len(unique_exp_paths)} completed experiments total.")
    return unique_exp_paths


def run_inference_for_experiment(exp_path: Path, device: torch.device):
    """Run inference for an experiment if not already done."""
    # Detect model name from directory structure or checkpoint filename
    # Old pattern: experiments/hyperparam_search/{model}/{experiment}/pai_training_{timestamp}/
    # Phase 1: experiments/exp*_{timestamp}/ with {model}_best.pth
    # Phase 2: experiments/{effnet|resnet|convnext}_exp*_{timestamp}/ with {model}_best.pth

    # Try old pattern first
    if exp_path.parent.parent.name in ['resnet50', 'efficientnet_b3', 'convnext_tiny']:
        model_name_key = exp_path.parent.parent.name
    # Phase 2 pattern: detect from directory name prefix
    elif exp_path.name.startswith("effnet_"):
        model_name_key = "efficientnet_b3"
    elif exp_path.name.startswith("resnet_"):
        model_name_key = "resnet50"
    elif exp_path.name.startswith("convnext_"):
        model_name_key = "convnext_tiny"
    # Phase 3, 4, 5, 6 patterns: detect from p*_model_* prefix
    elif exp_path.name.startswith("p3_") or exp_path.name.startswith("p4_") or exp_path.name.startswith("p5_") or exp_path.name.startswith("p6_"):
        # Extract model name from directory name (e.g., p5_resnet_*, p6_resnet_alpha_*)
        if "resnet" in exp_path.name.lower():
            model_name_key = "resnet50"
        elif "effnet" in exp_path.name.lower():
            model_name_key = "efficientnet_b3"
        elif "convnext" in exp_path.name.lower():
            model_name_key = "convnext_tiny"
        else:
            # Fallback to checkpoint detection
            checkpoint_files = list(exp_path.glob("*_best.pth"))
            if checkpoint_files:
                model_name_key = checkpoint_files[0].stem.replace("_best", "").replace("-", "_")
            else:
                print(f"  - WARNING: Cannot determine model for {exp_path.name}. Skipping.")
                return
    else:
        # Phase 1 pattern: detect from checkpoint filename
        checkpoint_files = list(exp_path.glob("*_best.pth"))
        if checkpoint_files:
            # Extract model from filename: "resnet50_best.pth" -> "resnet50"
            model_name_key = checkpoint_files[0].stem.replace("_best", "").replace("-", "_")
        else:
            print(f"  - WARNING: Cannot determine model for {exp_path.name}. Skipping.")
            return
    
    try:
        # Check if statistics report already exists
        report_path = exp_path / "statistics_report.txt"
        if report_path.exists():
            print(f"  - Statistics report already exists for '{exp_path.name}'. Skipping inference.")
            return
        
        # Find checkpoint
        checkpoint_filename = f"{get_model_config(model_name_key).name.lower().replace(' ', '_')}_best.pth"
        checkpoint_path = exp_path / checkpoint_filename
        
        if not checkpoint_path.exists():
            print(f"  - WARNING: Checkpoint not found at '{checkpoint_path}'. Skipping inference.")
            return
        
        print(f"  - Running inference for '{exp_path.name}'...")
        
        # Load inference config
        inference_config = get_inference_config()['inference']
        data_config = DataConfig()
        
        # Prepare test data
        df_parts = []
        for csv_p, root_d in zip(inference_config.test_csv_paths, inference_config.test_root_dirs):
            temp_df = pd.read_csv(csv_p)
            temp_df['root_dir'] = root_d
            df_parts.append(temp_df)
        test_df = pd.concat(df_parts, ignore_index=True)

        # Clean and validate PAI values (same as training)
        print(f"    Total test samples loaded: {len(test_df)}")

        # Convert PAI to numeric in case it's stored as string
        test_df['PAI'] = pd.to_numeric(test_df['PAI'], errors='coerce')

        # Check for and remove invalid PAI values
        invalid_mask = test_df['PAI'].isna()
        if invalid_mask.any():
            n_invalid = invalid_mask.sum()
            print(f"    WARNING: Found {n_invalid} test rows with invalid PAI values (NS, etc.)")
            test_df = test_df[~invalid_mask].copy()
            print(f"    Dropped {n_invalid} invalid rows. Remaining: {len(test_df)}")

        # Validate PAI values are in expected range (1-5)
        invalid_range = (test_df['PAI'] < 1) | (test_df['PAI'] > 5)
        if invalid_range.any():
            n_invalid_range = invalid_range.sum()
            print(f"    WARNING: Found {n_invalid_range} test rows with PAI values outside range [1-5]")
            test_df = test_df[~invalid_range].copy()
            print(f"    Dropped {n_invalid_range} out-of-range rows. Remaining: {len(test_df)}")

        test_df['PAI_0_indexed'] = test_df['PAI'] - 1
        print(f"    Valid test samples for inference: {len(test_df)}")
        
        # Load model and run inference
        model = load_model(model_name_key, str(checkpoint_path), device)
        test_transform = get_transforms(model_name_key, data_config.mean, data_config.std)
        test_dataset = CustomDataset(test_df, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
        
        predictions_df = run_batch_inference(model, test_loader, device)
        calculate_and_save_report(predictions_df, report_path)
        print(f"  - ✓ Statistics report saved.")
        
    except Exception as e:
        print(f"  - ✗ ERROR during inference for {exp_path.name}: {e}")


def parse_experiment_results(exp_path: Path, debug: bool = False) -> dict:
    """Parse comprehensive results from an experiment directory."""

    # Detect directory structure pattern and phase
    # Old pattern: experiments/hyperparam_search/{model}/{experiment}/pai_training_{timestamp}/
    # Phase 1: experiments/exp*_{timestamp}/ with {model}_best.pth
    # Phase 2: experiments/{effnet|resnet|convnext}_exp*_{timestamp}/ with {model}_best.pth
    # Phase 3: experiments/final_*_{timestamp}/
    # Phase 4: experiments/p4_*_{timestamp}/ or nested in Phase 2
    # Phase 5: experiments/p5_*_{timestamp}/
    # Phase 6: experiments/p6_*/pai_training_{timestamp}/

    phase = "Unknown"

    if exp_path.parent.parent.name in ['resnet50', 'efficientnet_b3', 'convnext_tiny']:
        # Old pattern
        model_name = exp_path.parent.parent.name
        experiment_name = exp_path.parent.name
        timestamp = exp_path.name
        phase = "Pre-Phase1"
    else:
        dir_name = exp_path.name

        # Check for Phase 6 nested structure (p6_*/pai_training_*)
        if exp_path.parent.name.startswith("p6_"):
            # Phase 6: experiments/p6_resnet_alpha_0_35/pai_training_20251029_095100/
            experiment_name = exp_path.parent.name  # p6_resnet_alpha_0_35
            timestamp = exp_path.name.replace("pai_training_", "")  # 20251029_095100
            phase = "Phase6"

            # Extract model from parent directory name
            if "resnet" in exp_path.parent.name.lower():
                model_name = "resnet50"
            elif "effnet" in exp_path.parent.name.lower():
                model_name = "efficientnet_b3"
            elif "convnext" in exp_path.parent.name.lower():
                model_name = "convnext_tiny"
            else:
                checkpoint_files = list(exp_path.glob("*_best.pth"))
                model_name = checkpoint_files[0].stem.replace("_best", "").replace("-", "_") if checkpoint_files else "unknown"

        # Check for Phase 4 nested in Phase 2 directory
        elif exp_path.parent.name.startswith("exp2_") and dir_name.startswith("p4_"):
            # Phase 4 nested: exp2_stronger_regularization_*/p4_champion_*_YYYYMMDD_HHMMSS/
            phase = "Phase4"
            if "resnet" in dir_name.lower():
                model_name = "resnet50"
            elif "effnet" in dir_name.lower():
                model_name = "efficientnet_b3"
            elif "convnext" in dir_name.lower():
                model_name = "convnext_tiny"
            else:
                checkpoint_files = list(exp_path.glob("*_best.pth"))
                model_name = checkpoint_files[0].stem.replace("_best", "").replace("-", "_") if checkpoint_files else "unknown"

            parts = dir_name.split("_")
            if len(parts) >= 3:
                timestamp = f"{parts[-2]}_{parts[-1]}"
                experiment_name = "_".join(parts[:-2])
            else:
                experiment_name = dir_name
                timestamp = "unknown"

        # Check for Phase 2 nested in Phase 1 directory
        elif exp_path.parent.name.startswith("exp") and (dir_name.startswith("resnet_") or dir_name.startswith("effnet_") or dir_name.startswith("convnext_")):
            # Phase 2 nested: exp2_stronger_regularization_*/resnet_exp*_YYYYMMDD_HHMMSS/
            phase = "Phase2"
            if dir_name.startswith("effnet_"):
                model_name = "efficientnet_b3"
            elif dir_name.startswith("resnet_"):
                model_name = "resnet50"
            elif dir_name.startswith("convnext_"):
                model_name = "convnext_tiny"
            else:
                checkpoint_files = list(exp_path.glob("*_best.pth"))
                model_name = checkpoint_files[0].stem.replace("_best", "").replace("-", "_") if checkpoint_files else "unknown"

            parts = dir_name.split("_")
            if len(parts) >= 3:
                timestamp = f"{parts[-2]}_{parts[-1]}"
                experiment_name = "_".join(parts[:-2])
            else:
                experiment_name = dir_name
                timestamp = "unknown"

        # Standard flat patterns
        else:
            # Detect phase from directory name prefix
            if dir_name.startswith("exp") and not any(dir_name.startswith(x) for x in ["effnet_", "resnet_", "convnext_"]):
                phase = "Phase1"
                checkpoint_files = list(exp_path.glob("*_best.pth"))
                model_name = checkpoint_files[0].stem.replace("_best", "").replace("-", "_") if checkpoint_files else "unknown"

            elif dir_name.startswith("effnet_"):
                phase = "Phase2"
                model_name = "efficientnet_b3"
            elif dir_name.startswith("resnet_"):
                phase = "Phase2"
                model_name = "resnet50"
            elif dir_name.startswith("convnext_"):
                phase = "Phase2"
                model_name = "convnext_tiny"

            elif dir_name.startswith("final_"):
                phase = "Phase3"
                checkpoint_files = list(exp_path.glob("*_best.pth"))
                model_name = checkpoint_files[0].stem.replace("_best", "").replace("-", "_") if checkpoint_files else "resnet50"

            elif dir_name.startswith("p3_"):
                phase = "Phase3"
                if "resnet" in dir_name.lower():
                    model_name = "resnet50"
                elif "effnet" in dir_name.lower():
                    model_name = "efficientnet_b3"
                elif "convnext" in dir_name.lower():
                    model_name = "convnext_tiny"
                else:
                    checkpoint_files = list(exp_path.glob("*_best.pth"))
                    model_name = checkpoint_files[0].stem.replace("_best", "").replace("-", "_") if checkpoint_files else "unknown"

            elif dir_name.startswith("p4_"):
                phase = "Phase4"
                if "resnet" in dir_name.lower():
                    model_name = "resnet50"
                elif "effnet" in dir_name.lower():
                    model_name = "efficientnet_b3"
                elif "convnext" in dir_name.lower():
                    model_name = "convnext_tiny"
                else:
                    checkpoint_files = list(exp_path.glob("*_best.pth"))
                    model_name = checkpoint_files[0].stem.replace("_best", "").replace("-", "_") if checkpoint_files else "unknown"

            elif dir_name.startswith("p5_"):
                phase = "Phase5"
                if "resnet" in dir_name.lower():
                    model_name = "resnet50"
                elif "effnet" in dir_name.lower():
                    model_name = "efficientnet_b3"
                elif "convnext" in dir_name.lower():
                    model_name = "convnext_tiny"
                else:
                    checkpoint_files = list(exp_path.glob("*_best.pth"))
                    model_name = checkpoint_files[0].stem.replace("_best", "").replace("-", "_") if checkpoint_files else "unknown"

            else:
                # Unknown pattern
                checkpoint_files = list(exp_path.glob("*_best.pth"))
                model_name = checkpoint_files[0].stem.replace("_best", "").replace("-", "_") if checkpoint_files else "unknown"

            # Parse timestamp from directory name
            parts = dir_name.split("_")
            if len(parts) >= 3:
                timestamp = f"{parts[-2]}_{parts[-1]}"
                experiment_name = "_".join(parts[:-2])
            else:
                experiment_name = dir_name
                timestamp = "unknown"

    # Add phase prefix to experiment name for clarity
    if phase != "Unknown" and not experiment_name.startswith(phase):
        experiment_name = f"{phase}:{experiment_name}"

    results = {
        'Model': model_name,
        'Phase': phase,  # Separate phase column for easy filtering
        'Experiment': experiment_name,
        'Timestamp': timestamp
    }
    
    if debug:
        print(f"\n  Parsing: {exp_path.name}")
    
    try:
        # Extract from training log
        log_file = exp_path / "training.log"
        if log_file.exists():
            content = log_file.read_text()
            
            # Extract hyperparameters
            hyperparams = extract_hyperparameters_from_log(content, debug=debug)
            results.update(hyperparams)
            
            # Extract training metrics
            train_metrics = extract_training_metrics_from_log(content, debug=debug)
            results.update(train_metrics)
            
            # Extract training duration
            duration_info = extract_training_duration(content, debug=debug)
            results.update(duration_info)
        else:
            if debug:
                print(f"  ⚠ Training log not found")
        
        # Extract test metrics from inference report
        report_file = exp_path / "statistics_report.txt"
        if report_file.exists():
            test_metrics = extract_test_metrics_from_report(report_file, debug=debug)
            results.update(test_metrics)
        else:
            if debug:
                print(f"  ⚠ Statistics report not found (inference may not have been run)")
            
    except Exception as e:
        print(f"  Warning: Error parsing {exp_path.name}: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        
    return results


def main():
    parser = argparse.ArgumentParser(description="Summarize Hyperparameter Search Results")
    parser.add_argument(
        "--exp_dir", 
        type=str, 
        default=str(PROJECT_ROOT / "experiments/hyperparam_search"), 
        help="Directory containing hyperparameter search results"
    )
    parser.add_argument(
        "--skip-inference", 
        action="store_true", 
        help="Skip inference step (only summarize existing reports)"
    )
    parser.add_argument(
        "--print-table", 
        action="store_true", 
        help="Print summary table to console"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug output"
    )
    args = parser.parse_args()
    
    base_experiments_dir = Path(args.exp_dir)
    if not base_experiments_dir.is_dir():
        print(f"ERROR: Experiments directory not found at '{base_experiments_dir}'")
        return
    
    # Find all completed experiments
    experiment_paths = find_completed_experiments(base_experiments_dir)
    if not experiment_paths:
        print("No completed experiments found. Exiting.")
        return
    
    # Run inference if requested
    if not args.skip_inference:
        print("\n--- Running Inference on Test Set ---")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for exp_path in experiment_paths:
            run_inference_for_experiment(exp_path, device)
    
    # Parse all experiment results
    print("\n--- Parsing All Experiment Results ---")
    all_results = []
    for p in experiment_paths:
        result = parse_experiment_results(p, debug=args.debug)
        all_results.append(result)
    
    if not all_results:
        print("No results to summarize.")
        return
    
    # Create DataFrame with exact column order
    summary_df = pd.DataFrame(all_results)
    
    # Define the exact column order from your CSV format
    column_order = [
        'Model', 'Phase', 'Experiment', 'Test QWK', 'Test Accuracy', 'Test F1 (Weighted)',
        'Best Val QWK', 'Epochs Run', 'Training Duration (hours)', 'Training Duration (min)',
        'Test Sens PAI 1', 'Test Sens PAI 2', 'Test Sens PAI 3', 'Test Sens PAI 4', 'Test Sens PAI 5',
        'LR', 'Max LR', 'Weight Decay', 'Dropout', 'Drop Path', 'Grad Clip',
        'Focal Gamma', 'Focal Alpha', 'Label Smoothing',
        'Batch Size', 'Input Size', 'Use Class Weights', 'Class Weights', 'Use Oversampling',
        'Mixup', 'Mixup Alpha', 'CutMix',
        'Epochs Configured', 'Patience', 'Min Delta',
        'Optimizer', 'Scheduler', 'PCT Start',
        'Loss Type', 'Use AMP', 'Num Workers',
        'Timestamp'
    ]
    
    # Add any columns that exist in the data but not in column_order
    existing_columns = [col for col in column_order if col in summary_df.columns]
    summary_df = summary_df[existing_columns]
    
    # Sort by Test QWK if available
    if 'Test QWK' in summary_df.columns and summary_df['Test QWK'].notna().any():
        summary_df = summary_df.sort_values(by="Test QWK", ascending=False, na_position='last')
    
    # Round numeric columns
    numeric_cols = summary_df.select_dtypes(include=['float64', 'float32']).columns
    summary_df[numeric_cols] = summary_df[numeric_cols].round(4)
    
    # Save CSV
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv_path = base_experiments_dir / f"hyperparameter_summary_{timestamp}.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("✓ SUMMARY COMPLETE")
    print(f"Summary table with {len(summary_df)} experiments saved to:")
    print(f"  {summary_csv_path}")
    
    # Check completeness
    print("\nData Completeness:")
    key_columns = ['LR', 'Weight Decay', 'Dropout', 'Class Weights', 'Training Duration (hours)', 
                   'Test QWK', 'Best Val QWK']
    for col in key_columns:
        if col in summary_df.columns:
            valid_count = summary_df[col].notna().sum()
            pct = valid_count/len(summary_df)*100 if len(summary_df) > 0 else 0
            print(f"  {col}: {valid_count}/{len(summary_df)} experiments ({pct:.1f}%)")
    
    # Duration statistics
    if 'Training Duration (hours)' in summary_df.columns:
        valid_duration = summary_df['Training Duration (hours)'].dropna()
        if len(valid_duration) > 0:
            print(f"\nTraining Duration Statistics:")
            print(f"  Longest:  {valid_duration.max():.2f} hours")
            print(f"  Mean:     {valid_duration.mean():.2f} hours")
            print(f"  Shortest: {valid_duration.min():.2f} hours")
            print(f"  Total:    {valid_duration.sum():.2f} hours")
    
    # QWK statistics
    if 'Test QWK' in summary_df.columns:
        valid_qwk = summary_df['Test QWK'].dropna()
        if len(valid_qwk) > 0:
            print(f"\nTest QWK Statistics:")
            print(f"  Best:   {valid_qwk.max():.4f}")
            print(f"  Mean:   {valid_qwk.mean():.4f}")
            print(f"  Median: {valid_qwk.median():.4f}")
            print(f"  Worst:  {valid_qwk.min():.4f}")
    
    print("="*80)
    
    # Print table if requested
    if args.print_table:
        print("\n--- Top 20 Experiments ---")
        display_cols = ['Model', 'Experiment', 'Test QWK', 'Test Accuracy', 'Best Val QWK', 
                       'Epochs Run', 'Training Duration (hours)', 'LR', 'Weight Decay', 
                       'Dropout', 'Focal Gamma']
        display_cols = [c for c in display_cols if c in summary_df.columns]
        with pd.option_context('display.max_rows', 20, 'display.max_columns', None, 
                               'display.width', 1000, 'display.max_colwidth', 30):
            print(summary_df[display_cols].head(20))


if __name__ == '__main__':
    main()