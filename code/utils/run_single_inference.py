#!/usr/bin/env python3
"""
================================================================================
Single Experiment Inference and CSV Formatter
================================================================================
Purpose:
  Runs test-set inference on a single, specified experiment folder and prints
  the results, including a ready-to-paste CSV row. This script bypasses the
  main summarizer's discovery mechanism, making it ideal for manually processing
  or debugging a single run.

Author: Gerald Torgersen (with AI Assistant)
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
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score, classification_report

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# --- Project Imports (ensure this works from your environment) ---
try:
    from code.config import DataConfig, get_model_config
    from code.training.data_utils import CustomDataset
    from code.test_inference.inference_gradcam import load_model, get_transforms, run_batch_inference
    # We need to borrow the log parsing function from the main summarizer
    from code.summarize_and_infer import extract_hyperparameters_from_log, extract_training_metrics_from_log, extract_training_duration
except ImportError as e:
    print(f"FATAL ERROR: Could not import a required module: {e}", file=sys.stderr)
    print("Please ensure you are running this script from the project's root directory.", file=sys.stderr)
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run inference on a single experiment and generate a CSV line.")
    parser.add_argument("exp_dir", type=str, help="Path to the specific timestamped experiment directory (e.g., '.../pai_training_20251014_091357').")
    args = parser.parse_args()

    exp_path = Path(args.exp_dir).resolve()
    if not exp_path.is_dir():
        print(f"ERROR: Directory not found: {exp_path}")
        return

    print(f"--- Processing Experiment: {exp_path.name} ---")

    # --- 1. Determine Model Name from Path ---
    try:
        model_name = exp_path.parent.parent.name
        print(f"Detected Model: '{model_name}'")
        # Validate model name
        _ = get_model_config(model_name)
    except (ValueError, IndexError) as e:
        print(f"ERROR: Could not determine model name from path structure. Expected '.../model_name/experiment_name/timestamp'. Error: {e}")
        return

    # --- 2. Find Checkpoint File ---
    checkpoints = list(exp_path.glob("*_best.pth"))
    if not checkpoints:
        print(f"ERROR: No '*_best.pth' checkpoint file found in {exp_path}")
        return
    checkpoint_path = checkpoints[0]
    print(f"Found checkpoint: {checkpoint_path.name}")

    # --- 3. Run Inference ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    data_config = DataConfig()
    model = load_model(model_name, str(checkpoint_path), device)
    test_transform = get_transforms(model_name, data_config.mean, data_config.std)

    # Load test data from config
    test_df = pd.read_csv(data_config.data_csv) # Using the main data CSV as a proxy for test data structure if needed, adjust if you have a separate test CSV
    test_df['root_dir'] = data_config.data_root
    
    # You might need to adjust this part if your test set is specified differently
    # For now, we assume the test set logic from your config is what's used.
    from code.config import get_inference_config
    inference_config = get_inference_config()['inference']
    df_parts = []
    for csv_p, root_d in zip(inference_config.test_csv_paths, inference_config.test_root_dirs):
        temp_df = pd.read_csv(csv_p)
        temp_df['root_dir'] = root_d
        df_parts.append(temp_df)
    test_df = pd.concat(df_parts, ignore_index=True)
    test_df['PAI_0_indexed'] = test_df['PAI'] - 1

    test_dataset = CustomDataset(test_df, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    
    predictions_df = run_batch_inference(model, test_loader, device)
    
    # --- 4. Calculate Metrics ---
    true_labels = predictions_df['True_Label_0_Indexed']
    pred_labels = predictions_df['Predicted_Label_0_Indexed']
    
    test_metrics = {}
    test_metrics['Test QWK'] = cohen_kappa_score(true_labels + 1, pred_labels + 1, weights='quadratic')
    test_metrics['Test Accuracy'] = accuracy_score(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, output_dict=True)
    test_metrics['Test F1 (Weighted)'] = report['weighted avg']['f1-score']
    
    for i in range(5): # PAI 1 to 5
        label_str = str(i)
        if label_str in report:
            test_metrics[f'Test Sens PAI {i+1}'] = report[label_str]['recall']
        else:
            test_metrics[f'Test Sens PAI {i+1}'] = 0.0

    print("\n--- TEST SET METRICS ---")
    for key, val in test_metrics.items():
        print(f"{key}: {val:.4f}")

    # --- 5. Parse Logs for Other Data ---
    log_file = exp_path / "training.log"
    if not log_file.exists():
        print(f"WARNING: training.log not found. CSV line will be incomplete.")
        log_content = ""
    else:
        log_content = log_file.read_text()
        
    hyperparams = extract_hyperparameters_from_log(log_content)
    train_metrics = extract_training_metrics_from_log(log_content)
    duration_metrics = extract_training_duration(log_content)

    # --- 6. Assemble and Print CSV Line ---
    all_data = {
        'Model': model_name,
        'Experiment': exp_path.parent.name,
        'Timestamp': exp_path.name,
        **test_metrics,
        **hyperparams,
        **train_metrics,
        **duration_metrics
    }

    # Define the exact column order to match your master CSV
    column_order = [
        'Model', 'Experiment', 'Test QWK', 'Test Accuracy', 'Test F1 (Weighted)',
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
    
    csv_row_list = []
    for col in column_order:
        value = all_data.get(col, '') # Get value or empty string if not found
        # Format floating point numbers to 4 decimal places
        if isinstance(value, float):
            value = f"{value:.4f}"
        csv_row_list.append(str(value))
        
    csv_row = ",".join(csv_row_list)

    print("\n" + "="*80)
    print("COPY AND PASTE THE LINE BELOW INTO YOUR CSV FILE:")
    print("="*80 + "\n")
    print(csv_row)
    print("\n" + "="*80)

if __name__ == '__main__':
    main()
