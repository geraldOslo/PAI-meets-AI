#!/usr/bin/env python3
"""
Multi-Model Simple Split Training for PAI Classification with Complete Logging

This version ensures all hyperparameters and metrics are logged in a parseable format
for automated result summarization.

Author: Gerald Torgersen
SPDX-License-Identifier: MIT
Copyright (c) 2025 Gerald Torgersen
"""

import os
import sys
import json
import argparse
import datetime
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, SubsetRandomSampler
from torch.amp import GradScaler
from torchvision import transforms
import timm

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score, mean_absolute_error
from tqdm import tqdm

# Setup paths and imports
SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = CODE_DIR.parent

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

# Import training utilities
try:
    from training import data_utils
    from training import train_utils
    print("✓ Successfully imported training utilities", file=sys.stderr)
except ImportError as e:
    print(f"✗ ERROR importing training utilities: {e}", file=sys.stderr)
    sys.exit(1)

# Import configuration
try:
    from config import (
        DataConfig, ModelConfig, TrainingConfig, AugmentationConfig,
        MODEL_CONFIGS, get_model_config, get_default_config, print_config
    )
    print("✓ Successfully imported config", file=sys.stderr)
except ImportError as e:
    print(f"✗ ERROR importing config: {e}", file=sys.stderr)
    sys.exit(1)


def setup_logging(output_dir: str, experiment_name: str) -> Tuple[Path, Any]:
    """Set up logging for the experiment."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(output_dir) / f"{experiment_name}_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "training.log"
    
    class Logger:
        def __init__(self, filepath):
            self.terminal = sys.stdout
            self.log = open(filepath, "a")
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    logger = Logger(log_file)
    sys.stdout = logger
    
    return log_dir, logger


def print_hyperparameters_for_parsing(
    train_config: TrainingConfig, 
    aug_config: AugmentationConfig,
    model_config: ModelConfig
):
    """
    Print hyperparameters in a structured, parseable format.
    This section is designed to be easily parsed by summarize_and_infer.py
    """
    print("\n" + "="*80)
    print("HYPERPARAMETERS (for parsing)")
    print("="*80)
    
    # Learning rates
    print(f"base_lr: {train_config.base_lr}")
    print(f"max_lr: {train_config.max_lr}")
    
    # Regularization
    print(f"weight_decay: {train_config.weight_decay}")
    print(f"dropout: {model_config.dropout}")
    print(f"drop_path: {model_config.drop_path}")
    print(f"grad_clip: {train_config.grad_clip}")
    
    # Loss function parameters
    print(f"loss_type: {train_config.loss_type}")
    print(f"focal_alpha: {train_config.focal_alpha}")
    print(f"focal_gamma: {train_config.focal_gamma}")
    print(f"label_smoothing: {train_config.label_smoothing}")
    
    # Class imbalance handling
    print(f"use_class_weights: {train_config.use_class_weights}")
    print(f"class_weights: {train_config.class_weights}")
    print(f"use_oversampling: {train_config.use_oversampling}")
    
    # Model-specific
    print(f"batch_size: {model_config.batch_size}")
    print(f"input_size: {model_config.input_size}")
    
    # Augmentation
    print(f"mixup: {aug_config.mixup}")
    print(f"mixup_alpha: {aug_config.mixup_alpha if aug_config.mixup else 0.0}")
    print(f"cutmix: {aug_config.cutmix}")
    
    # Training schedule
    print(f"epochs: {train_config.epochs}")
    print(f"patience: {train_config.patience}")
    print(f"min_delta: {train_config.min_delta}")
    
    # Optimizer and scheduler
    print(f"optimizer: {train_config.optimizer}")
    print(f"scheduler: {train_config.scheduler}")
    print(f"pct_start: {train_config.pct_start}")
    
    # Technical
    print(f"use_amp: {train_config.use_amp}")
    print(f"num_workers: {train_config.num_workers}")
    
    print("="*80 + "\n")


def create_weighted_sampler(dataset, indices: List[int]) -> WeightedRandomSampler:
    """Create weighted random sampler for handling class imbalance."""
    labels = [dataset.get_target(i) for i in indices]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[np.array(labels)]
    
    print(f"\nClass distribution in training subset ({len(labels)} samples):")
    for i, count in enumerate(class_counts):
        print(f"  PAI {i+1}: {count} samples (weight: {class_weights[i]:.4f})")
    
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


def create_model(model_config: ModelConfig, num_classes: int = 5) -> nn.Module:
    """Create a model using timm library."""
    return timm.create_model(
        model_config.timm_name,
        pretrained=model_config.pretrained,
        num_classes=num_classes,
        drop_rate=model_config.dropout,
        drop_path_rate=model_config.drop_path
    )


def get_criterion(train_config: TrainingConfig, device: torch.device = None) -> nn.Module:
    """Get loss function based on configuration."""
    if train_config.loss_type == 'FocalLoss':
        criterion = train_utils.FocalLoss(alpha=train_config.focal_alpha, gamma=train_config.focal_gamma)
    elif train_config.use_class_weights and train_config.class_weights:
        print(f"Using WeightedCrossEntropyLoss with weights: {train_config.class_weights}")
        criterion = train_utils.WeightedCrossEntropyLoss(
            class_weights=train_config.class_weights,
            label_smoothing=train_config.label_smoothing
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=train_config.label_smoothing)
    
    if device:
        criterion = criterion.to(device)
    return criterion


def get_transforms(data_config, aug_config, model_config, training=True):
    """Create transforms based on configuration."""
    transform_list = [
        transforms.Resize(model_config.input_size),
        transforms.CenterCrop(model_config.input_size),
    ]
    
    if training:
        if aug_config.random_flip:
            transform_list.append(transforms.RandomHorizontalFlip(p=aug_config.flip_prob))
        if aug_config.random_rotation:
            transform_list.append(transforms.RandomRotation(aug_config.rotation_degrees))
        if aug_config.color_jitter:
            transform_list.append(transforms.ColorJitter(brightness=aug_config.brightness, contrast=aug_config.contrast))
        if aug_config.random_affine:
            transform_list.append(transforms.RandomAffine(
                degrees=0,
                translate=(aug_config.translate, aug_config.translate),
                scale=(aug_config.scale_min, aug_config.scale_max)
            ))
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=data_config.mean, std=data_config.std)
    ])
    
    return transforms.Compose(transform_list)


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, train_config, aug_config):
    """Train the model for one epoch."""
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    scaler = GradScaler(enabled=train_config.use_amp)
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for inputs, labels, _ in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        if aug_config.mixup:
            inputs, targets_a, targets_b, lam = train_utils.mixup_data(inputs, labels, alpha=aug_config.mixup_alpha)
        
        with torch.amp.autocast(device_type=device.type, enabled=train_config.use_amp):
            outputs = model(inputs)
            if aug_config.mixup:
                loss = train_utils.mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        if train_config.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()
        
        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return {'loss': total_loss / len(all_labels), 'accuracy': accuracy_score(all_labels, all_preds)}


@torch.no_grad()
def validate(model, val_loader, criterion, device, train_config):
    """Validate the model."""
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    
    for inputs, labels, _ in tqdm(val_loader, desc="Validation", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        with torch.amp.autocast(device_type=device.type, enabled=train_config.use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    labels_1idx, preds_1idx = np.array(all_labels) + 1, np.array(all_preds) + 1
    return {
        'loss': total_loss / len(all_labels),
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, average='weighted'),
        'qwk': cohen_kappa_score(labels_1idx, preds_1idx, weights='quadratic'),
        'mae': mean_absolute_error(labels_1idx, preds_1idx)
    }


def train_model(model_config, data_config, train_config, aug_config, train_loader, val_loader, device, save_dir):
    """Train a single model with complete timing tracking."""
    print(f"\n{'='*80}\nTraining {model_config.name}\n{'='*80}")
    
    # Print all hyperparameters in parseable format
    print_hyperparameters_for_parsing(train_config, aug_config, model_config)
    
    # Start timing
    training_start_time = time.time()
    
    model = create_model(model_config, num_classes=5).to(device)
    criterion = get_criterion(train_config, device)
    
    optimizer = optim.AdamW(model.parameters(), lr=train_config.base_lr, weight_decay=train_config.weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=train_config.max_lr,
        epochs=train_config.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=train_config.pct_start
    )
    
    best_qwk, patience_counter = -1.0, 0
    best_metrics = {}
    history = {'train_loss': [], 'val_loss': [], 'val_qwk': [], 'val_f1': []}
    
    for epoch in range(train_config.epochs):
        print(f"\nEpoch {epoch + 1}/{train_config.epochs}")
        
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, train_config, aug_config)
        val_metrics = validate(model, val_loader, criterion, device, train_config)
        
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_qwk'].append(val_metrics['qwk'])
        history['val_f1'].append(val_metrics['f1'])
        
        print(f"Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
              f"Val QWK: {val_metrics['qwk']:.4f} | Val F1: {val_metrics['f1']:.4f}")
        
        if val_metrics['qwk'] > best_qwk + train_config.min_delta:
            best_qwk = val_metrics['qwk']
            best_metrics = val_metrics.copy()
            best_metrics['epoch'] = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), save_dir / f'{model_config.name.lower().replace(" ", "_")}_best.pth')
            print(f"New best QWK: {best_qwk:.4f}. Checkpoint saved.")
        else:
            patience_counter += 1
            if patience_counter >= train_config.patience:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break
    
    # End timing and calculate duration
    training_end_time = time.time()
    training_duration_seconds = training_end_time - training_start_time
    training_duration_minutes = training_duration_seconds / 60
    training_duration_hours = training_duration_minutes / 60
    
    # Print duration in parseable format
    print(f"\n{'='*80}")
    print("TRAINING DURATION (for parsing)")
    print(f"{'='*80}")
    print(f"Training Duration (seconds): {training_duration_seconds:.1f}")
    print(f"Training Duration (minutes): {training_duration_minutes:.1f}")
    print(f"Training Duration (hours): {training_duration_hours:.2f}")
    print(f"Elapsed wallclock time: {training_duration_minutes:.1f} minutes")
    print(f"Elapsed wallclock time: {training_duration_hours:.2f} hours")
    print(f"{'='*80}\n")
    
    # Save history
    with open(save_dir / f'{model_config.name.lower().replace(" ", "_")}_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    return {
        'model': model_config.name,
        'best_metrics': best_metrics,
        'history': history,
        'training_duration_seconds': training_duration_seconds,
        'training_duration_minutes': training_duration_minutes,
        'training_duration_hours': training_duration_hours
    }


def main():
    """Main training function."""
    start_time = datetime.datetime.now()
    
    print("="*80)
    print("PAI Classification Training")
    print(f"Job Started: {start_time.strftime('%a %b %d %I:%M:%S %p %Z %Y')}")
    print("="*80)
    print(f"Script location: {SCRIPT_DIR}")
    print(f"Code directory: {CODE_DIR}")
    print(f"Project root: {PROJECT_ROOT}")
    print("="*80)
    
    default_config = get_default_config()
    data_config = default_config['data']
    train_config = default_config['training']
    aug_config = default_config['augmentation']
    
    parser = argparse.ArgumentParser(description='Multi-Model PAI Training')
    parser.add_argument('--data_csv', type=str, default=data_config.data_csv)
    parser.add_argument('--data_root', type=str, default=data_config.data_root)
    parser.add_argument('--output_dir', type=str, default=data_config.output_dir)
    parser.add_argument('--models', nargs='+', default=['efficientnet_b3'], choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--epochs', type=int, default=train_config.epochs)
    parser.add_argument('--patience', type=int, default=train_config.patience)
    parser.add_argument('--use_oversampling', action='store_true', default=train_config.use_oversampling)
    parser.add_argument('--no-oversampling', action='store_false', dest='use_oversampling')
    parser.add_argument('--random_seed', type=int, default=data_config.random_seed)
    parser.add_argument('--base_lr', type=float, default=None)
    parser.add_argument('--max_lr', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--focal_gamma', type=float, default=None, help='Focal loss gamma parameter (default: 2.0)')
    parser.add_argument('--focal_alpha', type=float, default=None, help='Focal loss alpha parameter (default: 0.25)')
    parser.add_argument('--label_smoothing', type=float, default=None, help='Label smoothing factor (0.0-1.0)')
    parser.add_argument('--use_class_weights', action='store_true', default=None)
    parser.add_argument('--mixup', action='store_true', default=False)
    parser.add_argument('--mixup_alpha', type=float, default=0.2)
    parser.add_argument('--experiment_name', type=str, default='pai_training', help='Name for this experiment (used in output directory)')
    parser.add_argument('--print_config', action='store_true', help='Print configuration and exit')
    
    args = parser.parse_args()
    
    if args.print_config:
        print_config({'data': data_config, 'training': train_config, 'augmentation': aug_config})
        return
    
    # Update configs from args
    data_config.data_csv = args.data_csv
    data_config.data_root = args.data_root
    data_config.output_dir = args.output_dir
    data_config.random_seed = args.random_seed
    train_config.epochs = args.epochs
    train_config.patience = args.patience
    train_config.use_oversampling = args.use_oversampling
    
    if args.base_lr is not None: train_config.base_lr = args.base_lr
    if args.max_lr is not None: train_config.max_lr = args.max_lr
    if args.weight_decay is not None: train_config.weight_decay = args.weight_decay
    if args.focal_gamma is not None: train_config.focal_gamma = args.focal_gamma
    if args.focal_alpha is not None: train_config.focal_alpha = args.focal_alpha
    if args.label_smoothing is not None: train_config.label_smoothing = args.label_smoothing
    if args.use_class_weights is not None: train_config.use_class_weights = args.use_class_weights
    
    aug_config.mixup = args.mixup
    aug_config.mixup_alpha = args.mixup_alpha
    
    # Set random seeds
    np.random.seed(data_config.random_seed)
    torch.manual_seed(data_config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(data_config.random_seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_dir, _ = setup_logging(args.output_dir, args.experiment_name)
    
    print(f"\nStarting PAI Training")
    print(f"Device: {device}")
    print(f"Models: {args.models}")
    print(f"Output: {log_dir}")
    
    print(f"\n{'='*80}")
    print(f"Loading dataset from {data_config.data_csv}")
    
    all_results = []
    
    try:
        data = pd.read_csv(data_config.data_csv)
        data['root_dir'] = data_config.data_root

        print(f"Total samples loaded: {len(data)}")

        # Convert PAI to numeric in case it's stored as string
        data['PAI'] = pd.to_numeric(data['PAI'], errors='coerce')

        # Check for and remove invalid PAI values
        invalid_mask = data['PAI'].isna()
        if invalid_mask.any():
            n_invalid = invalid_mask.sum()
            print(f"WARNING: Found {n_invalid} rows with invalid PAI values")
            if n_invalid <= 10:  # Show details if not too many
                print("Invalid rows:")
                print(data[invalid_mask][['filename', 'PAI']].to_string() if 'filename' in data.columns else data[invalid_mask])
            data = data[~invalid_mask].copy()
            print(f"Dropped {n_invalid} invalid rows. Remaining: {len(data)}")

        # Validate PAI values are in expected range (1-5)
        invalid_range = (data['PAI'] < 1) | (data['PAI'] > 5)
        if invalid_range.any():
            n_invalid_range = invalid_range.sum()
            print(f"WARNING: Found {n_invalid_range} rows with PAI values outside range [1-5]")
            print(f"PAI value range: {data['PAI'].min()} to {data['PAI'].max()}")
            data = data[~invalid_range].copy()
            print(f"Dropped {n_invalid_range} out-of-range rows. Remaining: {len(data)}")

        data['PAI_0_indexed'] = data['PAI'] - 1

        print(f"Valid samples for training: {len(data)}")
        print(f"PAI distribution:\n{data['PAI'].value_counts().sort_index()}")
        
        train_idx, val_idx = train_test_split(
            np.arange(len(data)),
            test_size=data_config.val_split,
            stratify=data['PAI'],
            random_state=data_config.random_seed
        )
        
        print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
        
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        traceback.print_exc()
        return
    
    # Train each model
    for model_name in args.models:
        model_config = get_model_config(model_name)
        
        if args.dropout is not None:
            model_config.dropout = args.dropout
        
        print(f"\n{'='*80}")
        print(f"Preparing {model_config.name}")
        print(f"{'='*80}")
        
        train_transforms = get_transforms(data_config, aug_config, model_config, training=True)
        val_transforms = get_transforms(data_config, aug_config, model_config, training=False)
        
        train_dataset = data_utils.CustomDataset(data, transform=train_transforms)
        val_dataset = data_utils.CustomDataset(data, transform=val_transforms)
        
        train_sampler = create_weighted_sampler(train_dataset, train_idx) if train_config.use_oversampling else SubsetRandomSampler(train_idx)
        train_loader = DataLoader(train_dataset, batch_size=model_config.batch_size, sampler=train_sampler, num_workers=train_config.num_workers, pin_memory=train_config.pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=model_config.batch_size * 2, sampler=SubsetRandomSampler(val_idx), num_workers=train_config.num_workers, pin_memory=train_config.pin_memory)
        
        try:
            result = train_model(model_config, data_config, train_config, aug_config, train_loader, val_loader, device, log_dir)
            all_results.append(result)
        except Exception as e:
            print(f"ERROR training {model_name}: {e}")
            traceback.print_exc()
            continue
    
    # Save comprehensive summary
    summary = {
        'experiment_date': datetime.datetime.now().isoformat(),
        'dataset_size': len(data),
        'train_size': len(train_idx),
        'val_size': len(val_idx),
        'config': {
            'mean': data_config.mean,
            'std': data_config.std,
            'val_split': data_config.val_split,
            'oversampling': train_config.use_oversampling,
            'epochs': train_config.epochs,
            'patience': train_config.patience
        },
        'models': all_results
    }
    
    with open(log_dir / 'experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    if all_results:
        print(f"{'Model':<20} {'Best QWK':<12} {'Best F1':<12} {'Best Acc':<12} {'Epoch':<8}")
        print(f"{'-'*80}")
        for result in all_results:
            metrics = result['best_metrics']
            print(f"{result['model']:<20} "
                  f"{metrics.get('qwk', 0):<12.4f} "
                  f"{metrics.get('f1', 0):<12.4f} "
                  f"{metrics.get('accuracy', 0):<12.4f} "
                  f"{metrics.get('epoch', 0):<8d}")
        print(f"{'='*80}")
    
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    
    print(f"\nExperiment complete. Results saved to: {log_dir}")
    print(f"Job Finished: {end_time.strftime('%a %b %d %I:%M:%S %p %Z %Y')}")
    print(f"Total training time: {duration}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)