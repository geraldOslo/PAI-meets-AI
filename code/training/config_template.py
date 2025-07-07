# config_template.py
"""
Configuration template for the PAI Classification Deep Learning System.

This file centralizes all adjustable parameters for data paths, model architecture,
training hyperparameters, and data augmentation. Users should copy this file
to `config.py` in the project root and customize it to match their local environment
and experimental setup.

Author: Gerald Torgersen
SPDX-License-Identifier: MIT
Copyright (c) 2025 Gerald Torgersen, UiO
Contact: https://www.odont.uio.no/iko/english/people/aca/gerald/
"""

import os
import sys

# ============================================================================
# PROJECT DIRECTORY STRUCTURE
# ============================================================================
# Define paths relative to the project root for portability.
# It's assumed this 'config.py' (or 'config_template.py') file will reside
# directly in the main project root directory (e.g., PAI-meets-AI/).
#
# If you place this file elsewhere, adjust 'os.path.abspath(__file__)' accordingly
# to correctly point to your 'PAI-meets-AI' directory.
try:
    # Path to the directory containing this config file
    CURRENT_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback for interactive environments like Jupyter notebooks if __file__ is not defined.
    # Assumes the notebook is launched from the project root.
    CURRENT_CONFIG_DIR = os.getcwd()
    print(f"Warning: Could not determine __file__, using current working directory as config base: {CURRENT_CONFIG_DIR}")

# The root directory of your PAI-meets-AI project.
# This is usually the directory containing 'README.md', 'code/', 'model_checkpoints/', etc.
PROJECT_ROOT = os.path.abspath(CURRENT_CONFIG_DIR)

# Directory where trained models checkpoints, logs, and results will be saved.
# This will be created automatically if it doesn't exist.
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "model_checkpoints")


# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
# Specify paths to your dataset(s).
# Images should be organized relative to the 'root_dir' within the CSV file.
# PAI labels in CSVs are expected to be 1-indexed (1-5) and will be
# automatically converted to 0-indexed (0-4) during loading.
DATA_PATHS = {
    # List of root directories where your training/validation images are stored.
    # If images are spread across multiple locations, list them here.
    # Example: ["/path/to/your/images/dataset_A", "/path/to/your/images/dataset_B"]
    "root_dirs": ["/path/to/your/dataset/clips300"],

    # List of CSV files containing metadata (image filenames and PAI labels)
    # for the training and validation sets.
    # Example: ["/path/to/your/dataset/data.csv"]
    "csv_files": ["/path/to/your/dataset/data.csv"],

    # --- Optional: Separate Test Set Configuration ---
    # If you have a completely separate, held-out test dataset, specify its paths here.
    # If left as None or empty, the final evaluation step in training.ipynb will be skipped.
    "test_root_dir": "/path/to/your/test_dataset/clips300",
    "test_csv_file": "/path/to/your/test_dataset/test_data.csv",

    # --- Optional: Custom Column Names in CSV ---
    # If your CSV files use different column names for filenames or PAI labels,
    # specify them here. Otherwise, 'filename' and 'PAI' are assumed.
    # "filename_col": "image_name",
    # "label_col": "PAI_Score_Original",
}

# ============================================================================
# MODEL ARCHITECTURE AND HYPERPARAMETERS
# ============================================================================
# Dictionary mapping common EfficientNet model names to their recommended input image sizes.
# If using a different model, ensure the 'input_size' matches its expected input.
INPUT_SIZES = {
    "efficientnet_b0": 224,
    "efficientnet_b1": 240,
    "efficientnet_b2": 260,
    "efficientnet_b3": 300,
    "efficientnet_b4": 380,
    "efficientnet_b5": 456,
    "efficientnet_b6": 528,
    "efficientnet_b7": 600,
    "efficientnet_v2_s": 384, # Example: larger input for EfficientNetV2 models
    "efficientnet_v2_m": 480,
    "efficientnet_v2_l": 480
}

MODEL_CONFIG = {
    # --- Model Architecture Selection ---
    "model": "efficientnet_b3",  # Choose base model (e.g., 'efficientnet_b3', 'resnet50', 'efficientnet_v2_s')
    "num_classes": 5,            # Number of output categories for PAI classification (PAI 1-5 maps to 0-4)
    "dropout": 0.5,              # Dropout rate for the final classifier head (e.g., 0.5 for fine-tuning)
    "drop_path_rate": 0.1,       # Stochastic depth rate for the backbone (e.g., 0.1-0.2 for larger models)
                                 # Higher value means more aggressive regularization.

    # --- Fine-tuning Strategy ---
    "finetune_blocks": 6,        # Controls which parts of the pre-trained backbone are trainable:
                                 # -1: All model parameters (backbone + classifier) are trainable (full fine-tuning).
                                 # 0: Only the final classification head is trainable; backbone is frozen (feature extractor).
                                 # >0: Classifier + the last N blocks of the backbone are trainable; earlier blocks frozen.
                                 # (e.g., 6 for EfficientNet-B3 fine-tunes blocks 1-6 + classifier if total blocks is 7)

    # --- Core Training Hyperparameters ---
    "epochs": 90,                # Maximum number of training epochs.
    "patience": 20,              # Early stopping patience: number of epochs without validation F1 improvement.
    "min_delta": 0.001,          # Minimum change in validation F1 score to qualify as an improvement.
    "base_lr": 3e-4,             # Base learning rate for the optimizer (typically applied to backbone layers).
    "classifier_lr_multiplier": 5.0, # Multiplier for the classifier head's learning rate relative to 'base_lr'.
    "fast_group_lr_multiplier": 3.0, # Multiplier for the learning rate of recently unfrozen backbone blocks (e.g., last N blocks).
                                     # These blocks learn faster than 'base_lr' but slower than the classifier.
    "weight_decay": 5e-3,        # L2 regularization strength (e.g., 1e-4 to 5e-3) to prevent overfitting.
    "grad_clip_max_norm": 1.0,   # Gradient clipping threshold (e.g., 1.0-5.0) to prevent exploding gradients.
                                 # Set to None or 0 to disable.
    "use_amp": True,             # Enable Automatic Mixed Precision (AMP) training for faster training and reduced
                                 # GPU memory usage on compatible GPUs (highly recommended).

    # --- Data Balancing Strategy ---
    "use_oversampler": True,     # If True, `WeightedRandomSampler` will be used for the training DataLoader
                                 # to balance class distribution by oversampling minority classes.

    # --- Learning Rate Scheduler Parameters (OneCycleLR) ---
    "scheduler_pct_start": 0.3,  # Percentage of total training steps spent in the LR warmup phase.
                                 # (e.g., 0.3 means 30% of total steps are for increasing LR, then 70% for decreasing).
    "scheduler_max_lr_multiplier": 10.0, # Factor by which the base_lr (for each group) is multiplied to reach its peak.
    "warmup_epochs": 0,          # Number of epochs for an *optional, manual* linear LR warmup.
                                 # If `scheduler_pct_start` > 0 for OneCycleLR, this is generally redundant and can be 0.

    # --- Loss Function Configuration ---
    "loss_function_type": "FocalLoss", # Options: "CrossEntropyLoss", "FocalLoss".
                                       # Focal Loss is often good for imbalanced datasets.
    "label_smoothing": 0.1,          # Label smoothing factor (0.0 to 1.0), used only with "CrossEntropyLoss".
                                     # Helps prevent overfitting and improves calibration.
    "focal_loss_alpha": 0.25,        # Alpha parameter for "FocalLoss" (controls weighting of pos/neg examples).
    "focal_loss_gamma": 2.0,         # Gamma parameter for "FocalLoss" (controls focusing on hard examples).
    "use_class_weights": False,      # Set to True to apply manual `class_weights` to the loss function.
    "class_weights": [1.0, 1.0, 1.0, 1.0, 1.0], # Per-class weights for the loss (e.g., [1.0, 2.0, 3.0, 4.0, 5.0]).
                                     # Used if 'use_class_weights' is True.

    # --- Mixup Data Augmentation ---
    "mixup": False,              # Enable Mixup augmentation. If True, images and labels are linearly interpolated.
    "mixup_alpha": 0.2,          # Alpha parameter for Mixup's Beta distribution (controls interpolation strength).
                                 # Common values are 0.1 to 0.4. Set to 0 to effectively disable Mixup.
    "mixup_prob": 0.5,           # Probability that Mixup is applied to any given batch during training.
}

# Dynamically set the input image size based on the chosen model.
# This ensures consistency between the model and data preprocessing.
MODEL_CONFIG["input_size"] = INPUT_SIZES.get(MODEL_CONFIG["model"], 300) # Default to 300 if model not in dict

# ============================================================================
# DATALOADER CONFIGURATION
# ============================================================================
# Settings for PyTorch's DataLoader, controlling how data is batched and loaded.
DATALOADER_CONFIG = {
    "batch_size": 64,           # Number of samples per batch for training.
                                # Adjust based on GPU memory. Lower batch_size can be compensated
                                # by increasing 'accum_steps'.
    "accum_steps": 4,           # Number of gradient accumulation steps.
                                # Effective batch size = batch_size * accum_steps.
                                # Allows using larger effective batch sizes than GPU memory permits.
    "num_workers": 8,           # Number of subprocesses to use for data loading.
                                # Set to 0 for single-process loading (for debugging).
                                # Optimal value depends on CPU cores and dataset I/O speed.
    "prefetch_factor": 2,       # Number of batches loaded in advance by each worker.
                                # Set to None for default. Increase for faster data loading if CPU is bottleneck.
    "pin_memory": True,         # If True, copies Tensors into CUDA pinned memory before returning them.
                                # Speeds up host-to-device (CPU to GPU) data transfer.
    "persistent_workers": True, # If True, data loader workers will not be shut down after an epoch,
                                # speeding up subsequent epochs. Requires num_workers > 0.
    # Separate batch sizes for validation/testing (can be larger as no gradients are computed)
    "val_batch_size": 256,      # Batch size for validation set (default: 4x training batch size)
    "test_batch_size": 256,     # Batch size for dedicated test set (default: 4x training batch size)
}

# ============================================================================
# DATA AUGMENTATION AND PREPROCESSING
# ============================================================================
# Configuration for torchvision transforms applied to images during training and validation.
# Refer to data_utils.py for implementation details of these transforms.
DATA_TRANSFORM_CONFIG = {
    # Input image size, derived from MODEL_CONFIG, but can be overridden here.
    "input_size": MODEL_CONFIG["input_size"],

    # --- Random Transformation Parameters ---
    "rotation_degrees": 15,       # Max rotation angle in degrees (+/-) for RandomRotation/RandomAffine.
    "brightness_jitter": 0.2,     # Max brightness adjustment factor (0-1) for ColorJitter.
    "contrast_jitter": 0.2,       # Max contrast adjustment factor (0-1) for ColorJitter.
    "saturation_jitter": 0.2,     # Max saturation adjustment factor (0-1) for ColorJitter.
    "hue_jitter": 0.1,            # Max hue adjustment factor (0-0.5) for ColorJitter.
    "random_crop_scale": (0.8, 1.0), # Scale range for RandomResizedCrop (fraction of original image area).
    "random_crop_ratio": (0.75, 1.33), # Aspect ratio range for RandomResizedCrop.
    "translate_range": (0.05, 0.05), # Max translation fraction (x, y) for RandomAffine (e.g., 0.05 means 5% of width/height).
    "scale_range": (0.9, 1.1),    # Scale range for RandomAffine (e.g., 0.9-1.1 means scale by 90% to 110%).
    "shear_range": (-5, 5),       # Shear range in degrees for RandomAffine (e.g., (-5, 5) for +/- 5 degrees).
    "gamma_range": (0.9, 1.1),    # Gamma correction range (factor) for custom GammaTransform.
    "blur_sigma_range": (0.1, 1.0), # Standard deviation range for GaussianBlur.

    # --- Random Erasing Settings ---
    "random_erase_probability": 0.0,  # Probability of applying RandomErasing to an image.
                                    # Set to 0.0 to disable.
    "random_erase_scale": (0.02, 0.15), # Range for the fraction of the erased area.
    "random_erase_ratio": (0.3, 3.3),   # Aspect ratio range for the erased area.

    # --- Flags for Enabling/Disabling Specific Transformations ---
    # Set to True/False to include or exclude a transformation from the pipeline.
    # Note: Some transforms are mutually exclusive for spatial augmentation (e.g., pick one primary crop/affine strategy).
    "use_horizontal_flip": True,       # Apply random horizontal flipping.
    "use_rotation": True,              # Apply random rotations (used with RandomRotation or RandomAffine).
    "use_color_jitter": True,          # Apply random changes to brightness, contrast, saturation, hue.
    "use_random_crop": False,          # Primary spatial augmentation: Apply RandomResizedCrop.
    "use_centered_affine_scaling": True, # Primary spatial augmentation: Apply Resize + RandomAffine + CenterCrop.
                                       # If both use_random_crop and use_centered_affine_scaling are True,
                                       # use_centered_affine_scaling will be prioritized (see data_utils.py).
    "use_affine": True,                # Enable translation, scaling, and/or shearing within RandomAffine.
                                       # Only active if 'use_centered_affine_scaling' is True.
    "use_gamma": True,                 # Apply random gamma correction (custom transform).
    "use_blur": False,                 # Apply random Gaussian blur.
    "use_random_erase": False          # Apply RandomErasing (after normalization).
}


# DATA NORMALIZATION
# ============================================================================
# Pixel normalization values (mean and standard deviation).
# These are typically computed from your specific dataset to transform pixel values
# towards zero-mean, unit-variance, which is crucial for stable training.
# Use the compute_dataset_statistics.py script (or similar) to calculate these from your data.
NORMALIZATION = {
    # For a grayscale-like dataset (where R, G, B channels are identical),
    # the mean and std will be the same for all three channels.
    # Replace these with your actual computed values.
    "mean": [0.X] * 3,  # Example: [0.378] * 3 for 3 channels
    "std": [0.Y] * 3    # Example: [0.167] * 3 for 3 channels
    
    # Or, for distinct RGB channel values:
    # "mean": [R_mean, G_mean, B_mean],
    # "std": [R_std, G_std, B_std]
}