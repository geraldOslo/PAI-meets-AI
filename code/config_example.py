"""
Configuration management for PAI classification training

IMPORTANT: This is a template file!
1. Copy this file to config.py: cp config.example.py config.py
2. Update the paths in DataConfig for your environment
3. Never commit config.py (it's in .gitignore)

Author: Gerald Torgersen
SPDX-License-Identifier: MIT
Copyright (c) 2025 Gerald Torgersen
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    
    # Paths - UPDATE THESE FOR YOUR ENVIRONMENT
    data_csv: str = "/path/to/your/data.csv"           # ← UPDATE: Path to CSV with PAI labels
    data_root: str = "/path/to/your/images"            # ← UPDATE: Root directory for images
    output_dir: str = "/path/to/your/experiments"      # ← UPDATE: Where to save results
    
    # Data split
    val_split: float = 0.2          # Validation split ratio (20%)
    test_split: float = 0.0         # Test split (not used in simple split)
    random_seed: int = 42           # Random seed for reproducibility
    
    # Image preprocessing - UPDATED BY calculate_dataset_statistics.py
    # These are placeholder values - run calculate_dataset_statistics.py to get actual values
    mean: List[float] = None        # Will be set in __post_init__ if None
    std: List[float] = None         # Will be set in __post_init__ if None
    
    def __post_init__(self):
        """Set default normalization values if not provided."""
        if self.mean is None:
            # These are example values - replace with your dataset statistics
            # Run: python code/data_utils/calculate_dataset_statistics.py
            self.mean = [0.378, 0.378, 0.378]  # Example: ~0.378 for 8-bit images
        if self.std is None:
            self.std = [0.167, 0.167, 0.167]   # Example: ~0.167 for 8-bit images


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    
    name: str                       # Display name (e.g., "ResNet50")
    timm_name: str                  # timm model identifier
    input_size: int                 # Input image size (224 or 300)
    batch_size: int                 # Batch size (optimized for A100 40GB)
    dropout: float = 0.5            # Dropout rate
    drop_path: float = 0.2          # Drop path rate (for regularization)
    num_classes: int = 5            # PAI classes (1-5)
    pretrained: bool = True         # Use ImageNet pretrained weights
    lr: Optional[float] = None      # Model-specific learning rate
    weight_decay: Optional[float] = None # Model-specific weight decay


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""
    
    # Optimizer
    optimizer: str = "AdamW"        # Optimizer type
    base_lr: float = 3e-4           # Base learning rate
    weight_decay: float = 5e-3      # Weight decay (L2 regularization)
    
    # Scheduler
    scheduler: str = "OneCycleLR"   # Learning rate scheduler
    max_lr: float = 3e-3            # Maximum learning rate (10× base)
    pct_start: float = 0.3          # Warmup percentage
    
    # Loss
    loss_type: str = "FocalLoss"    # Loss function (FocalLoss or CrossEntropy)
    focal_alpha: float = 0.25       # Focal loss alpha parameter
    focal_gamma: float = 2.0        # Focal loss gamma parameter
    label_smoothing: float = 0.1    # Label smoothing (for CrossEntropy)
    
    # Training schedule
    epochs: int = 50                # Maximum epochs
    patience: int = 15              # Early stopping patience
    min_delta: float = 0.001        # Minimum improvement for early stopping
    
    # Regularization
    grad_clip: float = 1.0          # Gradient clipping norm
    use_amp: bool = True            # Use automatic mixed precision
    
    # Data loading
    num_workers: int = 4            # Number of data loading workers
    pin_memory: bool = True         # Pin memory for faster GPU transfer
    
    # Class imbalance handling
    use_oversampling: bool = True   # Use weighted random oversampling
    use_class_weights: bool = False # Use class weights in loss (alternative)


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    
    # Training augmentation
    random_flip: bool = True        # Random horizontal flip
    flip_prob: float = 0.5          # Flip probability
    
    random_rotation: bool = True    # Random rotation
    rotation_degrees: float = 10.0  # Maximum rotation angle
    
    color_jitter: bool = True       # Random color jittering
    brightness: float = 0.3         # Brightness jitter factor
    contrast: float = 0.3           # Contrast jitter factor
    
    random_affine: bool = True      # Random affine transformation
    translate: float = 0.04         # Translation factor
    scale_min: float = 0.92         # Minimum scale factor
    scale_max: float = 1.08         # Maximum scale factor
    
    # Advanced augmentation (disabled by default)
    random_erasing: bool = False    # Random erasing
    cutout: bool = False            # Cutout augmentation
    mixup: bool = False             # Mixup augmentation
    cutmix: bool = False            # CutMix augmentation


# Predefined model configurations optimized for A100 40GB
MODEL_CONFIGS = {
    'resnet50': ModelConfig(
        name='ResNet50',
        timm_name='resnet50',
        input_size=224,
        batch_size=64,           # Optimized for A100
        dropout=0.5,
        drop_path=0.1
    ),
    'efficientnet_b3': ModelConfig(
        name='EfficientNet-B3',
        timm_name='efficientnet_b3',
        input_size=300,          # EfficientNet uses 300×300
        batch_size=64,           # Lower batch size due to larger input
        dropout=0.5,
        drop_path=0.2
    ),
    'convnext_tiny': ModelConfig(
        name='ConvNeXt-Tiny',
        timm_name='convnext_tiny',
        input_size=224,
        batch_size=48,           # Balanced for A100
        dropout=0.5,
        drop_path=0.2,
        lr=1e-4,
        weight_decay=1e-3
    )
}


@dataclass
class InferenceConfig:
    """Configuration for model testing and inference."""
    base_experiments_dir: str = "/path/to/your/experiments"
    test_csv_paths: List[str] = field(default_factory=lambda: [
        "/path/to/test_data_1.csv",
        "/path/to/test_data_2.csv"
    ])

    test_root_dirs: List[str] = field(default_factory=lambda: [
        "/path/to/test_images_1",
        "/path/to/test_images_2"
    ])
    active_models: List[str] = field(default_factory=lambda: [
        'resnet50', 'efficientnet_b3', 'convnext_tiny',
    ])
    model_timestamps: Dict[str, str] = field(default_factory=lambda: {
        'resnet50': 'YYYYMMDD_HHMMSS',
        'efficientnet_b3': 'YYYYMMDD_HHMMSS',
        'convnext_tiny': 'YYYYMMDD_HHMMSS',
    })
    checkpoint_names: Dict[str, str] = field(default_factory=lambda: {
        'resnet50': 'resnet50_best.pth',
        'efficientnet_b3': 'efficientnet-b3_best.pth',
        'convnext_tiny': 'convnext-tiny_best.pth',
    })
    cam_method: str = 'gradcam++'
    heatmap_transparency: float = 0.5

    def get_checkpoint_path(self, model_name: str) -> str:
        timestamp = self.model_timestamps.get(model_name, '')
        checkpoint_name = self.checkpoint_names.get(model_name, f'{model_name}_best.pth')
        path = f"{self.base_experiments_dir}/{model_name}/pai_training_{timestamp}/{checkpoint_name}"
        return path

    def get_checkpoints(self) -> Dict[str, str]:
        return { model: self.get_checkpoint_path(model) for model in self.active_models }


def get_inference_config() -> Dict:
    return {'inference': InferenceConfig()}


def get_model_config(model_name: str) -> ModelConfig:
    """
    Get predefined model configuration.
    
    Args:
        model_name: Model identifier (resnet50, efficientnet_b3, convnext_tiny)
    
    Returns:
        ModelConfig instance
    
    Raises:
        ValueError: If model_name is not found
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Model '{model_name}' not found. "
            f"Available models: {list(MODEL_CONFIGS.keys())}"
        )
    return MODEL_CONFIGS[model_name]


def get_default_config() -> Dict:
    """
    Get default configuration for all components.
    
    Returns:
        Dictionary with data, training, and augmentation configs
    """
    return {
        'data': DataConfig(),
        'training': TrainingConfig(),
        'augmentation': AugmentationConfig()
    }


def print_config(config_dict: Dict = None) -> None:
    """
    Print current configuration in readable format.
    
    Args:
        config_dict: Dictionary of configs (if None, uses defaults)
    """
    if config_dict is None:
        config_dict = get_default_config()
    
    print("="*80)
    print("CONFIGURATION")
    print("="*80)
    
    for section, config in config_dict.items():
        print(f"\n{section.upper()} CONFIG:")
        print("-"*80)
        for key, value in vars(config).items():
            print(f"  {key}: {value}")
    
    print("="*80)


# Example usage
if __name__ == "__main__":
    print("PAI Classification Configuration Template")
    print("="*80)
    print("\nTo use this configuration:")
    print("1. Copy this file: cp config.example.py config.py")
    print("2. Update paths in DataConfig")
    print("3. Run calculate_dataset_statistics.py to get mean/std")
    print("4. Never commit config.py (it's in .gitignore)")
    print("\n" + "="*80)
    print("\nCurrent configuration (with placeholder paths):\n")
    
    # Print example configuration
    print_config()
    
    print("\n" + "="*80)
    print("Available models:")
    print("="*80)
    for model_name in MODEL_CONFIGS.keys():
        config = get_model_config(model_name)
        print(f"\n{config.name}:")
        print(f"  Input size: {config.input_size}×{config.input_size}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  timm name: {config.timm_name}")