"""
Model utilities for PAI classification

Author: Gerald Torgersen
SPDX-License-Identifier: MIT
Copyright (c) 2025 Gerald Torgersen
"""

import torch
import torch.nn as nn
import timm
from typing import Optional, List
from config import ModelConfig


def create_model(
    model_config: ModelConfig,
    pretrained: bool = True,
    checkpoint_path: Optional[str] = None
) -> nn.Module:
    """
    Create model from configuration.
    
    Args:
        model_config: Model configuration
        pretrained: Whether to use pretrained weights
        checkpoint_path: Path to checkpoint to load (optional)
    
    Returns:
        Model instance
    """
    # Create model using timm
    model = timm.create_model(
        model_config.timm_name,
        pretrained=pretrained,
        num_classes=model_config.num_classes,
        drop_rate=model_config.dropout,
        drop_path_rate=model_config.drop_path
    )
    
    # Load checkpoint if provided
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    return model


def get_model_info(model: nn.Module) -> dict:
    """
    Get information about model architecture.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'total_params_millions': total_params / 1e6,
        'trainable_params_millions': trainable_params / 1e6
    }


def freeze_layers(model: nn.Module, freeze_until: Optional[str] = None):
    """
    Freeze model layers up to a certain point.
    
    Args:
        model: PyTorch model
        freeze_until: Name of layer to freeze up to (None = freeze all)
    """
    freeze_all = freeze_until is None
    
    for name, param in model.named_parameters():
        if freeze_all or freeze_until not in name:
            param.requires_grad = False
        else:
            break


def unfreeze_layers(model: nn.Module):
    """Unfreeze all model layers."""
    for param in model.parameters():
        param.requires_grad = True


def get_layer_groups(model: nn.Module, model_type: str) -> List[nn.Module]:
    """
    Get layer groups for discriminative learning rates.
    
    Args:
        model: PyTorch model
        model_type: Type of model ('resnet', 'efficientnet', 'convnext')
    
    Returns:
        List of layer groups
    """
    if 'resnet' in model_type.lower():
        return [
            model.conv1,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.fc
        ]
    elif 'efficientnet' in model_type.lower():
        return [
            model.conv_stem,
            *model.blocks[:3],
            *model.blocks[3:6],
            *model.blocks[6:],
            model.classifier
        ]
    elif 'convnext' in model_type.lower():
        return [
            model.stem,
            model.stages[0],
            model.stages[1],
            model.stages[2],
            model.stages[3],
            model.head
        ]
    else:
        # Default: return model as single group
        return [model]


def get_gradcam_target_layer(model: nn.Module, model_type: str) -> nn.Module:
    """
    Get target layer for GradCAM visualization.
    
    Args:
        model: PyTorch model
        model_type: Type of model ('resnet', 'efficientnet', 'convnext')
    
    Returns:
        Target layer for GradCAM
    """
    if 'resnet' in model_type.lower():
        return model.layer4[-1]
    elif 'efficientnet' in model_type.lower():
        return model.blocks[-1]
    elif 'convnext' in model_type.lower():
        return model.stages[-1]
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters_by_layer(model: nn.Module) -> dict:
    """
    Count parameters in each layer.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary mapping layer names to parameter counts
    """
    param_counts = {}
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                param_counts[name] = params
    
    return param_counts


def model_surgery(
    model: nn.Module,
    num_classes: int,
    freeze_backbone: bool = False
) -> nn.Module:
    """
    Modify model for transfer learning.
    
    Args:
        model: Pretrained model
        num_classes: Number of output classes
        freeze_backbone: Whether to freeze backbone layers
    
    Returns:
        Modified model
    """
    # Identify classifier layer
    if hasattr(model, 'fc'):
        # ResNet-style
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'classifier'):
        # EfficientNet-style
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'head'):
        # ConvNeXt-style
        if isinstance(model.head, nn.Sequential):
            in_features = model.head[-1].in_features
            model.head[-1] = nn.Linear(in_features, num_classes)
        else:
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, num_classes)
    else:
        raise ValueError("Could not identify classifier layer")
    
    # Freeze backbone if requested
    if freeze_backbone:
        for name, param in model.named_parameters():
            # Don't freeze classifier
            if 'fc' not in name and 'classifier' not in name and 'head' not in name:
                param.requires_grad = False
    
    return model


class ModelEMA:
    """
    Exponential Moving Average of model parameters.
    
    Maintains a shadow copy of model weights that are updated as:
        shadow_weights = decay * shadow_weights + (1 - decay) * model_weights
    
    Often improves generalization and stability.
    
    Args:
        model: Model to track
        decay: EMA decay rate (typically 0.999 or 0.9999)
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow weights with current model weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply shadow weights to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original model weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> dict:
    """
    Load checkpoint with model, optimizer, and scheduler states.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
    
    Returns:
        Dictionary with checkpoint metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Return metadata
    metadata = {
        'epoch': checkpoint.get('epoch', 0),
        'best_metric': checkpoint.get('best_qwk', checkpoint.get('best_metric', 0)),
        'metrics': checkpoint.get('metrics', {})
    }
    
    return metadata


def save_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    metrics: dict,
    best_metric: float
):
    """
    Save checkpoint with model, optimizer, and scheduler states.
    
    Args:
        checkpoint_path: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer to save
        scheduler: Scheduler to save (optional)
        epoch: Current epoch
        metrics: Dictionary of metrics
        best_metric: Best metric value so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': best_metric,
        'best_qwk': best_metric,  # Legacy key
        'metrics': metrics
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, checkpoint_path)
