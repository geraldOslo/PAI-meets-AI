"""
Training utilities for PAI classification

Author: Gerald Torgersen
SPDX-License-Identifier: MIT
Copyright (c) 2025 Gerald Torgersen
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (2017)
        https://arxiv.org/abs/1708.02002
    
    Args:
        alpha: Weighting factor in [0, 1] or list of weights per class
        gamma: Focusing parameter (gamma >= 0)
        reduction: 'none', 'mean', or 'sum'
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions (B, C) - raw logits
            targets: Ground truth labels (B,) - class indices
        
        Returns:
            Focal loss value
        """
        # Get probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Get probability of correct class
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1])
        pt = (probs * targets_one_hot).sum(dim=1)
        
        # Calculate focal term: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # Calculate cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        # Apply alpha weighting
        if self.alpha is not None:
            alpha_weight = self.alpha
            focal_loss = alpha_weight * focal_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Cross Entropy Loss with class weights.
    
    Useful for handling class imbalance by assigning higher weights
    to minority classes.
    
    Args:
        class_weights: List or tensor of weights per class
        label_smoothing: Label smoothing parameter
    """
    
    def __init__(
        self,
        class_weights: Optional[List[float]] = None,
        label_smoothing: float = 0.0
    ):
        super(WeightedCrossEntropyLoss, self).__init__()
        
        if class_weights is not None:
            self.register_buffer('weight', torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.weight = None
        
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            inputs,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing
        )


class OrdinalRegressionLoss(nn.Module):
    """
    Loss function for ordinal regression.
    
    Treats classification as multiple binary classification problems,
    preserving the ordinal nature of PAI scores.
    
    For PAI with 5 classes (1-5), we have 4 binary thresholds:
        - Is PAI > 1?
        - Is PAI > 2?
        - Is PAI > 3?
        - Is PAI > 4?
    
    Args:
        num_classes: Number of classes (5 for PAI)
    """
    
    def __init__(self, num_classes: int = 5):
        super(OrdinalRegressionLoss, self).__init__()
        self.num_classes = num_classes
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions (B, num_classes-1) - logits for each threshold
            targets: Ground truth labels (B,) - class indices (0-4)
        
        Returns:
            Ordinal regression loss
        """
        batch_size = inputs.size(0)
        
        # Create ordinal labels
        # For target class k, thresholds 0 to k-1 should be 1, rest should be 0
        ordinal_labels = torch.zeros(batch_size, self.num_classes - 1, device=inputs.device)
        for i in range(batch_size):
            if targets[i] > 0:
                ordinal_labels[i, :targets[i]] = 1
        
        # Calculate binary cross entropy for each threshold
        loss = F.binary_cross_entropy_with_logits(inputs, ordinal_labels)
        
        return loss


class EarlyStopping:
    """
    Early stopping utility to stop training when validation metric stops improving.
    
    Args:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
        mode: 'min' for loss, 'max' for accuracy/QWK
        verbose: Whether to print messages
    """
    
    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 0.001,
        mode: str = 'max',
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'min':
            self.compare = lambda current, best: current < best - min_delta
        else:
            self.compare = lambda current, best: current > best + min_delta
    
    def __call__(self, score: float) -> bool:
        """
        Check if should stop training.
        
        Args:
            score: Current validation metric
        
        Returns:
            True if should stop training
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.compare(score, self.best_score):
            self.best_score = score
            self.counter = 0
            if self.verbose:
                print(f"Validation metric improved to {score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement for {self.counter}/{self.patience} epochs")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered")
                return True
        
        return False
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class MetricTracker:
    """
    Utility for tracking training metrics across epochs.
    
    Args:
        metrics: List of metric names to track
    """
    
    def __init__(self, metrics: List[str]):
        self.metrics = metrics
        self.history = {metric: [] for metric in metrics}
    
    def update(self, **kwargs):
        """Update metrics with new values."""
        for metric, value in kwargs.items():
            if metric in self.history:
                self.history[metric].append(value)
    
    def get_last(self, metric: str) -> float:
        """Get last value for a metric."""
        if metric in self.history and len(self.history[metric]) > 0:
            return self.history[metric][-1]
        return None
    
    def get_best(self, metric: str, mode: str = 'max') -> float:
        """Get best value for a metric."""
        if metric not in self.history or len(self.history[metric]) == 0:
            return None
        
        if mode == 'max':
            return max(self.history[metric])
        else:
            return min(self.history[metric])
    
    def to_dict(self) -> dict:
        """Convert history to dictionary."""
        return self.history.copy()


def calculate_class_weights(
    labels: np.ndarray,
    num_classes: int = 5,
    method: str = 'inverse'
) -> torch.Tensor:
    """
    Calculate class weights for handling imbalance.
    
    Args:
        labels: Array of labels (0-indexed)
        num_classes: Number of classes
        method: 'inverse' or 'effective'
    
    Returns:
        Tensor of class weights
    """
    # Count samples per class
    class_counts = np.bincount(labels, minlength=num_classes)
    
    if method == 'inverse':
        # Inverse frequency
        weights = 1.0 / (class_counts + 1e-6)
        weights = weights / weights.sum() * num_classes
    
    elif method == 'effective':
        # Effective number of samples (from Class-Balanced Loss paper)
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / (effective_num + 1e-6)
        weights = weights / weights.sum() * num_classes
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return torch.tensor(weights, dtype=torch.float32)


def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2
) -> tuple:
    """
    Apply Mixup data augmentation.
    
    Reference:
        Zhang et al. "mixup: Beyond Empirical Risk Minimization" (2018)
    
    Args:
        x: Input batch (B, C, H, W)
        y: Labels (B,)
        alpha: Mixup parameter
    
    Returns:
        Tuple of (mixed_x, y_a, y_b, lambda)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float
) -> torch.Tensor:
    """
    Calculate loss for mixup.
    
    Args:
        criterion: Loss function
        pred: Predictions
        y_a: First set of labels
        y_b: Second set of labels
        lam: Mixup lambda
    
    Returns:
        Mixed loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
