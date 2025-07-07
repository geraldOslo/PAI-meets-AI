# train_utils.py
"""
Utility functions for the PyTorch training pipeline, focusing on PAI classification.

This module provides a collection of functions and classes to streamline the
deep learning training process, including:
-   Customizable loss functions (e.g., Focal Loss).
-   Advanced optimizer and learning rate scheduler setup (e.g., AdamW with parameter grouping, OneCycleLR).
-   Helper functions for training stability (gradient clipping, learning rate warmup, gradual unfreezing).
-   Data augmentation techniques like MixUp.
-   Robust checkpoint management (saving and loading model/optimizer states).
-   Comprehensive logging, reporting, and summary generation in JSON and YAML formats.
-   Inference execution and detailed metric calculation (accuracy, F1, Kappa, MAE, binary metrics).

Author: Gerald Torgersen
SPDX-License-Identifier: MIT
Copyright (c) 2025 Gerald Torgersen, UiO
Contact: https://www.odont.uio.no/iko/english/people/aca/gerald/
"""

# Standard Library Imports
import os
import json
import math
import traceback
import datetime # Added for checkpoint timestamp
from collections import Counter # For counting class distributions

# Third-party Library Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import pandas as pd
import yaml # For saving configuration summaries
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_recall_fscore_support, cohen_kappa_score,
                             mean_absolute_error)
from tqdm.auto import tqdm # Automatically selects notebook-friendly or console-friendly tqdm

# Type Hinting Imports
from typing import Dict, List, Optional, Tuple, Union, Any # Added Any

# ============================================================================
# Constants (for colored terminal output)
# ============================================================================
# Define ANSI escape codes for colored console output to improve log readability.
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_GREEN = "\033[92m"
COLOR_BLUE = "\033[94m"
COLOR_RESET = "\033[0m"

# ============================================================================
# Loss Functions
# ============================================================================

class FocalLoss(nn.Module):
    """
    Implements a custom Focal Loss for classification tasks,
    designed to down-weight easy examples and focus training on hard ones.

    References:
        T. Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
        (https://arxiv.org/abs/1708.02002)

    Attributes:
        alpha (float): Weighting factor for the loss, balancing positive and negative samples.
                       Typically a scalar (e.g., 0.25). Defaults to 1.0 (no explicit balancing).
        gamma (float): Focusing parameter (>= 0). Higher values increase the focus on
                       hard-to-classify examples. Defaults to 2.0.
        class_weights (torch.Tensor, optional): A 1D tensor of shape (num_classes,)
                                                 containing weights for each class.
                                                 These weights are applied *inside* the
                                                 CrossEntropyLoss calculation before
                                                 the focal term. Defaults to None.
        epsilon (float): A small value added for numerical stability to prevent log(0) issues.
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, class_weights: Optional[torch.Tensor] = None):
        """
        Initializes the FocalLoss module.

        Parameters
        ----------
        alpha : float, optional
            The alpha parameter for Focal Loss, controls the weighting of positive vs. negative examples.
            Defaults to 1.0.
        gamma : float, optional
            The gamma parameter for Focal Loss, controls the focusing of training on hard examples.
            Defaults to 2.0.
        class_weights : torch.Tensor, optional
            A tensor of per-class weights. These weights are passed directly to `F.cross_entropy`.
            Defaults to None.

        Raises
        ------
        ValueError
            If `gamma` is negative.
        """
        super().__init__()
        if gamma < 0:
            raise ValueError("Focal Loss gamma must be non-negative.")
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights # Weights applied inside CE calculation
        self.epsilon = 1e-8 # Small value for numerical stability

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Focal Loss for a batch of inputs and targets.

        Parameters
        ----------
        inputs : torch.Tensor
            Logits (raw predictions) from the model, of shape (batch_size, num_classes).
        targets : torch.Tensor
            Ground truth labels, of shape (batch_size,), containing integer class indices (0 to num_classes-1).

        Returns
        -------
        torch.Tensor
            The mean Focal Loss over the batch.
        """
        # Calculate Cross Entropy loss per element.
        # This implicitly applies softmax and then NLLLoss.
        # `weight` argument applies per-class weights during the CE calculation.
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.class_weights)

        # Convert cross-entropy loss to the probability of the true class (pt).
        # log_pt is the negative log-probability of the true class, so pt = exp(-log_pt).
        log_pt = -ce_loss
        pt = torch.exp(log_pt)

        # Calculate the modulating factor: (1 - pt)^gamma.
        # Clamp pt to [epsilon, 1-epsilon] to prevent numerical issues if pt is exactly 0 or 1.
        # The (1 - pt) term ensures that as the probability of the true class (pt) approaches 1
        # (meaning the example is easy), this term approaches 0, reducing its contribution to the loss.
        focal_term = (1 - pt).clamp(min=self.epsilon)**self.gamma

        # Combine alpha, modulating factor, and cross-entropy loss to get the final focal loss.
        focal_loss = self.alpha * focal_term * ce_loss

        # Return the mean loss across all samples in the batch.
        return focal_loss.mean()


def get_criterion(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """
    Initializes and returns a loss function based on the provided configuration.

    Parameters
    ----------
    config : Dict
        The training configuration dictionary.
        Expected keys for all losses: 'loss_function_type'.
        Optional keys: 'use_class_weights', 'class_weights'.
        Specific keys for 'CrossEntropyLoss': 'label_smoothing'.
        Specific keys for 'FocalLoss': 'focal_loss_alpha', 'focal_loss_gamma'.
    device : torch.device
        The device (e.g., 'cuda' or 'cpu') on which the loss function (and its weights)
        should be instantiated.

    Returns
    -------
    nn.Module
        An instance of the configured loss function module.

    Raises
    ------
    ValueError
        If an unsupported `loss_function_type` is specified in the configuration.
    """
    loss_type = config.get('loss_function_type', 'CrossEntropyLoss')
    class_weights_list = config.get('class_weights')
    use_class_weights_flag = config.get('use_class_weights', False)

    class_weights_tensor: Optional[torch.Tensor] = None
    if use_class_weights_flag and isinstance(class_weights_list, list) and len(class_weights_list) > 0:
        print(f"Preparing class weights: {class_weights_list}")
        # Convert list of floats to a PyTorch tensor and move to the specified device
        class_weights_tensor = torch.tensor(class_weights_list, dtype=torch.float32).to(device)
    elif use_class_weights_flag and not class_weights_list:
        print(f"{COLOR_YELLOW}Warning: 'use_class_weights' is True but 'class_weights' list is empty or None. Class weights will not be applied.{COLOR_RESET}")

    criterion: Optional[nn.Module] = None
    print(f"--- Creating Loss Function: {loss_type} ---")

    if loss_type == "FocalLoss":
        alpha = config.get('focal_loss_alpha', 1.0)
        gamma = config.get('focal_loss_gamma', 2.0)
        print(f"  Using Custom Focal Loss (Alpha: {alpha}, Gamma: {gamma})")
        if class_weights_tensor is not None:
            print(f"  Applying per-class weights internally via CrossEntropyLoss component of FocalLoss: {class_weights_tensor.tolist()}")
            print(f"  {COLOR_YELLOW}Note: The interaction between Focal Loss 'alpha' and Cross Entropy 'weight' parameter can be complex. Ensure desired balancing effect.{COLOR_RESET}")
        criterion = FocalLoss(alpha=alpha, gamma=gamma, class_weights=class_weights_tensor)

    elif loss_type == "CrossEntropyLoss":
        label_smoothing = config.get('label_smoothing', 0.0)
        print(f"  Using nn.CrossEntropyLoss")
        if label_smoothing > 0:
            print(f"  Label Smoothing enabled: {label_smoothing}")
        if class_weights_tensor is not None:
            print(f"  Class Weights: {class_weights_tensor.tolist()}")
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=class_weights_tensor)

    else:
        supported_losses = ['CrossEntropyLoss', 'FocalLoss']
        raise ValueError(f"Unsupported loss_function_type: '{loss_type}'. Please choose from: {supported_losses}")

    print("--------------------------------------")
    # Ensure the loss function itself is on the correct device, especially for internal buffers
    return criterion.to(device)


# ============================================================================
# Optimizer and Scheduler Setup
# ============================================================================

def get_optimizer(model: nn.Module, config: Dict[str, Any],
                  fast_backbone_param_names: Optional[List[str]] = None) -> optim.Optimizer:
    """
    Constructs and returns an AdamW optimizer with configurable learning rate groups.

    This function sets up three distinct learning rate groups based on common fine-tuning strategies:
    1.  `backbone_slow`: Early layers of the backbone (base_lr).
    2.  `backbone_fast`: Later, potentially unfrozen, blocks of the backbone (base_lr * fast_mult).
        These are identified by `fast_backbone_param_names` provided by `model_utils.get_model`.
    3.  `classifier`: The final classification head (base_lr * classifier_mult).

    Parameters
    ----------
    model : nn.Module
        The PyTorch model for which the optimizer is to be created.
    config : Dict
        The training configuration dictionary.
        Expected keys: 'base_lr', 'classifier_lr_multiplier', 'weight_decay'.
        Optional keys: 'fast_group_lr_multiplier'.
    fast_backbone_param_names : Optional[List[str]], optional
        A list of full parameter names (e.g., 'blocks.5.conv_pw.weight') that
        should be assigned to the 'backbone_fast' learning rate group.
        If None, the optimizer will fall back to a simpler grouping or use
        `fast_param_patterns` from `config` if present (though this is now discouraged).
        Defaults to None.

    Returns
    -------
    torch.optim.Optimizer
        An instance of `torch.optim.AdamW` with specified parameter groups.

    Raises
    ------
    ValueError
        If no trainable parameters are found in the model, preventing optimizer creation.
    """
    base_lr: float = config["base_lr"]
    classifier_lr_multiplier: float = config["classifier_lr_multiplier"]
    fast_group_lr_multiplier: float = config.get("fast_group_lr_multiplier", 3.0)
    
    # Convert provided fast_backbone_param_names to a set for efficient lookup
    fast_param_names_set = set(fast_backbone_param_names) if fast_backbone_param_names is not None else set()

    # Initialize lists for different parameter groups
    slow_params, fast_params, classifier_params = [], [], []

    # Iterate through model parameters to assign them to appropriate groups
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue # Skip parameters that are currently frozen

        # Prioritize classifier parameters
        if "classifier" in name or "fc" in name or "head" in name: # Check common classifier names
            classifier_params.append(p)
        # Then check for parameters designated as 'fast' by model_utils
        elif name in fast_param_names_set:
            fast_params.append(p)
        # All other trainable parameters go to the 'slow' backbone group
        else:
            slow_params.append(p)

    # Log the counts of parameters in each group for verification
    print(f"Optimizer groups: Slow backbone: {len(slow_params)} parameters, "
          f"Fast backbone: {len(fast_params)} parameters, "
          f"Classifier head: {len(classifier_params)} parameters.")

    param_groups: List[Dict[str, Union[str, float, List[nn.Parameter]]]] = []

    # Add parameter groups to the list, with their respective learning rates
    if slow_params:
        param_groups.append({
            "name": "backbone_slow",
            "params": slow_params,
            "lr": base_lr
        })
    if fast_params:
        param_groups.append({
            "name": "backbone_fast",
            "params": fast_params,
            "lr": base_lr * fast_group_lr_multiplier
        })
    if classifier_params:
        param_groups.append({
            "name": "classifier",
            "params": classifier_params,
            "lr": base_lr * classifier_lr_multiplier
        })
    
    # If no specific param groups were created (e.g., if model has no trainable params
    # or the naming convention doesn't match for a frozen backbone), ensure at least one group is present.
    if not param_groups:
        print(f"{COLOR_YELLOW}Warning: No specific optimizer groups (slow/fast backbone, classifier) could be formed based on naming conventions or provided `fast_backbone_param_names`. Adding all trainable parameters to a single group.{COLOR_RESET}")
        all_trainable_params = [p for p in model.parameters() if p.requires_grad]
        if all_trainable_params:
            param_groups.append({
                "name": "all_trainable_params",
                "params": all_trainable_params,
                "lr": base_lr # Use base_lr as default for this group
            })
        else:
            raise ValueError(f"{COLOR_RED}Error: Model has no trainable parameters. Optimizer cannot be created.{COLOR_RESET}")

    # Initialize and return the AdamW optimizer with the defined parameter groups
    return optim.AdamW(param_groups, weight_decay=config["weight_decay"])


def get_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any], steps_per_epoch: int) -> Optional[OneCycleLR]:
    """
    Creates a learning rate scheduler for the given optimizer based on configuration.
    Currently primarily supports `OneCycleLR`.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer for which the scheduler is to be created.
    config : Dict
        The training configuration dictionary.
        Expected keys: 'epochs', 'scheduler_max_lr_multiplier', 'scheduler_pct_start'.
        Optional keys: 'scheduler_type', 'warmup_epochs' (if custom warmup is needed).
    steps_per_epoch : int
        The total number of optimizer steps (batches / accumulation steps)
        that will occur in one epoch.

    Returns
    -------
    Optional[torch.optim.lr_scheduler.OneCycleLR]
        An instance of the configured scheduler, or `None` if no scheduler is specified
        or an unsupported type is requested.

    Raises
    ------
    ValueError
        If `steps_per_epoch` is not a positive integer.
    """
    scheduler_type: str = config.get('scheduler_type', 'OneCycleLR').lower()

    # Validate `steps_per_epoch` to ensure correct scheduler behavior
    if not isinstance(steps_per_epoch, int) or steps_per_epoch <= 0:
        raise ValueError(f"Invalid `steps_per_epoch`: {steps_per_epoch}. Must be a positive integer for scheduler creation.")

    scheduler: Optional[OneCycleLR] = None
    print(f"--- Setting up Scheduler: {scheduler_type.upper()} ---")

    if scheduler_type == 'onecyclelr':
        # Calculate the maximum learning rates for each parameter group.
        max_lr_list: List[float] = []
        base_lr_list: List[float] = []
        for param_group in optimizer.param_groups:
            group_lr_from_optimizer = param_group['lr'] # Get the current LR set by get_optimizer
            base_lr_list.append(group_lr_from_optimizer)
            max_lr_list.append(group_lr_from_optimizer * config['scheduler_max_lr_multiplier'])

        print(f"  Base LRs (per group): {[f'{lr:.2e}' for lr in base_lr_list]}")
        print(f"  Max LRs (per group): {[f'{lr:.2e}' for lr in max_lr_list]}")
        print(f"  Total Steps: {config['epochs'] * steps_per_epoch} (calculated from {steps_per_epoch} steps/epoch * {config['epochs']} epochs)")
        print(f"  Warmup phase (pct_start): {config['scheduler_pct_start']}")
        
        # Create OneCycleLR with recommended parameters for smooth scheduling
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr_list, # List of max LRs, one for each parameter group
            epochs=config['epochs'],
            steps_per_epoch=steps_per_epoch, # Total number of steps *per epoch*
            pct_start=config['scheduler_pct_start'],
            anneal_strategy='cos', # Cosine annealing is a common and effective strategy
            div_factor=config['scheduler_max_lr_multiplier'], # Initial LR is max_lr / div_factor
            final_div_factor=1e4, # Final LR is max_lr / final_div_factor (makes it very small)
            verbose=False # Set to True for verbose console output from the scheduler
        )
        
    elif scheduler_type == 'none':
        print("  No learning rate scheduler will be used.")
    else:
        print(f"{COLOR_YELLOW}Warning: Scheduler type '{scheduler_type}' is not explicitly supported. No scheduler created.{COLOR_RESET}")
        # Extend this section to support other schedulers (e.g., StepLR, ReduceLROnPlateau) as needed.

    print("--------------------------------------")
    return scheduler


# ============================================================================
# Training Helper Functions
# ============================================================================

def check_model_for_nans(model: nn.Module) -> bool:
    """
    Checks all parameters and their gradients within a PyTorch model for NaN (Not a Number) values.
    NaNs in weights or gradients typically indicate unstable training.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to inspect.

    Returns
    -------
    bool
        True if any NaN value is detected in parameters or their gradients, False otherwise.
    """
    has_nans = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"{COLOR_RED}Warning: NaN detected in model parameter: {name}{COLOR_RESET}")
            has_nans = True
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"{COLOR_RED}Warning: NaN detected in gradient of parameter: {name}{COLOR_RESET}")
            has_nans = True
    if not has_nans:
        print(f"{COLOR_GREEN}No NaN values detected in model parameters or gradients.{COLOR_RESET}")
    return has_nans


def clip_gradient_norm(parameters: Union[nn.Module, List[torch.Tensor]], max_norm: float) -> torch.Tensor:
    """
    Clips the gradient norm of an iterable of parameters to a specified maximum value.
    This helps prevent exploding gradients, which can destabilize training.

    Parameters
    ----------
    parameters : Union[nn.Module, List[torch.Tensor]]
        An iterable of parameters or a `torch.nn.Module` (in which case its
        `parameters()` are used).
    max_norm : float
        The maximum allowed Euclidean norm of the gradients. Gradients exceeding this
        norm will be scaled down. If `max_norm` is <= 0, no clipping is performed.

    Returns
    -------
    torch.Tensor
        The total norm of the gradients (before clipping, or clipped if clipping occurred).
        Returns a zero tensor if no parameters were clipped.
    """
    if max_norm <= 0:
        # Return a zero tensor on the correct device if clipping is disabled or no parameters
        # Attempt to get device from the first parameter, otherwise default to 'cpu'
        device = parameters[0].device if isinstance(parameters, list) and parameters else ('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.tensor(0., device=device)

    # Ensure 'parameters' is an iterable of tensors for torch.nn.utils.clip_grad_norm_
    if isinstance(parameters, nn.Module):
        params_to_clip = [p for p in parameters.parameters() if p.grad is not None and p.requires_grad]
    else:
        params_to_clip = [p for p in parameters if p.grad is not None and p.requires_grad]

    if not params_to_clip:
        # Return a zero tensor if no relevant parameters are found
        device = 'cuda' if torch.cuda.is_available() else 'cpu' # Fallback device
        return torch.tensor(0., device=params_to_clip[0].device if params_to_clip else device)

    # Perform gradient clipping in-place
    total_norm = torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=max_norm)
    return total_norm


def apply_warmup(optimizer: optim.Optimizer, epoch: int, training_config: Dict[str, Any]):
    """
    Applies a linear learning rate warmup schedule for the initial epochs.

    During warmup, the learning rate for each parameter group is gradually increased
    from a small fraction up to its base learning rate. This can help stabilize
    training at the beginning, especially with large batch sizes or deep networks.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer whose learning rates are to be adjusted.
    epoch : int
        The current training epoch (0-indexed).
    training_config : Dict
        The training configuration dictionary, containing 'warmup_epochs',
        'base_lr', and 'classifier_lr_multiplier' (for initial LR inference).
    """
    warmup_epochs: int = training_config.get('warmup_epochs', 0)
    
    if warmup_epochs > 0 and epoch < warmup_epochs:
        # Calculate the linear scaling factor for the current epoch
        scale = (epoch + 1) / warmup_epochs
        print(f"  {COLOR_BLUE}Applying linear LR warmup for Epoch {epoch+1}/{warmup_epochs}: Scaling LR by {scale:.3f}{COLOR_RESET}")

        for i, param_group in enumerate(optimizer.param_groups):
            # Store initial_lr in param_group if not already present.
            # This is crucial for correctly restoring the base LR after warmup.
            if 'initial_lr' not in param_group:
                # Attempt to infer the initial LR based on how `get_optimizer` sets it.
                # This is a heuristic; direct storage in `param_group` by `get_optimizer` is more robust.
                group_name = param_group.get('name', '')
                if 'classifier' in group_name:
                    param_group['initial_lr'] = training_config['base_lr'] * training_config.get('classifier_lr_multiplier', 1.0)
                elif 'fast' in group_name:
                    param_group['initial_lr'] = training_config['base_lr'] * training_config.get('fast_group_lr_multiplier', 1.0)
                else:
                    param_group['initial_lr'] = training_config['base_lr']
                print(f"    (Inferred initial_lr for group '{group_name}' ({i}): {param_group['initial_lr']:.2e})")

            # Apply the linear scaling to the learning rate of the current group
            param_group['lr'] = param_group['initial_lr'] * scale
    elif warmup_epochs > 0 and epoch == warmup_epochs:
        print(f"  {COLOR_GREEN}Warmup phase completed after {warmup_epochs} epochs. Standard LR scheduling resumed.{COLOR_RESET}")


def apply_gradual_unfreezing(model: nn.Module, epoch: int, training_config: Dict[str, Any], optimizer: Optional[optim.Optimizer] = None) -> bool:
    """
    Gradually unfreezes layers of the model based on a predefined schedule.

    This technique allows earlier layers of a pre-trained model to become trainable
    at specific epochs, providing a phased approach to fine-tuning.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model whose layers might be unfrozen.
    epoch : int
        The current training epoch (0-indexed).
    training_config : Dict
        The training configuration dictionary, expected to contain:
        - 'gradual_unfreezing' (bool): Flag to enable/disable this feature.
        - 'unfreeze_schedule' (Dict[int, List[str]]): A dictionary mapping epoch numbers
          to lists of parameter name patterns to unfreeze at that epoch.
    optimizer : Optional[torch.optim.Optimizer], optional
        The optimizer. If provided, a warning will be issued about potential
        stale optimizer states for newly unfrozen parameters. Defaults to None.

    Returns
    -------
    bool
        True if any parameters' `requires_grad` status was changed in this epoch, False otherwise.

    Notes
    -----
    When `param.requires_grad` is changed from False to True, PyTorch's `optimizer`
    might not automatically pick up these new parameters or correctly manage their
    optimizer state (e.g., Adam's `exp_avg` and `exp_avg_sq` buffers) if the optimizer
    was initialized *before* these parameters became trainable. For robust gradual
    unfreezing, it is often recommended to **re-initialize the optimizer** (or at least
    update its parameter groups) *after* unfreezing parameters, especially for complex
    optimizers like Adam. This implementation attempts to extend optimizer groups but
    a full re-initialization might be necessary for full correctness.
    """
    if not training_config.get('gradual_unfreezing', False):
        return False # Gradual unfreezing is not enabled

    unfreeze_schedule: Dict[int, List[str]] = training_config.get('unfreeze_schedule', {})
    # Check if there are any patterns to unfreeze at the current epoch
    current_patterns_to_unfreeze: Optional[List[str]] = unfreeze_schedule.get(epoch, None)

    if current_patterns_to_unfreeze is None or not current_patterns_to_unfreeze:
        return False # No specific unfreezing defined for this epoch

    print(f"Epoch {epoch+1}: Applying gradual unfreezing schedule. Patterns to unfreeze: {current_patterns_to_unfreeze}")

    newly_unfrozen_params: List[nn.Parameter] = []
    changed_requires_grad = False

    # Iterate through all named parameters in the model
    for name, param in model.named_parameters():
        if not param.requires_grad: # Only consider parameters that are currently frozen
            for pattern in current_patterns_to_unfreeze:
                if pattern in name:
                    param.requires_grad = True # Mark parameter as trainable
                    newly_unfrozen_params.append(param)
                    changed_requires_grad = True
                    print(f"  Unfrozen parameter: {name}")
                    break # Move to the next parameter after a match

    if changed_requires_grad:
        # Log the updated count of trainable parameters
        trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params_count = sum(p.numel() for p in model.parameters())
        if total_params_count > 0:
            print(f"  Updated trainable parameters: {trainable_params_count:,} ({100 * trainable_params_count / total_params_count:.2f}% of total).")
        else:
            print(f"  {COLOR_YELLOW}Warning: Model contains 0 total parameters. Unfreezing may have no effect.{COLOR_RESET}")

        # Attempt to update the optimizer with the newly unfrozen parameters.
        # This part is a heuristic and might not be fully robust for all optimizers.
        if newly_unfrozen_params and optimizer is not None:
            print(f"  {COLOR_YELLOW}Warning: Newly unfrozen parameters added to optimizer. For full robustness, consider re-initializing the optimizer if issues arise (e.g., with Adam's state buffers).{COLOR_RESET}")
            # Try to add to an existing backbone group, or create a new one if necessary.
            # This is a simplified approach; a full re-initialization of the optimizer is often safer.
            found_group = False
            for group in optimizer.param_groups:
                # Check for group names that typically contain backbone parameters
                if group.get('name') in ['backbone_slow', 'backbone_fast', 'all_trainable_params']:
                    group['params'].extend(newly_unfrozen_params)
                    found_group = True
                    print(f"  Added {len(newly_unfrozen_params)} newly unfrozen parameters to existing optimizer group '{group.get('name')}'.")
                    break
            if not found_group:
                # If no suitable existing group, add as a new group.
                # This new group would default to `base_lr` unless specified otherwise.
                print(f"  {COLOR_YELLOW}Warning: No suitable existing optimizer group ('backbone_slow', 'backbone_fast', 'all_trainable_params') found. Newly unfrozen parameters might not be handled correctly, or will be added to a new default group. Consider re-creating the optimizer.{COLOR_RESET}")
                # A more robust solution might be to re-create the optimizer entirely,
                # which would automatically pick up all now-trainable parameters.
        return True
    else:
        # This block might be reached if `current_patterns_to_unfreeze` was not empty
        # but no actual parameters matched the patterns (e.g., patterns for blocks
        # that don't exist in the model, or blocks already unfrozen in previous epochs).
        print(f"  No new parameters were unfrozen for Epoch {epoch+1} matching the schedule.")
        return False


def mixup_data(x: torch.Tensor, y: torch.Tensor, device: torch.device, alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Applies MixUp data augmentation to a batch of inputs and targets.

    MixUp linearly interpolates input samples and their corresponding labels,
    encouraging the model to predict linearly between samples.

    References:
        H. Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018.
        (https://arxiv.org/abs/1710.09412)

    Parameters
    ----------
    x : torch.Tensor
        The input batch (e.g., images) of shape (batch_size, channels, height, width).
    y : torch.Tensor
        The target labels for the input batch, of shape (batch_size,).
    device : torch.device
        The device where the tensors reside (e.g., 'cuda' or 'cpu').
    alpha : float, optional
        The concentration parameter for the Beta distribution used to draw the
        interpolation coefficient `lam`. A value of 0 results in no MixUp.
        Defaults to 0.2.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]
        A tuple containing:
        - mixed_x (torch.Tensor): The interpolated input batch.
        - y_a (torch.Tensor): The original target labels.
        - y_b (torch.Tensor): The target labels of the permuted batch.
        - lam (float): The interpolation coefficient used.
    """
    if alpha <= 0:
        return x, y, y, 1.0 # If alpha is 0 or less, no MixUp is applied, return original data with lam=1

    # Draw lambda from a Beta distribution
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)

    # Permute the batch indices randomly
    index = torch.randperm(batch_size, device=device)

    # Linearly interpolate inputs: mixed_x = lam * x_original + (1 - lam) * x_permuted
    mixed_x = lam * x + (1 - lam) * x[index, :]

    # Keep original and permuted labels for loss calculation
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion: nn.Module, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
    """
    Calculates the loss when MixUp data augmentation is applied.

    The loss is computed as a linear interpolation of the loss calculated
    with respect to the original targets and the permuted targets.

    Parameters
    ----------
    criterion : nn.Module
        The original loss function (e.g., `nn.CrossEntropyLoss`, `FocalLoss`).
    pred : torch.Tensor
        The model's predictions (logits), of shape (batch_size, num_classes).
    y_a : torch.Tensor
        The original target labels for the batch.
    y_b : torch.Tensor
        The target labels from the permuted batch.
    lam : float
        The interpolation coefficient used in MixUp.

    Returns
    -------
    torch.Tensor
        The interpolated loss value.
    """
    # Calculate loss with respect to original targets (y_a) and permuted targets (y_b)
    # Then, combine them using the lambda (lam) coefficient
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================================
# Checkpoint Management
# ============================================================================

def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, best_metric_val: float, path: str, model_config: Dict[str, Any]):
    """
    Saves the current state of the model and optimizer to a checkpoint file.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to save.
    optimizer : optim.Optimizer
        The optimizer whose state needs to be saved.
    epoch : int
        The current epoch number (0-indexed).
    best_metric_val : float
        The value of the best validation metric achieved so far.
    path : str
        The full file path where the checkpoint will be saved (e.g., 'model_best.pth').
    model_config : Dict[str, Any]
        Dictionary containing model architecture parameters.
    """
    print(f"Saving checkpoint for epoch {epoch+1} to '{path}'...")
    try:
        # Create a dictionary containing the necessary state information
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_metric_val': best_metric_val,
            'timestamp': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            'model_config': model_config
        }
        # Save the state dictionary to the specified path
        torch.save(state, path)
        print(f"{COLOR_GREEN}Checkpoint saved successfully to '{os.path.basename(path)}'.{COLOR_RESET}")
    except Exception as e:
        print(f"{COLOR_RED}Error saving checkpoint to '{path}': {e}{COLOR_RESET}")
        traceback.print_exc()


def load_checkpoint(model: Optional[nn.Module], optimizer: Optional[optim.Optimizer], filename: str, device: torch.device, model_utils=None) -> Tuple[nn.Module, Optional[optim.Optimizer], int, float]:
    """
    Loads model and optimizer states from a checkpoint file.
    
    Parameters
    ----------
    model : Optional[nn.Module]
        The PyTorch model instance to load state into. Can be None if model_utils is provided.
    optimizer : Optional[optim.Optimizer]
        The optimizer instance to load state into.
    filename : str
        The full file path to the checkpoint file.
    device : torch.device
        The device to which the loaded tensors should be mapped.
    model_utils : Optional[object]
        Module containing get_model function for creating model from config.
    
    Returns
    -------
    Tuple[nn.Module, Optional[optim.Optimizer], int, float]
        model, optimizer, start_epoch, best_metric_val
    """
    if not os.path.isfile(filename):
        print(f"{COLOR_YELLOW}Checkpoint file not found: '{filename}'. Starting training from scratch.{COLOR_RESET}")
        return model, optimizer, -1, -float('inf')

    print(f"Loading checkpoint from '{filename}'...")
    try:
        checkpoint = torch.load(filename, map_location=device)
        
        # Check if model needs to be created from config
        if 'model_config' in checkpoint and model_utils is not None:
            print("  Creating model from saved configuration...")
            model_config = checkpoint['model_config']
            model = model_utils.get_model(
                model_name=model_config['model_name'],
                num_classes=model_config.get('num_classes', 2),  # Default if not saved
                dropout_rate=model_config.get('dropout_rate', 0.0),
                drop_path_rate=model_config.get('drop_path_rate', 0.0),
                finetune_blocks=model_config.get('finetune_blocks', 0)
            ).to(device)
        elif model is None:
            raise ValueError("Model is None and no model_config found in checkpoint")
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            print("  Optimizer state loaded successfully.")
        
        start_epoch = checkpoint.get('epoch', -1)
        best_metric_val = checkpoint.get('best_metric_val', -float('inf'))
        
        print(f"{COLOR_GREEN}Checkpoint loaded successfully. Resuming from epoch {start_epoch + 1}. Best metric: {best_metric_val:.4f}{COLOR_RESET}")
        return model, optimizer, start_epoch, best_metric_val
        
    except Exception as e:
        print(f"{COLOR_RED}Error loading checkpoint from '{filename}': {e}{COLOR_RESET}")
        traceback.print_exc()
        return model, optimizer, -1, -float('inf')


# ============================================================================
# Logging and Reporting
# ============================================================================

def save_history_json(history: Dict[str, List[Union[float, List[float]]]], path: str):
    """
    Saves the training history dictionary to a JSON file.

    Handles conversion of NumPy types and non-finite float values to standard Python
    types for proper JSON serialization.

    Parameters
    ----------
    history : Dict[str, List[Union[float, List[float]]]]
        A dictionary containing training metrics history (e.g., 'train_loss', 'val_acc', 'lr').
        Values are expected to be lists of floats or lists of lists of floats (e.g., for LR).
    path : str
        The full file path where the JSON history will be saved.
    """
    serializable_history: Dict[str, List[Union[float, List[float]]]] = {}
    print(f"Preparing history for JSON serialization and saving to '{path}'...")
    for key, values in history.items():
        if not values:
            serializable_history[key] = [] # Keep empty lists if no data
            continue
        try:
            # Check the type of the first non-None element to handle nested lists (e.g., for LR groups)
            first_valid_element = next((item for item in values if item is not None), None)

            if isinstance(first_valid_element, (list, tuple)):
                # Handle lists of lists/tuples (e.g., for multi-group LRs)
                converted_list_of_lists = []
                for sublist in values:
                    if isinstance(sublist, (list, tuple)):
                        converted_sublist = []
                        for item in sublist:
                            if isinstance(item, (int, float)):
                                converted_sublist.append(float(item) if math.isfinite(item) else 0.0) # Convert non-finite to 0.0
                            else:
                                converted_sublist.append(None) # Or other placeholder if not numeric
                        converted_list_of_lists.append(converted_sublist)
                    else:
                        converted_list_of_lists.append(None) # Or handle non-list elements as needed
                serializable_history[key] = converted_list_of_lists
            else:
                # Handle flat lists of numbers
                converted_flat_list = []
                for value in values:
                    if isinstance(value, (int, float)):
                        converted_flat_list.append(float(value) if math.isfinite(value) else 0.0) # Convert non-finite to 0.0
                    else:
                        converted_flat_list.append(None) # Or other placeholder
                serializable_history[key] = converted_flat_list
        except Exception as e:
            print(f"{COLOR_YELLOW}Warning: Could not serialize history key '{key}'. Skipping this key. Error: {e}{COLOR_RESET}")
            traceback.print_exc()
            serializable_history[key] = [] # Default to empty list on error

    try:
        with open(path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        print(f"{COLOR_GREEN}Training history saved successfully to '{os.path.basename(path)}'.{COLOR_RESET}")
    except Exception as e:
        print(f"{COLOR_RED}Error saving history JSON to '{path}': {e}{COLOR_RESET}")
        traceback.print_exc()


def save_summary_yaml(config: Dict[str, Any], history: Dict[str, Any],
                      best_metrics: Dict[str, Any], output_files: Dict[str, Any], path: str):
    """
    Saves a comprehensive summary of the training run to a YAML file.

    This summary includes configuration parameters, key performance metrics
    (best and final), dataset information, and paths to all generated output files.

    Parameters
    ----------
    config : Dict
        The complete training configuration dictionary used for the run.
    history : Dict
        The training history dictionary containing metrics logged per epoch.
    best_metrics : Dict
        A dictionary containing the best validation metrics achieved during training.
        Expected keys: 'f1', 'acc', 'loss', 'epoch'.
    output_files : Dict
        A dictionary mapping descriptive names to paths of all generated output files
        (e.g., best model, plots, history JSON).
    path : str
        The full file path where the YAML summary will be saved.
    """
    print(f"Attempting to save run summary to '{path}'...")
    try:
        epochs_completed = len(history.get('train_loss', []))

        # Determine if early stopping was triggered
        patience_counter = config.get('patience_counter', 0)
        patience_limit = config.get('patience', float('inf'))
        planned_epochs = config.get('epochs', float('inf'))
        early_stopped = (patience_counter >= patience_limit) and \
                        (epochs_completed < planned_epochs) # Only true if patience was hit before max epochs

        # Find the best epoch number based on the best F1 score recorded
        best_epoch_num: Union[int, str] = 'N/A'
        best_f1_metric = best_metrics.get('f1', -1.0)
        history_f1 = history.get('f1')

        if history_f1: # Check if history_f1 is not None or empty
            # Convert history_f1 to a NumPy array, handling potential NaNs by setting to -inf for argmax
            f1_scores = np.array([f if f is not None and math.isfinite(f) else -np.inf for f in history_f1])
            if len(f1_scores) > 0 and np.any(f1_scores > -np.inf): # Ensure there's at least one finite score
                best_epoch_idx_from_history = np.nanargmax(f1_scores) # Index of the maximum F1 in history
                # Check if the highest F1 in history matches the 'best_f1_metric'
                # Use a tolerance for float comparison
                if math.isclose(f1_scores[best_epoch_idx_from_history], best_f1_metric, rel_tol=1e-5):
                    best_epoch_num = best_epoch_idx_from_history + 1
                else:
                    # If they don't exactly match, try to find the epoch recorded in best_metrics
                    if best_metrics.get('epoch') is not None and best_metrics['epoch'] != -1:
                        best_epoch_num = int(best_metrics['epoch'])
                        print(f"{COLOR_YELLOW}Warning: Best F1 score found in history ({f1_scores[best_epoch_idx_from_history]:.4f}) doesn't exactly match recorded best ({best_f1_metric:.4f}). Using epoch {best_epoch_num} from 'best_metrics'.{COLOR_RESET}")
                    else:
                        best_epoch_num = best_epoch_idx_from_history + 1
                        print(f"{COLOR_YELLOW}Warning: Best F1 score from history ({f1_scores[best_epoch_idx_from_history]:.4f}) and recorded best ({best_f1_metric:.4f}) do not match, and no specific 'best_epoch' recorded. Using epoch {best_epoch_num} from history argmax.{COLOR_RESET}")
            else:
                print(f"{COLOR_YELLOW}Warning: No valid finite F1 scores found in history to determine best epoch.{COLOR_RESET}")
        else:
            print(f"{COLOR_YELLOW}Warning: F1 score history is empty or missing.{COLOR_RESET}")

        # Get the metrics from the last completed epoch for final state summary
        last_metrics = {
            'train_loss': history.get('train_loss', [float('nan')])[-1] if epochs_completed > 0 else float('nan'),
            'train_acc': history.get('train_acc', [0.0])[-1] if epochs_completed > 0 else 0.0,
            'val_loss': history.get('val_loss', [float('nan')])[-1] if epochs_completed > 0 else float('nan'),
            'val_acc': history.get('val_acc', [0.0])[-1] if epochs_completed > 0 else 0.0,
            'precision': history.get('precision', [0.0])[-1] if epochs_completed > 0 else 0.0,
            'recall': history.get('recall', [0.0])[-1] if epochs_completed > 0 else 0.0,
            'f1': history.get('f1', [0.0])[-1] if epochs_completed > 0 else 0.0,
        }

        # Retrieve potentially nested dictionaries from the main config for detailed logging
        # Note: Corrected `normalization` key to `NORMALIZATION` based on `training_config` creation in notebook
        data_paths_dict = config.get('DATA_PATHS', {})
        normalization_dict = config.get('NORMALIZATION', {})


        # Assemble the comprehensive run summary dictionary
        run_summary_data: Dict[str, Any] = {
            'run_details': {
                'timestamp': config.get('timestamp', 'N/A'),
                'file_prefix': config.get('file_prefix', 'N/A'),
                'model_name': config.get('model', 'N/A'),
                'device_info': config.get('device_info', 'N/A'),
                'mixed_precision_used': config.get('use_amp', False),
                'epochs_planned': config.get('epochs', 'N/A'),
                'epochs_completed': epochs_completed,
                'early_stopping_triggered': early_stopped,
                'early_stopping_patience': config.get('patience', 'N/A'),
                'early_stopping_min_delta': config.get('min_delta', 0.001),
            },
            'dataset_info': {
                'train_set_size_original': config.get('original_train_set_size', 'N/A'),
                'train_set_size_resampled': config.get('resampled_train_set_size', 'N/A'),
                'val_set_size': config.get('val_set_size', 'N/A'),
                'test_set_size': config.get('test_set_size', 'N/A'), # test_set_size will be dynamically updated if test is run
                'class_names': config.get('class_names', []),
                'csv_files_used': data_paths_dict.get('csv_files', []),
                'root_dirs_used': data_paths_dict.get('root_dirs', []),
                'test_csv_file': data_paths_dict.get('test_csv_file', 'N/A'),
                'test_root_dir': data_paths_dict.get('test_root_dir', 'N/A'),
            },
            'model_config': {
                'input_size': config.get('input_size', None),
                'num_classes': config.get('num_classes', None),
                'dropout': config.get('dropout', None),
                'drop_path_rate': config.get('drop_path_rate', None),
                'finetune_blocks': config.get('finetune_blocks', None),
            },
            'training_hyperparameters': {
                'batch_size': config.get('batch_size', 'N/A'),
                'accum_steps': config.get('accum_steps', 'N/A'),
                'num_workers': config.get('num_workers', 'N/A'),
                'pin_memory': config.get('pin_memory', 'N/A'),
                'persistent_workers': config.get('persistent_workers', 'N/A'),
                'prefetch_factor': config.get('prefetch_factor', 'N/A'),
                'optimizer_type': config.get('optimizer_type', 'AdamW'),
                'base_lr': config.get('base_lr', None),
                'classifier_lr_multiplier': config.get('classifier_lr_multiplier', None),
                'fast_group_lr_multiplier': config.get('fast_group_lr_multiplier', None),
                'fast_param_patterns': config.get('fast_param_patterns', []),
                'weight_decay': config.get('weight_decay', None),
                'grad_clip_max_norm': config.get('grad_clip_max_norm', None),
                'mixup_enabled': config.get('mixup', False),
                'mixup_alpha': config.get('mixup_alpha') if config.get('mixup') else None,
                'mixup_prob': config.get('mixup_prob') if config.get('mixup') else None,
                'use_oversampler': config.get('use_oversampler', None),
            },
            'scheduler_settings': {
                'type': config.get('scheduler_type', 'OneCycleLR'),
                'pct_start': config.get('scheduler_pct_start', None),
                'max_lr_multiplier': config.get('scheduler_max_lr_multiplier', None),
                'warmup_epochs': config.get('warmup_epochs', 0),
            },
            'loss_function_config': {
                'type': config.get('loss_function_type', None),
                'label_smoothing': config.get('label_smoothing') if config.get('loss_function_type') == 'CrossEntropyLoss' else None,
                'focal_loss_alpha': config.get('focal_loss_alpha') if config.get('loss_function_type') == 'FocalLoss' else None,
                'focal_loss_gamma': config.get('focal_loss_gamma') if config.get('loss_function_type') == 'FocalLoss' else None,
                'class_weights': config.get('class_weights') if config.get('use_class_weights', False) else None,
            },
            'normalization_values': {
                'mean': normalization_dict.get('mean', None),
                'std': normalization_dict.get('std', None),
            },
            'augmentation_config': {
                'use_horizontal_flip': config.get('use_horizontal_flip', None),
                'use_rotation': config.get('use_rotation', None),
                'use_color_jitter': config.get('use_color_jitter', None),
                'use_random_crop': config.get('use_random_crop', None),
                'use_centered_affine_scaling': config.get('use_centered_affine_scaling', None),
                'use_affine': config.get('use_affine', None),
                'use_gamma': config.get('use_gamma', None),
                'use_blur': config.get('use_blur', None),
                'use_random_erase': config.get('use_random_erase', None),
                'rotation_degrees': config.get('rotation_degrees', None),
                'brightness_jitter': config.get('brightness_jitter', None),
                'contrast_jitter': config.get('contrast_jitter', None),
                'saturation_jitter': config.get('saturation_jitter', None),
                'hue_jitter': config.get('hue_jitter', None),
                'random_crop_scale': config.get('random_crop_scale', None),
                'random_crop_ratio': config.get('random_crop_ratio', None),
                'translate_range': config.get('translate_range', None),
                'scale_range': config.get('scale_range', None),
                'shear_range': config.get('shear_range', None),
                'gamma_range': config.get('gamma_range', None),
                'blur_sigma_range': config.get('blur_sigma_range', None),
                'random_erase_probability': config.get('random_erase_probability', None),
                'random_erase_scale': config.get('random_erase_scale', None),
                'random_erase_ratio': config.get('random_erase_ratio', None),
            },
            'results_summary': {
                'best_val_accuracy': float(f"{best_metrics.get('acc', 0.0):.4f}") if best_metrics.get('acc') is not None else None,
                'best_val_loss': float(f"{best_metrics.get('loss', float('inf')):.6f}") if best_metrics.get('loss') is not None and math.isfinite(best_metrics.get('loss')) else None,
                'best_val_f1': float(f"{best_metrics.get('f1', -1.0):.6f}") if best_metrics.get('f1') is not None and math.isfinite(best_metrics.get('f1')) else None,
                'best_epoch': int(best_epoch_num) if isinstance(best_epoch_num, int) else (int(best_epoch_num) if isinstance(best_epoch_num, str) and best_epoch_num.isdigit() else best_epoch_num),
                'final_train_loss': float(f"{last_metrics['train_loss']:.6f}") if math.isfinite(last_metrics['train_loss']) else None,
                'final_train_accuracy': float(f"{last_metrics['train_acc']:.4f}") if last_metrics['train_acc'] is not None else None,
                'final_val_loss': float(f"{last_metrics['val_loss']:.6f}") if math.isfinite(last_metrics['val_loss']) else None,
                'final_val_accuracy': float(f"{last_metrics['val_acc']:.4f}") if last_metrics['val_acc'] is not None else None,
                'final_val_precision': float(f"{last_metrics['precision']:.6f}") if last_metrics['precision'] is not None else None,
                'final_val_recall': float(f"{last_metrics['recall']:.6f}") if last_metrics['recall'] is not None else None,
                'final_val_f1': float(f"{last_metrics['f1']:.6f}") if last_metrics['f1'] is not None else None,
            },
            'output_file_paths': output_files
        }

        # Define a custom representer for NumPy types and tuples to ensure clean YAML output
        def sanitize_for_yaml(dumper: yaml.Dumper, data: Any):
            # Convert NumPy integers to standard Python integers
            if isinstance(data, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                               np.uint8, np.uint16, np.uint32, np.uint64)):
                return dumper.represent_int(int(data))
            # Convert NumPy floats to standard Python floats, handling NaNs and Infs specifically for YAML
            elif isinstance(data, (np.float16, np.float32, np.float64)):
                if np.isnan(data):
                    return dumper.represent_scalar('tag:yaml.org,2002:float', '.nan')
                if np.isinf(data):
                    return dumper.represent_scalar('tag:yaml.org,2002:float', '.inf' if data > 0 else '-.inf')
                return dumper.represent_float(float(data))
            # Convert NumPy arrays to Python lists
            elif isinstance(data, np.ndarray):
                return dumper.represent_list(data.tolist())
            # Convert NumPy boolean to Python boolean
            elif isinstance(data, np.bool_):
                return dumper.represent_bool(bool(data))
            # Convert tuples to string representation (can be adjusted if specific tuple structure is desired)
            elif isinstance(data, tuple):
                return dumper.represent_str(str(data))
            return dumper.represent_data(data)

        # Register the custom representers with PyYAML
        # This allows PyYAML to correctly serialize NumPy types and tuples
        yaml.add_multi_representer(np.generic, custom_representer) # For all numpy scalar types
        yaml.add_representer(tuple, custom_representer) # For python tuples

        # Write the assembled summary to the YAML file
        with open(path, 'w') as f:
            yaml.dump(run_summary_data, f, default_flow_style=False, sort_keys=False, indent=2)
        print(f"{COLOR_GREEN}Run summary saved successfully to '{os.path.basename(path)}'.{COLOR_RESET}")

    except ImportError:
        print(f"{COLOR_RED}\nError: Could not save YAML summary. PyYAML library not found. Please install it: pip install PyYAML{COLOR_RESET}")
    except Exception as e:
        print(f"{COLOR_RED}\nAn unexpected error occurred during saving run summary to YAML: {e}{COLOR_RESET}")
        traceback.print_exc()


def save_validation_report(labels: List[int], predictions: List[int], filenames: List[str], num_classes: int, output_path: str) -> Optional[pd.DataFrame]:
    """
    Generates and saves a detailed validation report, including a confusion matrix
    and a CSV file with per-sample true/predicted labels and filenames.

    Parameters
    ----------
    labels : List[int]
        List or array of ground truth labels (0-based integers).
    predictions : List[int]
        List or array of predicted labels (0-based integers).
    filenames : List[str]
        List of corresponding filenames for each sample.
    num_classes : int
        The total number of classes (e.g., 5 for PAI 1-5).
    output_path : str
        The full file path where the CSV report will be saved.

    Returns
    -------
    Optional[pd.DataFrame]
        The confusion matrix as a pandas DataFrame if successful, otherwise None.
    """
    print("\n--- Generating Detailed Validation Report ---")
    cm_df: Optional[pd.DataFrame] = None

    try:
        # Convert inputs to NumPy arrays for efficient processing
        true_labels_np = np.array(labels)
        pred_labels_np = np.array(predictions)

        if len(true_labels_np) == 0 or len(pred_labels_np) == 0 or len(true_labels_np) != len(pred_labels_np):
            print(f"{COLOR_YELLOW}Warning: Empty or mismatched label/prediction arrays provided ({len(true_labels_np)} labels, {len(pred_labels_np)} predictions). Cannot generate report.{COLOR_RESET}")
            return None

        # Create confusion matrix (labels are 0-based for sklearn)
        matrix_labels = list(range(num_classes))
        cm = confusion_matrix(true_labels_np, pred_labels_np, labels=matrix_labels)

        # Create a DataFrame for better display of the confusion matrix
        cm_df = pd.DataFrame(cm,
                             index=[f"True_{i+1}" for i in matrix_labels], # Display 1-based PAI in index
                             columns=[f"Pred_{i+1}" for i in matrix_labels]) # Display 1-based PAI in columns

        # Add row and column sums for total true/predicted counts
        cm_df['Total_True'] = cm_df.sum(axis=1) # This column is created as int64 here
        
        # --- MINIMAL CHANGE START ---
        # Explicitly convert 'Total_True' column to object dtype *before* assigning a string
        cm_df['Total_True'] = cm_df['Total_True'].astype(object)
        # --- MINIMAL CHANGE END ---

        cm_df.loc['Total_Pred'] = cm_df.sum(axis=0)

        # Calculate overall accuracy
        total_correct = np.trace(cm) # Sum of diagonal elements is total correct predictions
        total_samples = len(true_labels_np)
        accuracy_str = "N/A"
        if total_samples > 0:
            accuracy_str = f"{total_correct}/{total_samples} ({total_correct/total_samples:.2%})"

        # Place overall accuracy in the bottom-right cell of the confusion matrix summary
        if 'Total_True' in cm_df.columns and 'Total_Pred' in cm_df.index:
            # The line below will now work without warning because 'Total_True' column is object dtype
            cm_df.loc['Total_Pred', 'Total_True'] = accuracy_str

        # Print the formatted confusion matrix to console
        print("Confusion Matrix (Rows: True Class, Cols: Predicted Class, PAI values are 1-indexed):")
        print(cm_df.to_string())

        # Check for length consistency between filenames and labels/predictions
        if len(filenames) != total_samples:
            print(f"{COLOR_YELLOW}Warning: Length mismatch between provided filenames ({len(filenames)}) and labels/predictions ({total_samples}). CSV report might be incomplete or misaligned.{COLOR_RESET}")
            filenames = filenames[:total_samples] # Truncate filenames to match if needed

        # Create a pandas DataFrame for the per-sample detailed report
        report_df = pd.DataFrame({
            'filename': filenames,
            'true_PAI': [label + 1 for label in true_labels_np], # Convert 0-based labels to 1-based PAI scores
            'predicted_PAI': [pred + 1 for pred in pred_labels_np] # Convert 0-based predictions to 1-based PAI scores
        })

        # Save the detailed report to a CSV file
        try:
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True) # Create directory if it doesn't exist

            report_df.to_csv(output_path, index=False)
            print(f"\nDetailed validation report saved to: '{os.path.basename(output_path)}'.")
        except Exception as e_csv:
            print(f"{COLOR_RED}Error saving validation report CSV to '{output_path}': {e_csv}{COLOR_RESET}")
            traceback.print_exc()

    except Exception as e_main:
        print(f"{COLOR_RED}An unexpected error occurred while generating the validation report: {e_main}{COLOR_RESET}")
        traceback.print_exc()
        return None # Return None on critical error

    return cm_df # Return the confusion matrix DataFrame

# ============================================================================
# Inference and Metrics Calculation
# ============================================================================

@torch.no_grad() # Decorator to disable gradient computations for efficiency during inference
def run_inference(model: torch.nn.Module,
                  dataloader: torch.utils.data.DataLoader,
                  device: torch.device,
                  num_classes: int,
                  class_names: Optional[List[str]] = None,
                  description: str = "Inference") -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Runs inference using a trained model on a given DataLoader.

    This function sets the model to evaluation mode, iterates through the dataloader,
    performs forward passes, and collects predictions and true labels without
    computing gradients.

    Parameters
    ----------
    model : torch.nn.Module
        The trained PyTorch model. It should already be moved to the correct device.
    dataloader : torch.utils.data.DataLoader
        The DataLoader for the dataset (e.g., validation or test set) to evaluate.
    device : torch.device
        The device (e.g., 'cuda' or 'cpu') on which to run inference.
    num_classes : int
        The total number of classes in the classification task.
    class_names : Optional[List[str]], optional
        A list of human-readable names for each class. Used for contextual logging.
        Defaults to None.
    description : str, optional
        A descriptive string for the progress bar, indicating the type of inference
        (e.g., "Validation Inference", "Test Set Inference"). Defaults to "Inference".

    Returns
    -------
    Tuple[Optional[np.ndarray], Optional[np.ndarray]]
        A tuple containing two NumPy arrays:
        - all_preds (np.ndarray): Array of predicted class labels (0-based integers).
        - all_labels (np.ndarray): Array of true class labels (0-based integers).
        Returns `(None, None)` if the dataloader is empty or a critical error occurs.
    """
    print(f"\n--- Running {description} ---")
    if not dataloader or len(dataloader.dataset) == 0:
        print(f"{COLOR_YELLOW}Warning: Dataloader for '{description}' is empty. Skipping inference.{COLOR_RESET}")
        return None, None

    model.eval() # Set the model to evaluation mode (disables dropout, batchnorm updates, etc.)
    all_preds_list: List[np.ndarray] = []
    all_labels_list: List[np.ndarray] = []

    # Use tqdm for a progress bar during inference
    progress_bar = tqdm(dataloader, desc=description, leave=False, unit="batch")
    try:
        for batch_data in progress_bar:
            # Flexible batch data unpacking, assuming (inputs, labels, filenames) or (inputs, labels)
            inputs: torch.Tensor
            labels: torch.Tensor
            try:
                # Attempt to unpack (image, label, filename) tuple
                inputs, labels, _ = batch_data
            except ValueError:
                try:
                    # Fallback to unpacking (image, label) if filename is not present
                    inputs, labels = batch_data
                except Exception as e:
                    print(f"{COLOR_YELLOW}\nError unpacking batch data for '{description}': {e}. Skipping this batch.{COLOR_RESET}")
                    # Log the error and skip to the next batch
                    continue

            # Move tensors to the specified device (GPU/CPU)
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Perform forward pass (no need for torch.amp.autocast during inference usually, float32 is fine)
            outputs = model(inputs)

            # Get predicted class indices by taking the argmax of the logits
            preds = torch.argmax(outputs, dim=1)

            # Collect predictions and true labels, moving them to CPU and converting to NumPy arrays
            all_preds_list.append(preds.cpu().numpy())
            all_labels_list.append(labels.cpu().numpy())

    except Exception as e:
        print(f"{COLOR_RED}\nError during inference loop for '{description}': {e}{COLOR_RESET}")
        traceback.print_exc()
        return None, None # Indicate that inference failed
    finally:
        progress_bar.close() # Ensure the progress bar is closed properly

    if not all_labels_list: # Check if any batches were successfully processed and collected
        print(f"{COLOR_YELLOW}Warning: No labels collected during '{description}' inference loop. No results to concatenate.{COLOR_RESET}")
        return None, None

    try:
        # Concatenate all collected predictions and labels into single NumPy arrays
        all_preds = np.concatenate(all_preds_list)
        all_labels = np.concatenate(all_labels_list)
        print(f"{description} complete. Processed {len(all_labels)} samples.")
        return all_preds, all_labels
    except ValueError as e: # Handle cases where `all_preds_list` or `all_labels_list` might be empty
        print(f"{COLOR_RED}Error concatenating inference results (likely no batches processed successfully): {e}{COLOR_RESET}")
        return None, None


def calculate_and_print_metrics(labels: np.ndarray,
                                 preds: np.ndarray,
                                 num_classes: int,
                                 class_names: Optional[List[str]] = None,
                                 results_title: str = "Test Set Results") -> Dict[str, Union[float, int, List, np.ndarray]]:
    """
    Calculates and prints standard and custom classification metrics for multi-class tasks.

    Includes overall accuracy, F1-score (weighted), Quadratic Weighted Kappa (QWK),
    Mean Absolute Error (MAE), confusion matrix, and specialized binary metrics
    (Healthy vs. Pathological) for PAI classification.

    Parameters
    ----------
    labels : np.ndarray
        A 1D NumPy array of true class labels (0-based integers, 0 to num_classes-1).
    preds : np.ndarray
        A 1D NumPy array of predicted class labels (0-based integers, 0 to num_classes-1).
    num_classes : int
        The total number of classes in the classification problem.
    class_names : Optional[List[str]], optional
        A list of string names for each class, corresponding to the 0-based label order.
        If None or if its length does not match `num_classes`, generic numeric class names
        will be used. Defaults to None.
    results_title : str, optional
        A title to display before the printed results section. Defaults to "Test Set Results".

    Returns
    -------
    Dict[str, Union[float, int, List, np.ndarray]]
        A dictionary containing all calculated metrics. Keys include:
        'accuracy', 'f1_weighted', 'qwk', 'mae', 'confusion_matrix', 'class_names',
        'binary_accuracy', 'binary_sensitivity', 'binary_specificity', 'binary_fp_rate',
        'binary_fn_rate', 'binary_tn', 'binary_fp', 'binary_fn', 'binary_tp'.
        Returns an empty dictionary `{}` if input data is invalid or if an error occurs during calculation.
    """
    print(f"\n--- {results_title} ---")

    # Validate input arrays
    if labels is None or preds is None or len(labels) == 0 or len(preds) == 0 or len(labels) != len(preds):
        print(f"{COLOR_RED}Error: Invalid or empty input arrays for metric calculation. Labels length: {len(labels) if labels is not None else 0}, Predictions length: {len(preds) if preds is not None else 0}.{COLOR_RESET}")
        return {}

    # Prepare class names for display and metric calculation
    effective_class_names: List[str]
    if class_names and len(class_names) == num_classes:
        effective_class_names = class_names
    else:
        print(f"{COLOR_YELLOW}Warning: Class names not provided or length mismatch ({len(class_names) if class_names else 0} vs {num_classes}). Using generic numeric class names.{COLOR_RESET}")
        effective_class_names = [f"PAI {i+1}" for i in range(num_classes)] # Use 1-based PAI names

    try:
        # --- Overall Metrics ---
        # Overall Accuracy
        acc = accuracy_score(labels, preds)

        # Weighted F1-score, Precision, Recall
        # `average='weighted'` accounts for class imbalance. `zero_division=0` handles classes with no true samples.
        # `labels=list(range(num_classes))` ensures all classes are considered, even if absent in current batch.
        precision_w, recall_w, f1_weighted, _ = precision_recall_fscore_support(
            labels, preds, average='weighted', zero_division=0, labels=list(range(num_classes))
        )

        # Quadratic Weighted Kappa (QWK) - commonly used for ordinal classification tasks like PAI
        qwk = cohen_kappa_score(labels, preds, weights='quadratic')

        # Mean Absolute Error (MAE) - useful for ordinal scales, indicates average misclassification distance
        mae = mean_absolute_error(labels, preds)

        print(f"Overall Accuracy          : {acc:.4f} ({int(acc * len(labels))}/{len(labels)})")
        print(f"Weighted F1-Score         : {f1_weighted:.4f}")
        print(f"Quadratic Weighted Kappa  : {qwk:.4f}")
        print(f"Mean Absolute Error (MAE) : {mae:.4f}")

        # --- Confusion Matrix ---
        print("\nConfusion Matrix (Rows: True Class, Cols: Predicted Class):")
        # Ensure confusion matrix is built for all defined classes
        cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
        cm_df = pd.DataFrame(cm, index=effective_class_names, columns=effective_class_names)
        print(cm_df.to_string()) # Use to_string() for full console display

        # --- Binary Metrics (Healthy vs. Pathological) ---
        print("\nBinary Classification Metrics (Healthy: PAI 1,2 vs Pathological: PAI 3,4,5):")
        # Map original 0-4 PAI labels to binary (0=Healthy, 1=Pathological)
        labels_binary = np.where(labels <= 1, 0, 1) # PAI 0,1 -> 0 (Healthy); PAI 2,3,4 -> 1 (Pathological)
        preds_binary = np.where(preds <= 1, 0, 1)

        # Calculate binary confusion matrix and extract True Negative, False Positive, False Negative, True Positive
        tn, fp, fn, tp = 0, 0, 0, 0 # Initialize counts
        try:
            # This can fail if labels_binary/preds_binary contain only one class (e.g., all healthy)
            cm_binary = confusion_matrix(labels_binary, preds_binary, labels=[0, 1])
            tn, fp, fn, tp = cm_binary.ravel()
        except ValueError:
            print(f"  {COLOR_YELLOW}Warning: Could not compute full binary confusion matrix (likely only one class present in labels or predictions). Deriving counts manually.{COLOR_RESET}")
            # Manual derivation if ravel() fails due to single class
            # This ensures metrics are still calculated even in edge cases
            if len(labels_binary) > 0:
                if np.all(labels_binary == 0) and np.all(preds_binary == 0): tn = len(labels_binary)
                elif np.all(labels_binary == 1) and np.all(preds_binary == 1): tp = len(labels_binary)
                elif np.all(labels_binary == 0) and np.all(preds_binary == 1): fp = len(labels_binary)
                elif np.all(labels_binary == 1) and np.all(preds_binary == 0): fn = len(labels_binary)


        total_healthy_true = tn + fp # Total actual healthy samples
        total_pathol_true = fn + tp # Total actual pathological samples
        total_samples_binary = tn + fp + fn + tp

        binary_acc = (tn + tp) / total_samples_binary if total_samples_binary > 0 else 0.0
        # Sensitivity (Recall) = TP / (TP + FN) -> ability to correctly identify pathological cases
        sensitivity = tp / total_pathol_true if total_pathol_true > 0 else 0.0
        # Specificity = TN / (TN + FP) -> ability to correctly identify healthy cases
        specificity = tn / total_healthy_true if total_healthy_true > 0 else 0.0
        # False Negative Rate = FN / (FN + TP) -> pathological missed as healthy
        fn_rate = fn / total_pathol_true if total_pathol_true > 0 else 0.0
        # False Positive Rate = FP / (FP + TN) -> healthy misclassified as pathological
        fp_rate = fp / total_healthy_true if total_healthy_true > 0 else 0.0

        print(f"  Overall Binary Accuracy : {binary_acc:.4f}")
        print(f"  Sensitivity (Recall)    : {sensitivity:.4f} (Pathological cases correctly identified)")
        print(f"  Specificity             : {specificity:.4f} (Healthy cases correctly identified)")
        print(f"  False Negative Rate (FN): {fn_rate:.4f} ({fn}/{total_pathol_true} -> Pathological samples incorrectly classified as Healthy)")
        print(f"  False Positive Rate (FP): {fp_rate:.4f} ({fp}/{total_healthy_true} -> Healthy samples incorrectly classified as Pathological)")
        print(f"     (True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}, True Positives: {tp})")
        print("----------------------------------------------------------")

        # Store all calculated metrics in a dictionary for easy access and logging
        metrics_dict: Dict[str, Union[float, int, List, np.ndarray]] = {
            'accuracy': acc,
            'f1_weighted': f1_weighted,
            'qwk': qwk,
            'mae': mae,
            'confusion_matrix': cm.tolist(), # Convert NumPy array to list for JSON/YAML serialization
            'class_names': effective_class_names,
            'binary_accuracy': binary_acc,
            'binary_sensitivity': sensitivity,
            'binary_specificity': specificity,
            'binary_fp_rate': fp_rate,
            'binary_fn_rate': fn_rate,
            'binary_tn': int(tn), # Convert to standard Python int for serialization
            'binary_fp': int(fp),
            'binary_fn': int(fn),
            'binary_tp': int(tp),
        }
        return metrics_dict

    except Exception as e:
        print(f"{COLOR_RED}Error calculating metrics for '{results_title}': {e}{COLOR_RESET}")
        traceback.print_exc()
        return {} # Return an empty dictionary on error