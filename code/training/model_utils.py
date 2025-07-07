# model_utils.py
"""
Utilities for building and configuring deep learning models for PAI classification.

This module focuses on the neural network architecture, enabling:
-   Creation of various pre-trained models (e.g., EfficientNet family) using the `timm` library.
-   Flexible configuration of dropout and drop path rates.
-   Advanced parameter freezing strategies (e.g., fine-tuning only the classifier,
    or the classifier plus a specified number of top-most backbone blocks).

Author: Gerald Torgersen
SPDX-License-Identifier: MIT
Copyright (c) 2025 Gerald Torgersen, UiO
Contact: https://www.odont.uio.no/iko/english/people/aca/gerald/
"""

# Standard Library Imports
import os
import sys

# Third-party Library Imports
import torch
import torch.nn as nn
import timm # PyTorch Image Models library

# Type Hinting Imports
from typing import Optional, List, Tuple

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
# Model Building Function
# ============================================================================

def get_model(model_name: str = 'efficientnet_b3', # Changed default to efficientnet_b3 as per notebook context
              num_classes: int = 5,
              pretrained: bool = True,
              dropout_rate: float = 0.5, # Adjusted default based on common practice for EffNet-B3 fine-tuning
              drop_path_rate: float = 0.2,
              finetune_blocks: int = -1) -> Tuple[nn.Module, List[str]]:
    """
    Builds a deep learning model using the `timm` library, with configurable
    pretrained weights, output classes, dropout, and stochastic depth.
    It also applies a flexible parameter freezing strategy.

    Parameters
    ----------
    model_name : str, optional
        The name of the model architecture to load from `timm`
        (e.g., 'efficientnet_b3', 'efficientnet_v2_s', 'resnet50').
        Defaults to 'efficientnet_b3'.
    num_classes : int, optional
        The number of output classes for the final classification layer.
        For PAI, typically 5 (0-indexed). Defaults to 5.
    pretrained : bool, optional
        If True, loads ImageNet pre-trained weights. Defaults to True.
    dropout_rate : float, optional
        The dropout rate applied to the final classification head (between 0 and 1).
        Defaults to 0.5.
    drop_path_rate : float, optional
        The stochastic depth `drop_path` rate for the backbone layers (between 0 and 1).
        Defaults to 0.2.
    finetune_blocks : int, optional
        Controls which parameters are trainable:
        -   `-1`: All model parameters are set to trainable (standard fine-tuning).
        -   `0`: Only the final classification head (classifier layer) is trainable;
            all backbone parameters are frozen.
        -   `>0`: The final classification head and the last `N` (`finetune_blocks`)
            sequential blocks of the backbone are trainable; other backbone layers are frozen.
        Defaults to -1.

    Returns
    -------
    Tuple[torch.nn.Module, List[str]]
        A tuple containing:
        - model (torch.nn.Module): The constructed and potentially partially-frozen PyTorch model.
        - fast_backbone_param_names (List[str]): A list of full parameter names (e.g., 'blocks.5.conv')
                                                 that were explicitly unfrozen as part of the
                                                 `finetune_blocks > 0` strategy. These are intended
                                                 for the 'fast' learning rate group in the optimizer.

    Raises
    ------
    Exception
        If there is an error creating the model using `timm` (e.g., invalid model name,
        network issues for pretrained weights).
    """
    print(f"--- Building Model: '{model_name}' ---")
    print(f"  Num Classes       : {num_classes}")
    print(f"  Pretrained        : {pretrained}")
    print(f"  Dropout Rate      : {dropout_rate}")
    print(f"  Drop Path Rate    : {drop_path_rate}")
    print(f"  Finetune Blocks   : {finetune_blocks} (-1: all, 0: classifier only, >0: classifier + last N blocks)")
    print("-----------------------------------")

    fast_backbone_param_names: List[str] = [] # Collect names of parameters in the 'fast' group

    try:
        # Create the model using timm.create_model.
        # timm automatically handles adapting the classifier head (`num_classes`)
        # and applying global dropout (`drop_rate`) and stochastic depth (`drop_path_rate`).
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout_rate,
            drop_path_rate=drop_path_rate
        )

        # --- Parameter Freezing Logic ---
        if finetune_blocks == -1:
            # If -1, all parameters are trainable (this is timm's default behavior)
            print(f"All model parameters (`finetune_blocks`={finetune_blocks}) are set to trainable.")
            # Ensure all parameters are explicitly marked as requiring grad, in case a pretrained model defaults differently
            for param in model.parameters():
                param.requires_grad = True
        else:
            # If `finetune_blocks` is 0 or positive, first freeze all parameters
            print(f"Initiating parameter freezing based on `finetune_blocks`={finetune_blocks}...")
            for param in model.parameters():
                param.requires_grad = False

            # --- Unfreeze Classifier Head ---
            # Iterate through common classifier attribute names to find and unfreeze the head
            classifier_unfrozen = False
            for head_name in ['classifier', 'fc', 'head']: # Common names for classifier layers
                if hasattr(model, head_name) and isinstance(getattr(model, head_name), nn.Module):
                    # Unfreeze all parameters within the identified classifier module
                    for param in getattr(model, head_name).parameters():
                        param.requires_grad = True
                    print(f"  {COLOR_GREEN}Classifier layer '{head_name}' (final classification head) unfrozen.{COLOR_RESET}")
                    classifier_unfrozen = True
                    break
            if not classifier_unfrozen:
                print(f"  {COLOR_YELLOW}Warning: Could not automatically find a common classifier layer name ('classifier', 'fc', 'head') to unfreeze. Ensure your model's head is being fine-tuned.{COLOR_RESET}")

            # --- Unfreeze Specific Backbone Blocks (if finetune_blocks > 0) ---
            if finetune_blocks > 0:
                block_layer_name: Optional[str] = None
                # Identify the attribute name for the main sequential blocks (common in EfficientNets)
                if hasattr(model, 'blocks'):
                    block_layer_name = 'blocks'
                # Add more conditions here for other model families if needed (e.g., 'layer4' for ResNets)
                # elif hasattr(model, 'layer4'): block_layer_name = 'layer4' # Example for ResNet's last block

                if block_layer_name and isinstance(getattr(model, block_layer_name), nn.Sequential):
                    all_blocks = getattr(model, block_layer_name)
                    total_blocks = len(all_blocks)
                    # Determine the actual number of blocks to unfreeze, clamped by total available blocks
                    num_to_unfreeze = min(finetune_blocks, total_blocks)

                    if num_to_unfreeze > 0:
                        print(f"  Unfreezing last {num_to_unfreeze} blocks in '{block_layer_name}' (total blocks: {total_blocks})...")
                        # Iterate backwards from the end to unfreeze the last N blocks
                        for i in range(total_blocks - num_to_unfreeze, total_blocks):
                            # Recursively set requires_grad for all sub-parameters in this block
                            for name, param in all_blocks[i].named_parameters():
                                param.requires_grad = True
                                # Collect the full name of these unfrozen parameters for the fast LR group
                                fast_backbone_param_names.append(f"{block_layer_name}.{i}.{name}")
                            print(f"    - Block {i} (index {i}) unfrozen and parameters collected for fast LR group.")
                    else:
                        print(f"  No backbone blocks unfrozen. Only the classifier is trainable as `finetune_blocks` is {finetune_blocks} or too small compared to total blocks.")

                elif finetune_blocks > 0:
                    print(f"  {COLOR_YELLOW}Warning: `finetune_blocks` is > 0, but a standard sequential block structure (`blocks` or similar) was not found in the model. Backbone blocks could not be selectively unfrozen.{COLOR_RESET}")
            else:
                print("  No backbone blocks will be unfrozen (`finetune_blocks` is 0). Only the classifier head is trainable.")

        # Log total trainable parameters for confirmation
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        if total_params > 0:
             print(f"Total trainable parameters in model: {trainable_params:,} ({100 * trainable_params / total_params:.2f}% of total).")
        else:
             print(f"{COLOR_YELLOW}Warning: Model contains 0 total parameters. This might indicate an issue.{COLOR_RESET}")

        return model, fast_backbone_param_names

    except Exception as e:
        print(f"{COLOR_RED}Error creating model '{model_name}' with timm: {e}{COLOR_RESET}")
        print("Please check the `model_name` and ensure `timm` is installed and functioning correctly.")
        raise # Re-raise the error for upstream handling (e.g., by the main training script)