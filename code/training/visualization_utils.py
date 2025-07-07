# visualization_utils.py
"""
Utility functions for visualizing training progress and results.

This module provides functions to generate and save plots of key metrics
over training epochs, including:
-   GPU memory utilization.
-   Learning rate schedules for different optimizer parameter groups.
-   Training and validation loss, accuracy, and F1-score.

These visualizations are crucial for monitoring model performance, diagnosing
training issues (e.g., overfitting, exploding/vanishing gradients), and
documenting experimental outcomes.

Author: Gerald Torgersen
SPDX-License-Identifier: MIT
Copyright (c) 2025 Gerald Torgersen, UiO
Contact: https://www.odont.uio.no/iko/english/people/aca/gerald/
"""

# Standard Library Imports
import math
import traceback

# Third-party Library Imports
import matplotlib.pyplot as plt
import numpy as np # Used for checking numerical types and nan/inf

# Type Hinting Imports
from typing import Dict, List, Union, Any, Optional

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
# Plotting Functions
# ============================================================================

def plot_gpu_memory_and_save(history: Dict[str, List[Union[float, Any]]], save_path: str):
    """
    Plots and saves the GPU memory usage over training epochs.

    Parameters
    ----------
    history : Dict[str, List[Union[float, Any]]]
        A dictionary containing training history, expected to have a 'gpu_mem_used' key
        with a list of memory usage values per epoch (in GB).
    save_path : str
        The full file path where the plot will be saved (e.g., 'gpu_memory.png').
    """
    print(f"\n--- Generating GPU Memory Usage Plot ---")
    
    # Safely retrieve GPU memory data, filtering out non-numeric or non-positive values
    gpu_mem_data = [
        mem for mem in history.get('gpu_mem_used', [])
        if isinstance(mem, (int, float)) and math.isfinite(mem) and mem >= 0
    ]

    if not gpu_mem_data:
        print(f"{COLOR_YELLOW}Warning: No valid GPU memory data found in history. Skipping plot generation.{COLOR_RESET}")
        return

    try:
        fig, ax = plt.subplots(figsize=(10, 6)) # Create figure and axes
        epochs = range(1, len(gpu_mem_data) + 1)

        ax.plot(epochs, gpu_mem_data, marker='o', linestyle='-', color='tab:blue')
        ax.set_title('GPU Memory Usage Over Epochs')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Memory Used (GB)')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(bottom=0) # Ensure y-axis always starts at 0

        # Save the plot
        plt.tight_layout() # Adjust plot to prevent labels from overlapping
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"{COLOR_GREEN}GPU memory usage plot saved to '{save_path}'.{COLOR_RESET}")
    except Exception as e:
        print(f"{COLOR_RED}Error plotting or saving GPU memory graph: {e}{COLOR_RESET}")
        traceback.print_exc() # Print full traceback for debugging
    finally:
        plt.close(fig) # Close the figure to free up memory


def plot_lr_schedule_and_save(history: Dict[str, List[Union[float, List[float]]]], save_path: str):
    """
    Plots and saves the learning rate schedule over training epochs.

    This function can handle both single-group (flat list of LRs) and multi-group
    (list of lists of LRs for different parameter groups) learning rate histories.

    Parameters
    ----------
    history : Dict[str, List[Union[float, List[float]]]]
        A dictionary containing training history, expected to have an 'lr' key
        with a list of learning rate values per epoch. Each value can be a float
        (for single LR group) or a list of floats (for multiple LR groups).
    save_path : str
        The full file path where the plot will be saved (e.g., 'lr_schedule.png').
    """
    print(f"\n--- Generating Learning Rate Schedule Plot ---")
    
    lr_data = history.get('lr', [])

    if not lr_data:
        print(f"{COLOR_YELLOW}Warning: No learning rate data found in history. Skipping plot generation.{COLOR_RESET}")
        return

    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        num_epochs = len(lr_data)
        epochs = range(1, num_epochs + 1)

        # Check if LR history is a list of lists (multiple parameter groups)
        if lr_data and isinstance(lr_data[0], (list, tuple)):
            num_groups = len(lr_data[0])
            for i in range(num_groups):
                # Extract LR for each group, filtering for valid numeric and finite values
                group_lrs = [
                    epoch_lrs[i] if isinstance(epoch_lrs, (list, tuple)) and len(epoch_lrs) > i and
                                    isinstance(epoch_lrs[i], (int, float)) and math.isfinite(epoch_lrs[i])
                    else None
                    for epoch_lrs in lr_data
                ]
                valid_lrs_points = [(e, lr) for e, lr in zip(epochs, group_lrs) if lr is not None]
                
                if valid_lrs_points:
                    valid_epochs, valid_lr_values = zip(*valid_lrs_points)
                    ax.plot(valid_epochs, valid_lr_values, marker='o', linestyle='-', label=f'Group {i+1}')
                else:
                    print(f"{COLOR_YELLOW}Warning: No valid data points for LR Group {i+1}. Skipping plotting this group.{COLOR_RESET}")
        elif lr_data and isinstance(lr_data[0], (int, float)): # Assume flat list if not list of lists
            # Filter for valid numeric and finite values for a single LR group
            valid_lrs_points = [
                (e, lr) for e, lr in zip(epochs, lr_data)
                if isinstance(lr, (int, float)) and math.isfinite(lr)
            ]
            if valid_lrs_points:
                valid_epochs, valid_lr_values = zip(*valid_lrs_points)
                ax.plot(valid_epochs, valid_lr_values, marker='o', linestyle='-', label='All Groups')
            else:
                print(f"{COLOR_YELLOW}Warning: No valid data points for single LR group. Skipping plotting.{COLOR_RESET}")
        else:
            print(f"{COLOR_YELLOW}Warning: Unexpected format for 'lr' history. Expected list of floats or list of lists/tuples of floats. Skipping plot.{COLOR_RESET}")
            plt.close(fig)
            return

        # Check if any lines were actually plotted before finalizing the figure
        if ax.get_lines():
            ax.set_title('Learning Rate Schedule Over Epochs')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_yscale('log') # Log scale is often more informative for LR changes
            ax.grid(True, which='both', linestyle='--', alpha=0.7) # Grid for both major and minor ticks on log scale
            ax.legend()
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"{COLOR_GREEN}Learning rate schedule plot saved to '{save_path}'.{COLOR_RESET}")
        else:
            print(f"{COLOR_YELLOW}No valid learning rate data points found after filtering. Skipping plot.{COLOR_RESET}")
    except Exception as e:
        print(f"{COLOR_RED}Error plotting or saving learning rate graph: {e}{COLOR_RESET}")
        traceback.print_exc()
    finally:
        plt.close(fig)


def plot_metrics_and_save(history: Dict[str, List[Union[float, Any]]], save_path: str, model_name: str = "Model"):
    """
    Plots and saves training/validation loss, accuracy, and F1 score over epochs.

    Parameters
    ----------
    history : Dict[str, List[Union[float, Any]]]
        A dictionary containing training history with keys:
        'train_loss', 'val_loss', 'train_acc', 'val_acc', 'f1'.
        Expected values are lists of floats.
    save_path : str
        The full file path where the combined metrics plot will be saved
        (e.g., 'metrics_summary.png').
    model_name : str, optional
        Name of the model to include in the plot titles. Defaults to "Model".
    """
    print(f"\n--- Generating Training/Validation Metrics Plot ({model_name}) ---")
    
    # Safely retrieve data, filtering out non-numeric or non-finite values
    train_loss = [l for l in history.get('train_loss', []) if isinstance(l, (int, float)) and math.isfinite(l)]
    val_loss = [l for l in history.get('val_loss', []) if isinstance(l, (int, float)) and math.isfinite(l)]
    train_acc = [a for a in history.get('train_acc', []) if isinstance(a, (int, float)) and math.isfinite(a)]
    val_acc = [a for a in history.get('val_acc', []) if isinstance(a, (int, float)) and math.isfinite(a)]
    val_f1 = [f for f in history.get('f1', []) if isinstance(f, (int, float)) and math.isfinite(f)]

    # Determine epochs based on the longest available data, defaulting to train_loss
    max_len = max(len(train_loss), len(val_loss), len(train_acc), len(val_acc), len(val_f1))
    if max_len == 0:
        print(f"{COLOR_YELLOW}Warning: No valid metric data found in history for plotting. Skipping.{COLOR_RESET}")
        return

    epochs = range(1, max_len + 1)

    try:
        fig, axs = plt.subplots(2, 1, figsize=(10, 12)) # 2 rows, 1 column for loss and accuracy/F1

        # --- Plot Loss ---
        ax_loss = axs[0]
        if train_loss:
            ax_loss.plot(range(1, len(train_loss) + 1), train_loss, label='Train Loss', marker='o', linestyle='-')
        if val_loss:
            ax_loss.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss', marker='o', linestyle='-')
        
        ax_loss.set_title(f'{model_name} - Loss Over Epochs')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.grid(True, alpha=0.5)
        ax_loss.legend()
        ax_loss.set_ylim(bottom=0) # Loss should typically be non-negative

        # --- Plot Accuracy & F1-Score ---
        ax_metrics = axs[1]
        if train_acc:
            ax_metrics.plot(range(1, len(train_acc) + 1), train_acc, label='Train Accuracy (%)', marker='o', linestyle='-')
        if val_acc:
            ax_metrics.plot(range(1, len(val_acc) + 1), val_acc, label='Validation Accuracy (%)', marker='o', linestyle='-')
        if val_f1:
            ax_metrics.plot(range(1, len(val_f1) + 1), val_f1, label='Validation F1-Score', marker='*', linestyle='--') # Distinct style for F1
        
        ax_metrics.set_title(f'{model_name} - Accuracy & F1 Score Over Epochs')
        ax_metrics.set_xlabel('Epoch')
        ax_metrics.set_ylabel('Score / Percentage')
        ax_metrics.grid(True, alpha=0.5)
        ax_metrics.legend()
        ax_metrics.set_ylim(bottom=0, top=100 if all(a <= 100 for a in train_acc + val_acc) else None) # Accuracy typically percentage, F1 0-1

        # Final adjustments and save
        plt.tight_layout() # Adjust plot to prevent labels from overlapping
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"{COLOR_GREEN}Metrics plot saved to '{save_path}'.{COLOR_RESET}")
    except Exception as e:
        print(f"{COLOR_RED}Error plotting or saving metrics graph: {e}{COLOR_RESET}")
        traceback.print_exc()
    finally:
        plt.close(fig) # Close the figure to free up memory