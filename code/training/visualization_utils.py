"""
Visualization utilities for PAI classification

Author: Gerald Torgersen
SPDX-License-Identifier: MIT
Copyright (c) 2025 Gerald Torgersen
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd


def plot_training_history(
    history: dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
):
    """
    Plot training history curves.
    
    Args:
        history: Dictionary with training history
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Loss curves
    if 'train_loss' in history and 'val_loss' in history:
        axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
        axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # QWK curves
    if 'val_qwk' in history:
        axes[1].plot(history['val_qwk'], label='Val QWK', color='green', linewidth=2)
        axes[1].axhline(y=0.70, color='r', linestyle='--', label='Target (0.70)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('QWK')
        axes[1].set_title('Validation Quadratic Weighted Kappa')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # F1 curves
    if 'val_f1' in history:
        axes[2].plot(history['val_f1'], label='Val F1', color='orange', linewidth=2)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('F1 Score')
        axes[2].set_title('Validation F1 Score')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels (1-indexed for PAI: 1-5)
        y_pred: Predicted labels (1-indexed for PAI: 1-5)
        class_names: List of class names
        normalize: Whether to normalize values
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    if class_names is None:
        class_names = [f'PAI {i}' for i in range(1, 6)]
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5])
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_class_distribution(
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot class distribution.
    
    Args:
        labels: Array of labels
        class_names: List of class names
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    if class_names is None:
        unique_labels = np.unique(labels)
        class_names = [f'PAI {int(i)}' for i in unique_labels]
    
    # Count samples per class
    unique, counts = np.unique(labels, return_counts=True)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(class_names, counts, color='steelblue', edgecolor='black')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{int(count)}\n({count/counts.sum()*100:.1f}%)',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    ax.set_xlabel('PAI Class', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_model_comparison(
    results: Dict[str, dict],
    metrics: List[str] = ['qwk', 'accuracy', 'f1'],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Plot comparison of multiple models.
    
    Args:
        results: Dictionary mapping model names to metric dictionaries
        metrics: List of metrics to plot
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    models = list(results.keys())
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        values = [results[model].get(metric, 0) for model in models]
        
        bars = axes[idx].bar(models, values, color='coral', edgecolor='black')
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[idx].text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{value:.3f}',
                ha='center',
                va='bottom',
                fontsize=10
            )
        
        axes[idx].set_ylabel(metric.upper(), fontsize=11)
        axes[idx].set_title(f'{metric.upper()} Comparison', fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, axis='y')
        axes[idx].set_xticklabels(models, rotation=15, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_learning_rate_schedule(
    lrs: List[float],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot learning rate schedule.
    
    Args:
        lrs: List of learning rates per step
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(lrs, linewidth=2, color='darkblue')
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_prediction_examples(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    indices: Optional[List[int]] = None,
    n_samples: int = 9,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 12)
):
    """
    Plot example predictions with images.
    
    Args:
        images: Array of images (N, H, W, 3)
        y_true: True labels (1-5)
        y_pred: Predicted labels (1-5)
        probabilities: Prediction probabilities (N, 5)
        indices: Specific indices to plot (optional)
        n_samples: Number of samples to plot
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    if indices is None:
        indices = np.random.choice(len(images), size=min(n_samples, len(images)), replace=False)
    
    n_cols = 3
    n_rows = (len(indices) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for idx, ax in enumerate(axes):
        if idx < len(indices):
            i = indices[idx]
            
            # Plot image
            ax.imshow(images[i])
            
            # Create title
            correct = y_true[i] == y_pred[i]
            color = 'green' if correct else 'red'
            
            title = f"True: PAI {y_true[i]}, Pred: PAI {y_pred[i]}\n"
            title += f"Prob: {probabilities[i][y_pred[i]-1]:.2f}"
            
            ax.set_title(title, color=color, fontweight='bold')
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_roc_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Plot ROC curves for multi-class classification (one-vs-rest).
    
    Args:
        y_true: True labels (0-4 indexed)
        y_probs: Prediction probabilities (N, 5)
        class_names: List of class names
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    if class_names is None:
        class_names = [f'PAI {i}' for i in range(1, 6)]
    
    n_classes = y_probs.shape[1]
    
    # Binarize labels for one-vs-rest
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    
    # Compute ROC curve and AUC for each class
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        
        ax.plot(
            fpr,
            tpr,
            color=colors[i],
            lw=2,
            label=f'{class_names[i]} (AUC = {roc_auc:.2f})'
        )
    
    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.50)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def create_results_summary_table(
    results: Dict[str, dict],
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Create formatted results summary table.
    
    Args:
        results: Dictionary mapping model names to metric dictionaries
        save_path: Path to save CSV (optional)
    
    Returns:
        DataFrame with results summary
    """
    rows = []
    
    for model_name, metrics in results.items():
        row = {
            'Model': model_name,
            'QWK': f"{metrics.get('qwk', 0):.4f}",
            'Accuracy': f"{metrics.get('accuracy', 0):.4f}",
            'F1-Score': f"{metrics.get('f1', 0):.4f}",
            'MAE': f"{metrics.get('mae', 0):.4f}"
        }
        
        if 'qwk_std' in metrics:
            row['QWK'] += f" ± {metrics['qwk_std']:.4f}"
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    if save_path:
        df.to_csv(save_path, index=False)
    
    return df
