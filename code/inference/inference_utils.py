# ==============================================================================
# Utility Functions and Classes for PAI Classification XAI Inference
# ==============================================================================

# Standard & Data Science Libraries
import json
import os
import sys
import importlib.util
import traceback
import math
import gc
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any, Iterable

# Data Processing Libraries
import pandas as pd
import numpy as np
from PIL import Image
import cv2

# PyTorch & Computer Vision Libraries
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Plotting & Evaluation Libraries
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns
from sklearn.metrics import confusion_matrix, cohen_kappa_score

# Progress bar
from tqdm.auto import tqdm

# Try to import optional libraries
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, LayerCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    PYTORCH_GRAD_CAM_AVAILABLE = True
except ImportError:
    PYTORCH_GRAD_CAM_AVAILABLE = False

# ==============================================================================
# Environment Setup Functions
# ==============================================================================

def setup_custom_package_path(custom_path):
    """Setup custom package path for specific environments (e.g., HPC)."""
    if custom_path in sys.path:
        sys.path.remove(custom_path)
    if os.path.exists(custom_path):
        sys.path.insert(0, custom_path)
        print(f"Added custom library path to sys.path: {custom_path}")
        return True
    else:
        print(f"Warning: Custom library path not found: {custom_path}")
        return False

def check_library_availability():
    """Check availability of required libraries."""
    print("Checking library availability:")
    print(f"  timm: {'Available' if TIMM_AVAILABLE else 'NOT AVAILABLE'}")
    print(f"  pytorch-grad-cam: {'Available' if PYTORCH_GRAD_CAM_AVAILABLE else 'NOT AVAILABLE'}")
    
    if not TIMM_AVAILABLE:
        print("Warning: timm library not found. Please install it (`pip install timm`).")
    
    if not PYTORCH_GRAD_CAM_AVAILABLE:
        print("Warning: 'pytorch-grad-cam' library not found. Install it using: pip install grad-cam")

# ==============================================================================
# Dataset Class
# ==============================================================================

class CustomDataset(Dataset):
    """
    Custom PyTorch Dataset for loading images, labels, quadrant, and filename for inference.
    """
    def __init__(self, dataframe: pd.DataFrame, transform: transforms.Compose = None):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform
        # Basic check for essential columns
        required_cols = ['filename', 'root_dir', 'PAI_0_indexed', 'quadrant']
        if not all(col in self.data.columns for col in required_cols):
            missing = [col for col in required_cols if col not in self.data.columns]
            print(f"Error: CustomDataset initialized with missing columns: {missing}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, str]:
        try:
            row = self.data.iloc[idx]
            img_name = str(row['filename'])
            root_dir = str(row['root_dir'])
            label_0idx = int(row['PAI_0_indexed'])
            quadrant_val = row.get('quadrant', -1)
            try:
                quadrant = int(quadrant_val)
            except (ValueError, TypeError):
                quadrant = -1

        except Exception as e:
            print(f"Error accessing DataFrame row at index {idx}: {e}")
            raise RuntimeError(f"Error accessing data for index {idx}: {e}")

        img_path = os.path.join(root_dir, img_name)

        # Load Image
        try:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file not found at {img_path} (Index: {idx})")
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_name} at index {idx}: {e}")
            raise

        # Apply transformations
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"Error applying transform to image {img_name} at index {idx}: {e}")
                raise

        return image, label_0idx, quadrant, img_name

# ==============================================================================
# Data Loading Functions
# ==============================================================================

def load_and_prepare_test_data(test_csv_file, test_root_dir, mean, std, batch_size=100):
    """Load and prepare test data with validation."""
    print(f"Loading test metadata from: {test_csv_file}")
    
    try:
        test_data = pd.read_csv(test_csv_file)
        if test_data.empty:
            raise ValueError("Test CSV file is empty.")
        print(f"Loaded {len(test_data)} rows.")
    except FileNotFoundError:
        raise FileNotFoundError(f"Test CSV file not found at {test_csv_file}")
    except Exception as e:
        raise RuntimeError(f"Error loading test CSV {test_csv_file}: {e}")

    # Add root directory path
    test_data['root_dir'] = test_root_dir

    # Adjust PAI labels to 0-indexed
    if 'PAI' not in test_data.columns:
        raise ValueError("'PAI' column not found in the CSV file.")
    
    try:
        test_data['PAI_0_indexed'] = pd.to_numeric(test_data['PAI'], errors='coerce').astype('Int64') - 1
        original_count = len(test_data)
        test_data = test_data.dropna(subset=['PAI_0_indexed']).copy()
        filtered_count = len(test_data)
        if filtered_count < original_count:
            print(f"Warning: Removed {original_count - filtered_count} rows with invalid PAI values.")

        test_data['PAI_0_indexed'] = test_data['PAI_0_indexed'].astype(int)

        if not test_data['PAI_0_indexed'].between(0, 4).all():
            print("Warning: 'PAI_0_indexed' column contains values outside the expected 0-4 range.")

        print("Adjusted 'PAI' column to 'PAI_0_indexed' (0-based). Original 'PAI' retained.")
    except Exception as e:
        raise RuntimeError(f"Error processing 'PAI' column: {e}")

    # Verify image existence
    print(f"Verifying image existence in: {test_root_dir}")
    
    def check_image_path(row):
        filename = row.get('filename')
        if not filename:
            return False
        return os.path.exists(os.path.join(row['root_dir'], str(filename)))

    test_data['image_exists'] = test_data.apply(check_image_path, axis=1)
    original_count = len(test_data)
    test_data = test_data[test_data['image_exists']].copy()
    filtered_count = len(test_data)
    print(f"Filtered dataset: {filtered_count} samples remaining after checking image existence (removed {original_count - filtered_count}).")
    
    if filtered_count == 0:
        raise ValueError("No valid image paths found. Check 'test_root_dir' and 'filename' columns.")
    
    test_data = test_data.drop('image_exists', axis=1)

    # Display statistics
    display_dataset_statistics(test_data, 'PAI_0_indexed', "Test")

    # Create transforms
    test_transforms = transforms.Compose([
        transforms.Resize(300),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Create dataset and dataloader
    print("Creating Test Dataset and DataLoader...")
    try:
        test_dataset = CustomDataset(test_data, transform=test_transforms)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                               num_workers=0, pin_memory=True, drop_last=False)
        print(f"DataLoader created with batch size {batch_size}.")
        return test_loader, test_data
    except Exception as e:
        raise RuntimeError(f"Error creating Dataset or DataLoader: {e}")

def display_dataset_statistics(df, label_col, name):
    """Prints class distribution statistics for a DataFrame."""
    if label_col not in df.columns:
        print(f"Warning: Label column '{label_col}' not found for {name} statistics.")
        return
    
    class_counts = Counter(df[label_col])
    total_samples = len(df)
    if total_samples == 0:
        print(f"\nStatistics for {name} set: No samples.")
        return

    sorted_classes = sorted(class_counts.keys())
    stats_data = {
        'Class (0-indexed)': [],
        'Count': [],
        'Percentage (%)': []
    }
    for cls in sorted_classes:
        count = class_counts[cls]
        percentage = (count / total_samples) * 100
        stats_data['Class (0-indexed)'].append(cls)
        stats_data['Count'].append(count)
        stats_data['Percentage (%)'].append(f"{percentage:.1f}")

    stats_df = pd.DataFrame(stats_data)
    print(f"\nStatistics for {name} set:")
    print(stats_df.to_string(index=False))

def display_dataset_samples(data, num_images=20, images_per_row=5, title=""):
    """Display sample images from the dataset."""
    rows = num_images // images_per_row
    fig, axes = plt.subplots(rows, images_per_row, figsize=(20, 4*rows))
    fig.suptitle(title, fontsize=16)
    
    for idx in range(num_images):
        row = idx // images_per_row
        col = idx % images_per_row
        
        img_path = os.path.join(data.iloc[idx]['root_dir'], data.iloc[idx]['filename'])
        img = Image.open(img_path)
        
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].axis('off')
        axes[row, col].set_title(f"{data.iloc[idx]['filename']}\nPAI: {data.iloc[idx]['PAI']}")
    
    plt.tight_layout()
    plt.show()

# ==============================================================================
# Model Loading Functions
# ==============================================================================

def load_model_for_inference(checkpoint_path: str, device: torch.device,
                             model_name: str = None, num_classes: int = 5,
                             dropout_rate: float = None, drop_path_rate: float = None) -> nn.Module:
    """
    Loads the model architecture using timm and loads weights from checkpoint.
    """
    if not TIMM_AVAILABLE:
        raise ImportError("timm library is required but not available.")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print("Checkpoint loaded successfully.")
    except Exception as e:
        print(f"Error loading checkpoint file: {e}")
        raise

    # Extract model configuration
    model_config = checkpoint.get('model_config', {})
    if not model_config:
        raise ValueError("No model_config found in checkpoint. This appears to be an old format checkpoint that's not supported.")
    
    print("Found model_config in checkpoint:")
    print(f"  Model name: {model_config.get('model_name')}")
    print(f"  Num classes: {model_config.get('num_classes')}")
    print(f"  Dropout rate: {model_config.get('dropout_rate')}")
    print(f"  Drop path rate: {model_config.get('drop_path_rate')}")
    print(f"  Finetune blocks: {model_config.get('finetune_blocks')}")
    
    final_model_name = model_config.get('model_name')
    final_num_classes = model_config.get('num_classes', 5)
    final_dropout_rate = model_config.get('dropout_rate', 0.0)
    final_drop_path_rate = model_config.get('drop_path_rate', 0.0)
    
    if not final_model_name:
        raise ValueError("model_name not found in checkpoint's model_config")
    
    print(f"Using model configuration from checkpoint.")
    print(f"Creating model architecture '{final_model_name}' with {final_num_classes} classes...")
    
    try:
        model = timm.create_model(
            final_model_name,
            pretrained=False,
            num_classes=final_num_classes,
            drop_rate=final_dropout_rate,
            drop_path_rate=final_drop_path_rate
        )
        print("Model architecture created successfully.")
    except Exception as e:
        print(f"Error creating model architecture '{final_model_name}': {e}")
        raise

    # Load weights
    print("Loading model weights...")
    try:
        model_state_dict = checkpoint.get('model_state_dict', checkpoint)
        if isinstance(model_state_dict, dict) and 'state_dict' in model_state_dict:
            model_state_dict = model_state_dict['state_dict']

        # Handle DataParallel prefix
        if list(model_state_dict.keys())[0].startswith('module.'):
            print("  Removing 'module.' prefix from state_dict keys.")
            model_state_dict = {k[len('module.'):]: v for k, v in model_state_dict.items()}

        model.load_state_dict(model_state_dict, strict=True)
        print("Model state_dict loaded successfully (strict=True).")

    except RuntimeError as e:
        print(f"Error loading state_dict from checkpoint: {e}")
        print("Attempting to load with strict=False (use with caution)...")
        try:
            model.load_state_dict(model_state_dict, strict=False)
            print("Model state_dict loaded successfully (strict=False).")
            print("Warning: Using strict=False means some weights were not loaded or some were unexpectedly present.")
        except Exception as e_false:
            print(f"Fallback loading with strict=False failed: {e_false}")
            raise RuntimeError(f"Failed to load model state_dict even with strict=False: {e_false}") from e_false

    except Exception as e:
        print(f"An unexpected error occurred during checkpoint loading: {e}")
        raise

    model.to(device)
    model.eval()
    print("Model loaded and set to evaluation mode.")
    
    if 'epoch' in checkpoint:
        print(f"Checkpoint info: Epoch {checkpoint['epoch']}, Best metric: {checkpoint.get('best_metric_val', 'N/A')}")
    if 'timestamp' in checkpoint:
        print(f"Checkpoint timestamp: {checkpoint['timestamp']}")
    
    return model

# ==============================================================================
# Evaluation Functions
# ==============================================================================

def evaluate_predictions(y_true: List[int], y_pred: List[int], num_classes: int = 5) -> Tuple[np.ndarray, float, float, float]:
    """Calculates confusion matrix, accuracy, MAE, and QWK."""
    labels = list(range(1, num_classes + 1))
    y_true_arr = np.array(y_true, dtype=int)
    y_pred_arr = np.array(y_pred, dtype=int)

    if len(y_true_arr) == 0 or len(y_true_arr) != len(y_pred_arr):
        print("Warning: Invalid or empty data provided for evaluation.")
        return np.zeros((num_classes, num_classes), dtype=int), 0.0, 0.0, np.nan

    try:
        cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    except ValueError as e:
        print(f"Error calculating confusion matrix: {e}")
        return np.zeros((num_classes, num_classes), dtype=int), 0.0, 0.0, np.nan

    accuracy = np.sum(y_true_arr == y_pred_arr) / len(y_true_arr)
    mae = np.mean(np.abs(y_true_arr - y_pred_arr))

    try:
        qwk = cohen_kappa_score(y_true_arr, y_pred_arr, weights='quadratic', labels=labels)
    except Exception as e:
        print(f"Error calculating Quadratic Weighted Kappa: {e}")
        qwk = np.nan

    print("Evaluation Metrics Calculated:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  Quadratic Weighted Kappa (QWK): {qwk:.4f}" if not np.isnan(qwk) else "  Quadratic Weighted Kappa (QWK): N/A")

    return cm, accuracy, mae, qwk

# ==============================================================================
# XAI Helper Functions
# ==============================================================================

def denormalize(tensor: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    """Denormalizes a PyTorch tensor image using mean and std."""
    tensor = tensor.clone().detach()
    mean_t = torch.tensor(mean, device=tensor.device).view(3, 1, 1)
    std_t = torch.tensor(std, device=tensor.device).view(3, 1, 1)
    tensor.mul_(std_t).add_(mean_t)
    tensor.clamp_(0, 1)
    return tensor

def apply_colormap(heatmap: np.ndarray, cmap=cv2.COLORMAP_JET) -> np.ndarray:
    """Applies a specified OpenCV colormap to a normalized [0, 1] heatmap."""
    if not isinstance(heatmap, np.ndarray) or heatmap.ndim != 2:
        print("Warning: Invalid heatmap format for colormap.")
        return np.zeros((*heatmap.shape, 3), dtype=np.uint8)

    heatmap = np.clip(heatmap, 0, 1)
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cmap)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
    return colored_heatmap

def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Normalizes a heatmap numpy array to the range [0, 1]."""
    if not isinstance(heatmap, np.ndarray) or heatmap.ndim != 2:
        print("Warning: Invalid heatmap format for normalization.")
        return np.zeros_like(heatmap, dtype=np.float32)

    min_val, max_val = np.min(heatmap), np.max(heatmap)
    if max_val - min_val > 1e-6:
        heatmap = (heatmap - min_val) / (max_val - min_val)
    else:
        heatmap = np.zeros_like(heatmap, dtype=np.float32)
    return heatmap

def create_circular_mask(h: int, w: int, center: Optional[Tuple[int, int]] = None, radius: Optional[int] = None) -> np.ndarray:
    """Creates a boolean circular mask."""
    if center is None:
        center = (int(w/2), int(h/2))
    if radius is None:
        radius = min(center[0], center[1], w-center[0], h-center[1])

    if not isinstance(center, tuple) or len(center) != 2 or not all(isinstance(c, int) for c in center):
        print(f"Warning: Invalid center {center} for mask. Using image center.")
        center = (int(w/2), int(h/2))
    if not isinstance(radius, int) or radius <= 0:
        print(f"Warning: Invalid radius {radius} for mask. Using default radius.")
        radius = min(h, w) // 2

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask

def apply_apex_mask(heatmap: np.ndarray, apex_center: Optional[Tuple[int, int]] = None, radius: Optional[int] = None) -> np.ndarray:
    """Applies a circular mask centered around the presumed apex."""
    if not isinstance(heatmap, np.ndarray) or heatmap.ndim != 2:
        print("Warning: Invalid heatmap format for masking. Skipping mask.")
        return heatmap

    h, w = heatmap.shape

    try:
        if radius is not None:
            radius = int(radius)
            if radius <= 0:
                print("Warning: Mask radius must be positive. Skipping mask.")
                return heatmap
    except (ValueError, TypeError):
        print(f"Warning: Invalid radius '{radius}' provided for mask. Skipping mask.")
        return heatmap

    if apex_center is not None:
        if not isinstance(apex_center, tuple) or len(apex_center) != 2 or not all(isinstance(c, int) for c in apex_center):
            print(f"Warning: Invalid apex_center {apex_center} for mask. Using image center.")
            apex_center = (int(w/2), int(h/2))
    else:
        apex_center = (int(w/2), int(h/2))

    try:
        mask = create_circular_mask(h, w, center=apex_center, radius=radius)
        return heatmap * mask
    except Exception as e:
        print(f"Error applying circular mask: {e}. Returning original heatmap.")
        return heatmap

# ==============================================================================
# CAM Generation Functions
# ==============================================================================

def get_cam_heatmap(
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    class_idx_0_based: Optional[int] = None,
    method: str = 'gradcam',
    target_size: Tuple[int, int] = (300, 300),
    preserve_range: bool = False
) -> Tuple[np.ndarray, int]:
    """
    FIXED: Enhanced CAM generation with better normalization control.
    
    Args:
        preserve_range (bool): If True, preserve original CAM range for better class differentiation
    """
    model.eval()

    if not PYTORCH_GRAD_CAM_AVAILABLE:
        target_class_0_based_used = -1
        if class_idx_0_based is None:
            try:
                with torch.no_grad():
                    output = model(input_tensor)
                    target_class_0_based_used = torch.argmax(output, dim=1)[0].item()
            except Exception:
                pass
        else:
            try:
                target_class_0_based_used = int(class_idx_0_based)
            except Exception:
                pass
        return np.zeros(target_size, dtype=np.float32), target_class_0_based_used

    # Determine target class
    actual_target_class_0_based = class_idx_0_based
    if actual_target_class_0_based is None:
        try:
            with torch.no_grad():
                output = model(input_tensor)
                actual_target_class_0_based = torch.argmax(output, dim=1)[0].item()
        except Exception as e:
            print(f"Error predicting target class for CAM: {e}")
            return np.zeros(target_size, dtype=np.float32), -1
    else:
        try:
            actual_target_class_0_based = int(actual_target_class_0_based)
        except (ValueError, TypeError):
            print(f"Error: Invalid target_class_0_based provided for CAM: {class_idx_0_based}")
            return np.zeros(target_size, dtype=np.float32), -1

    try:
        targets = [ClassifierOutputTarget(actual_target_class_0_based)]
        
        # Select CAM algorithm
        method_lower = method.lower()
        if method_lower == 'gradcam++':
            cam_algorithm = GradCAMPlusPlus
        elif method_lower == 'scorecam':
            cam_algorithm = ScoreCAM
        elif method_lower == 'layercam':
            cam_algorithm = LayerCAM
        elif method_lower == 'gradcam':
            cam_algorithm = GradCAM
        else:
            print(f"Warning: Unknown CAM method '{method}'. Using default GradCAM.")
            cam_algorithm = GradCAM

        cam = cam_algorithm(model=model, target_layers=[target_layer])
        
        # Ensure proper tensor dimensions
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)
        elif input_tensor.ndim != 4:
            print(f"Error: CAM input tensor has unexpected dims {input_tensor.shape}")
            return np.zeros(target_size, dtype=np.float32), actual_target_class_0_based

        # Generate CAM
        grayscale_cam = cam(input_tensor=input_tensor,
                           targets=targets,
                           aug_smooth=False,
                           eigen_smooth=False)[0, :]

        # Resize if needed
        if grayscale_cam.shape[:2] != target_size:
            grayscale_cam = cv2.resize(grayscale_cam, target_size[::-1], interpolation=cv2.INTER_LINEAR)

        # FIXED: Enhanced normalization options for better class differentiation
        if preserve_range:
            # Preserve the original range for better class differentiation
            heatmap_norm = np.clip(grayscale_cam, 0, 1)
        else:
            # Standard normalization to [0,1]
            heatmap_norm = normalize_heatmap(grayscale_cam)
        
        return heatmap_norm, actual_target_class_0_based

    except Exception as e:
        print(f"Error using pytorch-grad-cam method '{method}' for target class {actual_target_class_0_based}: {e}")
        traceback.print_exc()
        return np.zeros(target_size, dtype=np.float32), actual_target_class_0_based


def identify_target_layers(model: torch.nn.Module, layer_indices: List[int] = [1, 2, 3, 4, 5, 6]) -> Tuple[Optional[Dict[str, torch.nn.Module]], List[str]]:
    """Identifies target convolutional layers within the model's feature extractor."""
    target_layers_dict: Dict[str, torch.nn.Module] = {}
    layer_names_list: List[str] = []
    
    try:
        if not hasattr(model, 'blocks') or not isinstance(model.blocks, nn.Sequential):
            print("Error: Model does not have 'blocks' attribute of type nn.Sequential.")
            if hasattr(model, 'encoder') and isinstance(model.encoder, nn.Sequential):
                print("Attempting to use model.encoder instead of model.blocks.")
                blocks_module = model.encoder
            elif hasattr(model, 'stages') and isinstance(model.stages, nn.Sequential):
                print("Attempting to use model.stages instead of model.blocks.")
                blocks_module = model.stages
            else:
                raise AttributeError("Could not find standard blocks/encoder/stages attribute in model.")
        else:
            blocks_module = model.blocks

        total_blocks = len(blocks_module)
        print(f"Found {total_blocks} blocks in the model.")

        for idx in layer_indices:
            if 0 <= idx < total_blocks:
                layer_name = f"block_{idx}"
                target_layers_dict[layer_name] = blocks_module[idx]
                layer_names_list.append(layer_name)
            else:
                print(f"Warning: Layer index {idx} out of bounds for model blocks (length {total_blocks}). Skipping.")

        if not target_layers_dict:
            raise ValueError("No target layers could be identified with the provided indices.")

        print(f"Target layers identified: {list(target_layers_dict.keys())}")
        return target_layers_dict, layer_names_list

    except (AttributeError, IndexError, ValueError) as e:
        print(f"Error identifying target layers: {e}")
        return None, []

def get_multi_layer_gradcam(
    model: torch.nn.Module,
    target_layers_dict: Dict[str, torch.nn.Module],
    input_tensor: torch.Tensor,
    device: torch.device,
    target_class_0_based: Optional[int] = None,
    gradcam_method: str = 'gradcam',
    target_size: Tuple[int, int] = (300, 300)
) -> Tuple[Dict[str, np.ndarray], int]:
    """Computes CAM for multiple layers and combines them using weighted average."""
    model.eval()
    layer_names = list(target_layers_dict.keys())
    if not layer_names:
        print("Error: target_layers_dict is empty.")
        return {}, -1

    # Determine target class
    actual_target_class_0_based = target_class_0_based
    if actual_target_class_0_based is None:
        try:
            with torch.no_grad():
                output = model(input_tensor)
                actual_target_class_0_based = torch.argmax(output, dim=1)[0].item()
        except Exception as e:
            print(f"Error predicting target class in get_multi_layer_gradcam: {e}")
            return {}, -1
    else:
        try:
            actual_target_class_0_based = int(actual_target_class_0_based)
        except (ValueError, TypeError):
            print(f"Error: Invalid target_class_0_based provided: {target_class_0_based}")
            return {}, -1

    # Calculate CAM for each layer
    heatmaps: Dict[str, np.ndarray] = {}
    for name, layer_module in target_layers_dict.items():
        heatmap_norm, used_target_class = get_cam_heatmap(
            model=model,
            target_layer=layer_module,
            input_tensor=input_tensor,
            device=device,
            class_idx_0_based=actual_target_class_0_based,
            method=gradcam_method,
            target_size=target_size
        )

        if used_target_class == -1 or not isinstance(heatmap_norm, np.ndarray) or heatmap_norm.shape != target_size:
            print(f"    Warning: Failed to generate valid heatmap for layer '{name}'. Storing zero map.")
            heatmaps[name] = np.zeros(target_size, dtype=np.float32)
        elif used_target_class != actual_target_class_0_based:
            print(f"    Warning: Target class mismatch for layer '{name}'. Expected {actual_target_class_0_based}, got {used_target_class}.")
            heatmaps[name] = np.zeros(target_size, dtype=np.float32)
        else:
            heatmaps[name] = heatmap_norm

    # Calculate combined weighted heatmap
    combined = np.zeros(target_size, dtype=np.float32)
    combine_weights = np.linspace(0.5, 1.0, len(layer_names))
    valid_heatmaps_count = 0
    actual_weights_sum = 0.0

    for i, name in enumerate(layer_names):
        heatmap = heatmaps.get(name)
        if isinstance(heatmap, np.ndarray) and heatmap.shape == target_size and np.any(heatmap):
            weight = combine_weights[i]
            combined += weight * heatmap
            valid_heatmaps_count += 1
            actual_weights_sum += weight

    if valid_heatmaps_count > 0 and actual_weights_sum > 1e-6:
        combined /= actual_weights_sum
    else:
        combined = np.zeros(target_size, dtype=np.float32)

    heatmaps['combined'] = normalize_heatmap(combined)
    return heatmaps, actual_target_class_0_based

# ==============================================================================
# Visualization Functions
# ==============================================================================

def generate_save_individual_visualization(
    img_pil: Image.Image,
    heatmaps_orig: Dict[str, np.ndarray],
    true_pai_1idx: int,
    pred_pai_1idx: int,
    probs_np: np.ndarray,
    layer_names: List[str],
    current_filename: str,
    save_dir: str,
    example_index: int,
    gradcam_method: str,
    model: torch.nn.Module,
    last_layer_module: torch.nn.Module,
    last_layer_name: str,
    input_tensor: torch.Tensor,
    device: torch.device,
    save_dpi: int = 150,
    png_compress_level: int = 6,
    num_classes: int = 5
) -> bool:
    """Generates and saves the multi-panel XAI visualization for one sample."""
    try:
        num_layers = len(layer_names)
        classes_plt = np.arange(1, num_classes + 1)
        img_array_norm = np.array(img_pil) / 255.0

        # Create figure and layout
        fig_width = max(12, 3 * (num_layers + 2))
        fig_height = 16
        fig = plt.figure(figsize=(fig_width, fig_height))

        gs = gridspec.GridSpec(4, 6, figure=fig, height_ratios=[1, 1, 1, 1])
        fig.suptitle(f"PAI - True: {true_pai_1idx}, Pred: {pred_pai_1idx} ({os.path.basename(current_filename)}) | Method: {gradcam_method.upper()}", fontsize=18)

        # Row 1: Original Image & Probabilities
        ax_img = fig.add_subplot(gs[0, 0:2])
        ax_img.imshow(img_pil)
        ax_img.set_title("Original", fontsize=14)
        ax_img.axis('off')

        ax_prob = fig.add_subplot(gs[0, 2:6])
        colors_plt = []
        for c_idx in range(num_classes):
            c_1based = c_idx + 1
            is_true = (c_1based == true_pai_1idx)
            is_pred = (c_1based == pred_pai_1idx)
            if is_true and is_pred:
                colors_plt.append('green')
            elif is_true:
                colors_plt.append('orange')
            elif is_pred:
                colors_plt.append('red')
            else:
                colors_plt.append('blue')
        
        ax_prob.bar(classes_plt, probs_np, color=colors_plt)
        ax_prob.set_title('Probabilities', fontsize=14)
        ax_prob.set_xlabel('PAI', fontsize=12)
        ax_prob.set_ylabel('Prob', fontsize=12)
        ax_prob.set_ylim(0, 1.05)
        ax_prob.set_xticks(classes_plt)
        ax_prob.grid(True, axis='y', alpha=0.3)
        
        legend_elements = [
            Patch(facecolor='green', label='True & Pred'),
            Patch(facecolor='orange', label='True Only'),
            Patch(facecolor='red', label='Pred Only'),
            Patch(facecolor='blue', label='Other')
        ]
        ax_prob.legend(handles=legend_elements, loc='upper right', fontsize='medium')

        # Row 2: Layer Heatmaps
        block_titles = [name.replace("_", " ").title() for name in layer_names]
        for k, name in enumerate(layer_names):
            if k >= 6:
                print(f"Warning: More than 6 layers ({num_layers}) specified, only plotting first 6 in row 2.")
                break
            ax = fig.add_subplot(gs[1, k])
            hmap = heatmaps_orig.get(name, np.zeros((300, 300)))
            if isinstance(hmap, np.ndarray) and hmap.shape == (300, 300):
                hmap_norm = normalize_heatmap(hmap)
                colored_hmap = apply_colormap(hmap_norm) / 255.0
                overlay = cv2.addWeighted(img_array_norm, 0.6, colored_hmap, 0.4, 0)
                ax.imshow(overlay)
            else:
                ax.imshow(img_pil)
                ax.text(0.5, 0.5, 'Heatmap Error', ha='center', va='center', color='red', transform=ax.transAxes)

            title = block_titles[k] if k < len(block_titles) else f"Layer {k+1}"
            ax.set_title(title, fontsize=11)
            ax.axis('off')

        # Hide unused subplots in Row 2
        for k in range(num_layers, 6):
            fig.add_subplot(gs[1, k]).axis('off')

        # Row 3: Combined & Difference Maps
        ax_combined = fig.add_subplot(gs[2, 0:3])
        h_comb = heatmaps_orig.get('combined', np.zeros((300,300)))
        if isinstance(h_comb, np.ndarray) and h_comb.shape == (300, 300):
            h_comb_norm = normalize_heatmap(h_comb)
            colored_h_comb = apply_colormap(h_comb_norm) / 255.0
            overlay_comb = cv2.addWeighted(img_array_norm, 0.6, colored_h_comb, 0.4, 0)
            ax_combined.imshow(overlay_comb)
        else:
            ax_combined.imshow(img_pil)
            ax_combined.text(0.5, 0.5, 'Combined Error', ha='center', va='center', color='red', transform=ax_combined.transAxes)
        ax_combined.set_title("Combined Overlay", fontsize=13)
        ax_combined.axis('off')

        # Difference map
        ax_diff = fig.add_subplot(gs[2, 3:6])
        if last_layer_name and last_layer_module:
            h_last = heatmaps_orig.get(last_layer_name, np.zeros((300,300)))
            if isinstance(h_last, np.ndarray) and isinstance(h_comb, np.ndarray) and h_last.shape == h_comb.shape:
                diff = np.abs(normalize_heatmap(h_last) - normalize_heatmap(h_comb))
                vmax_val = np.max(diff) if np.max(diff) > 1e-6 else 1.0
                im = ax_diff.imshow(diff, cmap='viridis', vmin=0, vmax=vmax_val)
                ax_diff.set_title(f"Difference ({last_layer_name} vs Comb)", fontsize=13)
                ax_diff.axis('off')
                plt.colorbar(im, ax=ax_diff, fraction=0.046, pad=0.04)
            else:
                ax_diff.imshow(img_pil)
                ax_diff.set_title("Difference Map Error", fontsize=13)
                ax_diff.axis('off')
        else:
            ax_diff.imshow(img_pil)
            ax_diff.set_title("Difference Map Error (Layer not found)", fontsize=13)
            ax_diff.axis('off')

        # Row 4: Last Layer Heatmaps for ALL Classes
        if last_layer_module:
            for class_idx_0 in range(num_classes):
                ax = fig.add_subplot(gs[3, class_idx_0])
                heatmap_class, _ = get_cam_heatmap(
                    model=model,
                    target_layer=last_layer_module,
                    input_tensor=input_tensor,
                    device=device,
                    class_idx_0_based=class_idx_0,
                    method=gradcam_method,
                    target_size=(300, 300)
                )
                if isinstance(heatmap_class, np.ndarray) and heatmap_class.shape == (300, 300):
                    hmap_norm = normalize_heatmap(heatmap_class)
                    colored_hmap = apply_colormap(hmap_norm) / 255.0
                    overlay = cv2.addWeighted(img_array_norm, 0.6, colored_hmap, 0.4, 0)
                    ax.imshow(overlay)
                else:
                    ax.imshow(img_pil)
                    ax.text(0.5, 0.5, 'Heatmap Error', ha='center', va='center', color='red', transform=ax.transAxes)

                ax.set_title(f"PAI {class_idx_0 + 1}", fontsize=11)
                ax.axis('off')

            if num_classes < 6:
                fig.add_subplot(gs[3, num_classes]).axis('off')
        else:
            for k in range(6):
                fig.add_subplot(gs[3, k]).axis('off')
            print("Warning: Last layer module not found, skipping Row 4 heatmaps.")

        # Save plot
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        base_filename_no_ext = os.path.splitext(os.path.basename(current_filename))[0]
        save_filename = f"example_{example_index}_true_{true_pai_1idx}_pred_{pred_pai_1idx}_{gradcam_method}_{base_filename_no_ext}.png"
        save_path = os.path.join(save_dir, save_filename)

        os.makedirs(save_dir, exist_ok=True)
        pil_save_kwargs = {'optimize': True, 'compress_level': png_compress_level}
        plt.savefig(save_path, dpi=save_dpi, bbox_inches='tight', format='png', pil_kwargs=pil_save_kwargs)
        plt.close(fig)
        return True

    except Exception as plot_err:
        print(f"Error generating/saving plot for {current_filename}: {plot_err}")
        plt.close('all')
        return False

def calculate_and_save_average_heatmaps(
    avg_heatmap_storage,
    heatmap_type,
    average_heatmap_dir,
    num_classes,
    gradcam_method='gradcam',
    apply_mask_radius=None,
    validate_class_differences=True
):
    """
    FIXED: Enhanced average heatmap calculation with class difference validation.
    """
    print(f"\nCalculating and saving average '{heatmap_type}' heatmaps (Method: {gradcam_method.upper()})...")
    os.makedirs(average_heatmap_dir, exist_ok=True)

    all_class_averages = {}  # Store for cross-class comparison

    for pai_class in range(1, num_classes + 1):
        heatmaps_list = avg_heatmap_storage.get(pai_class, [])
        if heatmaps_list:
            valid_heatmaps = [h for h in heatmaps_list if isinstance(h, np.ndarray) and h.ndim == 2 and h.shape[0] > 0 and h.shape[1] > 0]
            count = len(valid_heatmaps)
            if not valid_heatmaps:
                print(f"  No valid numpy heatmaps for PAI {pai_class}. Skipping.")
                continue
            
            print(f"  Averaging {count} '{heatmap_type}' heatmaps for PAI class {pai_class}...")
            
            try:
                stacked_heatmaps = np.stack(valid_heatmaps, axis=0)
                
                # ADDED: Calculate statistics for debugging
                mean_intensity = np.mean(stacked_heatmaps)
                std_intensity = np.std(stacked_heatmaps)
                max_intensity = np.max(stacked_heatmaps)
                min_intensity = np.min(stacked_heatmaps)
                print(f"    PAI {pai_class} - Count: {count}, Mean: {mean_intensity:.4f}, Std: {std_intensity:.4f}, Min: {min_intensity:.4f}, Max: {max_intensity:.4f}")
                
                # Calculate average
                average_heatmap = np.mean(stacked_heatmaps, axis=0)
                
                # Store for comparison
                all_class_averages[pai_class] = average_heatmap.copy()
                
                # Standard normalization for now
                average_heatmap_norm = normalize_heatmap(average_heatmap)

                # Apply masking if requested
                plot_title = f'Avg {heatmap_type.replace("_", " ").title()} {gradcam_method.upper()} PAI {pai_class}\n(n={count}, Upper Jaw Flipped)'
                if apply_mask_radius is not None:
                    print(f"    Applying post-hoc apex mask (radius: {apply_mask_radius} pixels)...")
                    average_heatmap_norm = apply_apex_mask(average_heatmap_norm, radius=apply_mask_radius)
                    plot_title = f'Avg {heatmap_type.replace("_", " ").title()} {gradcam_method.upper()} PAI {pai_class} (Masked)\n(n={count}, Upper Jaw Flipped)'

                # Plotting
                fig, ax = plt.subplots(figsize=(8, 8))
                im = ax.imshow(average_heatmap_norm, cmap='jet', vmin=0, vmax=1)
                plt.colorbar(im, ax=ax)
                ax.set_title(plot_title)
                ax.axis('off')

                # Save
                avg_save_filename = f'average_{heatmap_type}_heatmap_{gradcam_method}_pai_{pai_class}.png'
                avg_save_path = os.path.join(average_heatmap_dir, avg_save_filename)
                pil_save_kwargs = {'optimize': True, 'compress_level': 6}
                plt.savefig(avg_save_path, dpi=150, bbox_inches='tight', format='png', pil_kwargs=pil_save_kwargs)
                plt.close(fig)
                print(f"  Saved average {heatmap_type} heatmap for PAI {pai_class} to {avg_save_path}")

            except Exception as avg_err:
                print(f"  Error averaging/saving {heatmap_type} PAI {pai_class}: {avg_err}")
                traceback.print_exc()
                plt.close('all')
        else:
            print(f"  No '{heatmap_type}' heatmaps found for PAI class {pai_class} to average.")

    # ADDED: Validate class differences
    if validate_class_differences and len(all_class_averages) > 1:
        print(f"\n  🔍 Validating class differences for {heatmap_type}:")
        
        # Calculate center of mass for each class
        for pai_class, avg_map in all_class_averages.items():
            # Calculate center of mass (weighted centroid)
            y_coords, x_coords = np.mgrid[0:avg_map.shape[0], 0:avg_map.shape[1]]
            total_intensity = np.sum(avg_map)
            if total_intensity > 0:
                center_y = np.sum(y_coords * avg_map) / total_intensity
                center_x = np.sum(x_coords * avg_map) / total_intensity
                
                # Calculate spread (standard deviation of intensity-weighted positions)
                spread_y = np.sqrt(np.sum(((y_coords - center_y) ** 2) * avg_map) / total_intensity)
                spread_x = np.sqrt(np.sum(((x_coords - center_x) ** 2) * avg_map) / total_intensity)
                avg_spread = (spread_y + spread_x) / 2
                
                print(f"    PAI {pai_class}: Center=({center_x:.1f}, {center_y:.1f}), Spread={avg_spread:.1f}")
            
        # Compare classes pairwise
        class_pairs = [(i, j) for i in all_class_averages.keys() for j in all_class_averages.keys() if i < j]
        
        for pai_1, pai_2 in class_pairs:
            map_1 = all_class_averages[pai_1]
            map_2 = all_class_averages[pai_2]
            
            # Calculate difference metrics
            mse = np.mean((map_1 - map_2) ** 2)
            correlation = np.corrcoef(map_1.flatten(), map_2.flatten())[0, 1]
            
            print(f"    PAI {pai_1} vs PAI {pai_2}: MSE={mse:.6f}, Correlation={correlation:.4f}")
            
            if mse < 1e-4:
                print(f"    ⚠️  WARNING: Very similar heatmaps between PAI {pai_1} and {pai_2} (MSE < 1e-4)")
            if correlation > 0.95:
                print(f"    ⚠️  WARNING: Very high correlation between PAI {pai_1} and {pai_2} (r > 0.95)")

# ==============================================================================
# Results Saving Functions
# ==============================================================================

def save_results_to_csv(results: List[Dict[str, Any]], output_dir: str, prefix: str = "test_results"):
    """Saves the inference results list to a CSV file with a timestamp."""
    if not results:
        print("No results to save to CSV.")
        return
    try:
        results_df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f'{prefix}_{timestamp}.csv')
        results_df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
    except Exception as e:
        print(f"Error saving results CSV to {filename}: {e}")

def save_confusion_matrix(cm: np.ndarray, output_dir: str, class_labels: List[str], prefix: str = "confusion_matrix"):
    """Saves the confusion matrix as a heatmap PNG file."""
    if cm is None or not isinstance(cm, np.ndarray):
        print("Confusion matrix is None or invalid format, cannot save.")
        return
    if len(class_labels) != cm.shape[0] or len(class_labels) != cm.shape[1]:
        print("Warning: Class labels length mismatch with confusion matrix shape. Using generic labels.")
        class_labels = [str(i) for i in range(cm.shape[0])]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted PAI')
    plt.ylabel('True PAI')
    plt.title('Confusion Matrix')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f'{prefix}_{timestamp}.png')
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {filename}")
    except Exception as e:
        print(f"Error saving confusion matrix plot to {filename}: {e}")
    plt.close()

def save_statistics(accuracy: float, mae: float, qwk: float, output_dir: str, prefix: str = "test_metrics"):
    """Saves evaluation metrics to a text file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f'{prefix}_{timestamp}.txt')
    try:
        with open(filename, 'w') as f:
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Mean Absolute Error: {mae:.4f}\n")
            f.write(f"Quadratic Weighted Kappa: {qwk:.4f}\n")
        print(f"Statistics saved to {filename}")
    except Exception as e:
        print(f"Error saving statistics text file to {filename}: {e}")

def calculate_and_save_final_metrics(all_labels_0idx: List[int], all_preds_0idx: List[int], 
                                   all_probs: List[np.ndarray], all_filenames: List[str], 
                                   output_dir: str, num_classes: int = 5) -> Optional[Dict[str, Any]]:
    """Calculates final performance metrics and saves them."""
    print("\nCalculating final performance metrics...")
    if not all_labels_0idx or len(all_labels_0idx) != len(all_preds_0idx) or len(all_labels_0idx) != len(all_probs) or len(all_labels_0idx) != len(all_filenames):
        print("Error: Input lists for metrics calculation have inconsistent or zero length.")
        return None

    # Convert to 1-indexed for evaluate_predictions
    all_labels_1idx = [l + 1 for l in all_labels_0idx]
    all_preds_1idx = [p + 1 for p in all_preds_0idx]

    # Calculate metrics
    conf_matrix, accuracy, mae, quadratic_kappa = evaluate_predictions(all_labels_1idx, all_preds_1idx, num_classes=num_classes)

    # Save detailed results
    results_data = [{'filename': f, 'true_PAI': t, 'predicted_PAI': p, 'probabilities': ','.join([f"{pr:.4f}" for pr in prob])}
                    for f, t, p, prob in zip(all_filenames, all_labels_1idx, all_preds_1idx, all_probs)]
    save_results_to_csv(results_data, output_dir, prefix="prediction_results")

    # Save confusion matrix
    class_labels_1_based = [f"PAI {i}" for i in range(1, num_classes + 1)]
    save_confusion_matrix(conf_matrix, output_dir, class_labels=class_labels_1_based, prefix="confusion_matrix")

    # Save metrics
    save_statistics(accuracy, mae, quadratic_kappa, output_dir, prefix="final_metrics")

    # Prepare and save JSON
    metrics: Dict[str, Any] = {
        'accuracy': float(accuracy),
        'mae': float(mae),
        'quadratic_weighted_kappa': float(quadratic_kappa) if not np.isnan(quadratic_kappa) else None,
        'confusion_matrix_1based_labels': class_labels_1_based,
        'confusion_matrix': conf_matrix.tolist()
    }

    metrics_save_path = os.path.join(output_dir, 'final_metrics.json')
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(metrics_save_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics dictionary to {metrics_save_path}")
    except Exception as e:
        print(f"Error saving metrics JSON to {metrics_save_path}: {e}")

    return metrics

# ==============================================================================
# Main Orchestration Function
# ==============================================================================

def run_xai_test(
    model_path: str,
    test_loader: DataLoader,
    mean: List[float],
    std: List[float],
    test_root_dir: str,
    output_dir: str,
    num_examples: int = 100,
    num_classes: int = 5,
    gradcam_method: str = 'gradcam',
    apply_mask_radius: Optional[int] = None,
    target_block_indices: List[int] = [1, 2, 3, 4, 5, 6]
) -> Optional[Dict[str, Any]]:
    """
    FIXED: Main orchestration function with corrected CAM generation logic.
    
    KEY FIX: Generate separate CAMs for predicted class (visualization) and true class (averaging).
    """
    start_time = datetime.now()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting XAI Test Run at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Using device: {device}")
    print(f"Using Grad-CAM Method: {gradcam_method}")

    # Setup output directories
    gradcam_individual_dir = os.path.join(output_dir, f"xai_individual_{gradcam_method.lower().replace('+', 'plus')}")
    average_heatmap_dir = os.path.join(output_dir, "average_heatmaps")
    try:
        os.makedirs(gradcam_individual_dir, exist_ok=True)
        os.makedirs(average_heatmap_dir, exist_ok=True)
        print(f"Output dirs ensured: {gradcam_individual_dir}, {average_heatmap_dir}")
    except OSError as e:
        print(f"Error creating output dirs: {e}")
        return None

    # Load model
    try:
        model = load_model_for_inference(
            checkpoint_path=model_path,
            device=device
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

    # Identify target layers
    target_layers_dict, layer_names = identify_target_layers(model, layer_indices=target_block_indices)
    if not target_layers_dict:
        print("Failed to identify target layers. Aborting.")
        return None

    last_layer_name = f"block_{target_block_indices[-1]}" if target_block_indices else None
    last_layer_module = target_layers_dict.get(last_layer_name) if last_layer_name else None

    if not last_layer_module:
        print(f"Error: Could not get module for last targeted layer '{last_layer_name}'. Aborting.")
        return None

    # Initialize tracking
    all_preds_0idx: List[int] = []
    all_labels_0idx: List[int] = []
    all_probs: List[np.ndarray] = []
    all_filenames: List[str] = []
    avg_heatmaps_combined: Dict[int, List[np.ndarray]] = defaultdict(list)
    avg_heatmaps_block_last: Dict[int, List[np.ndarray]] = defaultdict(list)
    processed_visualizations_count = 0
    total_samples_processed = 0

    print("\nStarting Inference and XAI processing...")
    
    # ==========================================================================
    # FIXED MAIN PROCESSING LOOP
    # ==========================================================================
    for batch_idx, data_batch in enumerate(tqdm(test_loader, desc="Processing Test Batches", leave=False)):
        if not isinstance(data_batch, (list, tuple)) or len(data_batch) != 4:
            print(f"Warning: Skipping batch {batch_idx} due to unexpected format.")
            continue

        inputs, labels_0idx, quadrants, batch_filenames = data_batch

        try:
            inputs = inputs.to(device)
            labels_0idx = labels_0idx.to(device)
            if inputs.numel() == 0 or labels_0idx.numel() == 0:
                print(f"Warning: Skipping batch {batch_idx} due to empty tensors after moving to device.")
                continue
        except Exception as e:
            print(f"Error moving batch {batch_idx} data to device: {e}. Skipping.")
            continue

        # Perform inference
        try:
            with torch.no_grad():
                outputs = model(inputs)
                if not torch.all(torch.isfinite(outputs)):
                    print(f"Warning: Non-finite outputs for batch {batch_idx}. Skipping.")
                    continue
                probabilities = F.softmax(outputs, dim=1)
                _, predicted_0idx = torch.max(outputs.data, 1)
        except Exception as e:
            print(f"Error during inference batch {batch_idx}: {e}. Skipping.")
            continue

        # Store batch results
        batch_preds_np = predicted_0idx.cpu().numpy()
        batch_labels_np = labels_0idx.cpu().numpy()
        all_preds_0idx.extend(batch_preds_np)
        all_labels_0idx.extend(batch_labels_np)
        all_probs.extend(probabilities.cpu().numpy())
        all_filenames.extend([str(f) for f in batch_filenames])
        total_samples_processed += inputs.size(0)

        # Process samples for XAI
        if processed_visualizations_count < num_examples:
            for i in range(inputs.size(0)):
                if processed_visualizations_count >= num_examples:
                    break

                current_filename = batch_filenames[i]
                img_tensor = inputs[i:i+1]
                true_label_0idx = batch_labels_np[i]
                pred_label_0idx = batch_preds_np[i]
                true_pai_1idx = true_label_0idx + 1
                pred_pai_1idx = pred_label_0idx + 1
                sample_probs_np = probabilities[i].cpu().numpy()
                try:
                    quadrant = int(quadrants[i].item())
                except Exception:
                    quadrant = -1

                try:
                    # =============================================================
                    # CRITICAL FIX: Generate TWO sets of CAMs
                    # 1. For PREDICTED class (for individual visualizations)
                    # 2. For TRUE class (for averaging)
                    # =============================================================
                    
                    # Generate CAMs for PREDICTED class (for visualization)
                    heatmaps_pred, used_pred_label_0idx = get_multi_layer_gradcam(
                        model,
                        target_layers_dict,
                        img_tensor,
                        device,
                        target_class_0_based=pred_label_0idx,  # PREDICTED class for display
                        gradcam_method=gradcam_method
                    )

                    # Generate CAMs for TRUE class (for averaging collection)
                    heatmaps_true, used_true_label_0idx = get_multi_layer_gradcam(
                        model,
                        target_layers_dict,
                        img_tensor,
                        device,
                        target_class_0_based=true_label_0idx,  # TRUE class for averaging
                        gradcam_method=gradcam_method
                    )

                    # Check CAM generation success
                    visualization_ok = (used_pred_label_0idx != -1 and heatmaps_pred and 'combined' in heatmaps_pred and last_layer_name in heatmaps_pred)
                    averaging_ok = (used_true_label_0idx != -1 and heatmaps_true and 'combined' in heatmaps_true and last_layer_name in heatmaps_true)

                    if not visualization_ok:
                        print(f"Warning: Failed CAM generation for visualization of sample {current_filename}. Skipping visualization.")
                    
                    if not averaging_ok:
                        print(f"Warning: Failed CAM generation for averaging of sample {current_filename}. Skipping averaging.")

                    # Skip entirely if both failed
                    if not visualization_ok and not averaging_ok:
                        continue

                    # ============================================================
                    # INDIVIDUAL VISUALIZATION (using predicted class CAMs)
                    # ============================================================
                    if visualization_ok:
                        try:
                            original_img_path = os.path.join(test_root_dir, current_filename)
                            if os.path.exists(original_img_path):
                                img_pil = Image.open(original_img_path).convert('RGB').resize((300, 300))
                            else:
                                print(f"Warning: Original image file not found: {original_img_path}. Using denormalized tensor.")
                                img_denorm = denormalize(img_tensor.squeeze(0).cpu(), mean, std)
                                img_pil = transforms.ToPILImage()(img_denorm)

                            plot_success = generate_save_individual_visualization(
                                img_pil=img_pil,
                                heatmaps_orig=heatmaps_pred,  # Use PREDICTED class heatmaps for visualization
                                true_pai_1idx=true_pai_1idx,
                                pred_pai_1idx=pred_pai_1idx,
                                probs_np=sample_probs_np,
                                layer_names=layer_names,
                                current_filename=current_filename,
                                save_dir=gradcam_individual_dir,
                                example_index=processed_visualizations_count,
                                gradcam_method=gradcam_method,
                                model=model,
                                last_layer_module=last_layer_module,
                                last_layer_name=last_layer_name,
                                input_tensor=img_tensor,
                                device=device,
                                num_classes=num_classes
                            )
                            if plot_success:
                                processed_visualizations_count += 1
                                if processed_visualizations_count % 10 == 0:
                                    print(f"  Saved {processed_visualizations_count}/{num_examples} individual visualizations.")

                        except Exception as plot_err:
                            print(f"Error during individual plot for {current_filename}: {plot_err}")

                    # ============================================================
                    # AVERAGING COLLECTION (using TRUE class CAMs)
                    # ============================================================
                    if averaging_ok:
                        combined_heatmap_avg = heatmaps_true['combined'].copy()  # Use TRUE class heatmaps
                        block_last_heatmap_avg = heatmaps_true[last_layer_name].copy()
                        
                        # Apply quadrant-based flips for anatomical alignment
                        if isinstance(combined_heatmap_avg, np.ndarray) and isinstance(block_last_heatmap_avg, np.ndarray):
                            # Horizontal flip for left side quadrants (2 and 3)
                            if quadrant in [2, 3]:
                                combined_heatmap_avg = np.fliplr(combined_heatmap_avg)
                                block_last_heatmap_avg = np.fliplr(block_last_heatmap_avg)
                            
                            # Vertical flip for upper jaw (quadrants 1 and 2)
                            if quadrant in [1, 2]:
                                combined_heatmap_avg = np.flipud(combined_heatmap_avg)
                                block_last_heatmap_avg = np.flipud(block_last_heatmap_avg)
                        
                        # Store for averaging using TRUE PAI class (this is now correct!)
                        if isinstance(combined_heatmap_avg, np.ndarray) and combined_heatmap_avg.shape == (300, 300):
                            avg_heatmaps_combined[true_pai_1idx].append(combined_heatmap_avg)
                        if isinstance(block_last_heatmap_avg, np.ndarray) and block_last_heatmap_avg.shape == (300, 300):
                            avg_heatmaps_block_last[true_pai_1idx].append(block_last_heatmap_avg)

                except Exception as sample_processing_err:
                    print(f"Error processing sample {current_filename}: {sample_processing_err}")
                    traceback.print_exc()
                    continue

    print(f"\nFinished processing {total_samples_processed} samples.")

    # Calculate and save average heatmaps
    calculate_and_save_average_heatmaps(
        avg_heatmaps_combined,
        'combined',
        average_heatmap_dir,
        num_classes,
        gradcam_method,
        apply_mask_radius=apply_mask_radius
    )
    calculate_and_save_average_heatmaps(
        avg_heatmaps_block_last,
        last_layer_name,
        average_heatmap_dir,
        num_classes,
        gradcam_method,
        apply_mask_radius=apply_mask_radius
    )

    # Calculate and save final metrics
    final_metrics = calculate_and_save_final_metrics(all_labels_0idx, all_preds_0idx, all_probs, all_filenames, output_dir, num_classes)

    # Final summary
    end_time = datetime.now()
    print(f"\nXAI Test Run finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {end_time - start_time}")
    print(f"Outputs saved in directory: {output_dir}")

    return final_metrics