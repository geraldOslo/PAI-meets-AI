# ==============================================================================
# Configuration Parameters for PAI Classification XAI Inference
# ==============================================================================

import os

# ==============================================================================
# Environment Setup
# ==============================================================================
# Custom Package Path (Modify/Remove if not needed)
CUSTOM_PACKAGE_PATH = '/fp/projects01/ec192/python_packages_2'

# ==============================================================================
# Data Configuration
# ==============================================================================
# Test data paths
TEST_CSV_FILE = "/projects/ec192/data/endo-radiographs/GSclips300/data.csv"
TEST_ROOT_DIR = "/projects/ec192/data/endo-radiographs/GSclips300"

# Model checkpoint path
CHECKPOINT_PATH = '/projects/ec192/for_publishing_PAI/PAI-meets-AI/model_checkpoints/efficientnet_b3_NVIDIA_H100_PCIe_MIG_1g.20gb_20250531_121520_best.pth'

# Output directory
OUTPUT_DIR = "/projects/ec192/for_publishing_PAI/PAI-meets-AI/test-inference/test_efficientnet_b3_NVIDIA_H100_PCIe_MIG_1g.20gb_20250531_121520"

# ==============================================================================
# Model Configuration
# ==============================================================================
# Normalization parameters (MUST match training)
MEAN = [0.3784975] * 3
STD = [0.16739018] * 3

# Inference batch size
BATCH_SIZE = 100

# Number of classes (always 5 for PAI classification)
NUM_CLASSES = 5

# ==============================================================================
# XAI Configuration
# ==============================================================================
# CAM Methods configuration
GRAD_CAM_METHODS = ['gradcam', 'gradcam++', 'scorecam', 'layercam']
OUTPUT_FOLDER_NAMES = ['gradcam', 'gradcamplusplus', 'scorecam', 'layercam']
USE_METHODS = [0, 1, 2, 3]  # Indices of methods to run

# Target layer configuration for multi-layer analysis
TARGET_BLOCK_INDICES = [1, 2, 3, 4, 5, 6]  # EfficientNet blocks to analyze

# Visualization settings
NUM_EXAMPLES_TO_VISUALIZE = 100
APPLY_MASK_RADIUS_PIXELS = None  # Radius for circular mask on average heatmaps (None to disable)

# Visualization quality settings
SAVE_DPI = 150
PNG_COMPRESS_LEVEL = 6

# ==============================================================================
# Display Configuration
# ==============================================================================
# Whether to show sample clips during data loading
SHOW_CLIPS = False
CLIPS_TO_SHOW = 20
CLIPS_PER_ROW = 5

# ==============================================================================
# Helper Functions for Configuration
# ==============================================================================
def ensure_output_directory():
    """Ensure the output directory exists."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR

def get_method_output_dir(method_name):
    """Get the output directory for a specific CAM method."""
    method_dir = os.path.join(OUTPUT_DIR, f"xai_individual_{method_name.lower().replace('+', 'plus')}")
    os.makedirs(method_dir, exist_ok=True)
    return method_dir

def get_average_heatmap_dir():
    """Get the directory for average heatmaps."""
    avg_dir = os.path.join(OUTPUT_DIR, "average_heatmaps")
    os.makedirs(avg_dir, exist_ok=True)
    return avg_dir

def validate_config():
    """Validate the configuration parameters."""
    errors = []
    
    # Check file paths
    if not os.path.exists(TEST_CSV_FILE):
        errors.append(f"Test CSV file not found: {TEST_CSV_FILE}")
    
    if not os.path.exists(TEST_ROOT_DIR):
        errors.append(f"Test root directory not found: {TEST_ROOT_DIR}")
    
    if not os.path.exists(CHECKPOINT_PATH):
        errors.append(f"Checkpoint file not found: {CHECKPOINT_PATH}")
    
    # Check configuration values
    if len(MEAN) != 3 or len(STD) != 3:
        errors.append("MEAN and STD must have exactly 3 values for RGB channels")
    
    if BATCH_SIZE <= 0:
        errors.append("BATCH_SIZE must be positive")
    
    if NUM_EXAMPLES_TO_VISUALIZE <= 0:
        errors.append("NUM_EXAMPLES_TO_VISUALIZE must be positive")
    
    # Check USE_METHODS indices
    if any(i < 0 or i >= len(GRAD_CAM_METHODS) for i in USE_METHODS):
        errors.append("USE_METHODS contains invalid indices")
    
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
    
    return True

# ==============================================================================
# Configuration Summary
# ==============================================================================
def print_config_summary():
    """Print a summary of the current configuration."""
    print("=== PAI XAI Inference Configuration ===")
    print(f"Test CSV: {TEST_CSV_FILE}")
    print(f"Test Root: {TEST_ROOT_DIR}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Output Dir: {OUTPUT_DIR}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Examples to Visualize: {NUM_EXAMPLES_TO_VISUALIZE}")
    print(f"CAM Methods: {[GRAD_CAM_METHODS[i] for i in USE_METHODS]}")
    print(f"Mask Radius: {APPLY_MASK_RADIUS_PIXELS}")
    print("=" * 40)