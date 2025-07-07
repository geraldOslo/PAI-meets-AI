# data_utils.py
"""
Utilities for loading PAI datasets, creating dataloaders, and handling data transformations.

This module provides a robust framework for managing dental radiographic data,
including:
- A custom PyTorch Dataset (`CustomDataset`) to efficiently load images and labels.
- Functions to create flexible data augmentation pipelines (`create_data_transforms`).
- Logic for loading and splitting data from CSVs, with image existence checks and
  1-based to 0-based PAI label mapping (`load_datasets`).
- Functionality for setting up PyTorch DataLoaders, including support for
  weighted random sampling to address class imbalance (`create_dataloaders`).
- Utilities for displaying dataset statistics (`display_dataset_statistics`).

Author: Gerald Torgersen
SPDX-License-Identifier: MIT
Copyright (c) 2025 Gerald Torgersen, UiO
Contact: https://www.odont.uio.no/iko/english/people/aca/gerald/
"""

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from collections import Counter
import random
import traceback # Import traceback for detailed error logs
from sklearn.model_selection import train_test_split # Ensure sklearn is installed

# Import necessary types from typing for clear function signatures
from typing import Dict, List, Optional, Tuple, Union

# ============================================================================
# Constants (for colored terminal output)
# ============================================================================
# Define ANSI escape codes for colored console output to improve log readability
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_GREEN = "\033[92m"
COLOR_BLUE = "\033[94m"
COLOR_RESET = "\033[0m"


# ============================================================================
# Custom Dataset Class
# ============================================================================

class CustomDataset(Dataset):
    """
    Custom PyTorch Dataset for loading PAI (Periapical Index) images.

    This dataset expects a pandas DataFrame with 'filename', 'root_dir', and 'PAI' columns.
    The 'PAI' column should already be mapped to 0-based integer indices (0-4)
    prior to initializing this dataset.

    Attributes:
        data (pd.DataFrame): The preprocessed DataFrame containing image paths and labels.
        transform (transforms.Compose, optional): Composed torchvision transforms to apply to images.
    """
    def __init__(self, dataframe: pd.DataFrame, transform: Optional[transforms.Compose] = None):
        """
        Initializes the CustomDataset.

        Parameters
        ----------
        dataframe : pd.DataFrame
            DataFrame containing the dataset metadata. Must include 'filename', 'root_dir',
            and 'PAI' columns. 'PAI' should be 0-based integer labels (0-4).
        transform : torchvision.transforms.Compose, optional
            Torchvision transforms to apply to each image. Defaults to None.

        Raises
        ------
        TypeError
            If the input `dataframe` is not a pandas DataFrame.
        ValueError
            If essential columns ('filename', 'root_dir', 'PAI') are missing or if the
            'PAI' column cannot be converted to integer type.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input 'dataframe' must be a pandas DataFrame.")
        if dataframe.empty:
            print(f"{COLOR_YELLOW}Warning: Initializing CustomDataset with an empty DataFrame.{COLOR_RESET}")

        # Reset index to ensure robust integer-location based indexing after potential
        # filtering or splitting operations performed upstream.
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform

        # Verify essential columns exist in the DataFrame
        required_cols = ['filename', 'root_dir', 'PAI']
        if not all(col in self.data.columns for col in required_cols):
            missing = [col for col in required_cols if col not in self.data.columns]
            raise ValueError(f"DataFrame must contain columns: {required_cols}. Missing: {missing}")

        # Ensure 'PAI' column is of integer type.
        # This helps in robust label retrieval and avoids issues with mixed types.
        try:
            # Using 'Int64' initially to handle potential NaN values during type conversion,
            # then dropping NaNs and converting to standard 'int'.
            self.data['PAI'] = self.data['PAI'].astype('Int64', errors='ignore')
            self.data.dropna(subset=['PAI'], inplace=True)
            self.data['PAI'] = self.data['PAI'].astype(int)

            # Check if PAI values are within the expected 0-4 range
            if not self.data['PAI'].between(0, 4).all():
                print(f"{COLOR_YELLOW}Warning: 'PAI' column contains values outside the expected 0-4 range ({self.data['PAI'].min()}-{self.data['PAI'].max()}). This might indicate incorrect label mapping.{COLOR_RESET}")
        except Exception as e:
            raise ValueError(f"Could not convert 'PAI' column to integer or encountered an issue during conversion: {e}")

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns
        -------
        int
            The number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Fetches the sample (image, label, filename) at the given index.

        Parameters
        ----------
        idx : int
            The integer index of the sample to fetch (0-based).

        Returns
        -------
        Tuple[torch.Tensor, int, str]
            A tuple containing:
            - image (torch.Tensor): The processed image tensor.
            - label (int): The 0-based PAI label for the image.
            - filename (str): The original filename of the image.

        Raises
        ------
        IndexError
            If the provided `idx` is out of bounds for the dataset.
        FileNotFoundError
            If the image file cannot be found at the specified path.
        RuntimeError
            If there's an error accessing DataFrame row, opening/converting the image,
            or applying transformations.
        """
        # Basic bounds checking for the index
        if not 0 <= idx < len(self.data):
            raise IndexError(f"Index {idx} out of bounds for dataset length {len(self.data)}")

        # Retrieve row data using .iloc for robust integer-location based indexing
        try:
            row = self.data.iloc[idx]
            filename = str(row['filename'])
            root_dir = str(row['root_dir'])
            label = int(row['PAI']) # PAI should already be 0-4 integer
        except Exception as e:
            raise RuntimeError(f"Error accessing DataFrame row at index {idx}: {e}")

        img_path = os.path.join(root_dir, filename)

        try:
            # Check for exact file existence first
            if not os.path.exists(img_path):
                # Fallback: Attempt case-insensitive match if exact path fails.
                # Note: This can be slow for directories with many files.
                base_dir = os.path.dirname(img_path)
                base_name_lower = os.path.basename(img_path).lower()
                found_path = None
                if os.path.isdir(base_dir):
                    try:
                        # Iterate through directory contents to find a case-insensitive match
                        for fname in os.listdir(base_dir):
                            if fname.lower() == base_name_lower:
                                found_path = os.path.join(base_dir, fname)
                                break
                    except Exception as listdir_err:
                        print(f"{COLOR_YELLOW}Warning: Could not list directory '{base_dir}' for case-insensitive check: {listdir_err}{COLOR_RESET}")

                if found_path:
                    img_path = found_path # Update path to the found case-corrected one
                else:
                    # If no match found after fallback, raise FileNotFoundError
                    raise FileNotFoundError(f"Image file not found: {img_path} (Index: {idx})")

            # Load image using PIL and ensure it's in RGB format for consistent processing
            with Image.open(img_path) as img:
                image = img.convert('RGB')

        except FileNotFoundError:
            # Re-raise specifically for clarity for missing files
            raise FileNotFoundError(f"Image file not found: {img_path} (Index: {idx})")
        except Exception as e:
            # Catch and wrap other potential image loading or conversion errors
            raise RuntimeError(f"Error opening or converting image '{img_path}' (Index: {idx}): {e}")

        # Apply torchvision transformations if specified
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                raise RuntimeError(f"Error applying transforms to image '{img_path}' (Index: {idx}): {e}")

        return image, label, filename

    def get_target(self, idx: int) -> int:
        """
        Helper method to retrieve the target label for a given sample index.
        This method is particularly useful when creating a `WeightedRandomSampler`.

        Parameters
        ----------
        idx : int
            The index of the sample.

        Returns
        -------
        int
            The integer label (PAI, 0-4) for the sample at `idx`.

        Raises
        ------
        IndexError
            If the index is out of bounds for the dataset.
        RuntimeError
            If there's an unexpected error accessing the label from the DataFrame.
        """
        # Sampler should generally provide valid indices, but a check ensures robustness.
        if not 0 <= idx < len(self.data):
            raise IndexError(f"get_target called with index {idx} out of bounds for dataset length {len(self.data)}")
        try:
            # Access the 'PAI' column and convert numpy.int64 to standard Python int.
            return int(self.data.iloc[idx]['PAI'].item())
        except Exception as e:
            raise RuntimeError(f"Error accessing PAI label at index {idx} in get_target: {e}")


# ============================================================================
# Custom Transforms
# ============================================================================

class GammaTransform:
    """
    Applies random gamma correction to a PIL image.

    This transform adjusts the brightness and contrast of an image by applying
    a power-law transformation.
    """
    def __init__(self, gamma_range: Tuple[float, float] = (0.8, 1.2)):
        """
        Initializes the GammaTransform.

        Parameters
        ----------
        gamma_range : Tuple[float, float], optional
            A tuple (min, max) defining the range for the gamma correction factor.
            Values should be positive. Defaults to (0.8, 1.2).

        Raises
        ------
        ValueError
            If `gamma_range` is not a valid tuple of two positive numbers
            where min <= max.
        """
        if not (isinstance(gamma_range, tuple) and len(gamma_range) == 2 and
                gamma_range[0] <= gamma_range[1] and gamma_range[0] > 0):
            raise ValueError("`gamma_range` must be a tuple of two positive numbers (min, max) where min <= max.")
        self.gamma_range = gamma_range

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Applies the gamma correction transform to the input image.

        Parameters
        ----------
        img : PIL.Image.Image
            The input PIL image.

        Returns
        -------
        PIL.Image.Image
            The gamma-corrected image.
        """
        # Randomly select a gamma value within the specified range
        gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
        return transforms.functional.adjust_gamma(img, gamma)


# ============================================================================
# Data Transform Creation
# ============================================================================

def create_data_transforms(transform_config: Dict[str, Union[float, int, bool, Tuple]],
                           normalization_config: Dict[str, List[float]],
                           create_train: bool = True,
                           create_val: bool = True
                           ) -> Dict[str, transforms.Compose]:
    """
    Builds torchvision transform pipelines for training and validation/testing from
    configuration dictionaries.

    This function allows flexible configuration of various data augmentation techniques,
    including different spatial transformation strategies (RandomResizedCrop, Centered Affine Scaling,
    or simple Resize+CenterCrop).

    Parameters
    ----------
    transform_config : dict
        A dictionary containing configuration parameters for transformations
        (e.g., `DATA_TRANSFORM_CONFIG` from `config.py`).
    normalization_config : dict
        A dictionary containing normalization `mean` and `std` values
        (e.g., `NORMALIZATION` from `config.py`).
    create_train : bool, optional
        If True, the training transform pipeline will be created. Defaults to True.
    create_val : bool, optional
        If True, the validation/test transform pipeline will be created. Defaults to True.

    Returns
    -------
    Dict[str, transforms.Compose]
        A dictionary containing 'train' and/or 'val' transformation pipelines.
        Keys are 'train' and 'val', values are `transforms.Compose` objects.

    Raises
    ------
    ValueError
        If required keys are missing in the configuration dictionaries or if
        parameter values are invalid.
    RuntimeError
        If an unexpected error occurs during the creation or initialization
        of any transform.
    """
    data_transforms: Dict[str, transforms.Compose] = {}
    print("\n--- Creating Data Transforms ---")

    try:
        # --- Get basic config values and validate them ---
        cfg_input_size = int(transform_config.get("input_size", 300))
        if not isinstance(cfg_input_size, int) or cfg_input_size <= 0:
            raise ValueError(f"Invalid 'input_size': {cfg_input_size}. Must be a positive integer.")

        mean = normalization_config.get("mean")
        std = normalization_config.get("std")

        # Validate and convert mean/std to lists of floats
        if not (isinstance(mean, list) and len(mean) == 3 and all(isinstance(x, (int, float)) for x in mean)):
            raise ValueError(f"Invalid normalization 'mean': {mean}. Must be a list of 3 numbers.")
        if not (isinstance(std, list) and len(std) == 3 and all(isinstance(x, (int, float)) for x in std)):
            raise ValueError(f"Invalid normalization 'std': {std}. Must be a list of 3 numbers.")
        mean = [float(m) for m in mean] # Ensure float type
        std = [float(s) for s in std]   # Ensure float type

        # --- Configure Training Pipeline (if requested) ---
        if create_train:
            train_transforms_list = []
            print("--- Configuring Training Transforms ---")

            # Conditional data augmentations applied before spatial transforms
            if transform_config.get("use_horizontal_flip", False):
                print("  + RandomHorizontalFlip")
                train_transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))

            if transform_config.get("use_color_jitter", False):
                brightness = transform_config.get("brightness_jitter", 0.0)
                contrast = transform_config.get("contrast_jitter", 0.0)
                saturation = transform_config.get("saturation_jitter", 0.0)
                hue = transform_config.get("hue_jitter", 0.0)
                # Only add ColorJitter if at least one parameter is non-zero
                if brightness > 0 or contrast > 0 or saturation > 0 or hue > 0:
                    print(f"  + ColorJitter (b={brightness}, c={contrast}, s={saturation}, h={hue})")
                    train_transforms_list.append(transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue))
                else:
                    print("    (Skipping ColorJitter as all parameters are zero)")

            # --- Primary Spatial Transformation Logic ---
            # Prioritize Centered Affine Scaling if both it and RandomResizedCrop are enabled
            use_random_crop = transform_config.get("use_random_crop", False)
            use_centered_affine_scaling = transform_config.get("use_centered_affine_scaling", False)

            if use_random_crop and use_centered_affine_scaling:
                print(f"{COLOR_YELLOW}Warning: Both 'use_random_crop' and 'use_centered_affine_scaling' are True. Prioritizing 'use_centered_affine_scaling'.{COLOR_RESET}")
                use_random_crop = False # Disable random crop to ensure affine scaling is used

            if use_centered_affine_scaling:
                print("  + Using Centered Affine Scaling (Resize + RandomAffine + CenterCrop)")
                train_transforms_list.append(transforms.Resize(cfg_input_size))
                print(f"    - Resize ({cfg_input_size})")

                # Configure RandomAffine parameters from config, checking 'use_affine' and individual flags
                affine_degrees = transform_config.get("rotation_degrees", 0) if transform_config.get("use_rotation", False) else 0
                affine_translate = transform_config.get("translate_range", None) if transform_config.get("use_affine", False) else None
                affine_scale = transform_config.get("scale_range", None) if transform_config.get("use_affine", False) else None
                affine_shear = transform_config.get("shear_range", None) if transform_config.get("use_affine", False) else None

                # Add RandomAffine only if any parameter is configured to be active
                if affine_degrees != 0 or affine_translate is not None or affine_scale is not None or affine_shear is not None:
                    print(f"    - RandomAffine (degrees={affine_degrees}, translate={affine_translate}, scale={affine_scale}, shear={affine_shear})")
                    try:
                        train_transforms_list.append(transforms.RandomAffine(
                            degrees=affine_degrees,
                            translate=affine_translate,
                            scale=affine_scale,
                            shear=affine_shear,
                            interpolation=transforms.InterpolationMode.BILINEAR, # Specify interpolation method
                            fill=0 # Fill value for areas outside the image boundaries
                        ))
                    except Exception as e:
                        print(f"{COLOR_RED}Error initializing RandomAffine: {e}. Skipping transform.{COLOR_RESET}")
                        traceback.print_exc()
                else:
                    print("    (Skipping RandomAffine as no active parameters are set)")

                train_transforms_list.append(transforms.CenterCrop(cfg_input_size))
                print(f"    - CenterCrop ({cfg_input_size})")

            elif use_random_crop:
                print("  + Using RandomResizedCrop")
                scale = transform_config.get("random_crop_scale", (0.8, 1.0))
                ratio = transform_config.get("random_crop_ratio", (0.75, 1.333))
                if isinstance(scale, (list, tuple)) and len(scale) == 2 and scale[0] > 0 and scale[0] <= scale[1]:
                    print(f"    - RandomResizedCrop (size={cfg_input_size}, scale={scale}, ratio={ratio})")
                    train_transforms_list.append(transforms.RandomResizedCrop(cfg_input_size, scale=scale, ratio=ratio))
                else:
                    print(f"{COLOR_YELLOW}Warning: Skipping RandomResizedCrop due to invalid 'scale' parameter: {scale}.{COLOR_RESET}")

                # If using RRC, additional rotation might be applied separately
                if transform_config.get("use_rotation", False):
                    degrees = transform_config.get("rotation_degrees", 0)
                    if degrees > 0:
                        print(f"    - RandomRotation (degrees={degrees})")
                        train_transforms_list.append(transforms.RandomRotation(degrees=degrees))
                    else:
                        print("      (Skipping RandomRotation as 'degrees' is 0)")

            else: # Default behavior: Resize + CenterCrop if no specific spatial transform is enabled
                print("  + Using default Resize + CenterCrop")
                train_transforms_list.append(transforms.Resize(cfg_input_size))
                print(f"    - Resize ({cfg_input_size})")
                train_transforms_list.append(transforms.CenterCrop(cfg_input_size))
                print(f"    - CenterCrop ({cfg_input_size})")

                # If using default Resize+CenterCrop, additional rotation might be applied separately
                if transform_config.get("use_rotation", False):
                    degrees = transform_config.get("rotation_degrees", 0)
                    if degrees > 0:
                        print(f"    - RandomRotation (degrees={degrees})")
                        train_transforms_list.append(transforms.RandomRotation(degrees=degrees))
                    else:
                        print("      (Skipping RandomRotation as 'degrees' is 0)")

            # Other augmentations applied on PIL Image before conversion to Tensor
            if transform_config.get("use_gamma", False):
                gamma_range = transform_config.get("gamma_range", (1.0, 1.0))
                try:
                    # GammaTransform must be defined in this module or imported
                    train_transforms_list.append(GammaTransform(gamma_range=gamma_range))
                    print(f"  + GammaTransform (range={gamma_range})")
                except NameError:
                    print(f"{COLOR_RED}Error: GammaTransform class not found. Please ensure it's defined or imported correctly. Skipping.{COLOR_RESET}")
                except ValueError as ve:
                    print(f"{COLOR_RED}Error initializing GammaTransform: {ve}. Skipping.{COLOR_RESET}")

            if transform_config.get("use_blur", False):
                sigma = transform_config.get("blur_sigma_range", (0.1, 2.0))
                # Ensure sigma is a tuple (min, max) for GaussianBlur
                if isinstance(sigma, (int, float)):
                    sigma = (max(0.1, float(sigma)), float(sigma))
                if isinstance(sigma, (list, tuple)) and len(sigma) == 2 and sigma[0] >= 0 and sigma[0] <= sigma[1]:
                    # kernel_size must be an odd integer, e.g., 3, 5, 7, 9...
                    # Using a fixed kernel_size=7 as an example; this could be configurable.
                    train_transforms_list.append(transforms.GaussianBlur(kernel_size=7, sigma=sigma))
                    print(f"  + GaussianBlur (kernel_size=7, sigma={sigma})")
                else:
                    print(f"{COLOR_YELLOW}Warning: Skipping GaussianBlur due to invalid 'sigma' parameter: {sigma}. Must be a positive float or a tuple of positive floats.{COLOR_RESET}")


            # --- Core Transforms: Convert to Tensor and Normalize ---
            print("  + ToTensor")
            train_transforms_list.append(transforms.ToTensor())
            print(f"  + Normalize (mean={mean}, std={std})")
            train_transforms_list.append(transforms.Normalize(mean=mean, std=std))

            # --- Post-Normalization Augmentation (e.g., RandomErasing) ---
            if transform_config.get("use_random_erase", False):
                prob = transform_config.get("random_erase_probability", 0.0)
                scale_erase = transform_config.get("random_erase_scale", (0.02, 0.33))
                ratio_erase = transform_config.get("random_erase_ratio", (0.3, 3.3))
                if prob > 0:
                    print(f"  + RandomErasing (p={prob}, scale={scale_erase}, ratio={ratio_erase})")
                    train_transforms_list.append(transforms.RandomErasing(p=prob, scale=scale_erase, ratio=ratio_erase, value=0)) # value=0 for black box
                else:
                    print("    (Skipping RandomErasing as 'probability' is 0)")

            data_transforms['train'] = transforms.Compose(train_transforms_list)

        # --- Configure Validation/Test Pipeline (if requested) ---
        if create_val:
            print("\n--- Configuring Validation/Test Transforms ---")
            # Validation and test sets typically use fixed-size resizing and center cropping
            val_transforms_list = [
                transforms.Resize(cfg_input_size),
                transforms.CenterCrop(cfg_input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ]
            print(f"  + Resize ({cfg_input_size})")
            print(f"  + CenterCrop ({cfg_input_size})")
            print("  + ToTensor")
            print(f"  + Normalize (mean={mean}, std={std})")
            data_transforms['val'] = transforms.Compose(val_transforms_list)

    except (KeyError, ValueError) as e:
        print(f"{COLOR_RED}Error: Invalid configuration for transforms or normalization. Please check config.py: {e}{COLOR_RESET}")
        traceback.print_exc()
        raise ValueError(f"Invalid configuration for transforms/normalization: {e}") from e
    except Exception as e:
        print(f"{COLOR_RED}An unexpected error occurred during data transform creation: {e}{COLOR_RESET}")
        traceback.print_exc()
        raise RuntimeError(f"Data transform creation failed: {e}") from e

    print("\nData transforms creation complete.")
    return data_transforms


# ============================================================================
# Display Statistics Utility
# ============================================================================

def display_dataset_statistics(data: pd.DataFrame, name: str, target_col: str = 'PAI'):
    """
    Prints class distribution statistics for a given pandas DataFrame.

    This function provides insights into the balance of classes within a dataset
    split, which is crucial for medical imaging tasks with potential class imbalance.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the data for which to display statistics.
    name : str
        A descriptive name for the dataset split (e.g., "Initial Train", "Validation Split").
    target_col : str, optional
        The name of the column in the DataFrame that contains the class labels (0-based).
        Defaults to 'PAI'.
    """
    print(f"\n--- Statistics for {name} set ---")
    if not isinstance(data, pd.DataFrame):
        print(f"{COLOR_RED}Error: Input for '{name}' is not a pandas DataFrame. Skipping statistics.{COLOR_RESET}")
        return
    if target_col not in data.columns:
        print(f"{COLOR_YELLOW}Warning: Target column '{target_col}' not found in data for '{name}'. Skipping statistics.{COLOR_RESET}")
        return
    if data.empty:
        print(f"Total Samples: 0")
        print("DataFrame is empty.")
        print("----------------------------------")
        return

    try:
        # Get labels, drop any NaNs, and ensure they are integer type
        labels = data[target_col].dropna()
        if labels.empty:
            print(f"Total Samples: {len(data)}")
            print(f"No valid labels found in column '{target_col}' after dropping NaNs.")
            print("----------------------------------")
            return

        try:
            labels = labels.astype(int)
        except ValueError:
            print(f"{COLOR_RED}Error: Could not convert labels in column '{target_col}' to integer for '{name}'. Skipping statistics.{COLOR_RESET}")
            return

        class_counts = Counter(labels)
        total_samples = len(labels)

        if total_samples == 0:
            print(f"Total Samples: 0")
            print("No valid integer labels found after processing.")
            print("----------------------------------")
            return

        # Determine the full range of observed classes for consistent display
        min_label = min(class_counts.keys())
        max_label = max(class_counts.keys())
        all_possible_classes = list(range(min_label, max_label + 1))

        # Fill in counts for classes that might be missing in the current subset
        full_class_counts = {i: class_counts.get(i, 0) for i in all_possible_classes}
        class_percentages = {cls: (count / total_samples * 100) if total_samples > 0 else 0.0 for cls, count in full_class_counts.items()}

        stats_df = pd.DataFrame({
            'Class (0-based)': list(full_class_counts.keys()),
            'Count': list(full_class_counts.values()),
            'Percentage (%)': [f"{class_percentages[cls]:.1f}" for cls in full_class_counts.keys()]
        })
        stats_df = stats_df.sort_values(by='Class (0-based)').reset_index(drop=True)

        print(f"Total Samples: {total_samples}")
        print(stats_df.to_string(index=False))
        print("----------------------------------")

    except Exception as e:
        print(f"{COLOR_RED}An unexpected error occurred while calculating statistics for '{name}': {e}{COLOR_RESET}")
        traceback.print_exc()


# ============================================================================
# Data Loading and Splitting Function
# ============================================================================

def load_datasets(data_paths: Dict[str, Union[str, List[str]]],
                  transforms: Dict[str, transforms.Compose],
                  target_col: str = 'PAI', # Assumed 1-based in raw CSV
                  filename_col: str = 'filename',
                  split_test_size: float = 0.2,
                  split_random_state: int = 42,
                  load_split: Optional[str] = None) -> Tuple[Optional[CustomDataset], Optional[CustomDataset], Optional[CustomDataset], Optional[List[str]]]:
    """
    Loads dataset(s) based on configuration, handles train/val splitting or test set loading.

    This function is the primary entry point for loading PAI data. It performs:
    1. Reading metadata from specified CSV files.
    2. Adding a 'root_dir' column based on the configuration.
    3. Verifying the existence of image files and filtering out missing entries.
    4. Mapping original PAI labels (assumed 1-based in raw CSVs) to 0-based indices (0-4).
    5. Optionally, performing a stratified train/validation split or loading a dedicated test set.
    6. Creating `CustomDataset` objects with the specified transformations.
    7. Determining human-readable class names based on the loaded data.

    Parameters
    ----------
    data_paths : Dict[str, Union[str, List[str]]]
        A dictionary containing all necessary paths:
        - For train/val: 'csv_files' (list of paths), 'root_dirs' (list of paths).
        - For test: 'test_csv_file' (single path), 'test_root_dir' (single path).
        - Also expects 'filename_col' and 'label_col' if different from defaults.
    transforms : Dict[str, transforms.Compose]
        A dictionary with 'train' and 'val' (or 'test') transformation pipelines.
    target_col : str, optional
        The name of the column in the raw CSVs containing the original PAI labels
        (assumed to be 1-based, e.g., 1-5). Defaults to 'PAI'.
    filename_col : str, optional
        The name of the column in the raw CSVs containing image filenames or relative paths.
        Defaults to 'filename'.
    split_test_size : float, optional
        The fraction of the dataset to allocate for the validation split (0.0 to 1.0).
        Only applicable when `load_split` is None. Defaults to 0.2.
    split_random_state : int, optional
        The random seed for data splitting, ensuring reproducibility of the split.
        Defaults to 42.
    load_split : Optional[str], optional
        If 'test', loads only the test set using 'test_csv_file' and 'test_root_dir'.
        If None, loads all available data and performs a train/validation split.
        Defaults to None.

    Returns
    -------
    Tuple[Optional[CustomDataset], Optional[CustomDataset], Optional[CustomDataset], Optional[List[str]]]
        A tuple containing:
        - train_dataset (CustomDataset or None): The training dataset.
        - val_dataset (CustomDataset or None): The validation dataset.
        - test_dataset (CustomDataset or None): The test dataset (if `load_split='test'`).
        - class_names (List[str] or None): A list of human-readable class names (e.g., ['PAI 1', 'PAI 2', ...]).
        Non-relevant dataset objects will be `None`. Returns `(None, None, None, None)` on critical errors.
    """
    print(f"--- data_utils.load_datasets called (load_split='{load_split or 'train/val'}') ---")
    class_names: Optional[List[str]] = None
    train_dataset, val_dataset, test_dataset = None, None, None

    # Get column names from config, with robust fallbacks
    filename_col_actual = data_paths.get("filename_col", filename_col)
    label_col_actual = data_paths.get("label_col", target_col)

    if load_split == 'test':
        # --- Logic for Loading Only the Test Set ---
        print("--- Loading Test Set Only ---")
        test_csv_path = data_paths.get("test_csv_file")
        test_root_dir = data_paths.get("test_root_dir")

        # Validate essential inputs for test set loading
        if not test_csv_path:
            print(f"{COLOR_RED}Error: 'test_csv_file' path is missing in data_paths for test loading.{COLOR_RESET}")
            return None, None, None, None
        if not test_root_dir:
            print(f"{COLOR_RED}Error: 'test_root_dir' path is missing in data_paths for test loading.{COLOR_RESET}")
            return None, None, None, None
        if not os.path.exists(test_csv_path):
            print(f"{COLOR_RED}Error: Test CSV file not found at: {test_csv_path}{COLOR_RESET}")
            return None, None, None, None
        if not os.path.isdir(test_root_dir):
            print(f"{COLOR_RED}Error: Test root directory not found or is not a directory: {test_root_dir}{COLOR_RESET}")
            return None, None, None, None
        if 'val' not in transforms:
            print(f"{COLOR_RED}Error: 'val' transforms (used for testing) not provided in the transforms dictionary.{COLOR_RESET}")
            return None, None, None, None

        try:
            test_df = pd.read_csv(test_csv_path)
            print(f"Loading test set from '{test_csv_path}'. Initial rows: {len(test_df)}.")

            # Verify required columns exist using the actual configured names
            if filename_col_actual not in test_df.columns:
                raise KeyError(f"Filename column '{filename_col_actual}' not found in test CSV.")
            if label_col_actual not in test_df.columns:
                raise KeyError(f"Label column '{label_col_actual}' not found in test CSV.")

            test_df['root_dir'] = test_root_dir

            # --- Check for Existing Image Files in the Test Set ---
            print("Checking if test image files exist...")
            test_df['image_path_temp'] = test_df.apply(lambda row: os.path.join(row['root_dir'], str(row[filename_col_actual])), axis=1)
            exists_mask = test_df['image_path_temp'].apply(os.path.exists)
            original_count = len(test_df)
            test_df = test_df[exists_mask].drop(columns=['image_path_temp'])
            print(f"  Kept {len(test_df)} / {original_count} rows with existing image files.")
            if test_df.empty:
                print(f"{COLOR_RED}Error: No test samples remain after checking image file existence. Test dataset is empty.{COLOR_RESET}")
                return None, None, None, None

            # --- Map Original PAI Labels (assumed 1-based) to 0-4 ---
            print(f"Mapping original '{label_col_actual}' column (assumed 1-based) to 0-4...")
            try:
                # Convert to numeric, coerce errors to NaN, then convert to nullable integer type
                test_df['PAI'] = pd.to_numeric(test_df[label_col_actual], errors='coerce').astype('Int64') - 1
                test_df.dropna(subset=['PAI'], inplace=True) # Drop rows where PAI conversion failed
                test_df['PAI'] = test_df['PAI'].astype(int) # Convert to non-nullable integer
            except Exception as e:
                raise ValueError(f"Error mapping labels from '{label_col_actual}' to 0-4: {e}")

            if test_df.empty:
                print(f"{COLOR_RED}Error: No test samples remain after label mapping and cleaning. Test dataset is empty.{COLOR_RESET}")
                return None, None, None, None

            # Standardize filename column name for CustomDataset compatibility
            if filename_col_actual != 'filename':
                test_df = test_df.rename(columns={filename_col_actual: 'filename'})

            # Display statistics for the loaded test set
            display_dataset_statistics(test_df, 'Test Set', target_col='PAI')

            # Create the CustomDataset object for the test set
            test_dataset = CustomDataset(
                dataframe=test_df,
                transform=transforms.get('val') # Use validation transforms for testing
            )

            # Determine class names from the unique mapped labels (0-4) in the test set
            unique_labels = sorted(test_df['PAI'].unique())
            class_names = [f"PAI {i+1}" for i in unique_labels]
            print(f"Test dataset created. Class names: {class_names}")

            return None, None, test_dataset, class_names

        except (KeyError, ValueError) as e:
            print(f"{COLOR_RED}Error loading test data from '{test_csv_path}': {e}{COLOR_RESET}")
            traceback.print_exc()
            return None, None, None, None
        except Exception as e:
            print(f"{COLOR_RED}An unexpected error occurred during test dataset loading: {e}{COLOR_RESET}")
            traceback.print_exc()
            return None, None, None, None
    else:
        # --- Logic for Loading Train/Validation Data and Performing Split ---
        print("--- Loading Train/Validation Data & Splitting ---")
        metadata_csv_paths = data_paths.get("csv_files", [])
        root_dirs_list = data_paths.get("root_dirs", [])

        # Validate essential inputs for train/val loading
        if not metadata_csv_paths or not root_dirs_list:
            print(f"{COLOR_RED}Error: Missing 'csv_files' or 'root_dirs' in data_paths config for train/val loading.{COLOR_RESET}")
            return None, None, None, None
        if len(metadata_csv_paths) != len(root_dirs_list):
            print(f"{COLOR_RED}Error: Number of 'csv_files' must match number of 'root_dirs'.{COLOR_RESET}")
            return None, None, None, None
        if 'train' not in transforms or 'val' not in transforms:
            print(f"{COLOR_RED}Error: 'train' or 'val' transforms not provided in the transforms dictionary.{COLOR_RESET}")
            return None, None, None, None
        if not (0.0 < split_test_size < 1.0):
            print(f"{COLOR_RED}Error: Invalid split_test_size: {split_test_size}. Must be between 0.0 and 1.0 (exclusive).{COLOR_RESET}")
            return None, None, None, None


        # --- Load all data from specified CSVs ---
        all_data_list = []
        rows_before_check = 0 # Track total rows across all CSVs before filtering
        files_processed_count = 0 # Track how many CSV files were successfully processed
        print(f"Attempting to load metadata from: {metadata_csv_paths}")

        for csv_file, root_dir in zip(metadata_csv_paths, root_dirs_list):
            if not os.path.exists(csv_file):
                print(f"{COLOR_YELLOW}Warning: Metadata CSV file not found at: {csv_file}. Skipping this file.{COLOR_RESET}")
                continue
            if not os.path.isdir(root_dir):
                print(f"{COLOR_YELLOW}Warning: Root directory not found or is not a directory: {root_dir} (for {os.path.basename(csv_file)}). Skipping this file.{COLOR_RESET}")
                continue

            try:
                data = pd.read_csv(csv_file)
                rows_before_check += len(data)

                # Verify required columns exist before processing
                if filename_col_actual not in data.columns:
                    raise KeyError(f"Filename column '{filename_col_actual}' missing in CSV: {csv_file}")
                if label_col_actual not in data.columns:
                    raise KeyError(f"Label column '{label_col_actual}' missing in CSV: {csv_file}")

                data['root_dir'] = root_dir # Add the root directory to each row for image loading

                # --- Check if image files exist for each entry ---
                data['image_path_temp'] = data.apply(lambda row: os.path.join(row['root_dir'], str(row[filename_col_actual])), axis=1)
                data['image_exists'] = data['image_path_temp'].apply(os.path.exists)
                missing_count = len(data) - data['image_exists'].sum()

                if missing_count > 0:
                    print(f"  {COLOR_YELLOW}Warning: {missing_count} image file(s) from '{os.path.basename(csv_file)}' not found. Corresponding rows will be dropped.{COLOR_RESET}")

                data = data[data['image_exists']].drop(columns=['image_exists', 'image_path_temp'])
                if not data.empty:
                    all_data_list.append(data)
                    files_processed_count += 1
                    print(f"  Processed '{os.path.basename(csv_file)}', kept {len(data)} rows.")
                else:
                    print(f"  {COLOR_YELLOW}Warning: No valid rows kept from '{os.path.basename(csv_file)}' after file existence check. Skipping this file.{COLOR_RESET}")

            except KeyError as e:
                print(f"{COLOR_RED}Error processing CSV '{csv_file}': Missing expected column: {e}. Skipping this file.{COLOR_RESET}")
                traceback.print_exc()
                continue # Continue to the next CSV file
            except Exception as e:
                print(f"{COLOR_RED}An unexpected error occurred while processing CSV '{csv_file}': {e}. Skipping this file.{COLOR_RESET}")
                traceback.print_exc()
                continue # Continue to the next CSV file

        if not all_data_list:
            print(f"{COLOR_RED}Critical Error: No valid data loaded from any of the specified CSV files. Cannot proceed.{COLOR_RESET}")
            return None, None, None, None

        # Concatenate all successfully loaded and filtered dataframes
        all_data = pd.concat(all_data_list, ignore_index=True)
        print(f"\nSuccessfully processed {files_processed_count} CSV file(s).")
        print(f"Total rows found in CSVs before filtering: {rows_before_check}")
        print(f"Total rows kept after checking image files: {len(all_data)}")

        if all_data.empty:
            print(f"{COLOR_RED}Critical Error: Combined DataFrame is empty after processing all files. Cannot proceed with splitting.{COLOR_RESET}")
            return None, None, None, None

        try:
            # --- Map Original PAI Labels (assumed 1-based) to 0-4 ---
            print(f"Mapping original '{label_col_actual}' column (assumed 1-based) to 0-4...")
            try:
                # Convert to numeric, coerce errors to NaN, then convert to nullable integer type
                all_data['PAI'] = pd.to_numeric(all_data[label_col_actual], errors='coerce').astype('Int64') - 1
                all_data.dropna(subset=['PAI'], inplace=True) # Drop rows where PAI conversion failed
                all_data['PAI'] = all_data['PAI'].astype(int) # Convert to non-nullable integer
            except Exception as e:
                raise ValueError(f"Error mapping labels from '{label_col_actual}' to 0-4: {e}")

            if all_data.empty:
                print(f"{COLOR_RED}Error: No samples remain after label mapping and cleaning. Dataset is empty.{COLOR_RESET}")
                return None, None, None, None

            # Standardize filename column name for CustomDataset compatibility
            if filename_col_actual != 'filename':
                all_data = all_data.rename(columns={filename_col_actual: 'filename'})

            # --- Perform Stratified Train/Validation Split ---
            print(f"\nSplitting data ({1-split_test_size:.0%} train, {split_test_size:.0%} val) with stratification...")
            # Check if stratification is possible (each class must have at least 2 samples for train_test_split)
            min_stratify_count = all_data['PAI'].value_counts().min()
            if min_stratify_count < 2:
                print(f"{COLOR_YELLOW}Warning: Minimum class count for stratification is {min_stratify_count}. Stratification might be problematic or impossible for some classes with very few samples.{COLOR_RESET}")
                if min_stratify_count == 0:
                    print(f"{COLOR_RED}Error: Cannot perform stratified split: At least one class has zero samples after filtering. Consider cleaning data or using a non-stratified split.{COLOR_RESET}")
                    raise ValueError("Cannot perform stratified split: Some classes have zero samples.")

            train_df, val_df = train_test_split(
                all_data,
                test_size=split_test_size,
                random_state=split_random_state,
                stratify=all_data['PAI'] # Stratify on the 0-4 PAI column
            )
            print(f"Splitting complete.")

            # Display statistics for all dataset subsets
            display_dataset_statistics(all_data, 'Combined Initial Data', target_col='PAI')
            display_dataset_statistics(train_df, 'Initial Train Split', target_col='PAI')
            display_dataset_statistics(val_df, 'Validation Split (Unbalanced)', target_col='PAI')

            # --- Create CustomDataset Objects ---
            print("\nCreating Dataset objects...")
            train_dataset = CustomDataset(
                dataframe=train_df,
                transform=transforms.get('train')
            )
            val_dataset = CustomDataset(
                dataframe=val_df,
                transform=transforms.get('val')
            )

            # Determine class names from the unique mapped labels (0-4) in the entire dataset
            unique_labels = sorted(all_data['PAI'].unique())
            class_names = [f"PAI {i+1}" for i in unique_labels]
            print(f"Train/Val datasets created. Class names: {class_names}")

            return train_dataset, val_dataset, None, class_names

        except (KeyError, ValueError) as e:
            print(f"{COLOR_RED}Error during data processing or splitting: {e}. Check configured column names ('{label_col_actual}', '{filename_col_actual}') and data values.{COLOR_RESET}")
            traceback.print_exc()
            return None, None, None, None
        except Exception as e:
            print(f"{COLOR_RED}An unexpected error occurred during train/val data splitting or dataset creation: {e}{COLOR_RESET}")
            traceback.print_exc()
            return None, None, None, None


# ============================================================================
# Data Loader Creation Function
# ============================================================================

def create_dataloaders(train_dataset: Optional[CustomDataset],
                       val_dataset: Optional[CustomDataset],
                       test_dataset: Optional[CustomDataset],
                       dataloader_config: Dict[str, Union[int, bool, float]],
                       use_oversampler: bool = True) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """
    Creates PyTorch DataLoaders for train, validation, and optionally test sets.

    This function sets up DataLoaders with configurable batch sizes, worker counts,
    and memory pinning. It includes robust support for balanced sampling
    (via `WeightedRandomSampler`) for the training set to mitigate class imbalance.

    Parameters
    ----------
    train_dataset : Optional[CustomDataset]
        The training dataset object. Can be `None` if not creating a train loader.
    val_dataset : Optional[CustomDataset]
        The validation dataset object. Can be `None` if not creating a validation loader.
    test_dataset : Optional[CustomDataset]
        The test dataset object. Can be `None` if not creating a test loader.
    dataloader_config : Dict[str, Union[int, bool, float]]
        A dictionary containing configuration parameters for DataLoaders.
        Expected keys: 'batch_size', 'num_workers'.
        Optional keys: 'val_batch_size', 'test_batch_size', 'pin_memory',
                        'persistent_workers', 'prefetch_factor'.
    use_oversampler : bool, optional
        If True, a `WeightedRandomSampler` will be used for the training set
        to balance class distribution. Defaults to True.

    Returns
    -------
    Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]
        A tuple containing:
        - train_loader (DataLoader or None): The DataLoader for the training set.
        - val_loader (DataLoader or None): The DataLoader for the validation set.
        - test_loader (DataLoader or None): The DataLoader for the test set.
        Non-applicable loaders will be `None`.
    """
    train_loader, val_loader, test_loader = None, None, None
    print("\n--- Creating DataLoaders ---")

    # Retrieve common dataloader arguments from configuration with sensible defaults
    num_workers = dataloader_config.get('num_workers', 0)
    pin_memory = dataloader_config.get('pin_memory', False)
    # Persistent workers are only beneficial if num_workers > 0 and if datasets are not frequently recreated
    persistent_workers = dataloader_config.get('persistent_workers', False) and num_workers > 0
    # Prefetch factor is only relevant if num_workers > 0
    prefetch_factor = dataloader_config.get('prefetch_factor', 2) if num_workers > 0 else None

    # --- Training DataLoader Creation ---
    if train_dataset:
        if not isinstance(train_dataset, CustomDataset):
            print(f"{COLOR_RED}Error: 'train_dataset' is not a CustomDataset object. Cannot create train loader.{COLOR_RESET}")
        else:
            train_batch_size = dataloader_config.get('batch_size', 64)
            if not isinstance(train_batch_size, int) or train_batch_size <= 0:
                print(f"{COLOR_YELLOW}Warning: Invalid 'batch_size' ({train_batch_size}) for training. Setting to default 64.{COLOR_RESET}")
                train_batch_size = 64

            # Optional: Heuristic to reduce batch size for specific models (e.g., EfficientNet-B4)
            # It's generally better to set appropriate batch_size in config.py based on GPU memory.
            if train_batch_size > 64 and "b4" in train_dataset.transform.__repr__().lower():
                orig_train_batch_size = train_batch_size
                train_batch_size = min(64, train_batch_size)
                print(f"{COLOR_YELLOW}Warning: Reduced training batch size from {orig_train_batch_size} to {train_batch_size} to mitigate potential memory issues with EfficientNet-B4. Consider adjusting 'batch_size' in config.py directly.{COLOR_RESET}")

            if use_oversampler:
                print("Creating balanced training sampler (WeightedRandomSampler)...")
                try:
                    # Collect all labels from the training dataset using get_target method
                    all_labels = [train_dataset.get_target(i) for i in range(len(train_dataset))]

                    # Calculate class frequencies for inverse weighting
                    class_counts = Counter(all_labels)
                    print(f"  Original class distribution: {dict(class_counts)}")

                    # Determine weights for each sample based on inverse class frequency
                    # Samples from minority classes will have higher weights, increasing their chance of being sampled
                    # A common approach is 1.0 / count, or max_count / count to make weights integers
                    weights = []
                    # Get the count of the most frequent class to normalize weights
                    max_count = float(max(class_counts.values()))
                    for idx in range(len(train_dataset)):
                        label = train_dataset.get_target(idx)
                        weights.append(max_count / class_counts[label]) # Weight = max_class_frequency / current_class_frequency

                    # Create a WeightedRandomSampler
                    sampler = WeightedRandomSampler(
                        weights=weights,
                        num_samples=len(train_dataset), # Draw samples equal to the original dataset size per epoch
                        replacement=True # Crucial for oversampling to allow drawing the same sample multiple times
                    )

                    train_loader = DataLoader(
                        train_dataset,
                        batch_size=train_batch_size,
                        sampler=sampler, # Use the custom sampler instead of shuffle
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        prefetch_factor=prefetch_factor,
                        persistent_workers=persistent_workers,
                        drop_last=False # Do not drop the last batch to use all samples, if any
                    )
                    print(f"  Using balanced sampling with WeightedRandomSampler.")

                except Exception as e:
                    print(f"{COLOR_RED}Error creating balanced sampler: {e}. Falling back to standard shuffling.{COLOR_RESET}")
                    traceback.print_exc()
                    use_oversampler = False # Fallback

            # If oversampling is disabled or failed, create a standard DataLoader
            if not use_oversampler:
                print("Creating standard training loader (with shuffling)...")
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=train_batch_size,
                    shuffle=True, # Shuffle for standard training
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    prefetch_factor=prefetch_factor,
                    persistent_workers=persistent_workers,
                    drop_last=False
                )

            if train_loader:
                print(f"Train loader created. Batches per epoch: {len(train_loader)}. Batch size: {train_batch_size}, Workers: {num_workers}.")

    # --- Validation DataLoader Creation ---
    if val_dataset:
        if not isinstance(val_dataset, CustomDataset):
            print(f"{COLOR_RED}Error: 'val_dataset' is not a CustomDataset object. Cannot create validation loader.{COLOR_RESET}")
        elif len(val_dataset) > 0:
            val_batch_size = dataloader_config.get('val_batch_size', dataloader_config.get('batch_size', 64) * 2)
            if not isinstance(val_batch_size, int) or val_batch_size <= 0:
                print(f"{COLOR_YELLOW}Warning: Invalid 'val_batch_size' ({val_batch_size}). Setting to default (train_batch_size * 2).{COLOR_RESET}")
                val_batch_size = dataloader_config.get('batch_size', 64) * 2

            if val_batch_size > 128 and "b4" in val_dataset.transform.__repr__().lower():
                orig_val_batch_size = val_batch_size
                val_batch_size = min(128, val_batch_size)
                print(f"{COLOR_YELLOW}Warning: Reduced validation batch size from {orig_val_batch_size} to {val_batch_size} for memory efficiency with EfficientNet-B4. Consider adjusting 'val_batch_size' in config.py directly.{COLOR_RESET}")

            val_loader = DataLoader(
                val_dataset,
                batch_size=val_batch_size,
                shuffle=False,  # Never shuffle validation data for reproducible evaluation
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=False, # Persistent workers usually not needed for validation
                prefetch_factor=prefetch_factor,
                drop_last=False # Do NOT drop the last batch in validation to evaluate all samples
            )
            print(f"Validation loader created. Batches: {len(val_loader)}, Batch size: {val_batch_size}.")
        else:
            print("Validation dataset is empty. Skipping validation DataLoader creation.")

    # --- Test DataLoader Creation ---
    if test_dataset:
        if not isinstance(test_dataset, CustomDataset):
            print(f"{COLOR_RED}Error: 'test_dataset' is not a CustomDataset object. Cannot create test loader.{COLOR_RESET}")
        elif len(test_dataset) > 0:
            test_batch_size = dataloader_config.get('test_batch_size', dataloader_config.get('batch_size', 64) * 2)
            if not isinstance(test_batch_size, int) or test_batch_size <= 0:
                print(f"{COLOR_YELLOW}Warning: Invalid 'test_batch_size' ({test_batch_size}). Setting to default (train_batch_size * 2).{COLOR_RESET}")
                test_batch_size = dataloader_config.get('batch_size', 64) * 2

            if test_batch_size > 128 and "b4" in test_dataset.transform.__repr__().lower():
                orig_test_batch_size = test_batch_size
                test_batch_size = min(128, test_batch_size)
                print(f"{COLOR_YELLOW}Warning: Reduced test batch size from {orig_test_batch_size} to {test_batch_size} for memory efficiency with EfficientNet-B4. Consider adjusting 'test_batch_size' in config.py directly.{COLOR_RESET}")

            test_loader = DataLoader(
                test_dataset,
                batch_size=test_batch_size,
                shuffle=False, # Never shuffle test data
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=False,
                prefetch_factor=prefetch_factor,
                drop_last=False # Do NOT drop the last batch in test
            )
            print(f"Test loader created. Batches: {len(test_loader)}, Batch size: {test_batch_size}.")
        else:
            print("Test dataset is empty. Skipping test DataLoader creation.")

    print("--- DataLoaders creation finished. ---")
    return train_loader, val_loader, test_loader
