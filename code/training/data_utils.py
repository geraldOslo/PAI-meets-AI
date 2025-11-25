"""
Data loading utilities for PAI classification

Author: Gerald Torgersen
SPDX-License-Identifier: MIT
Copyright (c) 2025 Gerald Torgersen
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Callable

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CustomDataset(Dataset):
    """
    Custom dataset for PAI classification.
    
    Expected CSV columns:
        - 'filename' or 'image_path': Path to image file
        - 'PAI': PAI score (1-5)
        - 'root_dir' (optional): Root directory for images
        - 'PAI_0_indexed' (optional): PAI score converted to 0-4
    
    Args:
        dataframe: pandas DataFrame with image paths and labels
        transform: Torchvision transforms to apply to images
        target_transform: Transforms to apply to labels (optional)
    """
    
    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform
        self.target_transform = target_transform
        
        # Determine image path column
        if 'filename' in self.data.columns:
            self.image_col = 'filename'
        elif 'image_path' in self.data.columns:
            self.image_col = 'image_path'
        else:
            raise ValueError("DataFrame must have 'filename' or 'image_path' column")
        
        # Determine root directory
        if 'root_dir' in self.data.columns:
            self.use_root = True
        else:
            self.use_root = False
        
        # Determine label column (prefer 0-indexed)
        if 'PAI_0_indexed' in self.data.columns:
            self.label_col = 'PAI_0_indexed'
        elif 'PAI' in self.data.columns:
            # Convert PAI (1-5) to 0-indexed (0-4)
            self.data['PAI_0_indexed'] = self.data['PAI'] - 1
            self.label_col = 'PAI_0_indexed'
        else:
            raise ValueError("DataFrame must have 'PAI' or 'PAI_0_indexed' column")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Get item by index.
        
        Returns:
            Tuple of (image_tensor, label, filename)
        """
        row = self.data.iloc[idx]
        
        # Get image path
        filename = row[self.image_col]
        if self.use_root:
            img_path = os.path.join(row['root_dir'], filename)
        else:
            img_path = filename
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")
        
        # Get label (0-indexed)
        label = int(row[self.label_col])
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label, filename
    
    def get_target(self, idx: int) -> int:
        """
        Get label for specific index without loading image.
        Used for creating weighted samplers.
        
        Returns:
            Label (0-indexed, 0-4)
        """
        return int(self.data.iloc[idx][self.label_col])
    
    def get_labels(self) -> np.ndarray:
        """Get all labels as numpy array."""
        return self.data[self.label_col].values
    
    def get_class_distribution(self) -> dict:
        """Get distribution of classes."""
        labels = self.get_labels()
        unique, counts = np.unique(labels, return_counts=True)
        return {int(label): int(count) for label, count in zip(unique, counts)}


def get_train_transforms(input_size: int = 224, 
                        mean: list = [0.378, 0.378, 0.378],
                        std: list = [0.167, 0.167, 0.167],
                        augment: bool = True) -> transforms.Compose:
    """
    Get training transforms with optional augmentation.
    
    Args:
        input_size: Target image size
        mean: Normalization mean (R, G, B)
        std: Normalization std (R, G, B)
        augment: Whether to apply data augmentation
    
    Returns:
        Composed transforms
    """
    if augment:
        return transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.04, 0.04),
                scale=(0.92, 1.08)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])


def get_val_transforms(input_size: int = 224,
                      mean: list = [0.378, 0.378, 0.378],
                      std: list = [0.167, 0.167, 0.167]) -> transforms.Compose:
    """
    Get validation/test transforms (no augmentation).
    
    Args:
        input_size: Target image size
        mean: Normalization mean (R, G, B)
        std: Normalization std (R, G, B)
    
    Returns:
        Composed transforms
    """
    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def denormalize_image(tensor: torch.Tensor,
                     mean: list = [0.378, 0.378, 0.378],
                     std: list = [0.167, 0.167, 0.167]) -> torch.Tensor:
    """
    Denormalize image tensor for visualization.
    
    Args:
        tensor: Normalized image tensor (C, H, W)
        mean: Normalization mean used
        std: Normalization std used
    
    Returns:
        Denormalized tensor
    """
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    return tensor * std_t + mean_t


def calculate_dataset_statistics(dataframe: pd.DataFrame,
                                image_col: str = 'filename',
                                root_col: str = 'root_dir',
                                sample_size: int = 1000) -> Tuple[list, list]:
    """
    Calculate mean and std of dataset for normalization.
    
    Args:
        dataframe: DataFrame with image paths
        image_col: Column name for image filenames
        root_col: Column name for root directory
        sample_size: Number of images to sample for calculation
    
    Returns:
        Tuple of (mean, std) as lists of 3 values (R, G, B)
    """
    # Sample subset for efficiency
    if len(dataframe) > sample_size:
        sample_df = dataframe.sample(n=sample_size, random_state=42)
    else:
        sample_df = dataframe
    
    pixel_values = []
    
    for idx, row in sample_df.iterrows():
        img_path = os.path.join(row[root_col], row[image_col])
        try:
            img = Image.open(img_path).convert('RGB')
            img = np.array(img) / 255.0  # Normalize to [0, 1]
            pixel_values.append(img.reshape(-1, 3))
        except Exception as e:
            print(f"Warning: Could not load {img_path}: {e}")
            continue
    
    # Concatenate all pixels
    all_pixels = np.concatenate(pixel_values, axis=0)
    
    # Calculate mean and std
    mean = all_pixels.mean(axis=0).tolist()
    std = all_pixels.std(axis=0).tolist()
    
    return mean, std


def verify_dataset(dataframe: pd.DataFrame,
                  image_col: str = 'filename',
                  root_col: str = 'root_dir',
                  check_images: bool = True,
                  verbose: bool = True) -> dict:
    """
    Verify dataset integrity.
    
    Args:
        dataframe: DataFrame to verify
        image_col: Column name for image filenames
        root_col: Column name for root directory
        check_images: Whether to check if image files exist
        verbose: Whether to print detailed information
    
    Returns:
        Dictionary with verification results
    """
    results = {
        'total_samples': len(dataframe),
        'missing_files': [],
        'class_distribution': {},
        'valid': True
    }
    
    # Check class distribution
    if 'PAI' in dataframe.columns:
        results['class_distribution'] = dataframe['PAI'].value_counts().sort_index().to_dict()
    elif 'PAI_0_indexed' in dataframe.columns:
        dist = dataframe['PAI_0_indexed'].value_counts().sort_index()
        results['class_distribution'] = {k+1: v for k, v in dist.to_dict().items()}
    
    # Check image files exist
    if check_images:
        for idx, row in dataframe.iterrows():
            if root_col in dataframe.columns:
                img_path = os.path.join(row[root_col], row[image_col])
            else:
                img_path = row[image_col]
            
            if not os.path.exists(img_path):
                results['missing_files'].append(img_path)
                results['valid'] = False
    
    if verbose:
        print(f"Dataset verification results:")
        print(f"  Total samples: {results['total_samples']}")
        print(f"  Class distribution: {results['class_distribution']}")
        print(f"  Missing files: {len(results['missing_files'])}")
        
        if results['missing_files']:
            print(f"\nFirst 5 missing files:")
            for path in results['missing_files'][:5]:
                print(f"    {path}")
    
    return results
