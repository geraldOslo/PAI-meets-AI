#!/usr/bin/env python3
"""
Calculate mean and standard deviation for dataset normalization.
Auto-detects image format (8-bit uint8 or 32-bit float32).

Can optionally take command-line arguments to override config paths.
"""

import os
import sys
import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm
import argparse  # <-- NECESSARY CHANGE: Import argparse

# Add parent directory to path to import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import DataConfig

# Configuration
UPDATE_CONFIG = False  # Set to True to automatically update config.py

def calculate_statistics(data_csv, data_root):
    """Calculate mean and std for PyTorch normalization."""
    
    print("="*80 + "\nDataset Statistics Calculation\n" + "="*80)
    print(f"Data CSV: {data_csv}")
    print(f"Data root: {data_root}\n")
    
    df = pd.read_csv(data_csv)
    print(f"Total images: {len(df)}")
    
    first_image_path = os.path.join(data_root, df.iloc[0]['filename'])
    sample_img = tifffile.imread(first_image_path)
    
    print(f"\nImage format detection:")
    print(f"  dtype: {sample_img.dtype}, shape: {sample_img.shape}, min: {sample_img.min()}, max: {sample_img.max()}")
    
    normalize_factor = 255.0 if np.issubdtype(sample_img.dtype, np.integer) or sample_img.max() > 1.0 else 1.0
    print(f"  → Using normalization factor: {normalize_factor}\n")
    
    pixel_sum = 0.0
    pixel_sum_sq = 0.0
    total_pixels = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        img_path = os.path.join(data_root, row['filename'])
        try:
            img = tifffile.imread(img_path).astype(np.float32) / normalize_factor
            pixel_sum += img.sum()
            pixel_sum_sq += (img ** 2).sum()
            total_pixels += img.size
        except Exception as e:
            print(f"Warning: Could not process {img_path}. Error: {e}")
    
    mean = pixel_sum / total_pixels
    std = np.sqrt((pixel_sum_sq / total_pixels) - (mean ** 2))
    
    print(f"\n{'='*80}\nRESULTS - Normalization Statistics\n{'='*80}")
    print(f"Mean: {mean:.6f}\nStd:  {std:.6f}")
    print(f"\nFor RGB (3 channels):\nmean = [{mean:.6f}, {mean:.6f}, {mean:.6f}]\nstd  = [{std:.6f}, {std:.6f}, {std:.6f}]")
    print(f"{'='*80}")
    
    stats_file = os.path.join(os.path.dirname(data_csv), "statistics.txt")
    with open(stats_file, 'w') as f:
        f.write(f"Mean: {mean:.6f}\nStd:  {std:.6f}\n")
    print(f"\n✓ Statistics saved to: {stats_file}")
    
    return mean, std


def update_config_file(config_path, mean, std):
    # This function remains unchanged
    print(f"\n{'='*80}\nUpdating config.py\n{'='*80}")
    print(f"Config file: {config_path}")
    try:
        with open(config_path, 'r') as f: lines = f.readlines()
        updated = False
        for i, line in enumerate(lines):
            if 'self.mean = [' in line:
                lines[i] = f'            self.mean = [{mean:.6f}, {mean:.6f}, {mean:.6f}]\n'
                updated = True
            elif 'self.std = [' in line:
                lines[i] = f'            self.std = [{std:.6f}, {std:.6f}, {std:.6f}]\n'
        if updated:
            with open(config_path, 'w') as f: f.writelines(lines)
            print("\n✓ Config file updated successfully!")
        else:
            print("\n⚠ Warning: Could not find mean/std lines in config.py. Please update manually.")
    except Exception as e:
        print(f"\n✗ ERROR updating config file: {e}")


if __name__ == "__main__":
    # --- NECESSARY CHANGE: Add argument parser and main logic block ---
    parser = argparse.ArgumentParser(description="Calculate dataset normalization statistics.")
    parser.add_argument("--data_csv", type=str, default=None, help="Optional: Path to the data CSV file to override config.")
    parser.add_argument("--data_root", type=str, default=None, help="Optional: Path to the image root folder to override config.")
    args = parser.parse_args()

    try:
        # Load defaults from config
        default_config = DataConfig()
        
        # Use override if provided, otherwise use default
        data_csv = args.data_csv if args.data_csv else default_config.data_csv
        data_root = args.data_root if args.data_root else default_config.data_root

        # Run the main calculation
        mean, std = calculate_statistics(data_csv, data_root)

        # Update config.py if requested
        if UPDATE_CONFIG:
            config_path = PROJECT_ROOT / 'code' / 'config.py'
            update_config_file(config_path, mean, std)
        else:
            print(f"\nTo update config.py automatically, set UPDATE_CONFIG = True in this script.")
            
    except Exception as e:
        print(f"\nERROR: An error occurred: {e}", file=sys.stderr)