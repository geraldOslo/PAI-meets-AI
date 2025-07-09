#!/usr/bin/env python3
"""
PAI Score Dataset Extractor for Deep Learning

This script generates a dataset for AI-based Periapical Index (PAI) scoring by:
1. Reading radiographic measurements from CSV files
2. Cropping standardized image segments centered on apex coordinates
3. Creating anonymized datasets with proper train/test separation
4. Excluding specific images that are part of designated test sets
5. Generating metadata files for both research and training purposes

The CSV input format includes detailed dental measurements including apex coordinates,
PAI scores, and anatomical landmarks for each root.

Author: [Author Name]
Institution: Faculty of Dentistry, University of Oslo
License: [License]
"""

import argparse
import configparser
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from tqdm import tqdm


class PAIDatasetExtractor:
    """
    A class for extracting and processing dental radiograph datasets for PAI scoring.
    
    This class handles the extraction of standardized image crops from dental radiographs
    based on apex coordinates, creating anonymized datasets suitable for machine learning.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the PAI Dataset Extractor.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.logger = self._setup_logging()
        self.config = self._load_config(config_path) if config_path else {}
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        return logging.getLogger(__name__)
    
    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dictionary containing configuration parameters
        """
        config = configparser.ConfigParser()
        try:
            config.read(config_path)
            return dict(config['DEFAULT']) if 'DEFAULT' in config else {}
        except Exception as e:
            self.logger.warning(f"Could not load config file {config_path}: {e}")
            return {}
    
    def get_image_resolution(self, image_path: str) -> Tuple[float, float]:
        """
        Extract X and Y resolution from TIFF metadata.
        
        Args:
            image_path: Path to the TIFF image file
            
        Returns:
            Tuple of (x_resolution, y_resolution) in pixels/mm
            
        Raises:
            Exception: If resolution metadata cannot be read
        """
        try:
            with tifffile.TiffFile(image_path) as tif:
                x_resolution = tif.pages[0].tags['XResolution'].value
                y_resolution = tif.pages[0].tags['YResolution'].value
                
                # Calculate actual resolution values (ratio of numerator/denominator)
                x_res = x_resolution[0] / x_resolution[1]
                y_res = y_resolution[0] / y_resolution[1]
                
                return x_res, y_res
        except Exception as e:
            raise Exception(f"Failed to read resolution from {image_path}: {e}")
    
    def validate_pai_score(self, pai_value) -> Optional[int]:
        """
        Validate and convert PAI score to integer.
        
        Args:
            pai_value: Raw PAI score value from CSV
            
        Returns:
            Valid PAI score (1-5) or None if invalid
        """
        try:
            pai_score = int(float(pai_value))
            if 1 <= pai_score <= 5:
                return pai_score
            else:
                self.logger.warning(f"PAI score {pai_score} outside valid range (1-5)")
                return None
        except (ValueError, TypeError):
            self.logger.warning(f"Invalid PAI score format: {pai_value}")
            return None
    
    def validate_coordinates(self, apex_x, apex_y) -> Optional[Tuple[float, float]]:
        """
        Validate and convert apex coordinates.
        
        Args:
            apex_x: X coordinate in mm
            apex_y: Y coordinate in mm
            
        Returns:
            Tuple of validated coordinates or None if invalid
        """
        try:
            x_coord = float(apex_x)
            y_coord = float(apex_y)
            return x_coord, y_coord
        except (ValueError, TypeError):
            self.logger.warning(f"Invalid coordinates: x={apex_x}, y={apex_y}")
            return None
    
    def crop_image_at_apex(self, image_path: str, apex_x_mm: float, apex_y_mm: float, 
                          crop_size: int = 300) -> Optional[np.ndarray]:
        """
        Crop image at specified apex coordinates.
        
        Args:
            image_path: Path to the source image
            apex_x_mm: X coordinate of apex in mm
            apex_y_mm: Y coordinate of apex in mm
            crop_size: Size of square crop in pixels
            
        Returns:
            Cropped image as numpy array or None if failed
        """
        try:
            # Get image resolution and convert coordinates to pixels
            x_res, y_res = self.get_image_resolution(image_path)
            apex_x_px = int(apex_x_mm * x_res)
            apex_y_px = int(apex_y_mm * y_res)
            
            # Load image and convert to RGB format
            img = np.array(Image.open(image_path).convert('L'))  # Grayscale
            img_rgb = np.stack([img] * 3, axis=-1)  # Convert to RGB
            
            # Calculate crop boundaries
            half_size = crop_size // 2
            left = apex_x_px - half_size
            right = apex_x_px + half_size
            top = apex_y_px - half_size
            bottom = apex_y_px + half_size
            
            # Calculate required padding if crop extends beyond image boundaries
            pad_top = max(0, -top)
            pad_bottom = max(0, bottom - img.shape[0])
            pad_left = max(0, -left)
            pad_right = max(0, right - img.shape[1])
            
            # Apply padding if needed
            if any([pad_top, pad_bottom, pad_left, pad_right]):
                img_rgb = np.pad(
                    img_rgb,
                    ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                    mode='constant',
                    constant_values=0
                )
                # Adjust coordinates for padding
                top += pad_top
                bottom += pad_top
                left += pad_left
                right += pad_left
            
            # Extract the crop
            crop_img = img_rgb[top:bottom, left:right]
            
            # Validate crop size
            if crop_img.shape[:2] != (crop_size, crop_size):
                self.logger.warning(f"Unexpected crop size: {crop_img.shape}")
                return None
                
            return crop_img
            
        except Exception as e:
            self.logger.error(f"Failed to crop image {image_path}: {e}")
            return None
    
    def load_exclusion_files(self, exclusion_paths: List[str]) -> pd.DataFrame:
        """
        Load and combine exclusion files.
        
        Args:
            exclusion_paths: List of file paths to exclusion CSVs
            
        Returns:
            Combined DataFrame of all exclusion files
        """
        exclusions_df_list = []
        
        for path in exclusion_paths:
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    exclusions_df_list.append(df)
                    self.logger.info(f"Loaded {len(df)} exclusions from {path}")
                except Exception as e:
                    self.logger.error(f"Error loading exclusion file {path}: {e}")
            else:
                self.logger.warning(f"Exclusion file not found: {path}")
        
        if not exclusions_df_list:
            return pd.DataFrame()
        
        return pd.concat(exclusions_df_list, ignore_index=True)
    
    def normalize_path(self, path: str) -> str:
        """
        Normalize a file path for consistent matching.
        
        Args:
            path: File path to normalize
            
        Returns:
            Normalized path string (basename with standardized separators)
        """
        return os.path.basename(str(path).replace('\\', '/'))
    
    def apply_exclusions(self, df: pd.DataFrame, exclusions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove excluded files from the main dataset.
        
        Args:
            df: Main dataset DataFrame
            exclusions_df: DataFrame containing files to exclude
            
        Returns:
            Filtered DataFrame with exclusions removed
        """
        if exclusions_df.empty:
            self.logger.info("No exclusions to apply")
            return df
        
        # Normalize paths for consistent matching
        df['match_key'] = df['file_path'].apply(self.normalize_path)
        exclusions_df['match_key'] = exclusions_df['filename'].apply(self.normalize_path)
        
        # Find matches
        matches = df[df['match_key'].isin(exclusions_df['match_key'])]
        self.logger.info(f"Excluding {len(matches)} images found in test sets")
        
        # Report unmatched exclusions
        unmatched = exclusions_df[~exclusions_df['match_key'].isin(df['match_key'])]
        if not unmatched.empty:
            self.logger.warning(f"{len(unmatched)} exclusion files not found in dataset")
            for filename in unmatched['filename'].head(5):
                self.logger.warning(f"  - {filename}")
        
        # Remove exclusions
        df_filtered = df[~df['match_key'].isin(exclusions_df['match_key'])].copy()
        df_filtered.drop('match_key', axis=1, inplace=True)
        
        self.logger.info(f"Dataset filtered: {len(df)} -> {len(df_filtered)} samples")
        return df_filtered
    
    def process_dataset(self, df: pd.DataFrame, base_folder: str, export_folder: str, 
                       crop_size: int = 300) -> None:
        """
        Process the dataset of radiograph measurements and create PAI dataset.
        
        Args:
            df: DataFrame containing measurements
            base_folder: Base directory for resolving relative image paths  
            export_folder: Directory where cropped images and data files will be saved
            crop_size: Size of square crop in pixels
        """
        # Create export directory
        Path(export_folder).mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking variables
        processed_count = 0
        output_data = []
        
        self.logger.info(f"Processing {len(df)} images...")
        
        # Process each row with progress bar
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
            # Resolve image path
            image_path = row['file_path']
            if not os.path.isabs(image_path):
                image_path = os.path.join(base_folder, image_path)
            
            if not os.path.exists(image_path):
                self.logger.warning(f"Image file not found: {image_path}")
                continue
            
            # Validate PAI score
            pai_score = self.validate_pai_score(row['PAI'])
            if pai_score is None:
                continue
            
            # Validate coordinates
            coords = self.validate_coordinates(row['apex_x'], row['apex_y'])
            if coords is None:
                continue
            
            apex_x_mm, apex_y_mm = coords
            
            # Crop image
            crop_img = self.crop_image_at_apex(image_path, apex_x_mm, apex_y_mm, crop_size)
            if crop_img is None:
                continue
            
            # Save cropped image with anonymized filename
            filename = f"k{processed_count:05d}.tif"
            crop_img_path = os.path.join(export_folder, filename)
            
            try:
                Image.fromarray(crop_img).save(crop_img_path)
                
                # Record successful processing
                output_data.append([
                    row['file_path'], filename,
                    row['quadrant'], row['tooth'], row['root'],
                    pai_score
                ])
                processed_count += 1
                
            except Exception as e:
                self.logger.error(f"Failed to save {crop_img_path}: {e}")
        
        # Save metadata files
        if output_data:
            self._save_metadata_files(output_data, export_folder)
            self.logger.info(f"Successfully processed {processed_count} images")
        else:
            self.logger.warning("No images were successfully processed")
    
    def _save_metadata_files(self, output_data: List, export_folder: str) -> None:
        """
        Save metadata files for the processed dataset.
        
        Args:
            output_data: List of processed image metadata
            export_folder: Directory to save metadata files
        """
        columns = ['image_path', 'filename', 'quadrant', 'tooth', 'root', 'PAI']
        output_df = pd.DataFrame(output_data, columns=columns)
        
        # Save full keyfile (with original paths for research use)
        keyfile_path = os.path.join(export_folder, 'keyfile.csv')
        output_df.to_csv(keyfile_path, index=False)
        self.logger.info(f"Saved keyfile to {keyfile_path}")
        
        # Save anonymized data file (for model training)
        data_path = os.path.join(export_folder, 'data.csv')
        output_df[['filename', 'quadrant', 'tooth', 'root', 'PAI']].to_csv(
            data_path, index=False
        )
        self.logger.info(f"Saved anonymized data to {data_path}")
        
        # Save dataset statistics
        self._save_dataset_statistics(output_df, export_folder)
    
    def _save_dataset_statistics(self, df: pd.DataFrame, export_folder: str) -> None:
        """
        Save dataset statistics and distribution information.
        
        Args:
            df: DataFrame containing processed dataset
            export_folder: Directory to save statistics
        """
        stats_path = os.path.join(export_folder, 'dataset_statistics.txt')
        
        with open(stats_path, 'w') as f:
            f.write(f"Dataset Statistics - Generated {datetime.now()}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total samples: {len(df)}\n\n")
            
            f.write("PAI Score Distribution:\n")
            pai_dist = df['PAI'].value_counts().sort_index()
            for score, count in pai_dist.items():
                percentage = (count / len(df)) * 100
                f.write(f"  PAI {score}: {count} ({percentage:.2f}%)\n")
            
            f.write("\nQuadrant Distribution:\n")
            quad_dist = df['quadrant'].value_counts().sort_index()
            for quad, count in quad_dist.items():
                percentage = (count / len(df)) * 100
                f.write(f"  Q{quad}: {count} ({percentage:.2f}%)\n")
            
            f.write("\nTooth Distribution:\n")
            tooth_dist = df['tooth'].value_counts().sort_index()
            for tooth, count in tooth_dist.items():
                percentage = (count / len(df)) * 100
                f.write(f"  Tooth {tooth}: {count} ({percentage:.2f}%)\n")
        
        self.logger.info(f"Saved statistics to {stats_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract PAI dataset from dental radiographs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input_csv',
        type=str,
        required=True,
        help='Path to input CSV file with measurements'
    )
    
    parser.add_argument(
        '--base_folder',
        type=str,
        required=True,
        help='Base directory for resolving relative image paths'
    )
    
    parser.add_argument(
        '--output_folder',
        type=str,
        required=True,
        help='Directory to save processed dataset'
    )
    
    parser.add_argument(
        '--exclusions',
        type=str,
        nargs='*',
        default=[],
        help='Paths to exclusion CSV files (test sets)'
    )
    
    parser.add_argument(
        '--crop_size',
        type=int,
        default=300,
        help='Size of square image crops in pixels'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--test_set_only',
        action='store_true',
        help='Process only test set (exclusion files)'
    )
    
    return parser.parse_args()


def main():
    """Main function to run the PAI dataset extractor."""
    args = parse_arguments()
    
    # Initialize extractor
    extractor = PAIDatasetExtractor(args.config)
    
    try:
        # Load input data
        extractor.logger.info(f"Loading measurement data from {args.input_csv}")
        df = pd.read_csv(args.input_csv)
        extractor.logger.info(f"Loaded {len(df)} measurements")
        
        # Load exclusions
        exclusions_df = extractor.load_exclusion_files(args.exclusions)
        
        if args.test_set_only:
            # Process only test set
            if exclusions_df.empty:
                extractor.logger.error("No exclusion files found for test set processing")
                return
            extractor.logger.info(f"Processing test set with {len(exclusions_df)} images")
            extractor.process_dataset(exclusions_df, args.base_folder, args.output_folder, args.crop_size)
        else:
            # Process main dataset (excluding test set)
            df_filtered = extractor.apply_exclusions(df, exclusions_df)
            extractor.process_dataset(df_filtered, args.base_folder, args.output_folder, args.crop_size)
        
        extractor.logger.info("Processing completed successfully")
        
    except Exception as e:
        extractor.logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
