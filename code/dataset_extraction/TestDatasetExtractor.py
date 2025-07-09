#!/usr/bin/env python3
"""
Test suite for PAI Dataset Extractor

This module contains unit tests and integration tests for the PAI dataset extraction tool.
"""

import os
import tempfile
import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from pai_dataset_extractor import PAIDatasetExtractor


class TestPAIDatasetExtractor(unittest.TestCase):
    """Test cases for PAI Dataset Extractor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = PAIDatasetExtractor()
        
        # Sample test data
        self.sample_data = {
            'file_path': ['image1.tif', 'image2.tif', 'image3.tif'],
            'apex_x': [10.0, 15.5, 12.3],
            'apex_y': [8.0, 9.2, 11.1],
            'PAI': [1, 3, 2],
            'quadrant': [1, 2, 3],
            'tooth': [6, 4, 7],
            'root': ['MB', '1', 'M']
        }
        self.df = pd.DataFrame(self.sample_data)
    
    def test_validate_pai_score_valid(self):
        """Test PAI score validation with valid scores."""
        self.assertEqual(self.extractor.validate_pai_score(1), 1)
        self.assertEqual(self.extractor.validate_pai_score(5), 5)
        self.assertEqual(self.extractor.validate_pai_score('3'), 3)
        self.assertEqual(self.extractor.validate_pai_score(2.0), 2)
    
    def test_validate_pai_score_invalid(self):
        """Test PAI score validation with invalid scores."""
        self.assertIsNone(self.extractor.validate_pai_score(0))
        self.assertIsNone(self.extractor.validate_pai_score(6))
        self.assertIsNone(self.extractor.validate_pai_score('invalid'))
        self.assertIsNone(self.extractor.validate_pai_score(None))
    
    def test_validate_coordinates_valid(self):
        """Test coordinate validation with valid coordinates."""
        result = self.extractor.validate_coordinates(10.5, 8.2)
        self.assertEqual(result, (10.5, 8.2))
        
        result = self.extractor.validate_coordinates('12.3', '9.1')
        self.assertEqual(result, (12.3, 9.1))
    
    def test_validate_coordinates_invalid(self):
        """Test coordinate validation with invalid coordinates."""
        self.assertIsNone(self.extractor.validate_coordinates('invalid', 8.2))
        self.assertIsNone(self.extractor.validate_coordinates(10.5, None))
        self.assertIsNone(self.extractor.validate_coordinates('', ''))
    
    def test_normalize_path(self):
        """Test path normalization."""
        # Test different path formats
        self.assertEqual(
            self.extractor.normalize_path('/path/to/image.tif'),
            'image.tif'
        )
        self.assertEqual(
            self.extractor.normalize_path('C:\\path\\to\\image.tif'),
            'image.tif'
        )
        self.assertEqual(
            self.extractor.normalize_path('image.tif'),
            'image.tif'
        )
    
    def test_load_exclusion_files_empty(self):
        """Test loading exclusion files when files don't exist."""
        result = self.extractor.load_exclusion_files(['nonexistent.csv'])
        self.assertTrue(result.empty)
    
    def test_apply_exclusions_no_exclusions(self):
        """Test applying exclusions when no exclusions exist."""
        empty_exclusions = pd.DataFrame()
        result = self.extractor.apply_exclusions(self.df, empty_exclusions)
        pd.testing.assert_frame_equal(result, self.df)
    
    def test_apply_exclusions_with_matches(self):
        """Test applying exclusions with matching files."""
        exclusions = pd.DataFrame({
            'filename': ['image1.tif', 'image3.tif']
        })
        
        result = self.extractor.apply_exclusions(self.df, exclusions)
        
        # Should exclude 2 files, leaving only image2.tif
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['file_path'], 'image2.tif')
    
    @patch('tifffile.TiffFile')
    def test_get_image_resolution(self, mock_tiff):
        """Test image resolution extraction."""
        # Mock TIFF file with resolution tags
        mock_page = Mock()
        mock_page.tags = {
            'XResolution': Mock(value=(300, 1)),  # 300 DPI
            'YResolution': Mock(value=(300, 1))   # 300 DPI
        }
        mock_tiff.return_value.__enter__.return_value.pages = [mock_page]
        
        x_res, y_res = self.extractor.get_image_resolution('test.tif')
        self.assertEqual(x_res, 300.0)
        self.assertEqual(y_res, 300.0)
    
    def test_save_metadata_files(self):
        """Test saving metadata files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_data = [
                ['original1.tif', 'k00001.tif', 1, 6, 'MB', 2],
                ['original2.tif', 'k00002.tif', 2, 4, '1', 1]
            ]
            
            self.extractor._save_metadata_files(output_data, temp_dir)
            
            # Check if files were created
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'keyfile.csv')))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'data.csv')))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'dataset_statistics.txt')))
            
            # Verify content
            keyfile = pd.read_csv(os.path.join(temp_dir, 'keyfile.csv'))
            self.assertEqual(len(keyfile), 2)
            self.assertIn('image_path', keyfile.columns)
            
            data_file = pd.read_csv(os.path.join(temp_dir, 'data.csv'))
            self.assertEqual(len(data_file), 2)
            self.assertNotIn('image_path', data_file.columns)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete workflow."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.extractor = PAIDatasetExtractor()
    
    def test_complete_workflow_no_images(self):
        """Test complete workflow with valid data but no actual images."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample CSV
            data = {
                'file_path': ['test1.tif', 'test2.tif'],
                'apex_x': [10.0, 15.0],
                'apex_y': [8.0, 12.0],
                'PAI': [1, 3],
                'quadrant': [1, 2],
                'tooth': [6, 4],
                'root': ['MB', '1']
            }
            df = pd.DataFrame(data)
            
            # Process dataset (will fail due to missing images, but tests structure)
            self.extractor.process_dataset(df, temp_dir, temp_dir)
            
            # Should create output directory
            self.assertTrue(os.path.exists(temp_dir))


def create_sample_test_data():
    """Create sample test data for demonstration."""
    sample_csv_content = """file_path,apex_x,apex_y,PAI,quadrant,tooth,root
sample1.tif,10.5,8.2,1,1,6,MB
sample2.tif,12.3,9.1,2,1,6,DB
sample3.tif,15.1,7.8,3,2,4,1
sample4.tif,9.8,10.5,1,3,7,M
sample5.tif,11.2,8.9,4,4,5,1"""
    
    exclusions_content = """filename
sample1.tif
sample3.tif"""
    
    print("Sample measurements.csv:")
    print(sample_csv_content)
    print("\nSample exclusions.csv:")
    print(exclusions_content)
    
    return sample_csv_content, exclusions_content


def run_example_workflow():
    """Demonstrate the complete workflow with sample data."""
    print("PAI Dataset Extractor - Example Workflow")
    print("=" * 50)
    
    # Create sample data
    measurements_csv, exclusions_csv = create_sample_test_data()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample CSV files
        measurements_path = os.path.join(temp_dir, 'measurements.csv')
        exclusions_path = os.path.join(temp_dir, 'exclusions.csv')
        
        with open(measurements_path, 'w') as f:
            f.write(measurements_csv)
        
        with open(exclusions_path, 'w') as f:
            f.write(exclusions_csv)
        
        # Initialize extractor
        extractor = PAIDatasetExtractor()
        
        # Load data
        df = pd.read_csv(measurements_path)
        exclusions_df = extractor.load_exclusion_files([exclusions_path])
        
        print(f"\nLoaded {len(df)} measurements")
        print(f"Loaded {len(exclusions_df)} exclusions")
        
        # Apply exclusions
        df_filtered = extractor.apply_exclusions(df, exclusions_df)
        print(f"Filtered dataset: {len(df_filtered)} samples remaining")
        
        # Show PAI distribution
        print("\nPAI Score Distribution:")
        pai_dist = df_filtered['PAI'].value_counts().sort_index()
        for score, count in pai_dist.items():
            print(f"  PAI {score}: {count} samples")
        
        print(f"\nExample workflow completed successfully!")
        print(f"Temporary directory: {temp_dir}")


if __name__ == '__main__':
    # Run tests
    print("Running unit tests...")
    unittest.main(verbosity=2, exit=False)
    
    print("\n" + "=" * 50)
    
    # Run example workflow
    run_example_workflow()
