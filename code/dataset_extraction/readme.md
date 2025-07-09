# PAI Dataset Extractor

A Python tool for extracting and processing dental radiograph datasets for AI-based Periapical Index (PAI) scoring.

## Description

This tool processes dental radiographs to create standardized datasets for machine learning applications. It extracts square image crops centered on root apex coordinates and generates anonymized datasets with proper train/test separation.

## Features

- **Automated image cropping**: Extracts 300×300 pixel crops centered on apex coordinates
- **Dataset anonymization**: Creates anonymized filenames while maintaining research keyfiles
- **Train/test separation**: Excludes designated test set images from training data
- **Flexible input formats**: Supports CSV-based measurement files with FDI notation
- **Quality validation**: Validates PAI scores (1-5) and coordinate data
- **Comprehensive logging**: Detailed processing logs and statistics
- **Configurable processing**: Command-line interface with configuration file support

## Installation

### Requirements

- Python 3.7+
- Required packages (install via pip):

```bash
pip install -r requirements.txt
```

### Dependencies

```
pandas>=1.3.0
numpy>=1.20.0
Pillow>=8.0.0
tifffile>=2021.1.0
tqdm>=4.60.0
```

## Usage

### Basic Usage

```bash
python pai_dataset_extractor.py \
    --input_csv /path/to/measurements.csv \
    --base_folder /path/to/radiograph/images \
    --output_folder /path/to/output/dataset
```

### Advanced Usage

```bash
python pai_dataset_extractor.py \
    --input_csv measurements.csv \
    --base_folder ./images \
    --output_folder ./output \
    --exclusions test_set_1.csv test_set_2.csv \
    --crop_size 300 \
    --config config.ini
```

### Extract Test Set Only

```bash
python pai_dataset_extractor.py \
    --input_csv measurements.csv \
    --base_folder ./images \
    --output_folder ./test_set \
    --exclusions test_set.csv \
    --test_set_only
```

## Input Data Format

### Measurements CSV

Required columns in the input CSV file:

- `file_path`: Path to radiograph image (relative to base_folder)
- `apex_x`: X coordinate of apex in mm
- `apex_y`: Y coordinate of apex in mm
- `PAI`: Periapical Index score (1-5)
- `quadrant`: Dental quadrant (1-4)
- `tooth`: Tooth number (FDI notation)
- `root`: Root type identifier

### Exclusion Files

CSV files containing images to exclude from training dataset:
- `filename`: Image filename to exclude

## Output

The tool generates:

- **Cropped images**: Anonymized TIFF files (k00001.tif, k00002.tif, ...)
- **keyfile.csv**: Full metadata including original paths (for research)
- **data.csv**: Anonymized metadata (for training)
- **dataset_statistics.txt**: Distribution statistics and summary

## Configuration

Create a `config.ini` file for default settings:

```ini
[DEFAULT]
crop_size = 300
base_folder = ./data/images
output_folder = ./output
```

## Dataset Structure

```
output_folder/
├── k00001.tif          # Anonymized cropped images
├── k00002.tif
├── ...
├── keyfile.csv         # Full metadata with original paths
├── data.csv           # Anonymized training metadata
└── dataset_statistics.txt  # Dataset summary
```

## PAI Scoring System

The Periapical Index (PAI) scoring system used:

1. **PAI 1**: Normal periapical structures
2. **PAI 2**: Small changes in bone structure
3. **PAI 3**: Changes in bone structure with mineral loss
4. **PAI 4**: Periodontitis with well-defined radiolucent area
5. **PAI 5**: Severe periodontitis with exacerbating features

## Citation

If you use this tool in your research, please cite:

```
[to be added]
```

## License

[MIT]




## Acknowledgments

- Faculty of Dentistry, University of Oslo

---

# Configuration File Example (config.ini)

```ini
[DEFAULT]
# Default crop size in pixels
crop_size = 300

# Default base folder for images
base_folder = ./data/radiographs

# Default output folder
output_folder = ./output/dataset

# Logging level (DEBUG, INFO, WARNING, ERROR)
log_level = INFO
```

---

# Requirements.txt

```
pandas>=1.3.0
numpy>=1.20.0
Pillow>=8.0.0
tifffile>=2021.1.0
tqdm>=4.60.0
```

---

# Example Usage Script (example_usage.py)

```python
#!/usr/bin/env python3
"""
Example usage of the PAI Dataset Extractor
"""

from pai_dataset_extractor import PAIDatasetExtractor

def main():
    # Initialize extractor
    extractor = PAIDatasetExtractor()
    
    # Example: Process a small dataset
    input_csv = "sample_measurements.csv"
    base_folder = "./sample_images"
    output_folder = "./sample_output"
    
    # Load data
    import pandas as pd
    df = pd.read_csv(input_csv)
    
    # Process dataset
    extractor.process_dataset(df, base_folder, output_folder, crop_size=300)
    
    print("Processing completed!")

if __name__ == "__main__":
    main()
```

---

# Sample CSV Structure (sample_measurements.csv)

```csv
file_path,apex_x,apex_y,PAI,quadrant,tooth,root
images/patient_001.tif,12.5,8.2,2,1,6,MB
images/patient_001.tif,13.1,9.8,2,1,6,DB
images/patient_002.tif,15.3,7.9,1,2,4,1
images/patient_003.tif,11.8,10.2,3,3,7,M
```
