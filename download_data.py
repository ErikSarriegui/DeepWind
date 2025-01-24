"""
DeepWind Turbine Dataset Downloader and Preprocessor

This script downloads and prepares a wind turbine dataset from Kaggle for YOLO object detection training.
The dataset contains images of wind turbines with annotations for dirt and damage detection.

Requirements:
- Kaggle API credentials in the PC (Disk:/Users/Username/.kaggle/kaggle.json)
- Python packages listed in requirements.txt

Dataset structure after processing:
yolo_dataset/
├── train/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── data.yaml
"""

from kaggle.api.kaggle_api_extended import KaggleApi
import shutil
import random
import yaml
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def download_dataset() -> None:
    """
    Downloads the wind turbine dataset from Kaggle using the Kaggle API.
    Requires authentication via KAGGLE_USERNAME and KAGGLE_KEY in .env file.
    """
    try:
        api = KaggleApi()
        api.authenticate()
        dataset = "ajifoster3/yolo-annotated-wind-turbines-586x371"
        logger.info(f"Downloading dataset: {dataset}")
        api.dataset_download_files(dataset, unzip=True)
        
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        logger.info("Please download the dataset manually")

def prepare_data() -> None:
    """
    Prepares the downloaded dataset for YOLO training by:
    1. Creating train/test/valid splits
    2. Organizing images and labels into YOLO format
    3. Generating data.yaml configuration file
    
    Split ratios:
    - Training: 80%
    - Testing: 10%
    - Validation: 10%
    """
    try:
        dataset_dir = "NordTank586x371"
        output_dir = "yolo_dataset"
        ratios = {
            "train": 0.8,
            "test": 0.1,
            "valid": 0.1
        }
        seed = 42  # For reproducibility

        # Create directory structure
        for split in ["train", "test", "valid"]:
            for subdir in ["images", "labels"]:
                os.makedirs(os.path.join(output_dir, split, subdir), exist_ok=True)
                logger.info(f"Created directory: {output_dir}/{split}/{subdir}")

        # Get list of images with corresponding labels
        image_files = [
            file for file in os.listdir(os.path.join(dataset_dir, "images"))
            if file.lower().endswith(('.png', '.jpg', '.jpeg'))
            and os.path.exists(os.path.join(dataset_dir, "labels", os.path.splitext(file)[0] + ".txt"))
        ]

        # Split dataset
        random.seed(seed)
        random.shuffle(image_files)
        total = len(image_files)
        train_end = int(ratios["train"] * total)
        test_end = train_end + int(ratios["test"] * total)

        splits = {
            "train": image_files[:train_end],
            "test": image_files[train_end:test_end],
            "valid": image_files[test_end:]
        }

        # Copy files to respective directories
        for split, files in splits.items():
            logger.info(f"Processing {split} split: {len(files)} files")
            for file in files:
                # Copy image
                shutil.copy(
                    os.path.join(dataset_dir, "images", file),
                    os.path.join(output_dir, split, "images", file)
                )
                
                # Copy label
                label_file = os.path.splitext(file)[0] + ".txt"
                shutil.copy(
                    os.path.join(dataset_dir, "labels", label_file),
                    os.path.join(output_dir, split, "labels", label_file)
                )

        # Generate YAML configuration
        data = {
            "names": ["Dirt", "Damage"],
            "nc": 2,
            "train": "train/images",
            "test": "test/images",
            "val": "valid/images"
        }

        yaml_path = f"{output_dir}/data.yaml"
        with open(yaml_path, "w") as file:
            yaml.dump(data, file, default_flow_style=False)
        logger.info(f"Created YAML configuration: {yaml_path}")

        logger.info("Dataset preparation completed!")
        logger.info(f"Total images with labels: {total}")
        logger.info("Distribution:")
        for split, files in splits.items():
            logger.info(f"- {split}: {len(files)} images")
    
    except Exception as e:
        logger.error(f"Error preparing dataset: {str(e)}")

def remove_original_dataset() -> None:
    """
    Removes the original downloaded dataset to save space.
    """
    try:
        shutil.rmtree("NordTank586x371")
        logger.info("Original dataset removed successfully")
    except Exception as e:
        logger.error(f"Error removing original dataset: {str(e)}")
        logger.info("Please remove the directory manually")

def download_prepare_data():
    download_dataset()
    prepare_data()
    remove_original_dataset()

if __name__ == "__main__":
    download_prepare_data()