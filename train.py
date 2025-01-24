"""
YOLO Model Training Script

This module provides functionality to train a YOLO (You Only Look Once) model
for object detection tasks. It uses the Ultralytics YOLO implementation.
"""

from ultralytics import YOLO
from typing import Dict, Any

# Data configuration path
DATA_YAML = "/content/datasets/yolo_dataset/data.yaml"

# Model configuration
YOLO_MODEL = "yolo11s.pt"

# Training hyperparameters
EPOCHS = 50
IMAGE_SIZE = 586
DEVICE = 0  # GPU device index (0 = first GPU)
OPTIMIZER = "AdamW"
BATCH_SIZE = 32

def train_model(
    data_yaml: str = DATA_YAML,
    yolo_model: str = YOLO_MODEL,
    epochs: int = EPOCHS,
    imgsz: int = IMAGE_SIZE,
    device: int = DEVICE,
    optimizer: str = OPTIMIZER,
    batch_size: int = BATCH_SIZE
) -> Dict[str, Any]:
    """
    Train a YOLO model with specified parameters.

    Args:
        data_yaml (str): Path to the data configuration file in YAML format
        yolo_model (str): Path to the YOLO model file or model name
        epochs (int): Number of training epochs
        imgsz (int): Size of input images
        device (int): GPU device index (0 = first GPU)
        optimizer (str): Name of the optimization algorithm
        batch_size (int): Number of images per training batch

    Returns:
        Dict[str, Any]: Training results containing metrics and model statistics

    Example:
        >>> results = train_model(
        ...     data_yaml="path/to/data.yaml",
        ...     yolo_model="yolo11s.pt",
        ...     epochs=100
        ... )
    """
    model = YOLO(yolo_model)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        optimizer=optimizer,
        batch=batch_size
    )

    return results