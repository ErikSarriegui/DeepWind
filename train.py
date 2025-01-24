from ultralytics import YOLO

DATA_YAML = "/content/datasets/yolo_dataset/data.yaml"
YOLO_MODEL = "yolo11n.pt"

EPOCHS = 50
IMAGE_SIZE = 586
DEVICE = 0 # GPU index
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
):
    model = YOLO(yolo_model)

    results = model.train(
        data = data_yaml,
        epochs = epochs,
        imgsz = imgsz,
        device = device,
        optimizer = optimizer,
        batch = batch_size
    )

    return results

if __name__ == "__main__":
    train_model()