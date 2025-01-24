from ultralytics import YOLO
from download_data import download_prepare_data

DATA_YAML = "yolo_dataset/data.yaml"
YOLO_MODEL = "yolo11n.pt"

EPOCHS = 5
IMAGE_SIZE = 586
DEVICE = 0 # GPU index
OPTIMIZER = "AdamW"
BATCH_SIZE = 16

def train_model():
    model = YOLO(YOLO_MODEL)

    results = model.train(
        data = DATA_YAML,
        epochs = EPOCHS,
        imgsz = IMAGE_SIZE,
        device = DEVICE,
        optimizer = OPTIMIZER,
        batch = BATCH_SIZE
    )

if __name__ == "__main__":
    download_prepare_data()
    train_model()