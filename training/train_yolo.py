import os
import sys
import argparse

import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


def train_yolo_pest(data_yaml: str = "pest_dataset.yaml", epochs: int = 40, img_size: int = 640, batch: int = 16):
    if not ULTRALYTICS_AVAILABLE:
        print("[Error] ultralytics package not found. Run: pip install ultralytics")
        return
    if not os.path.exists(data_yaml):
        print(f"[Error] dataset config not found: {data_yaml}")
        print("        Run: python training/prepare_yolo_dataset.py")
        return

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[YOLOv8n] Loading pretrained base model yolov8n.pt...")
    model = YOLO("yolov8n.pt")

    print(f"[YOLOv8n] Training on {data_yaml} for {epochs} epochs (device={device})...")
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch,
        name="yolov8n_pest_model",
        device=device,
        patience=10,
        workers=4,
        project=os.path.join("weights", "yolo_runs")
    )

    # Export best weights to the canonical location the server expects
    best = os.path.join("weights", "yolo_runs", "yolov8n_pest_model", "weights", "best.pt")
    dest = os.path.join("weights", "yolov8n_pest.pt")
    if os.path.exists(best):
        os.replace(best, dest)
        print(f"[YOLOv8n] Best weights exported -> {dest}")
    else:
        print(f"[YOLOv8n Warning] best.pt not found at {best}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8n Pest Detector")
    parser.add_argument("--data", type=str, default="pest_dataset.yaml")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    train_yolo_pest(data_yaml=args.data, epochs=args.epochs, img_size=args.imgsz, batch=args.batch)
