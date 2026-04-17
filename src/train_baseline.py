import os
from ultralytics import YOLO
import config

def train_baseline():
    """
    Trains a baseline YOLOv11 Nano model on the VisDrone dataset.
    This serves as the control model for our CMPE 401 experiments.
    """
    print("=== Training YOLO Baseline ===")
    print(f"Model: {config.MODEL_NAME}")
    print(f"Dataset: {config.DATASET}")
    
    # Load the baseline YOLO model
    model = YOLO(config.MODEL_NAME)
    
    results = model.train(
        data=config.DATASET,
        epochs=config.EPOCHS,
        imgsz=config.IMG_SIZE,
        batch=config.BATCH_SIZE,
        workers=config.WORKERS,
        project=config.RESULTS_DIR,
        name="baseline_run",
        device=config.DEVICE
    )
    
    print("\n[SUCCESS] Baseline training complete.")
    print("Results are saved in the 'results/baseline_run' directory.")

if __name__ == "__main__":
    train_baseline()
