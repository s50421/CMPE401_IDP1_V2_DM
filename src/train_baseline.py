"""
My script for running the basic YOLOv11 baseline.
This gives me the initial metrics so I can diagnose if the model is overfitting
or underfitting on the VisDrone dataset before I start tuning anything.
"""

import os
from ultralytics import YOLO
import config

def main():
    print("=== CMPE 401: YOLO Baseline Training ===")
    
    # Init the raw, pretrained YOLO model
    model = YOLO(config.MODEL_NAME)
    
    # Kick off the training run
    model.train(
        data=config.DATASET,
        epochs=config.EPOCHS,
        imgsz=config.IMG_SIZE,
        batch=config.BATCH_SIZE,
        workers=config.WORKERS,
        project=config.RESULTS_DIR,
        name="baseline_run",  # Saving it here so I can find the loss curves later
        device=config.DEVICE,
        exist_ok=True
    )
    
    print("\n[SUCCESS] Baseline training completed! Go check results/baseline_run/ for the logs.")

if __name__ == "__main__":
    main()
