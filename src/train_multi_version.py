"""
My script for training the older generation YOLO models.
Used for Part V of the rubric to prove that YOLOv11 actually is better than its predecessors.
"""

import os
from ultralytics import YOLO
import config

def train_version(model_name, run_name):
    """
    Helper function to load up a specific YOLO architecture and train it
    using my exact same base configurations.
    """
    print(f"\n{'='*50}")
    print(f"--- Training YOLO Version: {model_name} ---")
    print(f"{'='*50}\n")
    
    model = YOLO(model_name)
    
    model.train(
        data=config.DATASET,
        epochs=config.EPOCHS,
        imgsz=config.IMG_SIZE,
        batch=config.BATCH_SIZE,
        workers=config.WORKERS,
        project=config.RESULTS_DIR,
        name=run_name,
        device=config.DEVICE,
        exist_ok=True
    )

def main():
    print("=== CMPE 401: Multi-Version YOLO Comparison ===")
    
    # YOLOv8
    train_version("yolov8n.pt", "multi_yolov8n")
    
    # YOLOv9 (Tiny is the smallest official variant for v9)
    train_version("yolov9t.pt", "multi_yolov9t")
    
    # YOLOv10
    train_version("yolov10n.pt", "multi_yolov10n")
    
    print("\n[SUCCESS] Generational training finished.")

if __name__ == "__main__":
    main()
