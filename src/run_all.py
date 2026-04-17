"""
My main entry point for running all the different models and experiments.
Instead of running 7 different scripts manually, I wrote this master runner 
to just execute everything sequentially. I can just fire this up and go to sleep.
"""

import os
from ultralytics import YOLO
import config

def train_model(model_name, run_name, epochs):
    """
    A quick helper wrapper around the YOLO train function.
    Grabs all the shared settings from config.py to keep things DRY.
    """
    print(f"\n{'='*60}")
    print(f"--- STARTING RUN: {run_name} ({model_name} for {epochs} epochs) ---")
    print(f"{'='*60}\n")
    
    # Init the model architecture
    model = YOLO(model_name)
    
    # Kick off the training process
    model.train(
        data=config.DATASET,
        epochs=epochs,
        imgsz=config.IMG_SIZE,
        batch=config.BATCH_SIZE,
        workers=config.WORKERS,
        project=config.RESULTS_DIR,
        name=run_name,
        device=config.DEVICE,
        exist_ok=True # Overwrite if it already exists to avoid clutter
    )
    
    print(f"\n[SUCCESS] Completed run: {run_name}\n")

def main():
    print("=== CMPE 401 Master Runner ===")
    print("Executing all 7 experimental runs. Grab a coffee, this will take a while.\n")
    
    # --- Experiment 1: Epoch Scaling ---
    # Training the Nano model for 50 epochs to see if it fixes underfitting
    train_model("yolo11n.pt", "yolo11n_50e", 50)
    
    # And training it for 20 epochs to act as my baseline comparison
    train_model("yolo11n.pt", "yolo11n_20e", 20)
    
    # --- Experiment 2: Model Capacity ---
    # Scaling up from Nano to Small to see how the extra parameters handle dense objects
    train_model("yolo11s.pt", "yolo11s_20e", 20)
    
    # --- Part V: Generational Comparison ---
    # To compare the different YOLO versions fairly, I'm freezing the epochs at 10.
    train_model("yolo11n.pt", "yolo11n_10e", 10)
    train_model("yolo11s.pt", "yolo11s_10e", 10)
    train_model("yolov8n.pt", "yolov8n_10e", 10)
    
    # Note: Ultralytics uses the 'nu' suffix for their v5 implementations now
    train_model("yolov5nu.pt", "yolov5n_10e", 10)
    
    print("\n{'='*60}")
    print("[ALL DONE] All 7 models have been trained successfully!")
    print("Next step: Run `python src/evaluate.py` to dump the CSV metrics.")
    print("{'='*60}\n")

if __name__ == "__main__":
    main()
