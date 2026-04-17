import os
from ultralytics import YOLO
import config

def train_with_params(name, lr0, batch, epochs=5):
    """
    Helper function to train a model with specific hyperparameters.
    """
    print(f"\n--- Running Experiment: {name} ---")
    print(f"Hyperparameters -> lr0: {lr0}, batch: {batch}")
    
    model = YOLO(config.MODEL_NAME)
    model.train(
        data=config.DATASET,
        epochs=epochs,
        imgsz=config.IMG_SIZE,
        batch=batch,
        workers=config.WORKERS,
        lr0=lr0,
        project=config.RESULTS_DIR,
        name=name,
        device=config.DEVICE
    )

def main():
    """
    Runs controlled experiments demonstrating hyperparameter tuning 
    for the CMPE 401 project. This shows an understanding of how
    learning rates affect convergence and stability.
    """
    print("=== CMPE 401: YOLO Controlled Experiments ===")
    
    # Experiment 1: High Learning Rate
    # Demonstrates what happens when the model learns too fast (potential instability/overshooting)
    train_with_params(name="exp_high_lr", lr0=0.05, batch=16)
    
    # Experiment 2: Low Learning Rate
    # Demonstrates slow but potentially more stable convergence (might need more epochs)
    train_with_params(name="exp_low_lr", lr0=0.001, batch=16)
    
    print("\n[SUCCESS] All experiments completed. Check the 'results/' folder for comparisons.")

if __name__ == "__main__":
    main()
