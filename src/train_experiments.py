"""
My script for running iterative model improvements (Part IV).
I used this early on to play with heavy augmentations and regularizations 
before deciding that epoch scaling was the better path for VisDrone.
"""

import os
from ultralytics import YOLO
import config

def train_experiment(name, description, **kwargs):
    """
    A helper to let me dynamically pass in whatever hyperparams I want to test out.
    Keeps the code clean and lets me run multiple experiments in a row.
    """
    print(f"\n{'='*50}")
    print(f"--- Running Experiment: {name} ---")
    print(f"Description: {description}")
    print(f"Custom Parameters: {kwargs}")
    print(f"{'='*50}\n")
    
    model = YOLO(config.MODEL_NAME)
    
    # Setting up my base kwargs from the config file
    train_args = {
        "data": config.DATASET,
        "epochs": config.EPOCHS,
        "imgsz": config.IMG_SIZE,
        "batch": config.BATCH_SIZE,
        "workers": config.WORKERS,
        "project": config.RESULTS_DIR,
        "name": name,
        "device": config.DEVICE,
        "exist_ok": True
    }
    
    # Jamming in the custom kwargs for this specific experiment
    train_args.update(kwargs)
    
    # Train it!
    model.train(**train_args)

def main():
    print("=== CMPE 401: Iterative Model Improvement ===")
    
    # --- Experiment 1: High Learning Rate ---
    # Testing what happens when the gradients explode/overshoot
    train_experiment(
        name="exp_high_lr",
        description="Testing high learning rate to observe overshooting and gradient explosion.",
        lr0=0.05
    )
    
    # --- Experiment 2: Overfitting Control ---
    # Throwing the kitchen sink at it (mixup, mosaic, weight decay) to see if 
    # heavy regularization helps on the dense drone images.
    train_experiment(
        name="exp_overfit_control",
        description="Applying regularization (weight decay) and heavy augmentation (mixup) to control overfitting.",
        weight_decay=0.001,
        mixup=0.2,
        mosaic=1.0
    )
    
    print("\n[SUCCESS] Experiments completed.")

if __name__ == "__main__":
    main()
