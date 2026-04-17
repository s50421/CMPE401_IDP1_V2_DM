"""
My script to extract and plot the training vs validation loss curves.
Ultralytics creates a huge results.png, but I just want a clean plot of the box loss 
to stick into my README to prove convergence.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import config

def plot_loss_curves(run_dir, run_name):
    # Grab the raw training logs
    csv_path = os.path.join(run_dir, "results.csv")
    if not os.path.exists(csv_path):
        print(f"[WARNING] No results.csv found for {run_name}. Can't plot it.")
        return
        
    try:
        # Load the logs into pandas and clean up any whitespace in the column names
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        
        epochs = df['epoch']
        
        # Extracting just the bounding box loss (the most important one for this project)
        train_box = df['train/box_loss']
        val_box = df['val/box_loss']
        
        # Spin up a matplotlib figure
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_box, label='Train Box Loss', marker='o')
        plt.plot(epochs, val_box, label='Validation Box Loss', marker='x')
        
        # Dress it up nicely for the report
        plt.title(f"{run_name}: Training vs Validation Box Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        # Save it directly into the run's folder
        out_path = os.path.join(run_dir, "custom_loss_curve.png")
        plt.savefig(out_path)
        print(f"[INFO] Saved loss curve for {run_name} to {out_path}")
        
        # Clean up memory
        plt.close()
    except Exception as e:
        print(f"[ERROR] Failed to plot {run_name}. Details: {e}")

def main():
    print("=== CMPE 401: Generating Loss Curves ===")
    
    if not os.path.exists(config.RESULTS_DIR):
        print(f"[ERROR] Results directory {config.RESULTS_DIR} not found.")
        return
    
    # Scan through every folder in the results directory
    runs = [d for d in os.listdir(config.RESULTS_DIR) if os.path.isdir(os.path.join(config.RESULTS_DIR, d))]
    
    # Generate a plot for each one
    for run in runs:
        run_dir = os.path.join(config.RESULTS_DIR, run)
        plot_loss_curves(run_dir, run)
        
    print("\n[SUCCESS] Loss curves generated! Go check the folders.")

if __name__ == "__main__":
    main()
