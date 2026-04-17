import pandas as pd
import matplotlib.pyplot as plt
import os
import config

def plot_loss_curves(run_dir, title):
    """
    Extracts training logs from an Ultralytics run and plots 
    the training vs validation loss curves to analyze overfitting/underfitting.
    """
    csv_path = os.path.join(run_dir, "results.csv")
    if not os.path.exists(csv_path):
        print(f"[WARNING] No results.csv found in {run_dir}. Skipping plot.")
        return
        
    try:
        # Read the CSV (Ultralytics pads columns with spaces)
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        
        epochs = df['epoch']
        
        # We focus on Box loss and Class loss for object detection
        train_box = df['train/box_loss']
        val_box = df['val/box_loss']
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_box, label='Train Box Loss', marker='o')
        plt.plot(epochs, val_box, label='Validation Box Loss', marker='x')
        
        plt.title(f"{title}: Training vs Validation Box Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        out_path = os.path.join(run_dir, "custom_loss_curve.png")
        plt.savefig(out_path)
        print(f"[INFO] Saved custom loss curve to {out_path}")
        plt.close()
    except Exception as e:
        print(f"[ERROR] Failed to plot {title}. Details: {e}")

def main():
    """
    Generates professional plots for the project report.
    """
    print("=== CMPE 401: Generating Analysis Plots ===")
    plot_loss_curves(os.path.join(config.RESULTS_DIR, "baseline_run"), "Baseline Model (YOLOv11n)")
    plot_loss_curves(os.path.join(config.RESULTS_DIR, "exp_high_lr"), "High LR Model")
    plot_loss_curves(os.path.join(config.RESULTS_DIR, "exp_low_lr"), "Low LR Model")

if __name__ == "__main__":
    main()
