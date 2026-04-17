"""
Centralized Configuration for YOLO Training

This file allows you to easily tweak all important variables from one place.
"""

import os

# --- Project Paths ---
# Ensures results always save inside the project folder, regardless of where you run the script from
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# --- Hardware Acceleration ---
# Use 'mps' for Apple Silicon (M1/M2/M3) GPU acceleration.
# Use 'cpu' if you want to run it without the GPU.
DEVICE = "mps" 
WORKERS = 4  # Number of data loading threads. 4-8 is usually optimal for Macs.

# --- Fast First-Round Training Settings ---
# To make training very fast, you can:
# 1. Reduce EPOCHS (e.g., 3-5)
# 2. Reduce IMG_SIZE (e.g., to 320 or 416)
EPOCHS = 3           # Very short run to test the pipeline quickly
BATCH_SIZE = 16      # Decrease if you get memory errors
IMG_SIZE = 416       # Reduced from 640 for faster first rounds

# --- Model & Dataset ---
MODEL_NAME = "yolo11n.pt"   # YOLOv11 Nano (Fastest model)
DATASET = "VisDrone.yaml"   # Built-in VisDrone config
