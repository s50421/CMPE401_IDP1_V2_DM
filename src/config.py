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
# Use '0' for CUDA GPU acceleration on Windows/Linux.
# Use 'cpu' if you want to run it without the GPU.
DEVICE = "0" 
WORKERS = 8  # Number of data loading threads. 8 is usually optimal for good computers.

# --- Fast First-Round Training Settings ---
# To make training very fast, you can:
# 1. Reduce EPOCHS (e.g., 3-5)
# 2. Reduce IMG_SIZE (e.g., to 320 or 416)
EPOCHS = 20          # 10 epochs provides a good balance of accuracy and time for RTX
BATCH_SIZE = -1      # AutoBatch to maximize RTX VRAM utilization
IMG_SIZE = 640       # Increased from 416 for better small object detection

# --- Model & Dataset ---
MODEL_NAME = "yolo11n.pt"   # YOLOv11 Nano (Fastest model)
DATASET = "VisDrone.yaml"   # Built-in VisDrone config
