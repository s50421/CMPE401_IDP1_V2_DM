import os
from ultralytics import YOLO

def verify_environment():
    """
    Verifies the YOLO environment and prepares for dataset download.
    For the CMPE 401 project, we rely on Ultralytics' built-in VisDrone.yaml
    which automatically handles downloading the dataset from official sources.
    """
    print("=== CMPE 401: Deep Learning for Engineers ===")
    print("Verifying Environment for YOLO VisDrone Project...\n")
    
    # Load a lightweight model to verify Ultralytics is working
    try:
        model = YOLO("yolo11n.pt")
        print("[SUCCESS] YOLOv11 model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO model: {e}")
        return

    print("\n[INFO] Dataset Handling:")
    print("The VisDrone dataset is natively supported by Ultralytics.")
    print("It will automatically download (~1.5GB) into the `datasets/VisDrone` folder")
    print("the first time you run `src/train_baseline.py`.")
    print("Ensure you have a stable internet connection for the first run.")

if __name__ == "__main__":
    verify_environment()
