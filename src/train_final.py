"""
My script to train the ultimate 'Best Combination' model for the CMPE 401 competition.
Based on all my previous experiments, YOLO11s trained for 50 epochs mitigates underfitting 
and has the highest capacity for the dense VisDrone dataset. So this is my final submission!
"""

import os
import urllib.request
import zipfile
from ultralytics import YOLO
import config

def download_challenge_set(dataset_dir):
    """
    Downloads the VisDrone test-challenge dataset. 
    This is the blind dataset used for competitions (it has no ground truth labels!).
    """
    challenge_dir = os.path.join(dataset_dir, "VisDrone")
    os.makedirs(challenge_dir, exist_ok=True)
    zip_path = os.path.join(challenge_dir, "VisDrone2019-DET-test-challenge.zip")
    extract_path = os.path.join(challenge_dir, "VisDrone2019-DET-test-challenge")
    
    # Only download it if I haven't already
    if not os.path.exists(extract_path):
        print("\n[INFO] Downloading VisDrone test-challenge dataset...")
        url = "https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-test-challenge.zip"
        urllib.request.urlretrieve(url, zip_path)
        
        print("[INFO] Unzipping test-challenge dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(challenge_dir)
        print("[SUCCESS] Challenge dataset ready!")
    
    # Return the path to where the images live
    return os.path.join(extract_path, "images")

def main():
    print("=== CMPE 401: Final Competition Challenge Run ===")
    
    # Lock in my best settings
    model_name = "yolo11s.pt"
    run_name = "final_yolo11s_50e"
    epochs = 50
    
    print(f"\n[INFO] Initializing {model_name} for {epochs} epochs...")
    model = YOLO(model_name)
    
    # 1. Train the model
    model.train(
        data=config.DATASET,
        epochs=epochs,
        imgsz=config.IMG_SIZE,
        batch=config.BATCH_SIZE,
        workers=config.WORKERS,
        project=config.RESULTS_DIR,
        name=run_name,
        device=config.DEVICE,
        exist_ok=True
    )
    
    print("\n[SUCCESS] Final model training completed!")
    
    # Load up the best weights from the model I just trained
    best_weights = os.path.join(config.RESULTS_DIR, run_name, "weights", "best.pt")
    final_model = YOLO(best_weights)
    
    # 2. Evaluate on the TEST-DEV set
    # I'm doing this as a proxy for the README since I can't get mAP on the real challenge set
    print(f"\n[INFO] Evaluating {run_name} on the test-dev set (for README proxy metrics)...")
    metrics = final_model.val(data=config.DATASET, split="test", device=config.DEVICE)
    
    print("\n" + "="*50)
    print("TEST-DEV PROXY METRICS (Add to README)")
    print("="*50)
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print("="*50)

    # 3. Generate Predictions on TEST-CHALLENGE
    print("\n[INFO] Setting up the true testset-challenge...")
    datasets_root = os.path.join(config.PROJECT_ROOT, "datasets")
    challenge_images_path = download_challenge_set(datasets_root)
    
    print("\n[INFO] Running inference on test-challenge and saving prediction .txt files...")
    # This runs the model over all the blind images and spits out bounding box coordinates
    final_model.predict(
        source=challenge_images_path,
        save_txt=True,          # Save predictions in .txt format for the competition
        save_conf=True,         # Make sure to include confidence scores!
        project=config.RESULTS_DIR,
        name="competition_predictions",
        device=config.DEVICE
    )
    
    print(f"\n[SUCCESS] Prediction .txt files saved to {os.path.join(config.RESULTS_DIR, 'competition_predictions', 'labels')}")
    print("Ready to submit to the instructor!")

if __name__ == "__main__":
    main()
