from ultralytics import YOLO
import os
import config

def evaluate_model(model_path, title):
    print(f"\n=== Evaluating {title} ===")
    if not os.path.exists(model_path):
        print(f"[ERROR] Model weights not found at {model_path}.")
        return

    try:
        model = YOLO(model_path)
        # Evaluate on the validation set
        metrics = model.val(data="VisDrone.yaml")
        print(f"\n[INFO] {title} Evaluation Results:")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
        print(f"  mAP50:    {metrics.box.map50:.4f}")
    except Exception as e:
        print(f"[ERROR] Could not evaluate model. Details: {e}")

def main():
    """
    Evaluates the trained models to gather quantitative metrics (mAP, Precision, Recall).
    """
    print("=== CMPE 401: Model Evaluation ===")
    evaluate_model(os.path.join(config.RESULTS_DIR, "baseline_run/weights/best.pt"), "Baseline Model")
    evaluate_model(os.path.join(config.RESULTS_DIR, "exp_high_lr/weights/best.pt"), "High LR Model")
    evaluate_model(os.path.join(config.RESULTS_DIR, "exp_low_lr/weights/best.pt"), "Low LR Model")

if __name__ == "__main__":
    main()
