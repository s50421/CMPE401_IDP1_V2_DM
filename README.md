# CMPE 401: Deep Learning for Engineers - YOLO VisDrone Project

**Author:** David Manhart  
**Course:** CMPE 401 - Deep Learning for Engineers  

## 📌 Project Overview
This repository contains the implementation of a practical object detection system using **YOLOv11** on the **VisDrone Dataset**. The goal of this project is to demonstrate a clear understanding of deep learning concepts, structured experimentation, and practical engineering implementation. 

Rather than building an over-engineered framework, this repository focuses on:
- **Reproducible Training:** Clean, easy-to-run scripts.
- **Controlled Experimentation:** Exploring the effects of learning rates and batch sizes on convergence.
- **Evaluation & Metrics:** Analyzing Mean Average Precision (mAP), Precision, and Recall.
- **Professional Communication:** Clear tracking of results via loss curves and metric comparisons.

---

## 📂 Repository Structure

```text
CMPE401_IDP1_V2_DM/
│
├── src/                        # Source code for training and evaluation
│   ├── dataset_setup.py        # Verifies environment and dataset configuration
│   ├── train_baseline.py       # Trains the YOLO baseline model
│   ├── train_experiments.py    # Runs hyperparameter tuning experiments
│   ├── evaluate.py             # Computes mAP and evaluation metrics
│   └── plot_results.py         # Generates loss curves for the report
│
├── results/                    # Saved models, logs, and evaluation charts (generated after training)
├── data/                       # Local dataset directory (if used)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## ⚙️ Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/CMPE401_IDP1_V2_DM.git
   cd CMPE401_IDP1_V2_DM
   ```

2. **Install dependencies:**
   We use the official `ultralytics` package to handle the heavy lifting of the model architecture, allowing us to focus on the engineering loop.
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify the environment:**
   ```bash
   python src/dataset_setup.py
   ```

---

## 🚀 How to Run the Code

### 1. Training the Baseline
To train the baseline YOLOv11 model (which will also automatically download the ~1.5GB VisDrone dataset on the first run):
```bash
python src/train_baseline.py
```
*Note: The script is currently set to 5 epochs for testing. Open the file to adjust the `epochs` variable for full training (e.g., 50).*

### 2. Running Controlled Experiments
To run the hyperparameter tuning experiments (comparing high vs low learning rates):
```bash
python src/train_experiments.py
```

### 3. Evaluation
Once models are trained, evaluate their Mean Average Precision (mAP):
```bash
python src/evaluate.py
```

### 4. Analysis & Plotting
To generate the training vs validation loss curves:
```bash
python src/plot_results.py
```
This will save `custom_loss_curve.png` charts inside the respective `results/` folders.

---

## 📊 Findings & Analysis (To be completed post-training)

*Note: The following tables will be filled out after running the full 50-epoch training cycles.*

### Model Comparison
| Model Version | Learning Rate | Batch Size | mAP50 | mAP50-95 | Notes |
|---------------|---------------|------------|-------|----------|-------|
| Baseline      | 0.01 (auto)   | 16         | TBA   | TBA      | Standard convergence |
| Exp: High LR  | 0.05          | 16         | TBA   | TBA      | Investigating overshooting |
| Exp: Low LR   | 0.001         | 16         | TBA   | TBA      | Investigating stable but slow convergence |

### Core Concepts Demonstrated
1. **Transfer Learning:** Utilizing pre-trained YOLOv11 nano weights (`yolo11n.pt`) as a starting point.
2. **Hyperparameter Tuning:** Systematically altering the Learning Rate (`lr0`) to observe gradient descent behavior.
3. **Loss Analysis:** Tracking Box Loss to monitor overfitting (Training vs Validation divergence).

## 🔮 Future Improvements
- **Augmentation Testing:** Applying mixup and mosaic augmentations to improve small object detection (common in VisDrone).
- **Ensembling:** Combining predictions from multiple YOLO versions (e.g., YOLOv8 + YOLOv11).
- **Exporting:** Converting the trained `.pt` model to ONNX or TensorRT for faster edge deployment.
