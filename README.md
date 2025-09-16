# pneumonia-detection-swin-transformer
Deep learning pipeline using Swin Transformer for pneumonia detection from chest X-ray images. Achieved validation accuracy up to 98.95% and test accuracy of 76.6%, highlighting both the potential and challenges of deploying transformer-based architectures in medical imaging.
README.md
# Pneumonia Detection with Swin Transformer

This project implements a deep learning model based on **Swin Transformer** for detecting **Pneumonia** from chest X-ray images.  
It uses the publicly available **Chest X-Ray Pneumonia Dataset** (Kaggle) and leverages PyTorch with data augmentation, transfer learning, and comprehensive evaluation metrics.

## 📊 Key Results
- **Best Validation Accuracy:** 98.95%
- **Test Accuracy:** 76.60%
- **Sensitivity (Recall for Pneumonia):** 0.9974
- **Specificity (Recall for Normal):** 0.3803
- **Precision (Weighted):** 82.61%
- **F1 Score (Weighted):** 73.22%

The model demonstrates very strong sensitivity in detecting pneumonia but struggles with specificity for normal cases — a common challenge in medical image classification.

## 🚀 Features
- **Data Augmentation**: Random flips, rotations, color jitter.
- **Model**: Swin Transformer (`swin_tiny_patch4_window7_224`) pretrained on ImageNet.
- **Training Enhancements**: AdamW optimizer, ReduceLROnPlateau scheduler.
- **Visualization**: Training/validation curves, confusion matrix.
- **Evaluation**: Accuracy, Precision, Recall, F1-score, Sensitivity, Specificity.

---

## 📂 Dataset
Dataset: [Chest X-Ray Images (Pneumonia) on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

Folder structure:
chest_xray/
train/
NORMAL/
PNEUMONIA/
val/
NORMAL/
PNEUMONIA/
test/
NORMAL/
PNEUMONIA/

## ⚙️ Installation & Requirements
Clone the repository and install dependencies:
```bash
git clone https://github.com/<your-username>/pneumonia-detection-swin-transformer.git
cd pneumonia-detection-swin-transformer
pip install -r requirements.txt

Requirements:

Python 3.8+

PyTorch

Torchvision

timm

scikit-learn

matplotlib

seaborn

Pillow

pandas

▶️ Usage

Download and unzip the dataset into the project folder.

Run the training:

python train.py

Evaluate on test set:

python evaluate.py

The best model is saved as:

best_model.pth
pneumonia_detection_model_complete.pth

📈 Example Outputs

Training & Validation Loss Curves

Validation Accuracy Curve

Confusion Matrix Heatmap

Detailed Classification Report

🧪 Limitations

High sensitivity but lower specificity (tends to misclassify normal X-rays as pneumonia).

Performance could be improved with:

More balanced dataset

Advanced preprocessing

Model ensemble methods

📜 License

This project is released under the MIT License.

🙌 Acknowledgements

Dataset by Paul Mooney on Kaggle.

Swin Transformer architecture by Microsoft Research.

PyTorch & TIMM libraries.
