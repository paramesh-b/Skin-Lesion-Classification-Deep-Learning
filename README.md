# Skin Lesion Classification Using Deep Learning (Benign vs Malignant)

### Author: **Paramesh Kumar Bukya**  
*MSc Data Science (Distinction), Manchester Metropolitan University (MMU)*  
*Dissertation Project – 2024*

---

## 📘 Project Overview

This project investigates **deep learning techniques for binary skin lesion classification** (Benign vs Malignant) using dermoscopic images from the **ISIC 2016 Challenge Dataset**.

The goal is to support early detection of **melanoma**, one of the most aggressive forms of skin cancer, by evaluating multiple CNN architectures and analysing their performance, limitations, and suitability for clinical workflows.

The project includes:

- End-to-end training of multiple CNN models  
- Transfer learning with ImageNet-pretrained weights  
- Handling class imbalance  
- Comparison of model architectures  
- Evaluation using F1-score, ROC-AUC, Precision, Recall, and Accuracy  
- Insights on the impact of data augmentation  
- Best model demonstration and saved model usage

---

## 🔒 Dataset Access (Important)

This project uses the **ISIC 2016 Challenge Dataset**, which is publicly accessible but **cannot be re-distributed**.

### Therefore:  
- Image data is **NOT uploaded** to this repository  
- Only the notebook and dissertation PDF are included  

You may download the dataset directly from the official source:

🔗 https://challenge.isic-archive.com/landing/2016/

---

## 📂 Repository Contents

| File | Description |
|------|-------------|
| `classification_of_skin_lesion_dissertation_final.ipynb` | Complete training, evaluation, and model comparison notebook |
| `Paramesh Kumar Bukya dissertation.pdf` | Final MSc Dissertation document submitted to MMU |
| `README.md` | Documentation & dataset access instructions |

---

## 🧪 Deep Learning Methodology

### **1. Preprocessing**
- Image resizing (224×224 for ResNet/VGG; 299×299 for Inception)
- Normalisation to ImageNet means/std
- Train/validation splits
- **Class imbalance handling** using weighted loss

### **2. Models Evaluated**
All models use **transfer learning**:

| Model | F1-Score | ROC-AUC | Accuracy |
|-------|----------|----------|----------|
| **ResNet50 (No Augmentation)** | **0.8515** | **0.8638** | **0.8593** |
| ResNet18 | 0.7984 | 0.7369 | 0.7848 |
| VGG16 | 0.7867 | 0.7549 | 0.7849 |
| InceptionV3 | 0.7463 | 0.7712 | 0.7183 |
| EfficientNet-B0 | 0.6169 | 0.7812 | 0.5816 |
| ResNet50 (With Augmentation) | 0.7073 | 0.7279 | 0.6806 |

### 🌟 **Best Model: ResNet50 (without augmentation)**  
Achieved the highest F1-score and ROC-AUC.

---

## 📊 Key Findings

- **Deep learning is effective** for melanoma detection  
- **Transfer learning** significantly boosts performance  
- **ResNet50** outperforms all other tested architectures  
- **Data augmentation decreased performance**, likely due to distortion of clinical lesion features  
- Weighted loss effectively reduced bias from class imbalance  
- Model predictions provide meaningful decision support for dermatologists

---

## 🤖 Best Model Demonstration

The notebook includes:

- How to load the saved **ResNet50 pickle file**  
- How to preprocess a single dermoscopic image  
- How to generate a prediction (“Benign” or “Malignant”)

This demonstrates practical usage for deployment or integration.

---

## 🚀 Future Directions

- Expand to **multi-class classification** (various skin conditions)  
- Add **Explainable AI** techniques (e.g., Grad-CAM for heatmaps)  
- Explore **Vision Transformers (ViT)**  
- Validate model performance with **clinical partners**  
- Build larger, more diverse datasets

---

## ♻️ Reproducibility

To run the notebook:

1. Download the ISIC dataset  
2. Install dependencies:
   ```bash
   pip install torch torchvision numpy pandas matplotlib scikit-learn pillow opencv-python
