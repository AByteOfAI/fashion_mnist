# 👗 Fashion MNIST Classification with Deep CNN and Other Machine Learning Models

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
  <img src="https://img.shields.io/badge/Course-RAS598%20Fall%202024-8C1D40" />
</p>

A comprehensive machine learning pipeline that classifies Fashion MNIST images into 10 clothing categories, comparing deep learning architectures (CNN, ResNet) against traditional ML baselines (SVM, Random Forest, KNN, Logistic Regression, MLP).

> **Arizona State University — RAS598, Fall 2024**

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Models Implemented](#models-implemented)
- [Architecture Details](#architecture-details)
- [Results](#results)
- [Confusion Matrix Analysis](#confusion-matrix-analysis)
- [Grad-CAM Interpretability](#grad-cam-interpretability)
- [Installation & Usage](#installation--usage)
- [Project Structure](#project-structure)
- [Known Limitations](#known-limitations)
- [Future Work](#future-work)
- [Authors](#authors)

---

## Overview

This project builds and evaluates a full ML pipeline for image classification on the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset. The primary goal is to implement a deep Convolutional Neural Network (CNN) and a Residual Network (ResNet) using TensorFlow, then benchmark them against traditional machine learning algorithms to quantify the accuracy-complexity tradeoff.

**Key Questions Addressed:**
- How does a deep CNN compare to traditional ML models on Fashion MNIST?
- What preprocessing and optimization techniques improve classification accuracy?
- Can model predictions be made interpretable using Grad-CAM?

---

## Dataset

The [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset is a drop-in replacement for the classic MNIST digit dataset, containing **70,000 grayscale images** (28×28 pixels) across **10 categories**:

| Label | Class        | Label | Class      |
|-------|-------------|-------|------------|
| 0     | T-shirt/top | 5     | Sandal     |
| 1     | Trouser     | 6     | Shirt      |
| 2     | Pullover    | 7     | Sneaker    |
| 3     | Dress       | 8     | Bag        |
| 4     | Coat        | 9     | Ankle boot |

- **Training set:** 60,000 images
- **Test set:** 10,000 images
- **Balanced classes:** 6,000 samples per category in training
- **Pixel intensity:** 0–255 (normalized to [0, 1] during preprocessing)

---

## Models Implemented

| Model | Type | Test Accuracy | Key Characteristic |
|-------|------|:------------:|---------------------|
| **ResNet** | Deep Learning | **~91%** | Residual connections for gradient flow |
| **CNN** | Deep Learning | **~91%** | Progressive spatial feature extraction |
| **Random Forest** | Traditional ML | ~89% | Ensemble of decision trees |
| **SVM** | Traditional ML | ~87% | Kernel-based hyperplane separation |
| **MLP** | Traditional ML | ~85% | Fully connected layers, no spatial awareness |
| **Logistic Regression** | Traditional ML | ~83% | Linear decision boundary baseline |
| **KNN** | Traditional ML | — | Distance-based lazy learner |

---

## Architecture Details

### ResNet (Residual Network)

A lightweight ResNet implementation tailored for Fashion MNIST:

```
Input (28×28×1) → Conv2D(32) → MaxPool → ResBlock(32) → MaxPool → ResBlock(32) → GAP → Dense(10, Softmax)
```

| Feature | Implementation |
|---------|---------------|
| Filter Size | 32 throughout |
| Convolution | 3×3 kernels |
| Pooling | 2×2 MaxPool |
| Regularization | Batch Normalization |
| Skip Connections | Identity shortcuts |
| Final Layer | Softmax (10 classes) |
| Total Parameters | 38,154 |

### CNN (Convolutional Neural Network)

A sequential CNN with increasing filter complexity:

```
Input (28×28×1) → Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128) → Flatten → Dense(128, ReLU) → Dense(10, Softmax)
```

| Layer | Filters | Kernel | Output Shape |
|-------|---------|--------|-------------|
| Conv2D_1 | 32 | 3×3 | 26×26×32 |
| MaxPool_1 | — | 2×2 | 13×13×32 |
| Conv2D_2 | 64 | 3×3 | 11×11×64 |
| MaxPool_2 | — | 2×2 | 5×5×64 |
| Conv2D_3 | 128 | 3×3 | 3×3×128 |

---

## Results

### Training Performance

| Metric | ResNet (Training) | ResNet (Validation) | CNN (Training) | CNN (Validation) |
|--------|:-:|:-:|:-:|:-:|
| Final Accuracy | 95% | 91% | 96% | 91% |
| Final Loss | 0.10 | 0.30 | 0.12 | 0.28 |
| Epochs | 20 | 20 | 10 | 10 |

Both deep learning models converge well with a ~4–5% train-validation gap, indicating mild overfitting but good generalization.

### Per-Class Classification Report (ResNet)

| Class | Precision | Recall | F1-Score |
|-------|:---------:|:------:|:--------:|
| T-shirt/top | 0.81 | 0.91 | 0.86 |
| Trouser | 0.98 | 0.99 | 0.98 |
| Pullover | 0.87 | 0.88 | 0.88 |
| Dress | 0.95 | 0.88 | 0.91 |
| Coat | 0.88 | 0.87 | 0.88 |
| Sandal | 0.96 | 0.99 | 0.97 |
| Shirt | 0.79 | 0.71 | 0.74 |
| Sneaker | 0.94 | 0.98 | 0.96 |
| Bag | 0.97 | 0.99 | 0.98 |
| Ankle boot | 1.00 | 0.92 | 0.96 |
| **Overall** | **0.91** | **0.91** | **0.91** |

---

## Confusion Matrix Analysis

<p align="center">
  <img src="confusion_matrix.png" alt="Confusion Matrix" width="700"/>
</p>

**Best Performers:** Bag (99.3%), Sandal (98.9%), Trouser (98.7%) — visually distinct categories with unique silhouettes.

**Most Challenging:** Shirt (70.8% recall) — frequently confused with T-shirt/top (156 misclassifications), due to similar shapes at 28×28 resolution.

**Notable Confusion Pairs:**
- **Shirt ↔ T-shirt/top:** 156 Shirts misclassified as T-shirts (highest single off-diagonal value)
- **Coat ↔ Pullover:** Mutual confusion (40 and 43 misclassifications respectively)
- **Ankle boot ↔ Sneaker:** 56 Ankle boots predicted as Sneakers

---

## Grad-CAM Interpretability

<p align="center">
  <img src="grad_cam.png" alt="Grad-CAM Visualization" width="700"/>
</p>

[Grad-CAM](https://arxiv.org/abs/1610.02391) (Gradient-weighted Class Activation Mapping) is used to visualize which image regions drive the model's predictions:

- **Left:** Original grayscale input (True label: Ankle boot)
- **Middle:** Grad-CAM heatmap — bright regions (yellow/red) show where the model focuses
- **Right:** Superimposed overlay confirming the model attends to the boot's shaft and sole

This confirms the model relies on semantically meaningful features (shape, outline, sole) rather than background artifacts.

---

## Installation & Usage

### Prerequisites

```bash
Python >= 3.8
TensorFlow >= 2.x
scikit-learn
NumPy
Matplotlib
Seaborn
Plotly
```

### Setup

```bash
# Clone the repository
git clone https://github.com/AByteOfAI/fashion_mnist.git
cd fashion_mnist

# Install dependencies
pip install tensorflow scikit-learn numpy matplotlib seaborn plotly
```

### Quick Start

```python
import numpy as np
from tensorflow.keras.datasets import fashion_mnist

# Load data
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape for CNN/ResNet
X_train_cnn = X_train.reshape(-1, 28, 28, 1)
X_test_cnn = X_test.reshape(-1, 28, 28, 1)
```

---

## Project Structure

```
fashion_mnist/
├── README.md                 # Project documentation
├── RASFINALppt.pdf           # Presentation slides
├── Report_Part1.pdf          # Report — methodology, models & source code
├── Report_Part2.pdf          # Report — source code (continued)
├── confusion_matrix.png      # ResNet confusion matrix visualization
└── grad_cam.png              # Grad-CAM heatmap for ankle boot prediction
```

---

## Known Limitations

- **Dataset bias:** Fashion MNIST contains only grayscale images on white backgrounds, limiting generalization to real-world colored images with complex backgrounds.
- **Shirt vs. T-shirt confusion:** These categories share highly similar silhouettes at 28×28 resolution, resulting in the lowest per-class performance (F1: 0.74).
- **Mild overfitting:** ~4–5% gap between training and validation accuracy in both deep learning models.

---

## Future Work

- Extend to **colored image datasets** with diverse, complex backgrounds (e.g., DeepFashion)
- Address **Shirt/T-shirt misclassification** with attention mechanisms or targeted feature engineering
- Explore **hybrid models** combining traditional ML feature extraction with deep learning classifiers
- Add **dropout and stronger data augmentation** to reduce overfitting
- Implement **transfer learning** using pretrained models (e.g., MobileNet, EfficientNet)

---

## Authors

| Name | Email |
|------|-------|
| **Karan Athrey** | kathrey@asu.edu |
| **Abhijit Sinha** | asinh117@asu.edu |
| **Anusha Chatterjee** | achatt53@asu.edu |

**Arizona State University** — RAS598: Robotic and Autonomous Systems, Fall 2024

---

<p align="center">
  <i>If you find this project useful, consider giving it a ⭐!</i>
</p>
