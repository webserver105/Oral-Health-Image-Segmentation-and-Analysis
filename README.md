# Oral Lesion Segmentation using Deep Learning

<p align="center">
  <img src="images/banner.png" width="100%">
</p>

<p align="center">
  Deep Learning • Medical Imaging • Computer Vision • Segmentation
</p>

---

## Overview

This project focuses on automated Oral Lesion Classification and Segmentation using Deep Learning Techniques. The main goal was to explore whether Deep Learning models can help in identifying oral diseases from patient mouth images, especially Oral Potentially Malignant Disorders (OPMD) and Oral Cancer (OCA).

During this project, I explored multiple approaches starting from traditional machine learning models on tabular patient data and gradually moved towards CNN-based image models, hybrid architectures, and finally segmentation models using U-Net. This project uses Zenodo Dataset.

This project was mainly focused on:

* Handling severe class imbalance
* Learning from limited medical data
* Improving minority class detection
* Exploring medical image segmentation techniques

---

## Problem Statement

Oral cancer is often diagnosed at later stages, which makes treatment difficult. One of the major challenges in medical imaging datasets is class imbalance. In this dataset, healthy and common disease samples were much larger in number compared to Oral Cancer (OCA) samples.

Because of this, models easily become biased toward majority classes and fail to properly identify rare but important disease categories. The objective of this project was to explore different Deep Learning approaches to improve classification and segmentation performance on such challenging medical data.

---

## Project Pipeline

<p align="center">
  <img src="images/pipeline.png" width="90%">
</p>

The overall workflow followed in this project:

1. Exploratory Data Analysis (EDA)
2. Data Cleaning & Preprocessing
3. Image Cropping using Annotation Data
4. Data Augmentation
5. CNN Model Training
6. Hybrid Model Development
7. U-Net Segmentation
8. Model Evaluation

---

## Data Preprocessing

Several preprocessing techniques were applied before training:

* Mouth region extraction using annotation coordinates
* Image resizing to 256×256
* Random flips and rotations
* Class weighting
* Data augmentation
* Prevention of data leakage during splitting

Instead of using the full raw image, the model was trained only on the cropped mouth region to remove unnecessary background information.

---

## Models Explored

### 1. Tabular Machine Learning Models

Initially, I started with patient metadata and habit information (like whether he smokes or drinks etc) using:

* Neural Networks
* XGBoost

These models struggled because of severe class imbalance.

---

### 2. CNN Models

After limited performance from tabular models, I shifted towards image-based deep learning models.

The following architectures were explored:

* ResNet101
* DenseNet201
* EfficientNetB3

Transfer Learning and Fine-Tuning were used for training.

---

### 3. Hybrid Deep Learning Model

A hybrid architecture was also explored by combining:

* CNN image features
* Important tabular features

This approach slightly improved performance compared to standalone CNN models.

---

### 4. U-Net Segmentation

Finally, I experimented with semantic segmentation using U-Net.

The segmentation model was trained to identify lesion regions directly from oral images.

Techniques used:

* Dice Loss
* Focal Loss
* Class Weights
* Data Augmentation

---

## Techniques Used

* Transfer Learning
* Fine-Tuning
* Data Augmentation
* Class Weights
* Dice Loss
* Focal Loss
* SMOTE
* Early Stopping
* Medical Image Segmentation
* Hybrid Deep Learning

---

## Model Performance

| Model          | Accuracy |
| -------------- | -------- |
| ResNet101      | 56.72%   |
| DenseNet201    | 53.05%   |
| EfficientNetB3 | 48.17%   |
| Hybrid Model   | 59.69%   |

---

## CNN Results

<p align="center">
  <img src="images/resnet_results.png" width="80%">
</p>

The CNN models performed significantly better than the earlier tabular models.

Transfer Learning and class weighting helped improve minority class detection, especially for Oral Cancer samples.

---

## Hybrid Model Results

<p align="center">
  <img src="images/hybrid_results.png" width="80%">
</p>

The hybrid approach combined image features with selected patient metadata and achieved slightly better overall performance.

---

## U-Net Segmentation Results

<p align="center">
  <img src="images/unet_results.png" width="80%">
</p>

The segmentation experiments showed that lesion localization is much more challenging compared to classification because of:

* limited annotated data
* severe imbalance
* noisy medical images

Still, the experiments provided valuable insights into medical image segmentation workflows.

---

## Training Curves

<p align="center">
  <img src="images/training_curves.png" width="80%">
</p>

---

## Confusion Matrix

<p align="center">
  <img src="images/confusion_matrix.png" width="70%">
</p>

---

## Key Findings

* Severe class imbalance heavily affected model performance.
* CNNs performed better than tabular models.
* Transfer Learning improved convergence speed.
* Class weights helped improve minority class prediction.
* Segmentation remains difficult with limited medical annotations.
* Hybrid models showed slight improvements over standalone CNNs.

---

## Future Improvements

Some possible future improvements:

* Attention U-Net
* Vision Transformers
* Better lesion annotations
* Larger balanced datasets
* Self-supervised learning
* Multi-modal architectures
* Web deployment for screening assistance

---

## Repository Structure

```bash
├── images/
├── src/
├── results/
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Installation

```bash
git clone https://github.com/yourusername/oral-lesion-segmentation.git

cd oral-lesion-segmentation

pip install -r requirements.txt
```

---

## Requirements

Main libraries used:

* TensorFlow / Keras
* OpenCV
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* XGBoost

---

## Note

This repository is intended for research and educational purposes only.

The dataset used in this project is not included because it belongs to its respective owners.

---

## Author

Kunal Gandvane

Undergraduate Student
Indian Institute of Technology Bombay

---
