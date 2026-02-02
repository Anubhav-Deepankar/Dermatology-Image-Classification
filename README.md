# Robust Dermatology Image Classification (Noisy Labels)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)

## 📌 Project Overview
This repository contains a deep learning solution for classifying 28x28 grayscale dermatology images into 7 categories. The primary challenge is **Noisy Training Labels**: while the training data is unreliable, the model is optimized to perform on expert-verified (Gold Standard) validation data.

## 🚀 Key Features
- **Robust ResNet-18:** A modified ResNet backbone tailored for single-channel grayscale input.
- **Noise Mitigation:** Implements **Label Smoothing** and specific data normalization (Mean: 158.32, Std: 46.95) to prevent overfitting to incorrect labels.
- **Live Inference:** A production-ready script for real-time evaluation on hidden test datasets.

## 📊 Performance
- **Baseline Accuracy:** ~42% on hidden validation data.
- **Loss Strategy:** Optimized Cross-Entropy with smoothing to handle label uncertainty.

## 📁 Repository Structure
- `Healthwith42Accuracy.ipynb`: Full pipeline (EDA, Training, Evaluation).
- `best_model.pth`: Saved weights of the best-performing model.
- `candidate_dataset.npz`: Training and validation data arrays.

## 🛠️ Usage (Live Inference)
To run the model on a new hidden dataset (`.npz` format):
```python
from your_script import evaluate_hidden_dataset

# Provide path to the hidden dataset
accuracy = evaluate_hidden_dataset('path_to_test_file.npz')
print(f"Final Accuracy: {accuracy * 100:.2f}%")
