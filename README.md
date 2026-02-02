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

## 🛠️ Live Inference (Evaluation)
1. **Download Model Weights:** [best_model.pth].(https://drive.google.com/drive/folders/114bFLUBe_g5gJ2wtv0ILO_TTz0GDUCy_?usp=sharing)
2. **Setup:** Ensure `best_model.pth` and your script are in the same directory.
3. **Execution:** Run the following to evaluate on the hidden examiner dataset:

```python
# Short-form usage for live testing
from your_script import run_live_inference

# Path to the .npz file provided by the examiner
acc = run_live_inference('examiner_file.npz')
