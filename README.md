# 🎭 Multimodal Emotion Recognition using Deep Learning

A deep learning-based multimodal emotion recognition system that combines **audio (MFCC features)** and **video (facial frame features)** using CNN, BiLSTM, and attention-based late fusion.

---

## 🚀 Overview

This project performs emotion classification using both speech and facial expressions. It leverages:
- Audio feature extraction using MFCC
- Video feature extraction using MTCNN
- Temporal modeling using BiLSTM
- Attention-based late fusion for modality weighting

---

## 🧠 Model Architecture

- **Audio Branch:** CNN + BiLSTM
- **Video Branch:** MTCNN + BiLSTM
- **Fusion:** Attention-based Late Fusion
- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Adam
- **Training:** Mixed Precision (AMP)

---

## 📊 Results

- Accuracy: 85%
- F1 Score: 0.92

---

## 📁 Dataset

- RAVDESS dataset (preprocessed)
- Audio: MFCC features
- Video: Frame-based features (112×112×3)

---

## 🛠️ Tech Stack

- PyTorch
- Torchvision
- NumPy
- Scikit-learn
- OpenCV

---

## ⚙️ Training Pipeline

1. Load preprocessed audio and video features
2. Train audio and video models separately
3. Fuse embeddings using attention mechanism
4. Optimize using Adam + AMP
5. Evaluate using accuracy, precision, recall, and F1-score

---

## 📈 Visualizations

- Loss vs Epoch
- Accuracy vs Epoch
- Precision/Recall/F1 curves
- Confusion Matrix

---

## 💾 Checkpointing

- Best model saved based on F1-score
- Training progress saved for resumption
- Logs stored in JSON format

---

## 📌 Key Features

- Multimodal deep learning architecture
- Attention-based fusion mechanism
- GPU optimized training (T4 compatible)
- Fault-tolerant training with checkpoints

---

## 📂 How to Run

```bash
pip install -r requirements.txt
python main.py
