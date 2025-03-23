# Toxic Comment Classification - Kaggle Competition (2024)

Welcome to our project repository for the **Toxic Comment Classification** challenge, part of the *Foundations of Deep Learning* course at MSc in DSBA (2024–2025).

## 📝 Competition Description

Automated moderation of user-generated content plays a critical role in fostering safe and inclusive online environments. However, toxicity classifiers often inherit biases from their training data. This Kaggle competition frames the issue as a **subpopulation shift** problem, where different demographic identities represent distinct subpopulations.  

The objective is to **maintain high performance across all demographic groups**, not just on average, and to **reduce bias** in toxicity predictions, particularly for comments referencing specific demographic groups.

[competition page](https://www.kaggle.com/competitions/toxic-comment-classification-dsba-2025/overview)

[data](https://www.kaggle.com/competitions/toxic-comment-classification-dsba-2025/data)

## 👥 Team Members

- Piangpim Chancharunee  
- Ruxi He  
- I-Hsun Lu

## 📊 Dataset Overview

- **Total samples**: 269,038 (train set)
- **Task**: Binary classification of toxic vs. non-toxic comments
- **Challenge**:
  - Imbalanced label distribution (fewer toxic comments)
  - Overlapping demographic group labels
  - Subpopulation shift between train and test distributions

## ⚙️ Methods and Models

| Method | Model | Description | Accuracy (WGA) |
|--------|-------|-------------|----------------|
| #0 | CountVectorizer + MLP | Baseline with 1-hidden-layer MLP | **0.71** |
| #1 | FastText + ResNet + Attention | Lightweight ResNet with attention and demographic-weighted loss | **0.69** |
| #2 | DistilBERT + Adversarial Debiasing + DRO | Subpopulation-aware classifier with Gradient Reversal and group reweighting | **0.78** |
| #3 | BERT / BERT Large | Fine-tuned pre-trained language models for high accuracy | **0.78** |

> ✅ *Best performance achieved with Method #2 and #3, reaching 0.78 worst-group accuracy.*

## 🧪 Key Experiments

- **Method #1**:
  - FastText embedding for dimensionality reduction
  - Custom loss weighting based on demographic performance
  - Training time: ~3 minutes/epoch (Kaggle CPU)

- **Method #2**:
  - DistilBERT with adversarial debiasing and DRO
  - Group-aware loss functions (to be improved for overlapping groups)
  - Training time: ~40–60 minutes/epoch (Google Colab GPU)

- **Method #3**:
  - BERT and BERT Large tested for accuracy vs. overfitting
  - BERT outperformed BERT Large due to better generalization on limited data

## 🔍 Challenges and Learnings

- Handling **bias and fairness** is more complex when demographic groups overlap.
- Larger models like BERT require careful regularization to avoid overfitting.
- Lightweight models (FastText + ResNet) are efficient but trade off some accuracy.

## 🧠 Applications and Use Cases

- **Method #1**: On-device moderation, low-resource environments
- **Method #2**: Bias-sensitive applications (e.g. forums, health platforms)
- **Method #3**: High-accuracy systems in large-scale platforms (e.g. YouTube, X, Facebook)

## 📂 Structure of the Repository

```bash
├── kaggle_baseline.ipynb
├── Method#1.ipynb
├── Method#2.ipynb
├── Method#3-1.ipynb
├── Method#3-2.ipynb
└── README.md
