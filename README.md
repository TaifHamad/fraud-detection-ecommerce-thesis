# Fraud Detection in E-Commerce: Machine Learning Approaches for Imbalanced Data

**MSc Thesis | Taif Hamad AlWizainani | Umm Al-Qura University | May 2026**  
**Supervisor: Prof. Manal Adnan Al-Ghamdi**

---

## Overview

This repository contains the complete code and analysis for my Master's thesis on fraud detection in e-commerce transaction data. The study develops a four-stage machine learning pipeline using the **IEEE-CIS Fraud Detection dataset** (~590,000 transactions), with a focus on handling severe class imbalance.

**Best result:** LightGBM with SMOTETomek + Optuna achieved **PR-AUC = 0.8790**, while the soft-voting ensemble achieved the best **F1-Score = 0.8110**.

---

## Repository Structure

```
fraud-detection-ecommerce-thesis/
│
├── Methodology.ipynb            # Full pipeline: dataset → models → results
├── demo_analysis.ipynb          # Demo: loads saved models → generates all figures & tables
├── requirements.txt             # Required Python libraries
│
└── thesis_figures/              # All 9 thesis figures (PNG, 300 DPI)
    ├── fig3_1_class_distribution.png
    ├── fig3_2_pipeline_workflow.png
    ├── fig4_1_pr_stage1.png
    ├── fig4_2_pr_stage2.png
    ├── fig4_3_roc_stage3.png
    ├── fig4_4_pr_stage4.png
    ├── fig4_5_roc_stage4.png
    ├── fig4_6_confusion_matrix.png
    └── fig4_7_prauc_progression.png
```

---

## Experimental Pipeline

The study is structured as four sequential stages:

| Stage | Description | Best PR-AUC |
|-------|-------------|-------------|
| **Stage 1** | Baseline models on sampled data (LR, RF, XGBoost, Stacking) | 0.5578 |
| **Stage 2** | Feature engineering + LightGBM + Optuna (sampled) | 0.6532 |
| **Stage 3** | Full dataset without hyperparameter optimisation | 0.6535 |
| **Stage 4** | Full pipeline: SMOTETomek + Optuna + Soft-Voting Ensemble | **0.8790** |

---

## Stage 4 Results (Final Models)

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1-Score |
|-------|--------|---------|-----------|--------|----------|
| LightGBM ★ | **0.8790** | **0.9731** | **0.9838** | 0.6310 | 0.7689 |
| XGBoost | 0.8114 | 0.9642 | 0.8645 | 0.6884 | 0.7664 |
| CatBoost | 0.8189 | 0.9625 | 0.8636 | 0.7060 | 0.7769 |
| Soft-Voting Ensemble | 0.8511 | 0.9682 | 0.9441 | 0.7109 | **0.8110** |

---

## Key Findings

- **SMOTETomek** produced the largest single improvement: +0.2255 PR-AUC (Stage 3 → Stage 4)
- **LightGBM** achieved the highest PR-AUC (0.8790) with precision of 0.9838
- **Soft-Voting Ensemble** achieved the best F1-Score (0.8110) by combining all three models
- **PR-AUC** is a more informative metric than ROC-AUC under severe class imbalance

---

## How to Run

### Option A — Full Pipeline (trains from scratch)
```
1. Download the IEEE-CIS dataset from Kaggle:
   https://www.kaggle.com/c/ieee-fraud-detection

2. Open Methodology.ipynb in Google Colab

3. Run all cells sequentially
   (Note: full training takes several hours with GPU)
```

### Option B — Demo (loads saved models, generates all figures)
```
1. Open demo_analysis.ipynb in Google Colab

2. Upload best_models_v2.pkl when prompted

3. Run all cells → all figures and tables generated in minutes
```

---

## Dataset

**IEEE-CIS Fraud Detection Dataset**  
- Source: [Kaggle Competition](https://www.kaggle.com/c/ieee-fraud-detection)  
- Size: ~590,000 transactions, 400+ features  
- Fraud rate: ~3.5%  
- Released by: IEEE Computational Intelligence Society & Vesta Corporation

---

## Libraries

```
lightgbm
xgboost
catboost
optuna
scikit-learn
imbalanced-learn
pandas
numpy
matplotlib
seaborn
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## Methodology

- **Preprocessing:** Ordinal encoding, median imputation, 32-bit memory optimisation
- **Feature Engineering:** Temporal (hour, day), amount-based (log, decimal, rounded), email-domain features
- **Imbalance Handling:** SMOTETomek (ratio=0.30) → training fraud ratio raised from 3.5% to 22.96%
- **Optimisation:** Optuna TPE sampler, 30 trials, 3-fold stratified CV, maximising PR-AUC
- **Evaluation:** PR-AUC (primary), ROC-AUC, Precision, Recall, F1-Score
- **Threshold Optimisation:** F1-maximising threshold per model

---

*Department of Computer Science | College of Computers and Information Systems | Umm Al-Qura University*
