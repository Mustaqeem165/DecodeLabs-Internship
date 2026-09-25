# 🛡️ Enterprise Data Engineering & Financial Fraud Detection Pipeline

**Author:** Muhammad Mustaqeem  
**Program:** Data Science Industrial Training | Batch: 2026  
**Organization:** DecodeLabs  
**Repository:** [DecodeLabs-Internship](https://github.com/Mustaqeem165/DecodeLabs-Internship)

---

## 📌 Executive Summary

Modern enterprise payment gateways face an asymmetric risk challenge: legitimate transactions represent > 99% of overall volume, while fraudulent activities constitute a subtle minority. Standard off-the-shelf classifiers exhibit a false sense of security—achieving > 99% naive accuracy by simply classifying all transactions as legitimate, thereby missing catastrophic financial losses.

This repository hosts a production-grade machine learning system divided into two integrated milestones:
1. **Project 1: Advanced Input Fidelity & Vectorized Feature Engine** — Structural data sanitization, outlier Winsorization, orthogonal coordinate transformations, and collinearity eradication.
2. **Project 2: Leak-Free Supervised Learning Fraud Pipeline** — End-to-end SMOTE integration, scale-sensitive algorithmic pipelines, 5-fold cross-validation tuning, and strict evaluation under Recall, Precision, and ROC-AUC.

---

## 🏗️ System Architecture

```text
                                  DATA.csv
                       (1,200 Raw E-Commerce Records)
                                      │
                                      ▼
             ┌──────────────────────────────────────────────────┐
             │      STAGE 1: INPUT FIDELITY & SANITIZATION      │
             │   • Missingness Handling Matrix (CouponCode)     │
             │   • Non-destructive IQR Winsorization (Capping)  │
             └────────────────────────┬─────────────────────────┘
                                      │
                                      ▼
             ┌──────────────────────────────────────────────────┐
             │      STAGE 2: VECTORIZED FEATURE SYNTHESIS       │
             │   • C-Level SIMD Multi-Variable Interaction Math │
             │   • Orthogonal Coordinate Projection (One-Hot)   │
             │   • Pearson Collinearity Eradication (|r| > 0.80)│
             └────────────────────────┬─────────────────────────┘
                                      │
                                      ▼
                             cleaned_DATA.csv
                                      │
                                      ▼
             ┌──────────────────────────────────────────────────┐
             │    STAGE 3: ZERO-LEAKAGE SUPERVISED PIPELINE     │
             │   • Stratified 80/20 Train/Test Partition        │
             │   • imblearn Pipeline (SMOTE restricted to Train)│
             │   • Scaled Linear Engine vs Invariant Tree Engine│
             │   • 5-Fold StratifiedKFold GridSearchCV Tuning   │
             └────────────────────────┬─────────────────────────┘
                                      │
                                      ▼
              Rigorous Evaluation: Recall | Precision | ROC-AUC
