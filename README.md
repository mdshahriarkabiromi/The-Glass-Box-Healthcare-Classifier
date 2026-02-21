# 🧠 The “Glass Box” Stroke Healthcare Classifier

> **Engineered a transparent diagnostic model using SHAP to provide feature-level clinical justifications, bridging the gap between black-box AI and medical trust.**

---

## 📌 Project Overview

This project develops a **clinically calibrated stacked ensemble model** for stroke risk prediction using the Stroke Prediction Dataset.

Unlike traditional black-box systems, this model integrates:

- ✅ Strong predictive performance  
- ✅ Probability calibration  
- ✅ Clinically tuned threshold selection  
- ✅ SHAP-based global and local explanations  
- ✅ LIME patient-level explanations  

The goal is to combine **accuracy + transparency**, enabling trustworthy decision support in healthcare environments.

---

## 🗂 Dataset

**Source:** Kaggle Stroke Prediction Dataset  
**Target variable:** `stroke` (0 = No Stroke, 1 = Stroke)

### Key Features

- Age
- Hypertension
- Heart disease
- Average glucose level
- BMI
- Smoking status
- Work type
- Residence type
- Gender

The dataset is **highly imbalanced (~5% stroke cases)**, reflecting real-world screening scenarios.

---

## 🏗 Model Architecture

### 🔹 Preprocessing

- Median imputation (numeric features)
- Most frequent imputation (categorical features)
- Standard scaling (numeric features)
- One-hot encoding (categorical features)

### 🔹 Stacked Ensemble

**Base learners:**

- Random Forest
- Extra Trees
- HistGradientBoosting
- SVM (RBF kernel)

**Meta-learner:**

- Logistic Regression

### 🔹 Probability Calibration

Isotonic calibration applied for improved clinical probability reliability.

---

## 🎯 Threshold Strategy

Instead of using the default 0.5 threshold, we applied:

```yaml
strategy: recall_at_least
min_recall: 0.75
```

This prioritizes identifying stroke cases while improving precision.

Final selected threshold: 0.07

## 📊 Model Performance (Test Set)

| Metric | Value |
|--------|-------|
| ROC-AUC | **0.843** |
| PR-AUC | **0.278** |
| Recall (Sensitivity) | **0.80** |
| Precision | **0.152** |
| F1 Score | **0.255** |
| Brier Score | **0.0406** |

### Interpretation

- The model correctly identifies **80% of stroke cases**.
- PR-AUC (~0.278) is approximately **5× higher than the random baseline (~0.05)**.
- ROC-AUC (~0.84) indicates strong class discrimination.
- The Brier score (0.0406) suggests well-calibrated probability estimates.
- Precision is modest due to severe class imbalance (~5% positives), which is expected in screening systems.

Overall, the system functions as a **safety-oriented screening model**, prioritizing high recall to reduce missed stroke cases.

---

## 🔍 Explainability (Glass Box Layer)

Because stacked ensembles are complex, a **surrogate glass-box model** was trained to approximate ensemble predictions.

Explainability techniques applied:

- **SHAP** for global and local feature attribution
- **LIME** for patient-level explanations

---

## 🧠 SHAP Global Feature Importance

Top contributing features:

1. **Age** (strongest predictor)
2. **Average Glucose Level**
3. Hypertension
4. Heart Disease
5. BMI

### Clinical Alignment

- Stroke incidence increases significantly with age.
- Hyperglycemia is a major vascular risk factor.
- Hypertension strongly correlates with cerebrovascular events.

The model’s reasoning aligns with established epidemiological evidence, reinforcing its interpretability and trustworthiness.

---

## 🧾 Example Interpretation

> Elevated age and high average glucose levels significantly increased predicted stroke probability, while absence of hypertension reduced risk. This behavior is consistent with known stroke pathophysiology.

---

## 🏥 Clinical Framing

This system is designed as:

- A decision-support tool  
- Not a diagnostic replacement  
- Optimized for safety (high recall)  
- Transparent through feature-level attribution  

---

## 📚 Key Takeaway

This project demonstrates that high-performing ensemble models can be transformed into **transparent, clinically interpretable systems**, preserving both predictive power and medical trust.
