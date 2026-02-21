🧠 The “Glass Box” Stroke Healthcare Classifier

Engineered a transparent diagnostic model using SHAP to provide feature-level clinical justifications, bridging the gap between black-box AI and medical trust.

📌 Project Overview

This project develops a clinically calibrated stacked ensemble model for stroke risk prediction using the Stroke Prediction Dataset.

Unlike traditional black-box systems, this model integrates:

Strong predictive performance

Probability calibration

Clinically tuned threshold selection

SHAP-based global and local explanations

LIME patient-level explanations

The goal is to combine accuracy + transparency, enabling trustworthy decision support in healthcare environments.

🗂 Dataset

Source: Kaggle Stroke Prediction Dataset
Target variable: stroke (0 = No Stroke, 1 = Stroke)

Key Features

Age

Hypertension

Heart disease

Average glucose level

BMI

Smoking status

Work type

Residence type

Gender

The dataset is highly imbalanced (~5% stroke cases), reflecting real-world screening scenarios.

🏗 Model Architecture
🔹 Preprocessing

Median imputation (numeric features)

Most frequent imputation (categorical features)

Standard scaling (numeric features)

One-hot encoding (categorical features)

🔹 Stacked Ensemble

Base learners:

Random Forest

Extra Trees

HistGradientBoosting

SVM (RBF kernel)

Meta-learner:

Logistic Regression

🔹 Probability Calibration

Isotonic calibration applied for improved clinical probability reliability.

🎯 Threshold Strategy

Instead of using the default 0.5 threshold, we applied a recall-constrained strategy:

strategy: recall_at_least
min_recall: 0.75

This prioritizes identifying stroke cases while improving precision.

Final selected threshold: 0.07

📊 Model Performance (Test Set)
Metric	Value
ROC-AUC	0.843
PR-AUC	0.278
Recall (Sensitivity)	0.80
Precision	0.152
F1 Score	0.255
Brier Score	0.0406
Interpretation

The model correctly identifies 80% of stroke cases

PR-AUC is ~5× better than random baseline (~0.05)

Probability calibration is strong (low Brier score)

Precision is modest due to severe class imbalance (expected in screening systems)

This performance reflects a clinically reasonable screening model.

🔍 Explainability (Glass Box Layer)

Because stacked ensembles are complex, we implemented a surrogate glass-box model trained to mimic ensemble predictions.

We then applied:

SHAP (global + local explanations)

LIME (patient-level explanations)

🧠 SHAP Global Feature Importance

Top contributing features:

Age (Strongest predictor)

Average Glucose Level

Hypertension

Heart Disease

BMI

Clinical Alignment

Stroke incidence increases significantly with age

Hyperglycemia is a major vascular risk factor

Hypertension strongly correlates with cerebrovascular events

The model’s reasoning aligns with established epidemiological evidence, increasing trustworthiness.

🧾 Example Interpretation

Elevated age and high average glucose levels significantly increased predicted stroke probability, while absence of hypertension reduced risk. This aligns with clinical understanding of stroke pathophysiology.

🏥 Clinical Framing

This system is designed as:

A decision-support tool

Not a diagnostic replacement

Optimized for safety (high recall)

Transparent through feature-level attribution

📂 Project Structure
glass-box-stroke/
├─ configs/
├─ data/raw/
├─ models/
├─ reports/
└─ src/
🚀 How to Run
1️⃣ Install
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
2️⃣ Train
python -m src.train --config configs/default.yaml
3️⃣ Generate Explanations
python -m src.explain --config configs/default.yaml
📈 Why This Project Matters

Most healthcare ML systems suffer from:

Opaque predictions

Poor calibration

Arbitrary thresholds

No interpretability layer

This project addresses those gaps by combining:

Stacked ensemble performance

Calibration

Clinically tuned thresholding

SHAP glass-box explanations

📚 Key Takeaway

This project demonstrates that high-performing ensemble models can be transformed into transparent, clinically interpretable systems, preserving both predictive power and medical trust.