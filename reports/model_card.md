# Glass Box Stroke Classifier — Model Card

## Intended Use
Binary risk classification (stroke: 0/1) to support clinical decision-making. Not for autonomous diagnosis.

## Data
Stroke Prediction Dataset (Kaggle). Target: `stroke`.

## Preprocessing
- Numeric: median imputation + standard scaling
- Categorical: most-frequent imputation + one-hot encoding
- ID column removed

## Model
Stacked ensemble (RF, ExtraTrees, HGB, SVM) with Logistic Regression meta-model.
Optional probability calibration (isotonic/sigmoid).

## Thresholding
Validation-selected threshold. Default strategy: maximize precision while achieving recall ≥ configured minimum.

## Explainability
- SHAP: interpretable surrogate trained to mimic the ensemble (glass-box)
- LIME: local explanation on the real ensemble

## Limitations
Highly imbalanced labels. PR-AUC, recall, and calibration are emphasized.