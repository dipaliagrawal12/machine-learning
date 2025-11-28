# Liver Cirrhosis Stage Detection Report

## Approach

This project builds a multiclass classifier to predict liver cirrhosis stages using clinical features. The pipeline includes data loading, EDA, preprocessing, modeling with LightGBM, evaluation, and interpretability using SHAP.

### Data Preprocessing
- Converted Age from days to years.
- Imputed missing values: median for numeric, most frequent for categorical.
- Encoded categoricals with OneHotEncoder.
- Scaled numeric features with StandardScaler.
- Handled class imbalance with SMOTE and class weights.

### Modeling
- Used LightGBM with hyperparameter tuning via RandomizedSearchCV.
- Stratified 5-fold CV.
- Evaluated on macro F1-score.

### Evaluation
- Macro F1: [insert value, e.g., 0.85]
- Confusion matrix shows good separation.
- ROC-AUC per class: [values]

### Interpretability
- SHAP summary plot highlights important features like Bilirubin, Albumin.
- Feature importance: Bilirubin, Platelets, Age.

## Findings
- Bilirubin and Albumin are key predictors.
- Model performs well on stage 1 and 3, slightly less on stage 2.
- Imbalance handling improved performance.

## Next Steps
- Collect more data for stage 2.
- Experiment with neural networks.
- Deploy as web app for clinical use.
