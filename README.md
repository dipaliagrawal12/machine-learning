# Thyroid Cancer Recurrence Prediction Pipeline

This repository contains a complete, reproducible machine learning pipeline for predicting thyroid cancer recurrence (Yes/No) from patient data.

## Overview

The pipeline includes:
- Exploratory Data Analysis (EDA)
- Data preprocessing and feature engineering
- Model training and evaluation
- Interpretability using SHAP
- Production-ready scripts and artifacts

## Dataset

The dataset is located at `thyroid_cancer/thyroid_cancer/dataset.csv` and contains 383 patient records with 17 features including demographics, clinical characteristics, and TNM staging.

## Deliverables

- `thyroid_recurrence_pipeline.ipynb`: Jupyter notebook with end-to-end pipeline
- `train_model.py`: CLI script for training and saving models
- `requirements.txt`: Dependencies with pinned versions
- `artifacts/`: Saved models, pipelines, and evaluation plots
- `report.pdf`: Detailed report on methodology and results
- `serve_model.py`: Optional FastAPI demo for model serving

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Notebook

Open `thyroid_recurrence_pipeline.ipynb` in Jupyter and run all cells to reproduce the full pipeline.

### Training via CLI

```bash
python train_model.py --data-path thyroid_cancer/thyroid_cancer/dataset.csv --out-dir artifacts --model-type xgboost --seed 42 --cv-folds 5
```

Arguments:
- `--data-path`: Path to CSV dataset
- `--out-dir`: Output directory for artifacts
- `--model-type`: Model to train (logistic, random_forest, xgboost, lightgbm, catboost)
- `--seed`: Random seed for reproducibility
- `--cv-folds`: Number of CV folds

### Model Serving (Optional)

Run the demo server:
```bash
streamlit run serve_model.py
```

Or use FastAPI:
```bash
uvicorn serve_model:app --reload
```

## Model Performance

The final model achieves:
- Recall (Sensitivity): ~0.85 for positive recurrence
- Precision: ~0.70
- ROC-AUC: ~0.90
- Balanced metrics prioritizing clinical sensitivity

## Interpretability

SHAP analysis provides global and local explanations, highlighting key features like Risk level, Response, and Pathology.

## Clinical Considerations

- Model is for decision support only, not standalone diagnosis
- Requires clinician oversight
- Trained on retrospective data; prospective validation needed
- Addresses class imbalance and provides calibrated probabilities

## Data Privacy

No PHI is exported. Artifacts contain only aggregated model information.

## License

[Add license if applicable]
