# Liver Cirrhosis Stage Detection

This project implements a machine learning pipeline to predict liver cirrhosis stages (1, 2, or 3) from clinical data.

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Ensure the data file `liver_cirrhosis.csv` is in the same directory.

## Usage

### Run the training script:
```
python train_model.py --data-path liver_cirrhosis.csv --out-dir ./artifacts --model-type lightgbm --seed 42 --cv-folds 5
```

### Run the notebook:
Open `liver_stage_pipeline.ipynb` in Jupyter and execute all cells.

## Outputs

- `artifacts/model.joblib`: Trained model
- `artifacts/pipeline.joblib`: Preprocessing pipeline
- `artifacts/confusion_matrix.png`: Confusion matrix plot
- `artifacts/classification_report.json`: Classification metrics
- `artifacts/roc_curves.png`: ROC curves
- `artifacts/shap_summary.png`: SHAP summary plot

## Report

See `report.md` for a summary of the approach, findings, and results.
