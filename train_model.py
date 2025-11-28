import argparse
import os
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
import shap

# Set random seed
def set_seed(seed):
    np.random.seed(seed)
    import random
    random.seed(seed)

def load_data(data_path):
    df = pd.read_csv(data_path)
    # Convert Age to years if in days
    if 'Age' in df.columns:
        df['Age_years'] = df['Age'] / 365.25
        df.drop('Age', axis=1, inplace=True)
    return df

def preprocess_data(df):
    # Define features
    numeric_features = ['Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', 'Age_years']
    categorical_features = ['Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 'Drug', 'Status']

    # Impute missing values
    numeric_imputer = SimpleImputer(strategy='median')
    categorical_imputer = SimpleImputer(strategy='most_frequent')

    # Encode categoricals
    encoder = OneHotEncoder(drop='first', sparse=False)

    # Scale numeric
    scaler = StandardScaler()

    # Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[('imputer', numeric_imputer), ('scaler', scaler)]), numeric_features),
            ('cat', Pipeline(steps=[('imputer', categorical_imputer), ('encoder', encoder)]), categorical_features)
        ])

    return preprocessor, numeric_features + categorical_features

def train_model(X, y, model_type, cv_folds, seed):
    set_seed(seed)

    # Preprocessor
    preprocessor, feature_names = preprocess_data(pd.DataFrame(X, columns=feature_names))  # Assuming X is df

    # Model
    if model_type == 'lightgbm':
        model = lgb.LGBMClassifier(class_weight='balanced', random_state=seed)
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
    elif model_type == 'xgboost':
        model = xgb.XGBClassifier(scale_pos_weight=1, random_state=seed)  # Adjust for multiclass
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
    else:
        model = RandomForestClassifier(class_weight='balanced', random_state=seed)
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }

    # Full pipeline
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=seed)),
        ('model', model)
    ])

    # CV
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    # Search
    search = RandomizedSearchCV(pipeline, param_dist, n_iter=50, cv=cv, scoring='f1_macro', random_state=seed, n_jobs=-1)
    search.fit(X, y)

    return search.best_estimator_, search.best_params_

def evaluate_model(model, X_test, y_test, out_dir):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'))
    plt.close()

    # ROC curves
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(len(np.unique(y_test))):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    for i in range(len(np.unique(y_test))):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'roc_curves.png'))
    plt.close()

    # Save report
    with open(os.path.join(out_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f)

    return macro_f1

def interpret_model(model, X_train, out_dir):
    # SHAP
    explainer = shap.TreeExplainer(model.named_steps['model'])
    shap_values = explainer.shap_values(X_train)

    # Summary plot
    shap.summary_plot(shap_values, X_train, show=False)
    plt.savefig(os.path.join(out_dir, 'shap_summary.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train liver cirrhosis stage classifier')
    parser.add_argument('--data-path', type=str, required=True, help='Path to CSV data')
    parser.add_argument('--out-dir', type=str, default='./artifacts', help='Output directory')
    parser.add_argument('--model-type', type=str, default='lightgbm', choices=['lightgbm', 'xgboost', 'rf'], help='Model type')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--cv-folds', type=int, default=5, help='Number of CV folds')

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    # Load data
    df = load_data(args.data_path)
    X = df.drop('Stage', axis=1)
    y = df['Stage']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=args.seed)

    # Train
    model, best_params = train_model(X_train, y_train, args.model_type, args.cv_folds, args.seed)

    # Evaluate
    macro_f1 = evaluate_model(model, X_test, y_test, args.out_dir)

    # Interpret
    interpret_model(model, X_train, args.out_dir)

    # Save model and pipeline
    joblib.dump(model, os.path.join(args.out_dir, 'model.joblib'))
    joblib.dump(model.named_steps['preprocessor'], os.path.join(args.out_dir, 'pipeline.joblib'))

    print(f"Training complete. Macro F1: {macro_f1:.4f}")
    print(f"Best params: {best_params}")

if __name__ == '__main__':
    main()
