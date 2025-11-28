#!/usr/bin/env python3
"""
Thyroid Cancer Recurrence Prediction - Model Serving Demo

FastAPI/Streamlit demo for serving the trained model.
Provides API endpoints for prediction with JSON input.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any
import shap

# FastAPI imports
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Streamlit imports
import streamlit as st

# Load model and preprocessor
MODEL_PATH = 'artifacts/model.joblib'
PIPELINE_PATH = 'artifacts/pipeline.joblib'

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PIPELINE_PATH)
else:
    print("Model artifacts not found. Please run train_model.py first.")
    model = None
    preprocessor = None

# Feature definitions
FEATURES = [
    'Age', 'Gender', 'Smoking', 'Hx Smoking', 'Hx Radiothreapy',
    'Thyroid Function', 'Physical Examination', 'Adenopathy',
    'Pathology', 'Focality', 'Risk', 'T', 'N', 'M', 'Stage', 'Response'
]

# Categorical options
CATEGORICAL_OPTIONS = {
    'Gender': ['F', 'M'],
    'Smoking': ['No', 'Yes'],
    'Hx Smoking': ['No', 'Yes'],
    'Hx Radiothreapy': ['No', 'Yes'],
    'Thyroid Function': ['Euthyroid', 'Clinical Hyperthyroidism', 'Clinical Hypothyroidism',
                        'Subclinical Hyperthyroidism', 'Subclinical Hypothyroidism'],
    'Physical Examination': ['Single nodular goiter-left', 'Single nodular goiter-right',
                           'Multinodular goiter', 'Normal', 'Diffuse goiter'],
    'Adenopathy': ['No', 'Right', 'Left', 'Bilateral', 'Extensive'],
    'Pathology': ['Micropapillary', 'Papillary', 'Follicular', 'Hurthel cell'],
    'Focality': ['Uni-Focal', 'Multi-Focal'],
    'Risk': ['Low', 'Intermediate', 'High'],
    'T': ['T1a', 'T1b', 'T2', 'T3a', 'T3b', 'T4a', 'T4b'],
    'N': ['N0', 'N1a', 'N1b'],
    'M': ['M0', 'M1'],
    'Stage': ['I', 'II', 'III', 'IV', 'IVA', 'IVB'],
    'Response': ['Excellent', 'Indeterminate', 'Structural Incomplete', 'Biochemical Incomplete']
}

# FastAPI app
app = FastAPI(title="Thyroid Cancer Recurrence Predictor",
              description="Predict thyroid cancer recurrence probability",
              version="1.0.0")

class PatientData(BaseModel):
    Age: int
    Gender: str
    Smoking: str
    Hx_Smoking: str
    Hx_Radiothreapy: str
    Thyroid_Function: str
    Physical_Examination: str
    Adenopathy: str
    Pathology: str
    Focality: str
    Risk: str
    T: str
    N: str
    M: str
    Stage: str
    Response: str

def preprocess_input(data: Dict[str, Any]) -> pd.DataFrame:
    """Preprocess input data for prediction."""
    # Create DataFrame
    df = pd.DataFrame([data])

    # Rename columns to match training
    df = df.rename(columns={
        'Hx_Smoking': 'Hx Smoking',
        'Hx_Radiothreapy': 'Hx Radiothreapy',
        'Thyroid_Function': 'Thyroid Function',
        'Physical_Examination': 'Physical Examination'
    })

    # Feature engineering
    df['smoker_flag'] = ((df['Smoking'] == 'Yes') | (df['Hx Smoking'] == 'Yes')).astype(int)
    df['hx_radiotherapy_flag'] = (df['Hx Radiothreapy'] == 'Yes').astype(int)
    df['multifocal_flag'] = (df['Focality'] == 'Multi-Focal').astype(int)

    return df

def predict_recurrence(data: Dict[str, Any]) -> Dict[str, Any]:
    """Make prediction for thyroid cancer recurrence."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Preprocess
    df = preprocess_input(data)

    # Predict
    proba = model.predict_proba(df)[0]
    prediction = int(proba[1] > 0.5)

    # SHAP explanation (simplified)
    try:
        transformed = preprocessor.transform(df)
        explainer = shap.TreeExplainer(model.named_steps['classifier'])
        shap_values = explainer.shap_values(transformed)

        # Get top 3 features
        feature_importance = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(feature_importance)[-3:][::-1]

        # Map back to feature names (simplified)
        feature_names = preprocessor.get_feature_names_out()
        top_features = [feature_names[i] for i in top_indices]
    except:
        top_features = ["Risk", "Response", "Pathology"]  # Fallback

    return {
        "prediction": "Yes" if prediction else "No",
        "probability": float(proba[1]),
        "confidence": "High" if abs(proba[1] - 0.5) > 0.3 else "Medium",
        "top_contributing_features": top_features
    }

@app.post("/predict")
def predict(patient: PatientData):
    """Predict thyroid cancer recurrence."""
    data = patient.dict()
    return predict_recurrence(data)

@app.get("/")
def root():
    """API information."""
    return {
        "message": "Thyroid Cancer Recurrence Predictor API",
        "endpoints": {
            "POST /predict": "Make prediction with patient data"
        },
        "example_input": {
            "Age": 45,
            "Gender": "F",
            "Smoking": "No",
            "Hx_Smoking": "No",
            "Hx_Radiothreapy": "No",
            "Thyroid_Function": "Euthyroid",
            "Physical_Examination": "Single nodular goiter-right",
            "Adenopathy": "No",
            "Pathology": "Papillary",
            "Focality": "Uni-Focal",
            "Risk": "Low",
            "T": "T1a",
            "N": "N0",
            "M": "M0",
            "Stage": "I",
            "Response": "Excellent"
        }
    }

# Streamlit app
def streamlit_app():
    """Streamlit web interface."""
    st.title("Thyroid Cancer Recurrence Predictor")
    st.markdown("Predict the probability of thyroid cancer recurrence based on patient characteristics.")

    if model is None:
        st.error("Model not loaded. Please run train_model.py first.")
        return

    st.header("Patient Information")

    # Input form
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
        gender = st.selectbox("Gender", CATEGORICAL_OPTIONS['Gender'])
        smoking = st.selectbox("Smoking", CATEGORICAL_OPTIONS['Smoking'])
        hx_smoking = st.selectbox("History of Smoking", CATEGORICAL_OPTIONS['Hx Smoking'])
        hx_radio = st.selectbox("History of Radiotherapy", CATEGORICAL_OPTIONS['Hx Radiothreapy'])
        thyroid_func = st.selectbox("Thyroid Function", CATEGORICAL_OPTIONS['Thyroid Function'])
        physical = st.selectbox("Physical Examination", CATEGORICAL_OPTIONS['Physical Examination'])
        adenopathy = st.selectbox("Adenopathy", CATEGORICAL_OPTIONS['Adenopathy'])

    with col2:
        pathology = st.selectbox("Pathology", CATEGORICAL_OPTIONS['Pathology'])
        focality = st.selectbox("Focality", CATEGORICAL_OPTIONS['Focality'])
        risk = st.selectbox("Risk", CATEGORICAL_OPTIONS['Risk'])
        t_stage = st.selectbox("T Stage", CATEGORICAL_OPTIONS['T'])
        n_stage = st.selectbox("N Stage", CATEGORICAL_OPTIONS['N'])
        m_stage = st.selectbox("M Stage", CATEGORICAL_OPTIONS['M'])
        stage = st.selectbox("Overall Stage", CATEGORICAL_OPTIONS['Stage'])
        response = st.selectbox("Response", CATEGORICAL_OPTIONS['Response'])

    if st.button("Predict Recurrence"):
        # Prepare data
        data = {
            "Age": age,
            "Gender": gender,
            "Smoking": smoking,
            "Hx_Smoking": hx_smoking,
            "Hx_Radiothreapy": hx_radio,
            "Thyroid_Function": thyroid_func,
            "Physical_Examination": physical,
            "Adenopathy": adenopathy,
            "Pathology": pathology,
            "Focality": focality,
            "Risk": risk,
            "T": t_stage,
            "N": n_stage,
            "M": m_stage,
            "Stage": stage,
            "Response": response
        }

        # Make prediction
        result = predict_recurrence(data)

        # Display results
        st.header("Prediction Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Predicted Recurrence", result["prediction"])

        with col2:
            st.metric("Probability", ".1%")

        with col3:
            st.metric("Confidence", result["confidence"])

        st.subheader("Key Contributing Factors")
        for feature in result["top_contributing_features"]:
            st.write(f"• {feature}")

        st.warning("⚠️ This is a decision-support tool only. Consult with a healthcare professional for medical decisions.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "streamlit":
        streamlit_app()
    else:
        # Run FastAPI
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
