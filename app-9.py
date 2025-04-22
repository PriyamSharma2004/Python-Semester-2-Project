import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model and features
try:
    model = joblib.load('model.joblib')
    feature_names = joblib.load('features.joblib')
except FileNotFoundError:
    st.error("Model files not found! Please add:")
    st.code("""
    - model.joblib
    - features.joblib
    """)
    st.stop()

st.title(" Wine Quality Class Predictor")

# Create input form with REAL data ranges (example for wine)
FEATURE_RANGES = {
    'fixed acidity': (4.6, 15.9),
    'volatile acidity': (0.08, 1.58),
    'citric acid': (0.0, 1.66),
    'residual sugar': (0.9, 65.8),
    'chlorides': (0.01, 0.34),
    'free sulfur dioxide': (1.0, 289.0),
    'total sulfur dioxide': (6.0, 440.0),
    'density': (0.99, 1.04),
    'pH': (2.72, 3.82),
    'sulphates': (0.22, 1.08),
    'alcohol': (8.0, 14.9)
}

inputs = {}
for feature in feature_names:
    min_val, max_val = FEATURE_RANGES.get(feature, (0.0, 15.0))  # Default fallback
    inputs[feature] = st.slider(
        label=feature.replace('_', ' ').title(),
        min_value=float(min_val),
        max_value=float(max_val),
        value=float((min_val + max_val)/2)  # Midpoint
    )

if st.button("Predict Quality Class"):
    input_df = pd.DataFrame([inputs])[feature_names]  # Maintain feature order

    try:
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        st.success(f"Predicted Class: **{prediction}**")

        st.subheader("Probability Breakdown")
        for class_name, prob in zip(model.classes_, proba):
            st.progress(float(prob), text=f"{class_name}: {prob:.1%}")

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
