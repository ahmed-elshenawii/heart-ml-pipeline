import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
META_PATH = os.path.join(MODELS_DIR, "model_meta.json")

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title("❤️ Heart Disease Predictor")
st.write("Enter patient features to predict probability of heart disease using the trained model.")

@st.cache_resource
def load_model_and_meta():
    if not os.path.exists(BEST_MODEL_PATH) or not os.path.exists(META_PATH):
        st.error("Model is not trained yet. Please run the pipeline first.")
        st.stop()
    model = joblib.load(BEST_MODEL_PATH)
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    features = meta["feature_names"]
    return model, features, meta

model, features, meta = load_model_and_meta()

with st.form("pred_form"):
    inputs = {}
    for f in features:
        # Heuristics for numeric ranges
        if any(k in f.lower() for k in ["age"]):
            val = st.number_input(f, min_value=0.0, max_value=120.0, value=55.0, step=1.0)
        elif any(k in f.lower() for k in ["trestbps","chol","thalach"]):
            val = st.number_input(f, min_value=0.0, max_value=400.0, value=120.0, step=1.0)
        else:
            val = st.number_input(f, value=0.0, step=1.0)
        inputs[f] = val

    submitted = st.form_submit_button("Predict")

if submitted:
    X = pd.DataFrame([inputs])[features]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1][0]
        pred = int(proba >= 0.5)
    else:
        proba = float(model.decision_function(X))
        pred = int(proba >= 0.0)
    st.metric("Prediction (0=No Disease, 1=Disease)", pred)
    st.metric("Estimated probability", f"{proba:.3f}")
    st.success("Done.")
