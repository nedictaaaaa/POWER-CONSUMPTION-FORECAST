# DASHBOARD.py

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Page config
st.set_page_config(page_title="🔮 Power Consumption Forecasting", layout="centered")

# Constants
MODEL_PATH = "lstm_energy_forecast.h5"
SCALER_PATH = "scaler.save"
SEQUENCE_LENGTH = 24
REQUIRED_FEATURES = ["Temperature", "Humidity", "Total Power Consumption"]

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

# Load scaler
@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)

# Preprocess data
def preprocess_data(df, scaler):
    if not all(col in df.columns for col in REQUIRED_FEATURES):
        missing = list(set(REQUIRED_FEATURES) - set(df.columns))
        raise ValueError(f"Missing columns: {missing}")

    df = df[REQUIRED_FEATURES]
    scaled = scaler.transform(df)
    
    X = []
    for i in range(len(scaled) - SEQUENCE_LENGTH):
        X.append(scaled[i:i+SEQUENCE_LENGTH])
    return np.array(X)

# App UI
st.title("🔮 Power Consumption Forecasting")
st.markdown("Upload a CSV file with columns: `Temperature`, `Humidity`, `Total Power Consumption`")

uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        scaler = load_scaler()
        X = preprocess_data(df, scaler)

        model = load_model()
        predictions = model.predict(X)

        # Plot results
        st.subheader("📈 Forecasted Consumption")
        fig, ax = plt.subplots()
        ax.plot(predictions, label="Predicted Consumption", color='blue')
        ax.set_title("Predicted Power Consumption")
        ax.set_ylabel("Consumption")
        ax.set_xlabel("Hour")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("Please upload a CSV file to begin.")
