import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# App config
st.set_page_config(page_title="🔮 Power Consumption Forecast", layout="centered")

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

# Prepare sequences for prediction
def prepare_sequence(data, steps):
    sequence = data[-SEQUENCE_LENGTH:]  # Last 24 hours
    predictions = []

    for _ in range(steps):
        input_seq = sequence.reshape((1, SEQUENCE_LENGTH, -1))
        next_pred = model.predict(input_seq, verbose=0)[0][0]
        last_known = sequence[-1].copy()
        last_known[2] = next_pred  # Replace Consumption with prediction
        sequence = np.vstack([sequence[1:], last_known])
        predictions.append(next_pred)

    return predictions

# Interface
st.title("🔮 Power Consumption Forecasting")
st.markdown("Upload a CSV with `Temperature`, `Humidity`, `Total Power Consumption` to forecast future consumption.")

uploaded_file = st.file_uploader("📁 Upload CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Check required columns
        if not all(col in df.columns for col in REQUIRED_FEATURES):
            missing = list(set(REQUIRED_FEATURES) - set(df.columns))
            st.error(f"Missing columns: {missing}")
        elif len(df) < SEQUENCE_LENGTH:
            st.error(f"At least {SEQUENCE_LENGTH} rows are required for forecasting.")
        else:
            scaler = load_scaler()
            scaled_data = scaler.transform(df[REQUIRED_FEATURES])
            model = load_model()

            steps = st.slider("🔢 How many hours to forecast?", min_value=1, max_value=48, value=12)
            forecast = prepare_sequence(scaled_data, steps)
            
            # Inverse transform just the forecasted consumption
            dummy = np.zeros((steps, 3))  # 3 features
            dummy[:, 2] = forecast
            inv = scaler.inverse_transform(dummy)
            actual_forecast = inv[:, 2]

            # Show predictions
            st.subheader("📊 Forecasted Power Consumption")
            st.line_chart(actual_forecast)

            # Table
            forecast_df = pd.DataFrame({
                "Hour Ahead": np.arange(1, steps+1),
                "Predicted Consumption": actual_forecast
            })
            st.dataframe(forecast_df)

    except Exception as e:
        st.error(f"❌ Error: {e}")

else:
    st.info("Please upload a valid CSV to begin.")
