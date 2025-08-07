import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# --- Streamlit Page Config ---
st.set_page_config(page_title="🔮 Power Forecast", layout="centered")

# --- Constants ---
MODEL_PATH = "lstm_energy_forecast.h5"
SCALER_PATH = "scaler.save"
SEQUENCE_LENGTH = 24
REQUIRED_FEATURES = ["Temperature", "Humidity", "Total Power Consumption"]

# --- Load Model and Scaler ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)

# --- Preprocessing ---
def preprocess_data(df, scaler):
    df = df[REQUIRED_FEATURES]
    scaled = scaler.transform(df)
    X = []
    for i in range(len(scaled) - SEQUENCE_LENGTH):
        X.append(scaled[i:i+SEQUENCE_LENGTH])
    return np.array(X), scaled

# --- Forecasting Future Steps ---
def forecast_future(scaled_data, model, steps_ahead):
    input_seq = scaled_data[-SEQUENCE_LENGTH:]
    forecasts = []

    for _ in range(steps_ahead):
        X_input = input_seq.reshape(1, SEQUENCE_LENGTH, -1)
        next_pred = model.predict(X_input, verbose=0)[0][0]

        # Create next input row
        next_input = np.array([input_seq[-1][0], input_seq[-1][1], next_pred])
        input_seq = np.vstack([input_seq[1:], next_input])
        forecasts.append(next_pred)

    return forecasts

# --- App UI ---
st.title("🔮 Power Consumption Forecasting")
st.markdown("Upload a CSV file with **Temperature, Humidity, and Total Power Consumption**.")

uploaded_file = st.file_uploader("📁 Upload CSV", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        if not all(col in df.columns for col in REQUIRED_FEATURES):
            st.error(f"❌ Missing columns: {set(REQUIRED_FEATURES) - set(df.columns)}")
        elif len(df) < SEQUENCE_LENGTH:
            st.error(f"❌ Not enough rows. Need at least {SEQUENCE_LENGTH + 1} rows.")
        else:
            scaler = load_scaler()
            X, scaled_data = preprocess_data(df, scaler)
            model = load_model()

            # Predict for existing data
            preds = model.predict(X, verbose=0)

            # Forecast future
            st.subheader("🔧 Forecast Settings")
            steps = st.number_input("How many hours ahead to forecast?", min_value=1, max_value=168, value=24)
            future_preds = forecast_future(scaled_data, model, steps)

            # Inverse transform for readability
            dummy = np.zeros((steps, 3))
            dummy[:, 2] = future_preds
            future_consumption = scaler.inverse_transform(dummy)[:, 2]

            # --- Display results ---
            st.subheader("📊 Predicted Consumption (Historical + Forecast)")
            fig, ax = plt.subplots()
            ax.plot(df["Total Power Consumption"].values[-len(preds):], label="Actual")
            ax.plot(preds, label="Predicted (Historical)", linestyle="dashed")
            ax.plot(range(len(preds), len(preds)+steps), future_consumption, label="Forecast", color="green")
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("Power Consumption")
            ax.set_title("Power Consumption Forecast")
            ax.legend()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")
else:
    st.info("Upload a CSV file with the required structure to begin.")
