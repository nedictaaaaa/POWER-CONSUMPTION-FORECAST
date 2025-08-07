import streamlit as st
st.set_page_config(page_title="⚡ Power Forecast", layout="centered")

import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# --- SETTINGS ---
SEQUENCE_LENGTH = 24
MODEL_PATH = "lstm_energy_forecast.h5"
SCALER_PATH = "scaler.save"
REQUIRED_FEATURES = ["Temperature", "Humidity", "Total Power Consumption"]

# --- Load model and scaler ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)

model = load_model()
scaler = load_scaler()

# --- Prepare input sequence ---
def prepare_input(df):
    df = df.sort_values("Datetime")
    used = [f for f in REQUIRED_FEATURES if f in df.columns]
    missing = list(set(REQUIRED_FEATURES) - set(used))
    if missing:
        st.warning(f"Missing features: {', '.join(missing)}")
    if len(df) < SEQUENCE_LENGTH:
        st.error("Insufficient data. Need at least 24 rows.")
        return None
    df_scaled = scaler.transform(df[REQUIRED_FEATURES])
    input_seq = df_scaled[-SEQUENCE_LENGTH:]
    input_seq = input_seq.reshape(1, SEQUENCE_LENGTH, len(REQUIRED_FEATURES))
    return input_seq

# --- Recursive Forecast Function (fixed) ---
def recursive_forecast(df, model, scaler, steps=7):
    df_scaled = scaler.transform(df[REQUIRED_FEATURES])
    sequence = df_scaled[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, len(REQUIRED_FEATURES))
    forecasts = []

    for _ in range(steps):
        pred_scaled = model.predict(sequence)[0][0]
        next_input = np.array([[[0, 0, pred_scaled]]])  # shape (1, 1, 3)
        sequence = np.concatenate((sequence[:, 1:, :], next_input), axis=1)
        full_inverse = scaler.inverse_transform([[0, 0, pred_scaled]])
        forecasts.append(full_inverse[0][2])

    return forecasts

# --- Plot inputs ---
def plot_inputs(df):
    st.subheader("📈 Input Trends")
    fig, ax = plt.subplots()
    for col in REQUIRED_FEATURES:
        ax.plot(df["Datetime"].tail(SEQUENCE_LENGTH), df[col].tail(SEQUENCE_LENGTH), label=col)
    ax.legend()
    st.pyplot(fig)

# --- Plot forecasts ---
def plot_forecasts(forecasts):
    fig, ax = plt.subplots()
    ax.plot(range(1, len(forecasts) + 1), forecasts, marker='o')
    ax.set_title("🔮 Forecasted Power Consumption")
    ax.set_xlabel("Steps Ahead")
    ax.set_ylabel("Power (units)")
    st.pyplot(fig)

# --- Streamlit Layout ---
st.title("🔮 Power Consumption Forecasting")
uploaded_file = st.file_uploader("📤 Upload cleaned Excel or CSV file", type=["xlsx", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    st.write("### Preview of Uploaded Data")
    st.dataframe(df.head())

    input_seq = prepare_input(df)
    if input_seq is not None:
        st.subheader("📅 Select Forecast Horizon")
        forecast_steps = st.slider("How many steps ahead to forecast?", min_value=1, max_value=24, value=7)

        try:
            forecasts = recursive_forecast(df, model, scaler, steps=forecast_steps)
            st.success("✅ Forecast completed!")

            st.markdown("### 🔢 Forecasted Power Consumption Values")
            for i, val in enumerate(forecasts, 1):
                st.write(f"Step {i}: **{val:.2f} units**")

            plot_inputs(df)
            plot_forecasts(forecasts)

        except Exception as e:
            st.error(f"Forecasting failed: {str(e)}")

import streamlit as st
st.set_page_config(page_title="⚡ Power Forecast", layout="centered")

import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# --- SETTINGS ---
SEQUENCE_LENGTH = 24
MODEL_PATH = "lstm_energy_forecast.keras"
SCALER_PATH = "scaler.save"
REQUIRED_FEATURES = ["Temperature", "Humidity", "Total Power Consumption"]

# --- Load model and scaler ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)

model = load_model()
scaler = load_scaler()

# --- Prepare input sequence ---
def prepare_input(df):
    df = df.sort_values("Datetime")
    used = [f for f in REQUIRED_FEATURES if f in df.columns]
    missing = list(set(REQUIRED_FEATURES) - set(used))
    if missing:
        st.warning(f"Missing features: {', '.join(missing)}")
    if len(df) < SEQUENCE_LENGTH:
        st.error("Insufficient data. Need at least 24 rows.")
        return None
    df_scaled = scaler.transform(df[REQUIRED_FEATURES])
    input_seq = df_scaled[-SEQUENCE_LENGTH:]
    input_seq = input_seq.reshape(1, SEQUENCE_LENGTH, len(REQUIRED_FEATURES))
    return input_seq

# --- Recursive Forecast Function (fixed) ---
def recursive_forecast(df, model, scaler, steps=7):
    df_scaled = scaler.transform(df[REQUIRED_FEATURES])
    sequence = df_scaled[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, len(REQUIRED_FEATURES))
    forecasts = []

    for _ in range(steps):
        pred_scaled = model.predict(sequence)[0][0]
        next_input = np.array([[[0, 0, pred_scaled]]])  # shape (1, 1, 3)
        sequence = np.concatenate((sequence[:, 1:, :], next_input), axis=1)
        full_inverse = scaler.inverse_transform([[0, 0, pred_scaled]])
        forecasts.append(full_inverse[0][2])

    return forecasts

# --- Plot inputs ---
def plot_inputs(df):
    st.subheader("📈 Input Trends")
    fig, ax = plt.subplots()
    for col in REQUIRED_FEATURES:
        ax.plot(df["Datetime"].tail(SEQUENCE_LENGTH), df[col].tail(SEQUENCE_LENGTH), label=col)
    ax.legend()
    st.pyplot(fig)

# --- Plot forecasts ---
def plot_forecasts(forecasts):
    fig, ax = plt.subplots()
    ax.plot(range(1, len(forecasts) + 1), forecasts, marker='o')
    ax.set_title("🔮 Forecasted Power Consumption")
    ax.set_xlabel("Steps Ahead")
    ax.set_ylabel("Power (units)")
    st.pyplot(fig)

# --- Streamlit Layout ---
st.title("🔮 Power Consumption Forecasting")
uploaded_file = st.file_uploader("📤 Upload cleaned Excel or CSV file", type=["xlsx", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    st.write("### Preview of Uploaded Data")
    st.dataframe(df.head())

    input_seq = prepare_input(df)
    if input_seq is not None:
        st.subheader("📅 Select Forecast Horizon")
        forecast_steps = st.slider("How many steps ahead to forecast?", min_value=1, max_value=24, value=7)

        try:
            forecasts = recursive_forecast(df, model, scaler, steps=forecast_steps)
            st.success("✅ Forecast completed!")

            st.markdown("### 🔢 Forecasted Power Consumption Values")
            for i, val in enumerate(forecasts, 1):
                st.write(f"Step {i}: **{val:.2f} units**")

            plot_inputs(df)
            plot_forecasts(forecasts)

        except Exception as e:
            st.error(f"Forecasting failed: {str(e)}")

