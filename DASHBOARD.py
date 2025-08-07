# 📦 Imports
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt

# 🔧 Configuration
st.set_page_config(page_title="⚡ Power Forecast", layout="centered")
st.title("🔮 Power Consumption Forecasting")

SEQUENCE_LENGTH = 24
MODEL_PATH = "lstm_energy_forecast.h5"
SCALER_PATH = "scaler (1).save"
REQUIRED_FEATURES = ["Temperature", "Humidity", "Total Power Consumption"]

# 📥 Load Model and Scaler
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)

model = load_model()
scaler = load_scaler()

# 📤 File Upload
uploaded_file = st.file_uploader("Upload your power consumption CSV file", type=["csv"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # 🧼 Preprocess
        if not all(col in df.columns for col in REQUIRED_FEATURES):
            st.error(f"Missing required columns: {REQUIRED_FEATURES}")
        else:
            df = df[REQUIRED_FEATURES]
            scaled = scaler.transform(df)

            X = []
            for i in range(len(scaled) - SEQUENCE_LENGTH):
                X.append(scaled[i:i+SEQUENCE_LENGTH])
            X = np.array(X)

            # 🧠 Predict
            y_pred = model.predict(X)
            st.success("✅ Forecast completed!")

            # 🔢 Display last few predictions
            predictions = pd.DataFrame(y_pred, columns=["Predicted Consumption"])
            st.write("### 🔍 Predicted Consumption (last 10 rows)")
            st.dataframe(predictions.tail(10))

            # 📈 Plot
            st.write("### 📉 Forecast Plot")
            fig, ax = plt.subplots()
            ax.plot(predictions, label="Predicted")
            ax.set_title("Forecasted Power Consumption")
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("Consumption")
            ax.legend()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")
