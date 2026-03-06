import streamlit as st
import pandas as pd
import numpy as np
import mlflow.pyfunc
import yfinance as yf
from datetime import datetime, timedelta
import os

os.environ['MLFLOW_TRACKING_USERNAME'] = 'admin'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'mlflowadmin123'
# ==========================
# MLflow configuration
# ==========================
MLFLOW_TRACKING_URI = "https://5000-firebase-python-mlops-01-1771598143149.cluster-iktsryn7xnhpexlu6255bftka4.cloudworkstations.dev/"  # your MLflow server
MODEL_NAME = "IhsgRegressor"
MODEL_STAGE = "1"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

@st.cache_resource
def load_model():
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    model = mlflow.pyfunc.load_model(model_uri)
    return model

model_remote = load_model()

# ==========================
# Fetch latest IHSG data
# ==========================
st.title("📈 IHSG Prediction App (Live Data)")

st.markdown("""
Aplikasi ini mengambil **data IHSG 60 hari terakhir** dari [Yahoo Finance](https://finance.yahoo.com/)
dan menggunakan **30 hari terakhir** sebagai input model untuk memprediksi nilai IHSG hari berikutnya.
""")

# Fetch data
end_date = datetime.now()
start_date = end_date - timedelta(days=90)  # buffer for weekends/holidays
ticker = "^JKSE"

with st.spinner("📡 Mengambil data IHSG dari Yahoo Finance..."):
    data = yf.download(ticker, start=start_date, end=end_date)
    latest_data = data.tail(60)

st.subheader("📊 Data IHSG (60 Hari Terakhir)")
st.dataframe(data.tail(10))

# ==========================
# Prepare last 30 days as input
# ==========================
if len(latest_data) >= 30:
    st.line_chart(latest_data['Close'].tail(60))
    current_input_latest = latest_data['Close'].tail(30).values.reshape(1, -1)

    if st.button("🔮 Prediksi IHSG 15 Hari Ke Depan"):
        predicted_prices_latest = []
        current_predict_input = current_input_latest.copy()

        # Predict the next 15 days
        for _ in range(15):
            input_df = pd.DataFrame(current_predict_input, columns=[f"f{i}" for i in range(current_predict_input.shape[1])])
            next_price = model_remote.predict(input_df)[0]
            predicted_prices_latest.append(next_price)

            # Shift window and add the new predicted value
            current_predict_input = np.roll(current_predict_input, -1)
            current_predict_input[0, -1] = next_price

        # Generate future dates
        last_date = latest_data.index[-1]
        predicted_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=15, freq='D')

        # Combine results
        predicted_results_df = pd.DataFrame({
            'Date': predicted_dates,
            'Predicted Close Price': predicted_prices_latest
        })

        # Show prediction table
        st.subheader("📅 Hasil Prediksi 15 Hari Ke Depan")
        st.dataframe(predicted_results_df)

        # Combine historical and forecast for chart
        full_chart_df = pd.concat([
            latest_data[['Close']].rename(columns={'Close': 'Price'}),
            predicted_results_df.set_index('Date').rename(columns={'Predicted Close Price': 'Price'})
        ])

        st.subheader("📈 Grafik IHSG (Riil + Prediksi)")
        st.line_chart(full_chart_df)
else:
    st.warning("Data tidak cukup untuk membuat 30 lag features.")