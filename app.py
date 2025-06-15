import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Konfigurasi halaman
st.set_page_config(page_title="📊 Prediksi Komoditas Aceh", layout="wide")

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("data/data_perhari.csv", parse_dates=["Tanggal"])

@st.cache_data
def load_metrics():
    with open("metrics/evaluasi_model.json", "r") as f:
        return json.load(f)

@st.cache_resource
def load_models():
    return (
        load_model("models/model_lstm_komoditas.keras"),
        load_model("models/model_GRU_komoditas.keras")
    )

# Sidebar
st.sidebar.title("⚙️ Pengaturan")
komoditas_list = ["Beras Medium", "Bawang Merah", "Cabai Merah Keriting"]
komoditas = st.sidebar.selectbox("🧺 Pilih Komoditas", komoditas_list)

df = load_data()
metrics = load_metrics()
model_lstm, model_gru = load_models()

tanggal_pilihan = st.sidebar.selectbox("📅 Pilih Tanggal", df["Tanggal"].dt.strftime("%Y-%m-%d"))

# Fungsi Prediksi
def get_prediction(model, series):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))
    window = scaled[-10:].reshape(1, 10, 1)
    pred = model.predict(window, verbose=0)
    return scaler.inverse_transform(pred)[0][0]

# Buat Prediksi jika belum ada
harga_series = df[komoditas]
if f"Prediksi_LSTM_{komoditas}" not in df.columns:
    df[f"Prediksi_LSTM_{komoditas}"] = np.nan
    df[f"Prediksi_LSTM-GRU_{komoditas}"] = np.nan
    for i in range(10, len(df)):
        window_series = harga_series.iloc[i-10:i]
        df.loc[df.index[i], f"Prediksi_LSTM_{komoditas}"] = get_prediction(model_lstm, window_series)
        df.loc[df.index[i], f"Prediksi_LSTM-GRU_{komoditas}"] = get_prediction(model_gru, window_series)

# Judul Halaman
st.markdown(f"<h2 style='text-align: center; color: #4B8BBE;'>📈 Prediksi Harga Komoditas: {komoditas}</h2>", unsafe_allow_html=True)
st.markdown("---")

# Grafik Harga
st.subheader("📊 Visualisasi Harga Aktual vs Prediksi")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df["Tanggal"], harga_series, label="Aktual", color="#0077B6", linewidth=2)
ax.plot(df["Tanggal"], df[f"Prediksi_LSTM_{komoditas}"], label="LSTM", linestyle="--", color="#90E0EF")
ax.plot(df["Tanggal"], df[f"Prediksi_LSTM-GRU_{komoditas}"], label="LSTM-GRU", linestyle="--", color="#FFB703")
ax.set_xlabel("Tanggal")
ax.set_ylabel("Harga (Rp)")
ax.tick_params(axis='x', rotation=45)
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Analisis Peluang
st.subheader("🚚 Peluang Usaha Distribusi Komoditas")
def peluang_usaha_distribusi(prices):
    perubahan = prices.pct_change().dropna()
    rata2 = perubahan.mean()
    if rata2 > 0.02:
        return "📈 Harga cenderung naik ➜ Peluang distribusi **tinggi**"
    elif rata2 < -0.02:
        return "📉 Harga cenderung turun ➜ Distribusi **berisiko**"
    else:
        return "📊 Harga stabil ➜ Peluang distribusi **moderat**"
st.success(peluang_usaha_distribusi(harga_series))

# Prediksi Tanggal
st.subheader("📅 Prediksi Harga pada Tanggal Tertentu")
tanggal_dt = pd.to_datetime(tanggal_pilihan)
baris = df[df["Tanggal"] == tanggal_dt]

if not baris.empty:
    harga_aktual = baris[komoditas].values[0]
    pred_lstm = baris[f"Prediksi_LSTM_{komoditas}"].values[0]
    pred_gru = baris[f"Prediksi_LSTM-GRU_{komoditas}"].values[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("🎯 Harga Aktual", f"Rp {harga_aktual:,.0f}")
    col2.metric("🔵 Prediksi LSTM", f"Rp {pred_lstm:,.0f}")
    col3.metric("🟡 Prediksi LSTM-GRU", f"Rp {pred_gru:,.0f}")
else:
    st.warning("Tanggal ini belum memiliki data prediksi.")

# Metrik Evaluasi
st.subheader("📐 Evaluasi Kinerja Model")
metrik_lstm = metrics[komoditas]["LSTM"]
metrik_gru = metrics[komoditas]["LSTM-GRU"]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📉 MAPE - LSTM (%)", f"{metrik_lstm['MAPE']:.2f}")
    st.metric("📉 MAPE - GRU (%)", f"{metrik_gru['MAPE']:.2f}")
with col2:
    st.metric("✅ Accuracy - LSTM", f"{metrik_lstm['Accuracy']:.2f}%")
    st.metric("✅ Accuracy - GRU", f"{metrik_gru['Accuracy']:.2f}%")
with col3:
    st.metric("📏 RMSE - LSTM", f"{metrik_lstm['RMSE']:.2f}")
    st.metric("📏 RMSE - GRU", f"{metrik_gru['RMSE']:.2f}")

col4, col5 = st.columns(2)
with col4:
    st.metric("📌 MSE - LSTM", f"{metrik_lstm['MSE']:.2f}")
    st.metric("📌 MSE - GRU", f"{metrik_gru['MSE']:.2f}")
with col5:
    st.metric("📌 MAE - LSTM", f"{metrik_lstm['MAE']:.2f}")
    st.metric("📌 MAE - GRU", f"{metrik_gru['MAE']:.2f}")

# Footer
st.markdown("<hr style='margin-top: 2rem;'>", unsafe_allow_html=True)
st.markdown("<center><small style='color:gray;'>Made with ❤️ by <strong>Era Syafina</strong></small></center>", unsafe_allow_html=True)
