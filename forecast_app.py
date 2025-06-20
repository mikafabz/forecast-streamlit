import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

# 🧩 Configuração da página
st.set_page_config(page_title="Previsão BTC e GOOGL", layout="centered")

st.title("🔮 Previsão de Preços – BTC e GOOGL (14 dias)")

# 📂 Função para carregar dados
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df[['Close']]
    df = df.dropna()
    return df

# 📊 Função para preparar os dados para o modelo
def prepare_data(df, n_steps=30):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(n_steps, len(df_scaled) - 14):
        X.append(df_scaled[i - n_steps:i])
        y.append(df_scaled[i:i+14])  # Previsão dos 14 dias seguintes

    X = np.array(X)
    y = np.array(y)
    return X, y, scaler

# 🔁 Função para prever os próximos 14 dias
def predict_future(model, df, scaler, n_steps=30):
    last_sequence = df[-n_steps:]
    input_seq = scaler.transform(last_sequence)
    input_seq = input_seq.reshape(1, n_steps, 1)

    future_pred_scaled = model.predict(input_seq)
    future_pred = scaler.inverse_transform(future_pred_scaled[0])
    return future_pred

# 🧠 Carregar modelos
@st.cache_resource
def load_models():
    model_btc = load_model("model_btc_lstm.h5")
    model_googl = load_model("model_googl_lstm.h5")
    return model_btc, model_googl

# 🚀 Executar previsão e mostrar resultados
def run_forecast():
    model_btc, model_googl = load_models()

    # Carregar dados
    df_btc = load_data("BTC.csv")
    df_googl = load_data("GOOGL.csv")

    # Preparar dados
    _, _, scaler_btc = prepare_data(df_btc)
    _, _, scaler_googl = prepare_data(df_googl)

    # Prever próximos 14 dias
    forecast_btc = predict_future(model_btc, df_btc, scaler_btc)
    forecast_googl = predict_future(model_googl, df_googl, scaler_googl)

    # Criar datas futuras
    last_date = pd.to_datetime("2024-12-31")
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=14)

    # 📈 Plotar gráfico
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(future_dates, forecast_btc, label="BTC")
    ax.plot(future_dates, forecast_googl, label="GOOGL")
    ax.set_title("Previsão para os próximos 14 dias")
    ax.set_ylabel("Preço estimado (USD)")
    ax.legend()
    st.pyplot(fig)

    # 🧾 Tabela
    df_forecast = pd.DataFrame({
        "Data": future_dates,
        "BTC": forecast_btc,
        "GOOGL": forecast_googl
    })
    st.dataframe(df_forecast.set_index("Data").style.format("${:.2f}"))

# 📌 Rodar apenas uma vez
if st.button("📊 Gerar Previsão"):
    run_forecast()
else:
    st.info("Clique no botão acima para gerar a previsão para os próximos 14 dias.")
