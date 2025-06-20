import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from ta import add_all_ta_features
import traceback

st.set_page_config(page_title="Forecast BTC & GOOGL", layout="centered")
st.title("📈 Previsão Futura de Preços – BTC e GOOGL")
st.markdown("Este app mostra a **previsão dos próximos 14 dias** com base em dados históricos e aprendizado de máquina. O modelo é treinado uma única vez ao iniciar.")

# Função para carregar e preparar dados
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = [col.capitalize() for col in df.columns]
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(method='bfill').fillna(method='ffill')
    return df

# Função para previsão de 14 dias futuros
def forecast_next_14_days(df):
    df_model = df.drop(columns=['Date'])
    window_size = 30
    horizon = 14

    # Garantir que haja dados suficientes
    if len(df_model) < (window_size + horizon):
        return None, None

    X = []
    y = []

    for i in range(len(df_model) - window_size - horizon):
        X.append(df_model.iloc[i:i+window_size].values)
        y.append(df_model['Close'].iloc[i+window_size:i+window_size+horizon].values)

    X = np.array(X)
    y = np.array(y)

    X_train = X.reshape(X.shape[0], -1)
    y_train = y

    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Criar janela mais recente para prever o futuro
    last_window = df_model.tail(window_size).values.reshape(1, -1)
    future_pred = model.predict(last_window)[0]
    last_date = df['Date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=14)

    return future_dates, future_pred

# Previsão para BTC
try:
    btc_df = load_data("BTC.csv")
    st.subheader("🔮 BTC – Previsão para os próximos 14 dias")
    btc_dates, btc_forecast = forecast_next_14_days(btc_df)

    if btc_dates is not None:
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(btc_dates, btc_forecast, label="Previsão BTC")
        ax1.set_title("Bitcoin – Previsão de Preço")
        ax1.set_ylabel("Preço")
        ax1.set_xlabel("Data")
        ax1.legend()
        st.pyplot(fig1)
    else:
        st.warning("⚠️ Dados insuficientes para prever BTC")

except Exception as e:
    st.error("Erro ao prever BTC")
    st.code(traceback.format_exc())

# Previsão para GOOGL
try:
    googl_df = load_data("googl.csv")
    st.subheader("🔮 GOOGL – Previsão para os próximos 14 dias")
    googl_dates, googl_forecast = forecast_next_14_days(googl_df)

    if googl_dates is not None:
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(googl_dates, googl_forecast, label="Previsão GOOGL", color="orange")
        ax2.set_title("Google – Previsão de Preço")
        ax2.set_ylabel("Preço")
        ax2.set_xlabel("Data")
        ax2.legend()
        st.pyplot(fig2)
    else:
        st.warning("⚠️ Dados insuficientes para prever GOOGL")

except Exception as e:
    st.error("Erro ao prever GOOGL")
    st.code(traceback.format_exc())
