import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from datetime import timedelta

st.set_page_config(page_title="Previsão BTC e GOOGL", layout="wide")

st.title("📈 Previsão para os próximos 14 dias")
st.markdown("Este app usa um modelo Random Forest para prever os próximos 14 dias de fechamento do Bitcoin (BTC-USD) e ações da Google (GOOGL).")

# Função para previsão
def forecast_next_14_days(df):
    df = df[['Date', 'Close', 'Volume']].dropna()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # Usar janelas de 30 dias
    WINDOW_SIZE = 30
    X, y = [], []
    for i in range(WINDOW_SIZE, len(df) - 14):
        window = df.iloc[i - WINDOW_SIZE:i]
        features = window[['Close', 'Volume']].values.flatten()
        X.append(features)
        y.append(df['Close'].iloc[i])

    if len(X) == 0:
        raise ValueError("Dados insuficientes para treinar modelo.")

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Prever próximos 14 dias
    last_window = df[['Close', 'Volume']].iloc[-WINDOW_SIZE:].values.flatten().reshape(1, -1)
    forecasts = []
    forecast_dates = []
    base_date = df['Date'].iloc[-1]

    for i in range(14):
        pred = model.predict(last_window)[0]
        forecasts.append(pred)
        forecast_dates.append(base_date + timedelta(days=i + 1))

        # Atualiza janela com novo dia previsto
        new_row = np.array([pred, df['Volume'].iloc[-1]])
        last_window = np.append(last_window[:, 2:], new_row).reshape(1, -1)

    return forecasts, forecast_dates

# Carregar arquivos
def load_csv(path):
    df = pd.read_csv(path)
    if 'Date' not in df.columns or 'Close' not in df.columns or 'Volume' not in df.columns:
        raise ValueError("O CSV deve conter as colunas: Date, Close, Volume.")
    return df

# BTC
st.subheader("🔮 BTC – Previsão para os próximos 14 dias")
try:
    btc_df = load_csv("btc.csv")
    st.write(f"📄 Total de linhas disponíveis no btc.csv: {len(btc_df)}")
    btc_forecast, btc_dates = forecast_next_14_days(btc_df)

    df_btc_forecast = pd.DataFrame({'Date': btc_dates, 'Forecast': btc_forecast})
    st.line_chart(df_btc_forecast.set_index('Date'))

    st.download_button(
        label="📥 Baixar previsão BTC (CSV)",
        data=df_btc_forecast.to_csv(index=False).encode('utf-8'),
        file_name='forecast_btc.csv',
        mime='text/csv'
    )

except Exception as e:
    st.error(f"Erro com BTC: {e}")

# GOOGL
st.subheader("🔮 GOOGL – Previsão para os próximos 14 dias")
try:
    googl_df = load_csv("googl.csv")
    st.write(f"📄 Total de linhas disponíveis no googl.csv: {len(googl_df)}")
    googl_forecast, googl_dates = forecast_next_14_days(googl_df)

    df_googl_forecast = pd.DataFrame({'Date': googl_dates, 'Forecast': googl_forecast})
    st.line_chart(df_googl_forecast.set_index('Date'))

    st.download_button(
        label="📥 Baixar previsão GOOGL (CSV)",
        data=df_googl_forecast.to_csv(index=False).encode('utf-8'),
        file_name='forecast_googl.csv',
        mime='text/csv'
    )

except Exception as e:
    st.error(f"Erro com GOOGL: {e}")
