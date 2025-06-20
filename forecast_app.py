import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import streamlit as st
import datetime

# Parâmetros
WINDOW_SIZE = 60
FORECAST_DAYS = 14

# Função para carregar dados
@st.cache_data
def carregar_dados(caminho_arquivo, nome_ativo):
    df = pd.read_csv(caminho_arquivo)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df[['Date', 'Close']].dropna()
    df = df.sort_values('Date')
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna()
    return df

# Função de previsão
def prever_ativos_lstm(df, ativo):
    st.subheader(f"📈 Previsão para os próximos {FORECAST_DAYS} dias – {ativo}")
    
    if len(df) < WINDOW_SIZE + FORECAST_DAYS:
        st.warning(f"⚠️ Dados insuficientes para gerar previsão de {ativo}.")
        return

    # Normalização
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Close']])

    # Preparar dados de treino
    X, y = [], []
    for i in range(WINDOW_SIZE, len(scaled_data) - FORECAST_DAYS):
        X.append(scaled_data[i - WINDOW_SIZE:i])
        y.append(scaled_data[i])
    X, y = np.array(X), np.array(y)

    # Construir e treinar o modelo
    model = Sequential([
        LSTM(32, input_shape=(WINDOW_SIZE, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, batch_size=16, verbose=0)

    # Previsão
    forecast = []
    input_seq = scaled_data[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, 1)
    for _ in range(FORECAST_DAYS):
        pred = model.predict(input_seq, verbose=0)[0][0]
        forecast.append(pred)
        input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)

    # Reverter normalização
    forecast_inv = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
    forecast_dates = pd.date_range(df['Date'].max() + pd.Timedelta(days=1), periods=FORECAST_DAYS)
    forecast_df = pd.DataFrame({'Data': forecast_dates, 'Previsão': forecast_inv})

    # Visualizar gráfico
    st.line_chart(forecast_df.set_index('Data'))

# Interface Streamlit
st.title("🔮 Previsão de Ativos com LSTM")
st.write("Este aplicativo realiza a previsão de preços para os próximos 14 dias com base em dados históricos.")

# Carregar dados
btc_df = carregar_dados('BTC.csv', 'BTC')
googl_df = carregar_dados('GOOGL.csv', 'GOOGL')

# Visualizar série histórica
st.subheader("📊 Preços históricos")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(btc_df['Date'], btc_df['Close'], label='BTC')
ax.plot(googl_df['Date'], googl_df['Close'], label='GOOGL')
ax.set_title("Preço de Fechamento")
ax.legend()
st.pyplot(fig)

# Previsão
prever_ativos_lstm(btc_df, "BTC")
prever_ativos_lstm(googl_df, "GOOGL")
