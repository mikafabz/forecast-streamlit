import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import os

# 📌 Configuração do Streamlit
st.set_page_config(page_title="Previsão BTC e GOOGL", layout="centered")
st.title("📈 Previsão de Preços – BTC e GOOGL")

# 📌 Função para carregar e preparar os dados
@st.cache_data
def carregar_dados(arquivo, ativo):
    df = pd.read_csv(arquivo)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Date'].dt.year == 2024]
    df = df[['Date', 'Close']].copy()
    df.columns = ['Date', f'{ativo}_Close']
    df = df.sort_values('Date')
    df = df.ffill()
    return df

# 📌 Carregar dados filtrados de 2024
btc_df = carregar_dados('BTC_full.csv', 'BTC')
googl_df = carregar_dados('GOOGL_full.csv', 'GOOGL')

# 📌 Mostrar dados ao usuário
st.subheader("📊 Dados Históricos – 2024")
st.write("Total de dias disponíveis:")
st.write(f"BTC: {len(btc_df)} dias")
st.write(f"GOOGL: {len(googl_df)} dias")

# 📉 Plotar séries temporais
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(btc_df['Date'], btc_df['BTC_Close'], label='BTC', linewidth=2)
ax.plot(googl_df['Date'], googl_df['GOOGL_Close'], label='GOOGL', linewidth=2)
ax.set_title('Fechamento em 2024')
ax.legend()
st.pyplot(fig)

# 📌 Função de previsão (mock simples para 14 dias)
def prever(df, nome):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[[f'{nome}_Close']])
    ultimos = scaled[-30:]
    previsao = []
    valor_atual = ultimos[-1][0]

    for _ in range(14):
        prox = valor_atual + np.random.normal(0, 0.01)
        prox = np.clip(prox, 0, 1)
        previsao.append([prox])
        valor_atual = prox

    previsao_real = scaler.inverse_transform(previsao).flatten()
    datas_futuras = pd.date_range(df['Date'].max() + pd.Timedelta(days=1), periods=14)
    return pd.DataFrame({'Date': datas_futuras, 'Previsao': previsao_real})

# 🔮 Previsão para os próximos 14 dias
st.subheader("🔮 BTC – Previsão para os próximos 14 dias")
btc_forecast = prever(btc_df, 'BTC')
st.line_chart(btc_forecast.set_index('Date'))

st.subheader("🔮 GOOGL – Previsão para os próximos 14 dias")
googl_forecast = prever(googl_df, 'GOOGL')
st.line_chart(googl_forecast.set_index('Date'))

# 📁 Rodapé
st.markdown("---")
st.markdown("App desenvolvido para apresentação de projeto com dados de 2024.")
