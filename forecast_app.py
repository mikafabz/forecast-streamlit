import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 📌 Configuração do Streamlit
st.set_page_config(page_title="Previsão BTC e GOOGL", layout="centered")
st.title("📈 Previsão de Preços – BTC e GOOGL (Base: 2024)")

# 📌 Função para carregar e filtrar dados de 2024
@st.cache_data
def carregar_dados(caminho_csv, ativo):
    df = pd.read_csv(caminho_csv)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Date'].dt.year == 2024]
    df = df[['Date', 'Close']].copy()
    df.columns = ['Date', f'{ativo}_Close']
    df = df.sort_values('Date')
    df = df.ffill()
    return df

# 📊 Carregar dados
btc_df = carregar_dados('BTC_full.csv', 'BTC')
googl_df = carregar_dados('GOOGL_full.csv', 'GOOGL')

# 📉 Visualizar dados históricos
st.subheader("📊 Preços de Fechamento – 2024")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(btc_df['Date'], btc_df['BTC_Close'], label='BTC', linewidth=2)
ax.plot(googl_df['Date'], googl_df['GOOGL_Close'], label='GOOGL', linewidth=2)
ax.set_title('Fechamento de 2024')
ax.legend()
st.pyplot(fig)

# 🔮 Função de previsão simples para os próximos 14 dias
def prever_14_dias(df, col_name):
    scaler = MinMaxScaler()
    dados_2024 = df[[col_name]].values
    scaled = scaler.fit_transform(dados_2024)

    ultimos_valores = scaled[-30:]
    previsao = []
    ultimo = ultimos_valores[-1][0]

    for _ in range(14):
        proximo = ultimo + np.random.normal(0, 0.01)
        proximo = np.clip(proximo, 0, 1)
        previsao.append([proximo])
        ultimo = proximo

    previsao_real = scaler.inverse_transform(previsao).flatten()
    datas_futuras = pd.date_range(df['Date'].max() + pd.Timedelta(days=1), periods=14)
    return pd.DataFrame({'Date': datas_futuras, 'Previsao': previsao_real})

# 📈 Previsões com base apenas em 2024
st.subheader("🔮 Previsão BTC – Próximos 14 dias")
btc_forecast = prever_14_dias(btc_df, 'BTC_Close')
st.line_chart(btc_forecast.set_index('Date'))

st.subheader("🔮 Previsão GOOGL – Próximos 14 dias")
googl_forecast = prever_14_dias(googl_df, 'GOOGL_Close')
st.line_chart(googl_forecast.set_index('Date'))

# ℹ️ Rodapé
st.markdown("---")
st.markdown("As previsões são geradas com base exclusivamente nos dados de 2024.")
