import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

# Função para carregar dados
@st.cache_data
def carregar_dados(caminho_arquivo, nome_ativo):
    df = pd.read_csv(caminho_arquivo)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df[df['Date'].notnull()]
    df = df.sort_values('Date')
    df['Ativo'] = nome_ativo
    return df

# Função para criar features
def adicionar_features(df):
    df = df.copy()
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['Retorno'] = df['Close'].pct_change()
    df = df.dropna()
    return df

# Função para treinar e prever
def treinar_modelo(df):
    df = adicionar_features(df)
    df = df.dropna()

    # Usar apenas os últimos 100 dados para treinar
    df = df[-100:]

    # Features e alvo
    X = df[['SMA_5', 'EMA_5', 'Retorno']]
    y = df['Close']

    # Normalizar
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    # Previsão para os próximos 14 dias
    ultimos = X_scaled[-1].reshape(1, -1)
    previsoes = []
    for _ in range(14):
        pred = model.predict(ultimos)[0]
        previsoes.append(pred)
        nova_linha = np.array([[pred, pred, 0]])  # Simples aproximação para continuar
        ultimos = np.roll(ultimos, -1, axis=0)
        ultimos = nova_linha

    return previsoes

# Interface Streamlit
st.set_page_config(page_title="Forecast App", layout="centered")
st.title("📈 Previsão de Preços – BTC e GOOGL")

# Seleção de ativo
ativo = st.selectbox("Escolha o ativo:", ['BTC', 'GOOGL'])

# Carregar dados
btc = carregar_dados('BTC.csv', 'BTC')
googl = carregar_dados('GOOGL.csv', 'GOOGL')

# Selecionar ativo
df = btc if ativo == 'BTC' else googl

# Exibir dados recentes
st.subheader(f"📊 Últimos dados de {ativo}")
st.dataframe(df.tail())

# Rodar modelo e prever
st.subheader(f"🔮 Previsão de {ativo} para os próximos 14 dias")
previsoes = treinar_modelo(df)
st.line_chart(previsoes)

# Mostrar valores previstos
st.write("Valores previstos:")
st.write(previsoes)
