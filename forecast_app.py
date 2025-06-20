import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from ta import add_all_ta_features
import traceback

# Configuração da página
st.set_page_config(page_title="Forecast BTC & GOOGL", layout="centered")
st.title("📈 Previsão de Preços - BTC e GOOGL")
st.markdown("Este app usa aprendizado de máquina para prever os próximos 14 dias de preços de Bitcoin (BTC) e ações do Google (GOOGL).")

# Dicionário de arquivos CSV
file_map = {
    "BTC": "btc.csv",
    "GOOGL": "googl.csv"
}

# Seletor de ativo
asset = st.selectbox("Escolha o ativo:", ["BTC", "GOOGL"])

# Função para carregar e preparar dados
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = [col.capitalize() for col in df.columns]
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume")
    df = df.fillna(method='bfill').fillna(method='ffill')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(method='bfill').fillna(method='ffill')
    return df

# Função para criar janelas temporais
def create_windowed_dataset(data, target_column='Close', window_size=30, horizon=14):
    X, y = [], []
    for i in range(len(data) - window_size - horizon):
        X.append(data.iloc[i:i+window_size].values)
        y.append(data[target_column].iloc[i+window_size:i+window_size+horizon].values)
    return np.array(X), np.array(y)

# Tentativa de carregar dados e rodar o app
try:
    # Carregar dados
    df = load_data(file_map[asset])
    df_display = df[['Date', 'Close', 'Volume']].copy()

    # Visualização de histórico
    st.subheader("📊 Dados históricos")
    st.line_chart(df_display.set_index('Date'))

    # Preparar dataset para modelo
    df_model = df.drop(['Date'], axis=1)
    X, y = create_windowed_dataset(df_model, window_size=30, horizon=14)
    X_train, X_test, y_train, y_test = train_test_split(X.reshape(X.shape[0], -1), y, test_size=0.2, random_state=42)

    # Treinar modelo
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Exibir RMSE
    st.subheader("📉 Erro da Previsão (RMSE)")
    st.metric(label=f"Erro RMSE - {asset}", value=f"{rmse:.2f}")

    # Comparar previsão com valor real
    st.subheader("🔮 Comparativo de Previsão x Real")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y_test[0], label="Real")
    ax.plot(y_pred[0], label="Previsto")
    ax.set_title(f"Previsão de 14 dias - {asset}")
    ax.legend()
    st.pyplot(fig)

    # Mostrar últimos indicadores
    st.subheader("📋 Últimos Indicadores Técnicos")
    st.dataframe(df.tail(5).reset_index(drop=True))

except Exception as e:
    st.error("Erro ao rodar o app. Verifique se os arquivos CSV estão no repositório.")
    st.code(traceback.format_exc())
