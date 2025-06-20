import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from ta import add_all_ta_features
import traceback

st.set_page_config(page_title="Forecast BTC & GOOGL", layout="centered")
st.title("📈 Previsão de Preços - BTC e GOOGL")
st.markdown("Este app usa aprendizado de máquina para prever os **próximos 14 dias** de preços do Bitcoin e das ações do Google.")

# Mapear os arquivos dos ativos
file_map = {
    "BTC": "btc.csv",
    "GOOGL": "googl.csv"
}

# Seletor de ativo
asset = st.selectbox("Escolha o ativo:", list(file_map.keys()))

# Função de carregamento
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

# Criar janelas para treino/teste
def create_windowed_dataset(data, target_column='Close', window_size=30, horizon=14):
    X, y = [], []
    for i in range(len(data) - window_size - horizon):
        X.append(data.iloc[i:i+window_size].values)
        y.append(data[target_column].iloc[i+window_size:i+window_size+horizon].values)
    return np.array(X), np.array(y)

# Executar pipeline
try:
    # Carregar e exibir dados
    df = load_data(file_map[asset])
    df_display = df[['Date', 'Close', 'Volume']]
    st.subheader("📊 Histórico de Preços")
    st.line_chart(df_display.set_index("Date"))

    # Preparar dados
    df_model = df.drop(columns=['Date'])
    X, y = create_windowed_dataset(df_model, window_size=30, horizon=14)
    X_train, X_test, y_train, y_test = train_test_split(X.reshape(X.shape[0], -1), y, test_size=0.2, random_state=42)

    # Treinar modelo
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    st.subheader("📉 Erro de Treinamento (RMSE)")
    st.metric(label=f"Erro RMSE - {asset}", value=f"{rmse:.2f}")

    # 🔮 Previsão real para os próximos 14 dias
    last_window = df_model.tail(30).values.reshape(1, -1)
    future_pred = model.predict(last_window)[0]
    last_date = df['Date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=14)

    st.subheader("🔮 Previsão para os Próximos 14 Dias")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(future_dates, future_pred, label="Previsão")
    ax.set_title(f"Previsão de 14 dias futuros - {asset}")
    ax.set_ylabel("Preço")
    ax.set_xlabel("Data")
    ax.legend()
    st.pyplot(fig)

    # Mostrar últimos indicadores
    st.subheader("📋 Últimos Indicadores Técnicos")
    st.dataframe(df.tail(5).reset_index(drop=True))

except Exception as e:
    st.error("❌ Erro ao rodar o app. Verifique se os arquivos CSV estão presentes e formatados corretamente.")
    st.code(traceback.format_exc())
