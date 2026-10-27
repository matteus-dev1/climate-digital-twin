"""Dashboard Streamlit do Digital Twin: monitoramento do sensor ENV-01 e análise de ML.

Execução (a partir da raiz do projeto):  streamlit run dashboard/app.py
"""

import sys
import time
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # permite importar climate_twin

from climate_twin.config import ID_SENSOR, LIMITE_ALERTA, LIMITE_CRITICO
from climate_twin.data import carregar_leituras
from climate_twin.twin import classificar_status
from ml_tab import renderizar_ml

JANELA = 35  # máximo de leituras mantidas nos gráficos

COR_FUNDO = "#0E1117"
COR_TEXTO = "#FAFAFA"
COR_GRADE = "#3A3F4B"

plt.rcParams.update({
    "figure.facecolor": COR_FUNDO,
    "axes.facecolor": COR_FUNDO,
    "axes.edgecolor": COR_GRADE,
    "axes.labelcolor": COR_TEXTO,
    "axes.titlecolor": COR_TEXTO,
    "xtick.color": COR_TEXTO,
    "ytick.color": COR_TEXTO,
    "grid.color": COR_GRADE,
    "legend.facecolor": COR_FUNDO,
    "legend.labelcolor": COR_TEXTO,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 2,
})

# Coluna do CSV, rótulo e unidade de cada métrica (o 8º card é o status)
METRICAS = [
    ("temperature", "Temp", "°C"),
    ("humidity", "Umid", "%"),
    ("pressure", "Press", "hPa"),
    ("wind_speed", "Vento", "km/h"),
    ("wind_direction", "Direc", "°"),
    ("precipitation", "Precp", "mm"),
    ("solar_radiation", "Radia", "W/m²"),
]

# Coluna, título, unidade e cor de cada gráfico
GRAFICOS = [
    ("temperature", "Temperatura (°C)", "°C", "#FF6B6B"),
    ("humidity", "Umidade (%)", "%", "#4DABF7"),
    ("solar_radiation", "Radiação (W/m²)", "W/m²", "#FFA94D"),
]


def plotar(espaco, janela, coluna, titulo, unidade, cor):
    """Desenha um gráfico de linha de `coluna` ao longo do tempo."""
    fig, ax = plt.subplots(figsize=(7, 2.4))
    ax.plot(janela["timestamp"], janela[coluna], color=cor)
    ax.set_title(titulo, fontsize=10)
    ax.set_ylabel(unidade)
    ax.grid(True, alpha=0.4)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh"))
    fig.autofmt_xdate()
    espaco.pyplot(fig)
    plt.close(fig)  # libera a memória da figura a cada volta do laço


def rodar_leituras(df, cards, espacos_graficos, resumo, n_leituras, pausa):
    """Exibe as leituras uma a uma, atualizando cards e gráficos."""
    for contador, leitura in enumerate(df.itertuples(index=False), start=1):
        janela = df.iloc[:contador].tail(JANELA)

        resumo.caption(
            f"🕐 {leitura.timestamp:%H:%M}  ·  leitura {contador}/{n_leituras}"
            f"  ·  janela guarda {len(janela)} de {JANELA}"
        )
        for card, (coluna, rotulo, unidade) in zip(cards, METRICAS):
            card.metric(rotulo, f"{getattr(leitura, coluna)} {unidade}")
        cards[-1].metric("Status", leitura.status)

        for espaco, (coluna, titulo, unidade, cor) in zip(espacos_graficos, GRAFICOS):
            plotar(espaco, janela, coluna, titulo, unidade, cor)

        time.sleep(pausa)


def descrever(leitura):
    """Texto de uma leitura para o resumo final: data e hora | temperatura | status."""
    return f"{leitura['timestamp']:%Y-%m-%d %H:%M} | {leitura['temperature']} °C | {leitura['status']}"


def resumo_final(df):
    """Mostra a primeira leitura, o início da janela e a leitura mais recente."""
    st.write("---")
    if df.empty:
        st.warning("Nenhuma leitura encontrada no arquivo CSV.")
        return

    janela = df.tail(JANELA)
    st.write(f"📊 **Twin** {ID_SENSOR} — **{len(df)}** leituras lidas, "
            f"**{len(janela)}** na janela")
    st.write(f"**Primeira leitura:** {descrever(df.iloc[0])}")
    st.write(f"**Início da janela:** {descrever(janela.iloc[0])}")  # mais antigo que sobreviveu
    st.write(f"**Mais recente:** {descrever(janela.iloc[-1])}")


def aba_monitoramento(limite_alerta, limite_critico, n_leituras, pausa, limites_invalidos):
    """Monitoramento em tempo simulado do sensor."""
    st.write("Clique no botão abaixo para iniciar o monitoramento")

    # Áreas da tela, preenchidas a cada leitura
    cards = [coluna.empty() for coluna in st.columns(4) + st.columns(4)]
    espacos_graficos = [st.empty() for _ in GRAFICOS]
    resumo = st.empty()

    if st.button("▶️ Iniciar Monitoramento", disabled=limites_invalidos):
        df = carregar_leituras(n_leituras)
        df["status"] = classificar_status(df["temperature"], limite_alerta, limite_critico)
        rodar_leituras(df, cards, espacos_graficos, resumo, n_leituras, pausa)
        st.success("✅ Monitoramento finalizado")
        resumo_final(df)


def main():
    st.set_page_config(page_title="Digital Twin", page_icon="📊", layout="wide")
    st.title("📊 Digital Twin - Monitoramento Climático")

    st.sidebar.header("⚙️ Controles")
    st.sidebar.info("Ajuste os limiares ANTES de iniciar — mexer durante o rodar reseta tudo")
    limite_alerta = st.sidebar.slider("Limite Alerta", 20.0, 40.0, LIMITE_ALERTA, 0.5)
    limite_critico = st.sidebar.slider("Limite Crítico", 30.0, 45.0, LIMITE_CRITICO, 0.5)
    n_leituras = st.sidebar.slider("Quantidade de leituras", 5, 100, 35, 5)
    pausa = st.sidebar.slider("Pausa (s)", 0.0, 2.0, 0.5, 0.1)

    limites_invalidos = limite_alerta >= limite_critico
    if limites_invalidos:
        st.sidebar.warning("O Limite Alerta precisa ser MENOR que o Limite Crítico")

    aba_monitor, aba_ml = st.tabs(["Monitoramento", "Machine Learning"])
    with aba_monitor:
        aba_monitoramento(limite_alerta, limite_critico, n_leituras, pausa, limites_invalidos)
    with aba_ml:
        renderizar_ml()


main()
