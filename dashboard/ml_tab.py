"""Aba de ML do dashboard: previsões, anomalias e insights."""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # permite importar climate_twin

from climate_twin.config import CAMINHO_CSV, DURACAO_MINIMA, N_DESVIOS
from climate_twin.ml.exploration import anomalias_por_hora, comparar_detectores
from climate_twin.ml.insights import dia_com_mais_anomalias, resumir_periodo
from climate_twin.ml.model import comparar_com_baseline
from climate_twin.ml.pipeline import analisar, treinar
from climate_twin.ml.plots import figura_previsoes, figura_residuos

CAIXA_POR_SEVERIDADE = {"ALTA": st.error, "MÉDIA": st.warning, "BAIXA": st.info}


@st.cache_data(show_spinner="Treinando os modelos...")
def _treinar(caminho, modificado_em):
    """Treino em cache; o parâmetro `modificado_em` refaz o treino quando o CSV muda."""
    return treinar(caminho)


def _mostrar_figura(fig):
    st.pyplot(fig)
    plt.close(fig)


def _aba_anomalias(analise):
    col_total, col_eventos = st.columns(2)
    col_total.metric("Anomalias detectadas", len(analise["indices"]))
    col_eventos.metric(f"Eventos com {DURACAO_MINIMA}+ anomalias seguidas",
                        len(analise["eventos_filtrados"]))

    _mostrar_figura(figura_residuos(analise["ts_alvo"], analise["residuos"],
                                    analise["limite"], analise["indices"]))

    if analise["eventos_filtrados"]:
        eventos = pd.DataFrame(
            [{"Início": ev[0], "Fim": ev[-1], "Anomalias seguidas": len(ev)}
            for ev in analise["eventos_filtrados"]])
        st.dataframe(eventos, hide_index=True)


def _aba_insights(insights):
    if not insights:
        st.info("Nenhuma anomalia detectada com esta sensibilidade.")
        return

    datas = [i["timestamp"] for i in insights]
    dia_padrao = dia_com_mais_anomalias(insights).date()
    col_ini, col_fim = st.columns(2)
    inicio = col_ini.date_input("De", value=dia_padrao,
                                min_value=min(datas).date(), max_value=max(datas).date())
    fim = col_fim.date_input("Até", value=dia_padrao,
                            min_value=min(datas).date(), max_value=max(datas).date())
    if fim < inicio:
        st.warning("A data final precisa ser igual ou posterior à inicial.")
        return

    resumo = resumir_periodo(insights, inicio, pd.Timestamp(fim) + pd.Timedelta(days=1))
    if resumo["total"] == 0:
        st.info(f"Nenhuma anomalia em {resumo['titulo']}.")
    else:
        colunas = st.columns(4)
        colunas[0].metric("Anomalias", resumo["total"])
        for coluna, (sev, qtd) in zip(colunas[1:], resumo["por_severidade"].items()):
            coluna.metric(sev, qtd)
        st.write(f"Causa mais frequente: **{resumo['causa_principal']}** "
                f"({resumo['qtd_causa_principal']} de {resumo['total']}).")

        st.subheader("Mais graves")
        for i in resumo["mais_graves"]:
            CAIXA_POR_SEVERIDADE[i["severidade"]](f"**{i['severidade']}** · {i['frase']}")

    with st.expander("Todas as anomalias"):
        tabela = pd.DataFrame(insights)[
            ["timestamp", "real", "previsto", "residuo", "severidade", "causa", "tendencia"]]
        tabela.columns = ["Horário", "Real (°C)", "Previsto (°C)", "Erro (°C)",
                        "Severidade", "Causa provável", "Tendência"]
        numericas = ["Real (°C)", "Previsto (°C)", "Erro (°C)"]
        tabela[numericas] = tabela[numericas].round(2)
        st.dataframe(tabela, hide_index=True)


def _aba_avancada(resultados, analise):
    st.subheader("Anomalias por hora do dia")
    st.dataframe(anomalias_por_hora(analise["ts_alvo"], analise["residuos"],
                                    analise["indices"]))

    st.subheader("Comparação de detectores")
    st.bar_chart(comparar_detectores(analise["ts_alvo"], analise["residuos"],
                                    analise["limite"], resultados[1]))
    st.caption("Com poucas amostras por hora, o detector por quantil tende a marcar "
                "cerca de uma anomalia por hora. O calibrado no treino aprende o "
                "normal sem olhar o teste.")


def renderizar_ml(caminho=CAMINHO_CSV):
    """Desenha a aba de ML. Pode ser chamada dentro de qualquer container do Streamlit."""
    if not os.path.exists(caminho):
        st.error(f"Arquivo {caminho} não encontrado. Rode a coleta de dados primeiro.")
        return

    resultados = _treinar(str(caminho), os.path.getmtime(caminho))

    colunas = st.columns(len(resultados))
    for coluna, (h, res) in zip(colunas, resultados.items()):
        m = comparar_com_baseline(res)
        coluna.metric(f"Erro médio +{h}h", f"{m['modelo']:.2f} °C",
                    f"{m['modelo'] - m['baseline']:+.2f} °C vs baseline",
                    delta_color="inverse")

    n_desvios = st.slider(
        "Sensibilidade do detector (desvios padrão acima do erro médio)",
        1.0, 4.0, float(N_DESVIOS), 0.5, help="Valores menores marcam mais anomalias.")
    analise = analisar(resultados, n_desvios)

    aba_prev, aba_anom, aba_ins, aba_av = st.tabs(
        ["Previsões", "Anomalias", "Insights", "Análise avançada"])
    with aba_prev:
        _mostrar_figura(figura_previsoes(resultados))
    with aba_anom:
        _aba_anomalias(analise)
    with aba_ins:
        _aba_insights(analise["insights"])
    with aba_av:
        _aba_avancada(resultados, analise)
