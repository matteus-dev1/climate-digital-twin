"""Orquestra o pipeline: treino dos modelos e análise das anomalias (sem prints nem gráficos)."""

from climate_twin.config import CAMINHO_CSV, HORIZONTES, N_DESVIOS
from climate_twin.ml.anomalies import agrupar_eventos, detectar_anomalias, filtrar_eventos
from climate_twin.ml.features import carregar_dados
from climate_twin.ml.insights import montar_insights
from climate_twin.ml.model import treinar_todos


def treinar(caminho=CAMINHO_CSV, horizontes=HORIZONTES):
    """Carrega os dados e treina um modelo por horizonte."""
    return treinar_todos(carregar_dados(caminho), horizontes)


def analisar(resultados, n_desvios=N_DESVIOS):
    """Detecta anomalias com o horizonte +1h e monta eventos e insights."""
    res_h1 = resultados[1]
    ts_alvo = res_h1["timestamps_alvo_teste"]

    residuos, limite, indices = detectar_anomalias(
        res_h1["y_teste"], res_h1["previsoes"], n_desvios)
    eventos = agrupar_eventos(ts_alvo.iloc[indices].sort_values())

    return {
        "ts_alvo": ts_alvo,
        "residuos": residuos,
        "limite": limite,
        "indices": indices,
        "eventos": eventos,
        "eventos_filtrados": filtrar_eventos(eventos),
        "insights": montar_insights(resultados, residuos, indices, limite),
    }
