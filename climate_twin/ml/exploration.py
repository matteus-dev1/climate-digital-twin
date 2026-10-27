"""Análises exploratórias das anomalias: por hora do dia e comparação de detectores."""

import pandas as pd

from climate_twin.config import N_DESVIOS, QUANTIL


def montar_tabela(timestamps, residuos):
    """Cria uma tabela com timestamp, resíduo e hora do dia."""
    tabela = pd.DataFrame({
        "timestamp": pd.Series(timestamps).reset_index(drop=True),
        "residuo": residuos,
    })
    tabela["hora"] = tabela["timestamp"].dt.hour
    return tabela


def anomalias_por_hora(ts, residuos, indices_anomalias):
    """Quantidade e erro médio das anomalias em cada hora do dia."""
    tabela = montar_tabela(ts.iloc[indices_anomalias], residuos[indices_anomalias])
    return (
        tabela.groupby("hora")["residuo"]
        .agg(quantidade="count", erro_medio="mean")
        .round(2)
        .sort_values("quantidade", ascending=False)
    )


def comparar_detectores(ts, residuos, limite_global, res_h1, n_desvios=N_DESVIOS):
    """Conta as anomalias de 4 formas de definir o limite; devolve uma Series."""
    df_res = montar_tabela(ts, residuos)
    residuo = df_res["residuo"]
    por_hora = df_res.groupby("hora")["residuo"]
    q = QUANTIL * 100

    # Por hora do dia: média + N desvios de cada hora
    stats = por_hora.agg(["mean", "std"])
    limite_hora = df_res["hora"].map(stats["mean"] + n_desvios * stats["std"])

    # Por hora do dia: percentil de cada hora
    limite_quantil = df_res["hora"].map(por_hora.quantile(QUANTIL))

    # Calibrado no treino: aprende o "normal" no treino e aplica no teste
    df_treino = montar_tabela(res_h1["timestamps_alvo_treino"], res_h1["residuos_treino"])
    limite_calibrado = df_res["hora"].map(
        df_treino.groupby("hora")["residuo"].quantile(QUANTIL))

    return pd.Series({
        f"Global (média + {n_desvios}sd)": (residuo > limite_global).sum(),
        f"Por hora (média + {n_desvios}sd)": (residuo > limite_hora).sum(),
        f"Por hora (quantil {q:.0f})": (residuo > limite_quantil).sum(),
        f"Calibrado no treino (q{q:.0f})": (residuo > limite_calibrado).sum(),
    }, name="anomalias")
