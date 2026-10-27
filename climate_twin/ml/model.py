"""Treino de um modelo de regressão linear por horizonte e comparação com o baseline.

Convenção de tempo: "leitura" é o instante em que a previsão é feita e "alvo" é o instante
previsto (leitura + h horas). Os horários exibidos ao usuário são sempre o alvo.
"""

import numpy as np
from sklearn.linear_model import LinearRegression

from climate_twin.config import FEATURES, PROPORCAO_TREINO

# Condições do horário alvo usadas pelas regras de causa provável (colunas *_alvo)
CONDICOES = ["precipitation", "solar_radiation", "wind_speed"]


def treinar_horizonte(df, h):
    """Treina o modelo da temperatura daqui a h horas e devolve o que as outras etapas usam."""
    df_h = df.copy()
    df_h["temp_futura"] = df_h["temperature"].shift(-h)     # alvo do modelo
    df_h["timestamp_alvo"] = df_h["timestamp"].shift(-h)    # horário a que o alvo se refere
    df_h = df_h.join(df_h[CONDICOES].shift(-h).add_suffix("_alvo"))
    df_h = df_h.dropna(subset=["temp_futura"]).reset_index(drop=True)

    X = df_h[FEATURES]
    y = df_h["temp_futura"]

    # Divisão em ordem cronológica (nunca embaralhar série temporal)
    corte = int(len(df_h) * PROPORCAO_TREINO)
    X_treino, X_teste = X.iloc[:corte], X.iloc[corte:]
    y_treino, y_teste = y.iloc[:corte], y.iloc[corte:]

    modelo = LinearRegression()
    modelo.fit(X_treino, y_treino)

    return {
        "X_teste": X_teste.reset_index(drop=True),
        "y_teste": y_teste.to_numpy(),
        "previsoes": modelo.predict(X_teste),
        # Baseline: a temperatura daqui a h horas será igual à de agora
        "baseline": X_teste["temperature"].to_numpy(),
        # Todos os itens abaixo seguem a mesma ordem de y_teste e previsoes
        "timestamps_teste": df_h["timestamp"].iloc[corte:].reset_index(drop=True),  # leitura
        "timestamps_alvo_teste": df_h["timestamp_alvo"].iloc[corte:].reset_index(drop=True),
        "condicoes_alvo_teste": df_h[[f"{c}_alvo" for c in CONDICOES]]
                                .iloc[corte:].reset_index(drop=True),
        # Dados de treino, usados pelo detector calibrado
        "timestamps_alvo_treino": df_h["timestamp_alvo"].iloc[:corte].reset_index(drop=True),
        "residuos_treino": np.abs(y_treino.to_numpy() - modelo.predict(X_treino)),
        "n_treino": len(X_treino),
        "n_teste": len(X_teste),
    }


def treinar_todos(df, horizontes):
    """Treina um modelo por horizonte e devolve {h: resultado}."""
    return {h: treinar_horizonte(df, h) for h in horizontes}


def comparar_com_baseline(res):
    """Erros médios e piores erros do modelo e do baseline no conjunto de teste."""
    erros_modelo = np.abs(res["y_teste"] - res["previsoes"])
    erros_baseline = np.abs(res["y_teste"] - res["baseline"])
    return {
        "modelo": float(erros_modelo.mean()),
        "baseline": float(erros_baseline.mean()),
        "pior_modelo": float(erros_modelo.max()),
        "pior_baseline": float(erros_baseline.max()),
        "modelo_ganhou": bool(erros_modelo.mean() < erros_baseline.mean()),
    }
