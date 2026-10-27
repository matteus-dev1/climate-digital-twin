"""Figuras do matplotlib. Devolvem a figura; quem chama decide como exibir."""

import matplotlib.pyplot as plt

COR_REAL = "#4DABF7"
COR_PREVISAO = "#FF6B6B"
COR_RESIDUO = "#868E96"


def figura_previsoes(resultados, n=100):
    """Um painel por horizonte, cada previsão contra o seu real (eixo = horário alvo)."""
    fig, eixos = plt.subplots(len(resultados), 1, sharex=True, squeeze=False,
                              figsize=(12, 3.5 * len(resultados)))
    eixos = eixos.ravel()

    for eixo, (h, res) in zip(eixos, resultados.items()):
        alvo = res["timestamps_alvo_teste"].iloc[:n]
        eixo.plot(alvo, res["y_teste"][:n], label=f"Real (+{h}h)", color=COR_REAL)
        eixo.plot(alvo, res["previsoes"][:n], label=f"Previsão +{h}h",
                color=COR_PREVISAO, linestyle="--")
        eixo.set_title(f"Horizonte +{h}h")
        eixo.set_ylabel("Temperatura (°C)")
        eixo.legend()

    eixos[-1].set_xlabel("Data e hora")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def figura_residuos(ts, residuos, limite, indices_anomalias, n=300):
    """Resíduos ao longo do tempo, com a linha do limite e as anomalias."""
    fig, eixo = plt.subplots(figsize=(12, 5))
    eixo.plot(ts.iloc[:n], residuos[:n], label="Erro (resíduo)", color=COR_RESIDUO)
    eixo.axhline(limite, color=COR_PREVISAO, linestyle="--",
                label=f"Limite ({round(limite, 2)} °C)")

    # Só as anomalias dentro do trecho mostrado (primeiras n horas)
    visiveis = indices_anomalias[indices_anomalias < n]
    eixo.scatter(ts.iloc[visiveis], residuos[visiveis],
                color=COR_PREVISAO, zorder=3, label="Anomalias")

    eixo.set_title("Resíduos do modelo (+1h) e anomalias detectadas")
    eixo.set_xlabel("Data")
    eixo.set_ylabel("Erro (°C)")
    eixo.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig
