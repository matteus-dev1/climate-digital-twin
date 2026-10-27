"""Detecção de anomalias pelo erro do modelo e agrupamento em eventos."""

import numpy as np
import pandas as pd

from climate_twin.config import DURACAO_MINIMA, N_DESVIOS


def detectar_anomalias(y_real, y_prev, n_desvios=N_DESVIOS):
    """Detector global: erro > média + n_desvios * desvio padrão.

    Devolve os resíduos, o limite e as posições (dentro do teste) acima do limite.
    """
    residuos = np.abs(y_real - y_prev)
    limite = residuos.mean() + n_desvios * residuos.std(ddof=1)  # ddof=1, como no pandas
    indices_anomalias = np.where(residuos > limite)[0]
    return residuos, limite, indices_anomalias


def agrupar_eventos(timestamps_anomalias):
    """Junta horários consecutivos (intervalo de até 1h) em eventos."""
    eventos = []

    if len(timestamps_anomalias) == 0:
        return eventos

    evento_atual = [timestamps_anomalias.iloc[0]]

    for i in range(1, len(timestamps_anomalias)):
        dif = timestamps_anomalias.iloc[i] - timestamps_anomalias.iloc[i - 1]

        if dif <= pd.Timedelta(hours=1):
            evento_atual.append(timestamps_anomalias.iloc[i])
        else:
            eventos.append(evento_atual)
            evento_atual = [timestamps_anomalias.iloc[i]]

    eventos.append(evento_atual)  # o último evento
    return eventos


def filtrar_eventos(eventos, duracao_minima=DURACAO_MINIMA):
    """Fica só com os eventos de pelo menos duracao_minima anomalias."""
    return [ev for ev in eventos if len(ev) >= duracao_minima]
