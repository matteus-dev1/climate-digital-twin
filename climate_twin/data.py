"""Leitura do CSV de leituras climáticas."""

import pandas as pd

from climate_twin.config import CAMINHO_CSV


def carregar_leituras(n_leituras=None, caminho=CAMINHO_CSV):
    """Lê as primeiras n_leituras do CSV (todas, se n_leituras for None)."""
    return pd.read_csv(caminho, nrows=n_leituras, parse_dates=["timestamp"])
