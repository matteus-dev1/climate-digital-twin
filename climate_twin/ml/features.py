"""Features do modelo: hora do dia e temperaturas defasadas."""

from climate_twin.config import CAMINHO_CSV
from climate_twin.data import carregar_leituras


def carregar_dados(caminho=CAMINHO_CSV):
    """Lê o CSV e cria as features de hora e de temperatura defasada."""
    df = carregar_leituras(caminho=caminho)
    df["hora"] = df["timestamp"].dt.hour
    df["temp_menos_1"] = df["temperature"].shift(1)
    df["temp_menos_2"] = df["temperature"].shift(2)
    return df.dropna()  # as 2 primeiras linhas ficam com NaN por causa do shift
