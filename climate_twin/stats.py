"""Maior temperatura registrada pelo sensor e quantas leituras atingiram o limite crítico."""

from climate_twin.config import LIMITE_CRITICO
from climate_twin.data import carregar_leituras


def main():
    df = carregar_leituras()

    pos_maximo = df["temperature"].idxmax()
    print(f"Maior temperatura: {df.loc[pos_maximo, 'temperature']} °C "
        f"em {df.loc[pos_maximo, 'timestamp']:%Y-%m-%d %H:%M} (leitura {pos_maximo + 1})")

    criticas = df[df["temperature"] >= LIMITE_CRITICO]
    if criticas.empty:
        print("Nenhuma leitura atingiu o limite crítico.")
    else:
        print(f"O CRITICO tocaria {len(criticas)} vezes; "
            f"primeira em {criticas['timestamp'].iloc[0]:%Y-%m-%d %H:%M}")


if __name__ == "__main__":
    main()
