"""Gêmeo digital: simula o sensor ENV-01, classifica cada leitura e mantém o histórico."""

import time

import numpy as np

from climate_twin.config import ID_SENSOR, LIMITE_ALERTA, LIMITE_CRITICO
from climate_twin.data import carregar_leituras

PAUSA = 1.0  # segundos entre leituras
N_LEITURAS = 35


def classificar_status(temperaturas, limite_alerta=LIMITE_ALERTA, limite_critico=LIMITE_CRITICO):
    """Status de cada leitura conforme a temperatura (a primeira condição verdadeira vence)."""
    return np.select(
        [temperaturas >= limite_critico, temperaturas >= limite_alerta],
        ["CRÍTICO", "ALERTA"],
        default="NORMAL",
    )


def simular_leituras(historico):
    """Imprime as leituras uma a uma, com pausa entre elas."""
    # index=False: sem índice; enumerate só numera as leituras a partir de 1
    for contador, leitura in enumerate(historico.itertuples(index=False), start=1):
        print(
            f"[{ID_SENSOR}] {contador} | hora:{leitura.timestamp:%H:%M} | "
            f"Temp:{leitura.temperature} °C | Umid:{leitura.humidity} % | "
            f"Press:{leitura.pressure} hPa | Vento:{leitura.wind_speed} km/h | "
            f"Direc:{leitura.wind_direction} ° | Precp:{leitura.precipitation} mm | "
            f"Radia:{leitura.solar_radiation} W/m² | Status:{leitura.status}"
        )
        time.sleep(PAUSA)


def imprimir_resumo(historico):
    """Mostra a primeira e a mais recente leitura do histórico."""
    inicial = historico.iloc[0]
    recente = historico.iloc[-1]

    print(f"\n[Twin] {ID_SENSOR} — {len(historico)} leituras no histórico")
    print(f"{'  Inicial:':<16}{inicial['timestamp']:%Y-%m-%d %H:%M} | "
        f"{inicial['temperature']} °C | {inicial['status']}")
    print(f"{'  Mais recente:':<16}{recente['timestamp']:%Y-%m-%d %H:%M} | "
        f"{recente['temperature']} °C | {recente['status']}")


def main():
    historico = carregar_leituras(N_LEITURAS)
    historico["status"] = classificar_status(historico["temperature"])
    simular_leituras(historico)
    imprimir_resumo(historico)


if __name__ == "__main__":
    main()
