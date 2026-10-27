"""Sensor virtual: lê weather_data.csv e simula o envio de leituras, uma por vez."""

import time

from climate_twin.config import ID_SENSOR
from climate_twin.data import carregar_leituras

PAUSA = 1.0  # segundos entre leituras
N_LEITURAS = 5


def main():
    df = carregar_leituras(N_LEITURAS)

    for linha in df.itertuples(index=False):
        print(
            f"{ID_SENSOR} | Hora:{linha.timestamp:%H:%M} "
            f"Temperatura:{linha.temperature}°C | Umidade:{linha.humidity}% | "
            f"Pressão:{linha.pressure}hPa | Velocidade:{linha.wind_speed}km/h | "
            f"Direção:{linha.wind_direction}° | Precipitação:{linha.precipitation}mm | "
            f"Radiação:{linha.solar_radiation}W/m²"
        )
        time.sleep(PAUSA)

    print(f"\n[Twin] {ID_SENSOR} enviou {len(df)} leituras.")


if __name__ == "__main__":
    main()
