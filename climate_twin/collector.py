"""Baixa dados climáticos horários da Open-Meteo e salva em data/weather_data.csv."""

import pandas as pd
import requests

from climate_twin.config import CAMINHO_CSV

URL = (
    "https://archive-api.open-meteo.com/v1/archive"
    "?latitude=-5.795"
    "&longitude=-35.209"
    "&start_date=2021-01-01"
    "&end_date=2026-09-24"
    "&hourly=temperature_2m,relative_humidity_2m,precipitation,"
    "surface_pressure,wind_speed_10m,wind_direction_10m,shortwave_radiation"
    "&timezone=America%2FSao_Paulo"
)

# Nome na API -> nome no CSV (a ordem do dicionário define a ordem das colunas)
COLUNAS = {
    "time": "timestamp",
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "surface_pressure": "pressure",
    "wind_speed_10m": "wind_speed",
    "wind_direction_10m": "wind_direction",
    "precipitation": "precipitation",
    "shortwave_radiation": "solar_radiation",
}


def baixar_dados(url=URL):
    """Faz a requisição e devolve os dados horários como DataFrame."""
    resposta = requests.get(url, timeout=120)
    print("Status:", resposta.status_code)
    if resposta.status_code != 200:
        raise SystemExit(f"Erro na API: status {resposta.status_code}")

    return pd.DataFrame(resposta.json()["hourly"])[list(COLUNAS)].rename(columns=COLUNAS)


def main():
    df = baixar_dados()
    CAMINHO_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CAMINHO_CSV, index=False, encoding="utf-8")
    print(f"{len(df)} leituras salvas em {CAMINHO_CSV}")


if __name__ == "__main__":
    main()
