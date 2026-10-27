"""Constantes compartilhadas pelo projeto."""

from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_CSV = RAIZ / "data" / "weather_data.csv"

ID_SENSOR = "ENV-01"
LIMITE_ALERTA = 30.0   # °C
LIMITE_CRITICO = 32.0  # °C

# ===== Machine learning =====
HORIZONTES = [1, 2, 3]   # horas à frente que cada modelo prevê (precisa incluir 1)
PROPORCAO_TREINO = 0.8   # 80% treino, 20% teste
N_DESVIOS = 2            # limite = média + N_DESVIOS * desvio padrão
QUANTIL = 0.99           # usado no detector por percentil
DURACAO_MINIMA = 2       # anomalias seguidas para contar como evento

# Regras de causa provável e de severidade (insights)
LIMITE_CHUVA = 0.1             # mm: a partir daqui considera que está chovendo
LIMITE_RADIACAO = 500          # W/m²: radiação solar forte
LIMITE_VENTO = 20              # km/h: vento forte
LIMITE_VARIACAO = 2.0          # °C: mudança brusca entre a leitura e o horário alvo
LIMITE_TENDENCIA = 0.5         # °C: variação menor que isso conta como estabilidade
RAZAO_SEVERIDADE_ALTA = 1.5    # resíduo / limite de anomalia
RAZAO_SEVERIDADE_MEDIA = 1.2

FEATURES = [
    "temperature", "humidity", "pressure", "wind_speed",
    "precipitation", "solar_radiation", "hora",
    "temp_menos_1", "temp_menos_2",
]
