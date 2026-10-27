"""Previsão da temperatura da hora seguinte: baselines (persistência e média móvel) e regressão linear."""

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from climate_twin.config import PROPORCAO_TREINO
from climate_twin.data import carregar_leituras

LIMIAR_ERRO = 2.0  # °C: acima disso o erro é considerado grande
FEATURES = ["temperature", "humidity", "solar_radiation"]


def preparar_dados(df):
    """Adiciona a leitura da hora seguinte (colunas *_prox) e descarta a última linha.

    A última linha não tem "seguinte".
    """
    return df.join(df.shift(-1).add_suffix("_prox")).iloc[:-1]


def mostrar_pior_erro(dados, erros, previstas):
    """Imprime o maior erro e a leitura em que ele ocorreu."""
    i = erros.idxmax()
    print("Pior erro:", erros[i])
    print("Aconteceu na leitura:", dados.loc[i, "timestamp"])
    print("  Previsto:", previstas[i], "| Real:", dados.loc[i, "temperature_prox"])


def baseline_persistencia(dados):
    """Temperatura futura = temperatura atual. Devolve os erros absolutos."""
    erros = (dados["temperature_prox"] - dados["temperature"]).abs()
    print("Erro medio (baseline):", erros.mean())
    mostrar_pior_erro(dados, erros, dados["temperature"])
    return erros


def baseline_media_movel(dados):
    """Média móvel de 3 horas como previsão."""
    media_3h = dados["temperature"].rolling(3).mean()
    erros_mm = (dados["temperature_prox"] - media_3h).abs().dropna()
    print("Erro medio (media movel):", erros_mm.mean())
    mostrar_pior_erro(dados, erros_mm, media_3h)


def analisar_erros_grandes(dados, erros):
    """Conta os erros grandes do baseline e quantos aconteceram com chuva."""
    grandes = dados[erros > LIMIAR_ERRO]
    print(f"Erros acima de {LIMIAR_ERRO:g} °C: {len(grandes)} de {len(erros)}")

    com_chuva = grandes[grandes["precipitation_prox"] > 0]
    print(f"Erros grandes COM chuva: {len(com_chuva)} de {len(grandes)}")

    # Primeiro erro grande sem chuva
    sem_chuva = grandes[grandes["precipitation_prox"] == 0]
    if not sem_chuva.empty:
        ex = sem_chuva.iloc[0]
        print(ex["timestamp"], "->", ex["temperature_prox"],
            "| umidade:", ex["humidity_prox"],
            "| radiação:", ex["solar_radiation_prox"])


def regressao_linear(dados):
    """Treina uma regressão linear e devolve (valores reais, previsões) do teste.

    Divisão cronológica: o modelo treina no início da série e é avaliado no restante,
    comparado com a persistência no mesmo trecho.
    """
    corte = int(len(dados) * PROPORCAO_TREINO)
    treino, teste = dados.iloc[:corte], dados.iloc[corte:]

    modelo = LinearRegression()
    modelo.fit(treino[FEATURES], treino["temperature_prox"])

    y = teste["temperature_prox"]
    previsoes = modelo.predict(teste[FEATURES])
    print("Previsão do modelo para a 1ª leitura do teste:", previsoes[0])
    print("Valor real que deveria ser:", y.iloc[0])

    print("Erro médio no teste (Modelo ML):", (y - previsoes).abs().mean())
    print("Erro médio no teste (baseline):", (y - teste["temperature"]).abs().mean())
    return y, previsoes


def grafico_real_vs_previsto(y, previsoes, n=100):
    """Gráfico das primeiras n horas, para não poluir."""
    plt.figure(figsize=(12, 6))
    plt.plot(y.iloc[:n].to_numpy(), label="Temperatura Real", color="blue")
    plt.plot(previsoes[:n], label="Previsão ML", color="red", linestyle="--")
    plt.legend()
    plt.title("Temperatura Real vs Prevista pelo Modelo ML")
    plt.show()


def main():
    df = carregar_leituras()
    print("Leituras carregadas:", len(df))

    dados = preparar_dados(df)
    erros = baseline_persistencia(dados)
    baseline_media_movel(dados)
    analisar_erros_grandes(dados, erros)
    y, previsoes = regressao_linear(dados)
    grafico_real_vs_previsto(y, previsoes)


if __name__ == "__main__":
    main()
