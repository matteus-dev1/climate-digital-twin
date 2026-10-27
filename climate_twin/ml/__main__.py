"""Executa o pipeline de ML no terminal com saída resumida.

Uso:
    python -m climate_twin.ml                    # resumo do dia com mais anomalias
    python -m climate_twin.ml --dia 2026-03-31   # resumo de um dia específico
    python -m climate_twin.ml --graficos         # abre também os gráficos
"""

import argparse
import sys

import matplotlib.pyplot as plt
import pandas as pd

from climate_twin.config import DURACAO_MINIMA
from climate_twin.ml.insights import formatar_resumo, resumir_periodo
from climate_twin.ml.model import comparar_com_baseline
from climate_twin.ml.pipeline import analisar, treinar
from climate_twin.ml.plots import figura_previsoes, figura_residuos


def data(texto):
    """Converte o --dia em Timestamp; data inválida vira erro do argparse, sem traceback."""
    try:
        return pd.Timestamp(texto)
    except ValueError:
        raise argparse.ArgumentTypeError(f"data inválida: {texto!r} (use AAAA-MM-DD)")


def main():
    # No Windows, a saída redirecionada usa cp1252, que não tem caracteres como "→"
    sys.stdout.reconfigure(encoding="utf-8")

    parser =argparse.ArgumentParser(prog="python -m climate_twin.ml",
                                    description="Previsão, anomalias e insights.")
    parser.add_argument("--dia", type=data, help="dia do resumo (AAAA-MM-DD); padrão: dia com mais anomalias")
    parser.add_argument("--graficos", action="store_true", help="abre os gráficos ao final")
    args = parser.parse_args()

    resultados = treinar()
    analise = analisar(resultados)

    print("Erro médio por horizonte (modelo vs baseline, em °C):")
    for h, res in resultados.items():
        m = comparar_com_baseline(res)
        vencedor = "modelo melhor" if m["modelo_ganhou"] else "baseline melhor"
        print(f"  +{h}h: {m['modelo']:.2f} vs {m['baseline']:.2f} ({vencedor})")

    print(f"\nAnomalias: {len(analise['indices'])} (limite de {analise['limite']:.2f} °C) | "
        f"eventos com {DURACAO_MINIMA}+ seguidas: {len(analise['eventos_filtrados'])}\n")

    resumo = resumir_periodo(analise["insights"], inicio=args.dia)
    print("\n".join(formatar_resumo(resumo)))

    if args.graficos:
        figura_previsoes(resultados)
        figura_residuos(analise["ts_alvo"], analise["residuos"],
                        analise["limite"], analise["indices"])
        plt.show()


if __name__ == "__main__":
    main()
