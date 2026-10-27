"""Insights em linguagem natural: WHAT, WHY, WHAT MAY HAPPEN, severidade e resumo."""

import pandas as pd

from climate_twin.config import (
    LIMITE_CHUVA, LIMITE_RADIACAO, LIMITE_TENDENCIA, LIMITE_VARIACAO, LIMITE_VENTO,
    RAZAO_SEVERIDADE_ALTA, RAZAO_SEVERIDADE_MEDIA,
)

SEVERIDADES = ("ALTA", "MÉDIA", "BAIXA")


# WHAT: o que aconteceu
def gerar_insight_what(timestamp, residuo, real, previsto):
    """Transforma uma anomalia em uma frase legível."""
    quando = timestamp.strftime("%d/%m/%Y %H:%M")
    return (
        f"[{quando}] ALERTA: temperatura real de {real:.1f}°C, "
        f"mas o modelo previa {previsto:.1f}°C "
        f"(erro de {residuo:.1f}°C)."
    )


# WHY: causa provável
def classificar_causa(alvo, temp_leitura, real, previsto):
    """Causa provável por regras simples; a primeira regra verdadeira vence.

    `alvo` traz as condições no horário da anomalia (colunas *_alvo), não as da leitura.
    """
    erro = real - previsto           # > 0: mais quente que o previsto
    variacao = real - temp_leitura   # mudança da leitura até o horário alvo

    if alvo["precipitation_alvo"] >= LIMITE_CHUVA and erro < 0:
        return "chuva esfriando o ar"
    elif alvo["solar_radiation_alvo"] >= LIMITE_RADIACAO and erro > 0:
        return "radiação solar forte aquecendo mais que o previsto"
    elif alvo["wind_speed_alvo"] >= LIMITE_VENTO:
        return "vento forte alterando a temperatura"
    elif abs(variacao) >= LIMITE_VARIACAO:
        return "variação brusca em relação à hora anterior"
    else:
        return "sem causa clara nas outras variáveis"


# WHAT MAY HAPPEN: tendência a partir das previsões de +1h, +2h e +3h
def indexar_previsoes(resultados):
    """Devolve {h: previsões indexadas pelo horário da leitura}.

    O teste de cada horizonte tem tamanho diferente (o shift(-h) remove h linhas),
    então o alinhamento entre horizontes é feito pelo timestamp, não pela posição.
    """
    return {
        h: pd.Series(res["previsoes"], index=res["timestamps_teste"])
        for h, res in resultados.items()
    }


def descrever_tendencia(temp_atual, previsoes):
    """Compara a temperatura da leitura com a previsão do horizonte mais distante."""
    if not previsoes:
        return "Tendência indisponível"

    delta = previsoes[max(previsoes)] - temp_atual

    if delta >= LIMITE_TENDENCIA:
        sentido = "alta"
    elif delta <= -LIMITE_TENDENCIA:
        sentido = "queda"
    else:
        sentido = "estabilidade"

    trajeto = " → ".join(f"+{h}h: {t:.1f}°C" for h, t in sorted(previsoes.items()))
    return f"Tendência de {sentido} ({trajeto})"


def classificar_severidade(residuo, limite):
    """Classifica pelo quanto o erro passou do limite de anomalia."""
    razao = residuo / limite
    if razao >= RAZAO_SEVERIDADE_ALTA:
        return "ALTA"
    elif razao >= RAZAO_SEVERIDADE_MEDIA:
        return "MÉDIA"
    else:
        return "BAIXA"


def montar_insights(resultados, residuos, indices_anomalias, limite):
    """Cria um dicionário de insight para cada anomalia detectada."""
    res_h1 = resultados[1]
    previsoes_indexadas = indexar_previsoes(resultados)
    insights = []

    for pos in indices_anomalias:  # pos = posição da anomalia dentro do teste
        linha = res_h1["X_teste"].iloc[pos]
        real = res_h1["y_teste"][pos]
        previsto = res_h1["previsoes"][pos]
        residuo = residuos[pos]
        leitura = res_h1["timestamps_teste"].iloc[pos]
        alvo = res_h1["timestamps_alvo_teste"].iloc[pos]

        condicoes_alvo = res_h1["condicoes_alvo_teste"].iloc[pos]
        causa = classificar_causa(condicoes_alvo, linha["temperature"], real, previsto)
        previsoes = {h: s[leitura] for h, s in previsoes_indexadas.items()
                    if leitura in s.index}
        tendencia = descrever_tendencia(linha["temperature"], previsoes)

        frase = (f"{gerar_insight_what(alvo, residuo, real, previsto)} "
                f"Causa provável: {causa}. {tendencia}.")

        insights.append({
            "timestamp": alvo,
            "real": real,
            "previsto": previsto,
            "residuo": residuo,
            "severidade": classificar_severidade(residuo, limite),
            "causa": causa,
            "tendencia": tendencia,
            "frase": frase,
        })

    return insights


def dia_com_mais_anomalias(insights):
    """Devolve a data (à meia-noite) com mais anomalias."""
    datas = pd.Series([i["timestamp"] for i in insights]).dt.normalize()
    return datas.value_counts().idxmax()


def resumir_periodo(insights, inicio=None, fim=None, qtd=3):
    """Resume as anomalias de um período e devolve um dicionário.

    Sem `inicio`, usa o dia com mais anomalias. Sem `fim`, o período dura 1 dia
    (`fim` é exclusivo). Com total == 0, só `titulo` e `total` estão presentes.
    """
    if not insights:
        return {"titulo": "", "total": 0}

    if inicio is None:
        inicio = dia_com_mais_anomalias(insights)
    inicio = pd.Timestamp(inicio)
    fim = inicio + pd.Timedelta(days=1) if fim is None else pd.Timestamp(fim)

    if fim - inicio == pd.Timedelta(days=1):
        titulo = f"{inicio:%d/%m/%Y}"
    else:
        titulo = f"{inicio:%d/%m/%Y} a {fim - pd.Timedelta(days=1):%d/%m/%Y}"

    selecionados = [i for i in insights if inicio <= i["timestamp"] < fim]
    if not selecionados:
        return {"titulo": titulo, "total": 0}

    horarios = [i["timestamp"] for i in selecionados]
    causas = pd.Series([i["causa"] for i in selecionados]).value_counts()
    severidades = pd.Series([i["severidade"] for i in selecionados]).value_counts()

    return {
        "titulo": titulo,
        "total": len(selecionados),
        "primeiro": min(horarios),
        "ultimo": max(horarios),
        "por_severidade": {s: int(severidades.get(s, 0)) for s in SEVERIDADES},
        "causa_principal": causas.index[0],
        "qtd_causa_principal": int(causas.iloc[0]),
        "mais_graves": sorted(selecionados, key=lambda i: i["residuo"], reverse=True)[:qtd],
    }


def formatar_resumo(resumo):
    """Transforma o dicionário de resumir_periodo em linhas de texto para o terminal."""
    if resumo["total"] == 0:
        if resumo["titulo"]:
            return [f"Resumo de {resumo['titulo']}: nenhuma anomalia."]
        return ["Nenhuma anomalia detectada."]

    por_severidade = ", ".join(
        f"{qtd} {sev}" for sev, qtd in resumo["por_severidade"].items() if qtd)
    linhas = [
        f"===== RESUMO DE {resumo['titulo']} =====",
        f"{resumo['total']} anomalias entre {resumo['primeiro']:%d/%m %H:%M} "
        f"e {resumo['ultimo']:%d/%m %H:%M} ({por_severidade}).",
        f"Causa mais frequente: {resumo['causa_principal']} "
        f"({resumo['qtd_causa_principal']} de {resumo['total']}).",
        f"As {len(resumo['mais_graves'])} mais graves:",
    ]
    linhas += [f"  [{i['severidade']}] {i['frase']}" for i in resumo["mais_graves"]]
    return linhas
