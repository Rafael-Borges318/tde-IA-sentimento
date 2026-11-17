import os

import pandas as pd


def explorar(caminho_csv="data/avaliacoes.csv",
             caminho_saida="resultados/exploracao.txt"):
    avaliacoes = pd.read_csv(caminho_csv)
    total_linhas = len(avaliacoes)

    # número de palavras de cada texto
    avaliacoes["qtd_palavras"] = avaliacoes["texto"].str.split().str.len()

    contagem_por_classe = avaliacoes["sentimento"].value_counts()
    media_palavras_por_classe = (
        avaliacoes.groupby("sentimento")["qtd_palavras"].mean()
    )

    linhas_relatorio = []
    linhas_relatorio.append("EXPLORAÇÃO DOS DADOS")
    linhas_relatorio.append("=" * 40)
    linhas_relatorio.append(f"Número de linhas: {total_linhas}")

    linhas_relatorio.append("\nDistribuição das classes:")
    for classe, quantidade in contagem_por_classe.items():
        percentual = quantidade / total_linhas
        linhas_relatorio.append(f"  {classe}: {quantidade} ({percentual:.1%})")

    linhas_relatorio.append(
        "\nTamanho medio dos textos por classe:"
    )
    for classe, media in media_palavras_por_classe.items():
        linhas_relatorio.append(f"  {classe}: {media:.2f}")

    linhas_relatorio.append("\n3 exemplos de cada classe:")
    for classe in sorted(avaliacoes["sentimento"].unique()):
        linhas_relatorio.append(f"\n  [{classe}]")
        avaliacoes_da_classe = avaliacoes[avaliacoes["sentimento"] == classe]
        exemplos = avaliacoes_da_classe.sample(3, random_state=42)
        for texto in exemplos["texto"]:
            linhas_relatorio.append(f"   - {texto}")

    pasta_saida = os.path.dirname(caminho_saida)
    os.makedirs(pasta_saida, exist_ok=True)
    with open(caminho_saida, "w", encoding="utf-8") as arquivo:
        arquivo.write("\n".join(linhas_relatorio) + "\n")

    return avaliacoes
