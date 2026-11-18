import os

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.modelos import CONFIGURACOES, montar_pipeline, tamanho_vocabulario


def comparar_configuracoes(textos_treino, classes_treino,
                           caminho_saida="resultados/comparacao.csv"):
    # cv so nos dados de treino
    validacao = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metricas = {"acuracia": "accuracy", "f1": "f1_macro"}  # f1 macro: rotulos em texto

    linhas = []
    for nome, (remover_stopwords, classe_vetorizador) in CONFIGURACOES.items():
        pipeline = montar_pipeline(nome)

        resultado = cross_validate(
            pipeline, textos_treino, classes_treino,
            cv=validacao, scoring=metricas,
        )

        # fit so para contar vocabulario
        pipeline.fit(textos_treino, classes_treino)

        linhas.append({
            "config": nome,
            "stopwords": "removidas" if remover_stopwords else "mantidas",
            "vetorizador": "BoW" if classe_vetorizador is CountVectorizer else "TF-IDF",
            "vocabulario": tamanho_vocabulario(pipeline),
            "acc_cv_media": resultado["test_acuracia"].mean(),
            "acc_cv_desvio": resultado["test_acuracia"].std(),
            "f1_cv_media": resultado["test_f1"].mean(),
            "f1_cv_desvio": resultado["test_f1"].std(),
        })

    tabela = pd.DataFrame(linhas)

    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    tabela.round(4).to_csv(caminho_saida, index=False)
    return tabela


def escolher_melhor_configuracao(tabela):
    # escolha pelo cv, nunca pelo teste
    ordenada = tabela.sort_values(["acc_cv_media", "f1_cv_media"], ascending=False)
    return ordenada.iloc[0]["config"]


def avaliar_final(nome_config, textos_treino, textos_teste, classes_treino,
                  classes_teste, pasta_saida="resultados"):
    # teste usado uma unica vez
    pipeline = montar_pipeline(nome_config)
    pipeline.fit(textos_treino, classes_treino)

    previstas = pipeline.predict(textos_teste)
    probabilidades = pipeline.predict_proba(textos_teste)
    nomes_classes = list(pipeline.classes_)

    relatorio = classification_report(classes_teste, previstas, digits=3)
    with open(f"{pasta_saida}/relatorio_classificacao.txt", "w", encoding="utf-8") as arquivo:
        arquivo.write(f"configuracao: {nome_config}\n\n{relatorio}")

    matriz = confusion_matrix(classes_teste, previstas, labels=nomes_classes)
    pd.DataFrame(
        matriz,
        index=[f"real_{c}" for c in nomes_classes],
        columns=[f"previsto_{c}" for c in nomes_classes],
    ).to_csv(f"{pasta_saida}/matriz_confusao.csv")

    erros = []
    for texto, real, prevista, probs in zip(textos_teste, classes_teste,
                                            previstas, probabilidades):
        if real != prevista:
            erros.append({
                "texto": texto,
                "real": real,
                "previsto": prevista,
                "probabilidade": round(probs.max(), 3),
            })
    pd.DataFrame(erros, columns=["texto", "real", "previsto", "probabilidade"]).to_csv(
        f"{pasta_saida}/erros.csv", index=False
    )

    salvar_palavras_influentes(pipeline, pasta_saida)
    return pipeline


def salvar_palavras_influentes(pipeline, pasta_saida="resultados"):
    palavras = pipeline.named_steps["vetorizador"].get_feature_names_out()
    # coef positivo empurra para positivo
    coeficientes = pipeline.named_steps["modelo"].coef_[0]

    indices = coeficientes.argsort()
    mais_positivas = indices[::-1][:10]
    mais_negativas = indices[:10]

    linhas = []
    for j in mais_positivas:
        linhas.append({"direcao": "positivo", "palavra": palavras[j],
                       "coeficiente": round(coeficientes[j], 3)})
    for j in mais_negativas:
        linhas.append({"direcao": "negativo", "palavra": palavras[j],
                       "coeficiente": round(coeficientes[j], 3)})

    pd.DataFrame(linhas).to_csv(f"{pasta_saida}/palavras_influentes.csv", index=False)


def avaliar_casos_dificeis(pipeline, caminho_csv="data/casos_dificeis.csv",
                           pasta_saida="resultados"):
    casos = pd.read_csv(caminho_csv)
    posicao_positivo = list(pipeline.classes_).index("positivo")
    prob_positivo = pipeline.predict_proba(casos["texto"])[:, posicao_positivo]

    saida = pd.DataFrame({
        "texto": casos["texto"],
        "tipo": casos["tipo"],
        "esperado": casos["sentimento_esperado"],
        "previsto": pipeline.predict(casos["texto"]),
        "prob_positivo": prob_positivo.round(3),
    })
    saida.to_csv(f"{pasta_saida}/casos_dificeis_predicoes.csv", index=False)
    return saida
