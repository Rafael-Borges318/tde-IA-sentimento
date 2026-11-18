import os

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
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
