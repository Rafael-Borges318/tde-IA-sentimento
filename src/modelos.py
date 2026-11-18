from functools import partial

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.processamento import preprocessar

CONFIGURACOES = {
    "A": (False, CountVectorizer),
    "B": (False, TfidfVectorizer),
    "C": (True, CountVectorizer),
    "D": (True, TfidfVectorizer),
}


def dividir_treino_teste(caminho_csv="data/avaliacoes.csv"):
    avaliacoes = pd.read_csv(caminho_csv)

    textos_treino, textos_teste, classes_treino, classes_teste = train_test_split(
        avaliacoes["texto"],
        avaliacoes["sentimento"],
        test_size=0.25,
        stratify=avaliacoes["sentimento"],
        random_state=42,
    )
    return textos_treino, textos_teste, classes_treino, classes_teste


def montar_pipeline(nome_config):
    remover_stopwords, classe_vetorizador = CONFIGURACOES[nome_config]

    vetorizador = classe_vetorizador(
        preprocessor=partial(preprocessar, remover_stopwords=remover_stopwords),
        token_pattern=r"\S+",
    )

    return Pipeline([
        ("vetorizador", vetorizador),
        ("modelo", LogisticRegression(max_iter=1000)),
    ])


def tamanho_vocabulario(pipeline_treinado):
    return len(pipeline_treinado.named_steps["vetorizador"].vocabulary_)
