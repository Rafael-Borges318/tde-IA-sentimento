from src.avaliacao import (
    avaliar_casos_dificeis,
    avaliar_final,
    comparar_configuracoes,
    escolher_melhor_configuracao,
)
from src.exportacao import explorar
from src.modelos import dividir_treino_teste

if __name__ == "__main__":
    explorar()

    textos_treino, textos_teste, classes_treino, classes_teste = dividir_treino_teste()

    tabela = comparar_configuracoes(textos_treino, classes_treino)
    print(tabela.round(3))

    melhor = escolher_melhor_configuracao(tabela)
    print("Configuração escolhida pela validação cruzada:", melhor)

    modelo_final = avaliar_final(
        melhor, textos_treino, textos_teste, classes_treino, classes_teste
    )
    avaliar_casos_dificeis(modelo_final)
