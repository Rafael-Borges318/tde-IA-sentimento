from src.exportacao import explorar
from src.processamento import preprocessar

if __name__ == "__main__":
    explorar()

    print(preprocessar("Não gostei do produto, 10 de 10!"))
    # o esperado é "não gostei produto"
    print(preprocessar("Não gostei do produto", remover_stopwords=False))
    # o esperado é "não gostei do produto"
