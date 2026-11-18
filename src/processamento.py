import re 


STOPWORDS = {
    "o", "a", "os", "as", "um", "uma",
    "e", "é", "de", "do", "da", "que",
    "para", "com", "mas", "muito", "foi", "em"
}

def preprocessar(texto, remover_stopwords=True):
    palavras = re.findall(
        r"[a-záàâãéêíóôõúçü]+",
        texto.lower()
    )

    if remover_stopwords:
        palavras = [p for p in palavras if p not in STOPWORDS]

    return " ".join(palavras)

