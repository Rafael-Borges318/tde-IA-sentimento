import re

from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer
)

import pandas as pd

STOP = {
    "o", "a", "os", "as", "um", "uma",
    "e", "é", "de", "do", "da", "que",
    "para", "com", "mas", "muito", "foi", "em"
}

def prep(texto):
    toks = re.findall(
        r"[a-záàâãéêíóôõúç]+",
        texto.lower()
    )

    return " ".join(
        palavra for palavra in toks
        if palavra not in STOP
    )