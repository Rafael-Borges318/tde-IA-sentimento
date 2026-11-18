# RELATORIO — Análise de Sentimento com PLN

**Nome:** Rafael Borges de Souza Lima
**Repositório:** https://github.com/Rafael-Borges318/tde-IA-sentimento
**Data:** 17/11/2026

> Responda cada pergunta com **4 a 12 linhas**, com suas palavras e usando números dos arquivos de `resultados/`. Apague as instruções em itálico ao entregar.

## Tabela-resumo (preencha com os seus resultados)

| Config | Stopwords | Vetorizador | Vocabulário | Acurácia CV (média +- desvio) | F1 CV (média +- desvio) |
|---|---|---|---|---|---|
| A | mantidas | BoW | 425 | 0,841 +- 0,071 | 0,840 +- 0,072 |
| B | mantidas | TF-IDF | 425 | 0,821 +- 0,092 | 0,820 +- 0,092 |
| C | removidas | BoW | 407 | 0,841 +- 0,092 | 0,841 +- 0,092 |
| D | removidas | TF-IDF | 407 | 0,851 +- 0,091 | 0,851 +- 0,091 |

**Configuração escolhida:** D **Acurácia no teste:** ____ **F1 no teste:** ____

---

## 1. Pré-processamento
*Quais decisões você tomou (minúsculas, pontuação, números, stopwords)? Qual o tamanho do vocabulário com e sem remoção de stopwords? O que você fez com a palavra "não" e por quê?*

primeiro passei todo o texto para minusculo, depois usei uma expressao regular que pega so letras, entao pontuacao e numeros somem sozinhos. depois criei uma lista propria de stopwords com 18 palavras. a remocao de stopwords é ligada ou desligada por um parametro da funcao preprocessar. O vocabulario aprendido no treino ficou com 425 palavras mantendo as stopwords e 407 removendo, ou seja caiu 18 palavras. a palavra "não" eu deixei de fora da lista, entao ela é mantida. fiz isso porque ela inverte o sentido: sem ela, "gostei do produto" e "não gostei do produto" ficariam quase iguais e o modelo nao ia saber diferenciar.


## 2. BoW vs. TF-IDF
*Compare as quatro configurações levando em conta o desvio padrão. As diferenças são relevantes? Qual você escolheu e com base em quê? Explique a diferença conceitual entre as duas representações.*

Na validacao cruzada as acuracias medias foram 0,841 (A), 0,821 (B), 0,841 (C) e 0,851 (D). o desvio padrao ficou entre 0,07 e 0,09, bem maior que a diferenca entre as configuracoes, entao as diferencas não são relevantes, o resultado e praticamente um empate. mesmo assim escolhi a D porque teve a maior media na validacao cruzada, e nao olhei o teste pra decidir. o bow so conta quantas vezes cada palavra aparece em cada texto. ja o tf-idf da menos peso pra palavras que aparecem em muitos textos e mais peso pra palavras mais raras, que tem mais caracteristicas de um texto.


## 3. Avaliação no teste
*Mostre a matriz de confusão e interprete precisão, recall e F1 de cada classe. Para uma loja, qual erro é mais grave: classificar uma reclamação como elogio ou o contrário? Por que a acurácia é menos enganosa aqui do que no exemplo de fraude da Aula 8?*

a configuracao D teve acuracia de 0,831 no teste, um pouco abaixo dos 0,851 da validacao cruzada. na matriz de confusao, dos 33 negativos reais o modelo acertou 26 e errou 7, foi previsto positivo. dos 32 positivos reais acertou 28 e errou 4, previu negatico. pra classe negativo a precisao foi 0,867, o recall 0,788 e o f1 0,825. pra classe positivo a precisao foi 0,800, o recall 0,875 e o f1 0,836. pra uma loja, o erro mais grave e classificar reclamacao como elogio, porque o cliente insatisfeito fica sem resposta. o contrario so gera um atendimento a toa. a acuracia aqui engana menos porque as classes sao balanceadas


## 4. Validação cruzada vs. divisão única
*Por que a validação cruzada é mais confiável do que um único conjunto de teste para comparar configurações? Por que a escolha não pode ser feita olhando o teste? O que seria vazamento de dados neste trabalho e como você o evitou?*

a validacao cruzada testa o modelo em 5 divisoes diferentes do treino e tira a media, entao o resultado depende menos da sorte de uma divisao so. com um unico teste de 65 textos, um ou dois acertos a mais ja mudam bastante o numero. a escolha nao pode ser feita olhando o teste porque ai o teste deixa de ser dado novo, e a nota final fica otimista. vazamento de dados aqui seria ajustar o vetorizador com o dataset inteiro, porque o vocabulario ja carregaria informacao do teste. eu evitei usando um pipeline, que reajusta o vetorizador so com o treino de cada fold, e usei o teste so no final uma vez.

## 5. Análise de erros e palavras influentes
*Escolha 3 textos de `erros.csv` e explique por que o modelo pode ter errado. Observe as 10 palavras mais influentes de cada classe: há alguma que não carrega sentimento por si só (ex.: palavra funcional ou de domínio)? O que isso diz sobre o que o modelo realmente aprendeu?*

o modelo errou 11 textos e quase todos com probabilidade perto de 0,5, ele estava em duvida. em "a qualidade é muito boa" a palavra "muito" foi tirada como stopword e sobrou "qualidade boa", e "qualidade" aparece em varios negativos do treino, como "péssima qualidade". em "camiseta encolheu e desbotou depois da primeira lavagem" (negativo, previsto positivo) a palavra "encolheu" aparece no treino em "não encolheu depois da lavagem", que e positivo. em "cheguei animado, mas me arrependi, tudo estava sujo" (negativo, previsto positivo) a palavra "tudo" tem peso positivo por causa de "tudo certo" e "tudo perfeito". nas palavras mais influentes, tem varias que nao carregam sentimento sozinhas: "tudo", "dez", "bem", "mais", "toda" e "dinheiro". "bastante" e a mais positiva provavelmente porque "gostei bastante" aparece umas 10 vezes no dataset. a palavra "não" foi a mais negativa, o que mostra que mante-la foi uma boa decisao. isso mostra que o modelo aprendeu tambem padroes do dataset e nao so sentimento de verdade.


## 6. Casos difíceis e limitações
*Com base em `casos_dificeis_predicoes.csv`, como o modelo se saiu em negação, ironia e frases mistas? Explique por que BoW/TF-IDF têm essa limitação, proponha duas melhorias (cite técnicas vistas na Aula 11) e discuta duas limitações do dataset e um risco de usar este modelo em produção.*

dos 10 casos com resposta positivo ou negativo o modelo acertou 8. na negacao acertou 3 de 4: não gostei nada do atendimento, não é ruim, gostei bastante e não tenho nada a reclamar deram certo, mas sem defeitos, funciona direitinho saiu negativo. na ironia acertou 2 de 3, mas com probabilidades perto de 0,5, sendo 0,48 e 0,43, o que parece mais sorte, e que ótimo, o produto chegou quebrado saiu positivo com 0,667 porque ótimo pesa muito. as frases mistas, como o produto é bom, mas a entrega foi péssima, foram previstas como positivo com 0,59, e o neutro chegou ontem também como positivo com 0,515, pois o modelo so conhece 2 classes. bow e tf-idf so contam palavras, entao ignoram ordem e contexto e nao entendem que que ótimo e sarcasmo. duas melhorias seriam usar n-gramas, por exemplo bigramas como não gostei, e tratar a negacao antes de vetorizar, ou usar embeddings pra pegar contexto. sobre o dataset: ele e pequeno, com 260 textos, e tem frases repetidas e geradas por padrao, entao os resultados sao otimistas, e nao tem exemplos neutros nem mistos. em producao o risco e tratar uma reclamacao irônica ou mista como elogio e deixar um cliente sem resposta.


---

## Declaração de uso de IA

Marque uma opção e complete:

- [x] Não usei IA generativa em nenhuma parte deste trabalho.
- [x] Usei IA generativa **somente** para: para me guiar e criar modelos para mim me basear e colocar no codigo de forma "refatorada". (não é permitido gerar código ou relatório).