# E16: lote acima de B_simple

> Linha de pesquisa: questão 2 do instituto; fecha a H14b, que o E15 não alcançou. Critérios registrados no git antes da execução (fe356c4).

## Pergunta

No E15, o SNR do minilote ficou abaixo de 1 em todos os lotes, e a penalidade do canal de 1 bit não cresceu com o lote. A leitura posterior foi que nenhum lote do E15 passou de $`B_{\text{simple}}`$ (de cerca de 565 a 2.730). Com lotes de 8.192 e 32.768, o ruído do minilote fica bem abaixo do sinal. Nesse regime, o canal de 1 bit passa a ser o gargalo de cada projeção?

## Como foi feito

Mesma mecânica do E15. O script importa as funções de `ruido_do_minilote.py` sem alterá-las.

- Rede: N = 118.282, r = 32, 1500 passos, Adam lr 1e-3.
- Lote B em {8.192, 32.768}, com sorteio com reposição entre as 60.000 imagens de treino.
- Canais: float, ruído (b = 1, 2, 4) e quant (b = 1, 2).
- SNR do minilote medido como no E15. Duas sementes.

**Previsão (H14 aplicada a este regime).** Com $`\mathrm{SNR}_B`$ acima de 3, o canal de 1 bit domina o erro de cada projeção, $`b_{\text{ef}}`$ cai para perto de 1 bit e a penalidade em relação ao float cresce.

## Critérios

- **R0 (regime alcançado):** mediana de $`\mathrm{SNR}_B`$ em B = 32.768 de pelo menos 10 nas duas sementes. Se falhar, o E16 é inconclusivo.
- **H14b′ (a penalidade aparece):** pen(32.768) = acc(float) − acc(ruído, b = 1) ≥ pen do E15 em B = 128 (0,91 ponto) + 1 ponto.
- **H14 (com o E15):** aprovada se R0 e H14b′; reprovada se R0 vale e H14b′ não.
- **H10″ (informação efetiva em toda a faixa):** nas 23 células ruído/float do E15 e do E16, acc ~ a + c·log2($`b_{\text{ef}}`$) com R² ≥ 0,80. Registra-se também o R² com b nominal.
- **Inconclusivo adicional:** desvio entre sementes acima de 2 pontos em mais de 1/4 das células do E16.

## Resultado

**Inconclusivo: R0 reprovado.** A mediana de $`\mathrm{SNR}_B`$ em B = 32.768 foi 4,31 (semente 0) e 8,59 (semente 1), abaixo de 10. Pelo critério registrado, H14b′, H14 e H10″ não recebem veredito. A H14 continua parcial, como no E15.

| B | $`\mathrm{SNR}_B`$ (s0 / s1) | $`B_{\text{simple}}`$ | float | ruído b = 1 | 2 | 4 | quant b = 1 | 2 |
|---|---|---|---|---|---|---|---|---|
| 8.192 | 2,68 / 2,22 | ≈ 3.343 | 82,1 | 81,2 | 81,8 | 82,1 | 76,0 | 81,1 |
| 32.768 | 4,31 / 8,59 | ≈ 5.079 | 82,0 | 81,3 | 81,9 | 82,0 | 76,4 | 81,3 |

Acurácia de teste em %, média de 2 sementes. O desvio entre sementes ficou abaixo de 0,2 ponto em todas as células.

**Por que o regime não foi alcançado.** $`B_{\text{simple}}`$ cresce à medida que a perda cai (McCandlish et al., 2018), e os lotes maiores chegaram a perdas menores: de cerca de 565 em B = 128 (E15) para cerca de 5.079 em B = 32.768. O lote cresceu 256 vezes e $`B_{\text{simple}}`$, cerca de 9 vezes. Com 60.000 imagens de treino, não há lote muito maior disponível nesta tarefa.

**Leituras exploratórias (não são vereditos):**

- Nas 23 células ruído/float do E15 e do E16, $`\log_2 b_{\text{ef}}`$ explica a acurácia com $`R^2 = 0{,}83`$; com b nominal, $`R^2 = 0{,}008`$. O valor ficaria acima do limite de 0,80 da H10″, mas o critério não se aplica porque R0 falhou.
- Mesmo com $`\mathrm{SNR}_B`$ médio de 6,5, acima do SNR do canal de 1 bit, a penalidade desse canal foi 0,66 ponto (0,96 em B = 8.192). A acurácia do float parou em cerca de 82% nos dois lotes. Nesse ponto o gargalo passa a ser o número de projeções (r = 32), e nem o minilote nem o canal mudam muito a acurácia final.
- A quantização de 1 bit custou de 5,6 a 6,1 pontos, e a de 2 bits, de 0,7 a 1,1.

## Como reproduzir

```bash
poetry install --extras torch
make e16
```

- Sementes: as do E15 (s = 0, 1).
- Ambiente: `poetry.lock`, Python 3.12, CPU com 2 threads por processo.
