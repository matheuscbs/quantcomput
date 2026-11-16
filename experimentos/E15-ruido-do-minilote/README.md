# E15: ruído do minilote e informação efetiva por medida

> Linha de pesquisa: questão 2 do instituto (medidas, precisão e informação); H14 e a forma corrigida de H10. Critérios registrados no git antes da execução (88dca6c). No roteiro anterior, este teste se chamava E13d.

## Pergunta

No E12, 1 a 2 bits por projeção bastaram. A explicação proposta (H14) é que o minilote já põe tanto ruído em cada projeção que o ruído do canal abaixo dele não aparece. Se for isso, a informação útil de uma medida é limitada pelo pior dos dois ruídos, e é ela, não a capacidade nominal do canal, que deveria explicar a acurácia.

## Como foi feito

Cada projeção recebida tem dois ruídos. Com sinal $S = \mathbb{E}[(v^\top g)^2]$ (gradiente da perda inteira), variância do minilote $\sigma_B^2$ e canal gaussiano de $\mathrm{SNR}_c = 2^{2b} - 1$:

$$
\mathrm{SNR}_B = \frac{S}{\sigma_B^2}, \qquad
\mathrm{SNR}_{\text{ef}} = \frac{1}{1/\mathrm{SNR}_B + (1 + 1/\mathrm{SNR}_B)/\mathrm{SNR}_c}, \qquad
b_{\text{ef}} = \tfrac12 \log_2(1 + \mathrm{SNR}_{\text{ef}})
$$

$\mathrm{SNR}_B$ é medido durante o treino: a cada 100 passos, as r direções daquele passo são aplicadas aos gradientes de 8 minilotes independentes (gerador separado, sem mexer na trajetória). A variância entre eles dá $\sigma_B^2$.

Grade: N = 118.282, r = 32, lote B em {128, 512, 2048}, canais `float`, `ruido` (b = 1, 2, 4, 8) e `quant` (b = 1, 2, 4), iguais aos do E12; 1500 passos, 2 sementes.

## Critérios

Médias das 2 sementes.

- **H14a (o minilote domina em B = 128):** mediana de $\mathrm{SNR}_B$ abaixo de 3 (o SNR do canal de 1 bit) nas duas sementes.
- **H14b (a penalidade do canal aparece com o minilote limpo):** pen(B) = acc(float) − acc(ruído, b = 1); aprovado se pen(2048) ≥ pen(128) + 1 ponto.
- **H14:** aprovada se H14a e H14b; reprovada se nenhum; parcial se só um.
- **H10′ (informação efetiva governa):** nas 15 células (3 B × {float, ruído b = 1, 2, 4, 8}), acc ~ a + c·log2($b_{\text{ef}}$) com R² ≥ 0,80. Registra-se também o R² com b nominal.
- **Inconclusivo:** desvio entre sementes acima de 2 pontos em mais de 1/4 das células; ou amplitude das 15 acurácias abaixo de 2 pontos (só para H10′); ou sinal estimado $S \le 0$ em mais da metade das medidas de algum B.

## Resultado

**H14 parcial (H14a aprovado, H14b reprovado); H10′ aprovada.** Nenhum critério de inconclusivo disparou.

SNR do minilote por projeção (mediana dos 15 pontos de medida, execução float):

| Lote B | semente 0 | semente 1 | média | $B_{\text{simple}} = B/\mathrm{SNR}_B$ |
|---|---|---|---|---|
| 128 | 0,212 | 0,241 | 0,227 | ≈ 565 |
| 512 | 0,310 | 0,348 | 0,329 | ≈ 1.556 |
| 2048 | 0,839 | 0,662 | 0,750 | ≈ 2.729 |

Acurácia de teste (%), média de 2 sementes, com $b_{\text{ef}}$ entre parênteses:

| B | float | ruído b = 1 | 2 | 4 | 8 | quant b = 1 | 2 | 4 |
|---|---|---|---|---|---|---|---|---|
| 128 | 77,6 (0,15) | 76,7 (0,11) | 77,3 (0,14) | 77,5 (0,15) | 77,6 (0,15) | 73,3 | 76,7 | 77,4 |
| 512 | 79,7 (0,21) | 79,1 (0,15) | 79,7 (0,19) | 79,7 (0,20) | 79,8 (0,21) | 75,3 | 79,0 | 79,9 |
| 2048 | 81,1 (0,40) | 80,5 (0,28) | 81,1 (0,37) | 81,2 (0,40) | 81,2 (0,40) | 75,8 | 80,6 | 81,3 |

- **H14a:** o SNR do minilote ficou abaixo de 3 em todos os lotes, não só em B = 128. Cada projeção carrega mais ruído de amostragem que sinal.
- **H14b:** a penalidade do canal de 1 bit não cresceu com o lote: 0,91 ponto em 128, 0,59 em 512 e 0,65 em 2048. A leitura posterior (exploratória) é que mesmo B = 2048 ficou abaixo de $B_{\text{simple}}$, ou seja, o teste não alcançou o regime em que o canal deveria pesar. O E16 testa esse regime.
- **H10′:** $\log_2 b_{\text{ef}}$ explica a acurácia das 15 células com $R^2 = 0{,}887$. Com b nominal no lugar, $R^2 = 0{,}018$. A amplitude das acurácias foi de 4,5 pontos, acima do mínimo de 2.
- O lote maior aumenta a acurácia (77,6 → 79,7 → 81,1% no float) com o mesmo número de projeções. A quantização de 1 bit custa de 4,3 a 5,3 pontos, e 2 bits custam de 0,5 a 0,9.
- O float com B = 128 repete o E12 (77,6% com r = 32 e N = 118.282).

A relação $\mathrm{SNR}_B = B/B_{\text{simple}}$ vem de aplicar a Eq. 2.10 de McCandlish et al. (2018) a uma direção aleatória de norma unitária; ver [leituras](../../docs/08-leituras.md).

## Como reproduzir

```bash
poetry install --extras torch
make e15
```

- Sementes: pesos `manual_seed(s)`, direções `Generator().manual_seed(1000 + s)`, lotes `default_rng(s)`, canal `Generator().manual_seed(2000 + s)`, lotes extras da medida de SNR `default_rng(3000 + s)`; s = 0, 1.
- Ambiente: `poetry.lock`, Python 3.12, CPU com 2 threads por processo.
- Nota de registro: igual à do E14 (processos iniciados segundos antes do commit 88dca6c, com script idêntico).
