# E14: perturbação real, sem backpropagation

> Linha de pesquisa: questão 1 do instituto (como as projeções foram calculadas e o que entra no custo). Critérios registrados no git antes da execução (88dca6c).

## Pergunta

No E11 e no E12, cada projeção $v^\top g$ saiu do gradiente exato, calculado por backpropagation. Um hardware sem backprop não tem esse gradiente. Ele precisa medir a projeção perturbando os pesos e lendo a perda. O resultado do E11 sobrevive a essa troca? E o que passa a custar caro?

## Como foi feito

Não há passo de volta. Cada projeção é estimada por diferença central da perda no mesmo minilote:

$$
p_i = \frac{L(\theta + \varepsilon v_i) - L(\theta - \varepsilon v_i)}{2\varepsilon}, \qquad \|v_i\| = 1,\ \varepsilon = 10^{-2}
$$

Isso custa 2 avaliações da perda (passes de ida) por projeção, ou seja, $2r$ por passo. O Adam recebe $V p$, como no E11. Mesmas redes, sementes, lotes e direções do E11. O valor de $\varepsilon$ foi escolhido antes dos critérios, numa calibração com correlação de 0,9999 entre a diferença central e a projeção exata ([auxiliares/calibrar_eps.log](auxiliares/calibrar_eps.log)).

| Leitura | O que faz | Bits por passo |
|---|---|---|
| `exata` | $V^\top g$ por backprop (controle, deve repetir o E11) | $32r$ |
| `dif` | diferença central em float32 | $64r$ |
| `abs-b` | cada perda é lida por um conversor de b bits na faixa fixa [0, 4], e a diferença é feita depois | $2rb$ |
| `difer-b` | a diferença é lida diretamente (leitura diferencial, como num lock-in), com faixa adaptada a cada passo e quantização estocástica (o canal `quant` do E12) | $rb + 32$ |

Grade: N em {13.002, 118.282}, r em {8, 32, 128}, b em {8, 12, 16, 20} para `abs` e {1, 2, 4} para `difer`, 1500 passos, 2 sementes.

**Previsão escrita antes de rodar.** A leitura absoluta precisa resolver uma diferença de $2\varepsilon |v^\top g|$, da ordem de $10^{-5}$ a $10^{-4}$, numa faixa de 4. Isso pede uns 15 a 17 bits. A leitura diferencial deve herdar o E12: 2 bits bastam.

## Critérios

Médias das 2 sementes; célula = (N, r).

- **P0 (controle):** `exata` repete o E11 com diferença de no máximo 0,5 ponto em todas as 6 células.
- **P1 (diferença finita basta):** `dif` fica a no máximo 1 ponto de `exata` em pelo menos 5 das 6 células.
- **P2 (leitura absoluta exige muitos bits):** o menor b com acc(`abs-b`) ≥ acc(`dif`) − 1 ponto é de 16 ou mais (ou nenhum b basta) em pelo menos 5 das 6 células.
- **P3 (leitura diferencial herda o E12):** acc(`difer-2`) ≥ acc(`dif`) − 1,5 ponto em pelo menos 5 das 6 células.
- **P4 (sem penalidade em N com medida real):** acc(118.282) ≥ acc(13.002) − 1 ponto em pelo menos 5 das 6 combinações de leitura (`dif`, `difer-2`) e r.
- **Inconclusivo:** desvio entre sementes acima de 2 pontos em mais de 1/4 das células.

## Resultado

**P0, P1, P2, P3 e P4 aprovados, todos em 6 de 6 células.** O critério de inconclusivo não disparou.

Acurácia de teste (%), média de 2 sementes:

| N | r | exata | dif | abs 8 | abs 12 | abs 16 | abs 20 | difer 1 | difer 2 | difer 4 |
|---|---|---|---|---|---|---|---|---|---|---|
| 13.002 | 8 | 69,6 | 69,5 | 27,7 | 56,4 | 69,6 | 69,5 | 61,6 | 70,4 | 69,6 |
| 13.002 | 32 | 75,8 | 75,8 | 34,3 | 70,3 | 75,7 | 75,7 | 69,4 | 74,8 | 75,9 |
| 13.002 | 128 | 78,8 | 78,9 | 48,5 | 75,8 | 78,8 | 78,8 | 73,7 | 78,1 | 78,7 |
| 118.282 | 8 | 74,2 | 74,1 | 27,4 | 59,9 | 73,6 | 74,0 | 69,6 | 73,0 | 74,2 |
| 118.282 | 32 | 77,6 | 77,6 | 34,0 | 69,7 | 77,4 | 77,5 | 73,1 | 76,5 | 77,5 |
| 118.282 | 128 | 80,7 | 80,7 | 53,7 | 73,8 | 80,3 | 80,7 | 76,1 | 80,1 | 80,7 |

- **P0:** a leitura `exata` repetiu o E11 com diferença 0,0 nas 6 células.
- **P1:** a diferença central, sem backpropagation, ficou a no máximo 0,13 ponto da projeção exata em todas as células.
- **P2:** o menor b da leitura absoluta que chega a 1 ponto da diferença float32 foi 16 em todas as 6 células. Com 8 bits a rede fica entre 27 e 54%; com 12 bits perde de 3 a 14 pontos. A previsão registrada antes era da ordem de 16 bits. A conta exploratória com a mediana medida de $|v^\top g|$, $\log_2(4/(2\varepsilon\,\mathrm{med}|p|))$, dá de 14,8 a 15,2 bits na rede menor e de 16,2 a 16,5 na maior.
- **P3:** com a leitura diferencial de 2 bits, a perda máxima foi 1,1 ponto (rede maior, r = 32). Com 1 bit, a perda vai de 4,5 a 7,9 pontos; com 4 bits, empata.
- **P4:** a rede de 118.282 parâmetros não perdeu acurácia em relação à de 13.002 em nenhuma das 6 combinações (dif e difer-2 × r).

**Custo contado.** Por passo, a leitura exata custa 1 ida e 1 volta (contadas como 3 idas), e a diferença central custa $2r$ idas. Em 1500 passos: 4.500 idas contra 24.000 (r = 8), 96.000 (r = 32) e 384.000 (r = 128). Em computador digital, a medida por perturbação custa de 5 a 85 vezes mais avaliações da perda que o backpropagation para a mesma acurácia. A vantagem só aparece se a avaliação física da perda for muito mais barata que a digital.

**Leituras e bits.** A leitura absoluta faz $2r$ conversões de 16 bits por passo ($32r$ bits); a diferencial faz $r$ conversões de 2 bits mais a escala ($2r + 32$ bits). Para r = 128, são 6,1 milhões de bits contra 0,43 milhão em 1500 passos, com a mesma acurácia a menos de 1 ponto. A diferença de energia por leitura é maior que a de bits, porque a energia de um conversor na fronteira quadruplica a cada bit acima de cerca de 9 bits (ver [leituras](../../docs/08-leituras.md)).

Limites que continuam: a perda é calculada em float32 digital, e o único ruído físico simulado é a quantização do conversor. Não há ruído de avaliação, deriva nem não linearidade de dispositivo.

## Como reproduzir

```bash
poetry install --extras torch
make e14
```

- Sementes: as do E11 (pesos `manual_seed(s)`, direções `Generator().manual_seed(1000 + s)`, lotes `default_rng(s)`), mais o conversor estocástico `Generator().manual_seed(2000 + s)`; s = 0, 1.
- Ambiente: `poetry.lock` (torch 2.15.0, numpy 2.5.4), Python 3.12, CPU com 2 threads por processo; as duas sementes rodam em processos paralelos.
- Nota de registro: os dois processos oficiais foram iniciados alguns segundos antes do commit 88dca6c, por um erro na cadeia de comandos. O script executado é idêntico ao commitado (`git diff` vazio), e nenhum resultado existia no momento do commit.
