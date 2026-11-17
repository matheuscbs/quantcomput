# E13: desenho fatorial para definir I

> Linha de pesquisa: questão 3 do instituto (como definir I separando tarefa, arquitetura, inicialização e algoritmo); H11 e H12. Critérios registrados no git antes da execução (d033730).

## Pergunta

A definição proposta na nota técnica é operacional: $`I_{\mathcal{A}}(\varepsilon \mid \text{tarefa}, \text{arquitetura}, \text{inicialização})`$ é o total de bits que o canal entrega até o algoritmo $`\mathcal{A}`$ atingir a acurácia alvo $`\varepsilon`$. Para que $`I`$ diga algo sobre a tarefa, a maior parte da sua variação precisa vir da tarefa, e não da arquitetura, da inicialização, do algoritmo ou das interações entre eles. Isso acontece? E $`I`$ cresce com o número de classes, como sugerem Gur-Ari et al. (2018)?

## Como foi feito

Cada passo entrega r = 32 projeções aleatórias do gradiente por um canal gaussiano de 2 bits por projeção (o canal `ruido` do E12), ou seja, 64 bits por passo. A acurácia é avaliada a cada 10 passos, e $`I`$ = 64 × passos até a primeira avaliação que atinge o alvo.

| Fator | Níveis |
|---|---|
| Tarefa (T) | Fashion-MNIST com as classes 0..k−1, k ∈ {2, 5, 10} |
| Arquitetura (A) | `prof` = 784-128-128-k; `larga` = 784-512-k (cerca de 3,4 vezes mais pesos) |
| Inicialização (Z) | `padrao` (Kaiming uniforme do PyTorch); `xavier` (Xavier normal, viés zero) |
| Algoritmo (G) | `adam` (lr 1e-3); `sgdm` (SGD com momento 0,9, lr 1,0) |

São 24 células e 3 sementes. O alvo de cada tarefa é $`1/k + 0{,}8\,(\text{ref}_k - 1/k)`$, com $`\text{ref}_k`$ igual à acurácia média do backprop com Adam em 1500 passos, nas 4 combinações de arquitetura e inicialização e nas 3 sementes. O teto é de 3000 passos (192 mil bits); quem não chega é censurado.

**Calibração feita antes dos critérios** ([auxiliares/calibracao.log](auxiliares/calibracao.log)), em k = 10, `prof`, `padrao`, semente 0:

- lr do SGD com momento: 0,01 → 11,7%; 0,03 → 17,7%; 0,1 → 36,0%; 0,3 → 59,1%; **1,0 → 75,3%**; 3,0 → 66,3%.
- Fração do alvo: com F = 0,8, o Adam atinge o alvo de k = 10 em 400 a 580 passos, longe do teto.

## Critérios

$`y = \log_2 I`$. A soma dos quadrados entre as 24 médias de célula é decomposta com contrastes ortogonais (o desenho é balanceado) em efeitos principais T, A, Z, G e todas as interações.

- **H11 (a tarefa domina):** aprovada se SQ(T) ≥ 50% do total e as interações somam no máximo 20%. Reprovada se algum outro efeito principal superar T, ou se as interações somarem 50% ou mais. Parcial nos demais casos.
- **H12 (I cresce com k):** em cada uma das 8 combinações (A, Z, G), $`y`$ médio estritamente crescente em k = 2, 5, 10. Aprovada se vale em pelo menos 7 das 8; reprovada se em 4 ou menos; parcial com 5 ou 6.
- **Exploratório:** razão geométrica $`I(\text{larga})/I(\text{prof})`$; inclinação de $`\log_2 I`$ contra $`\log_2 k`$.
- **Inconclusivo:** mais de 1/4 das 72 execuções censuradas, ou desvio-padrão de $`y`$ entre sementes acima de 1 em mais de 1/4 das células.

## Resultado

**H11 aprovada; H12 aprovada (8 de 8).** As 72 execuções chegaram ao alvo; nenhuma foi censurada, e o critério de inconclusivo não disparou.

Alvos: 89,3% (k = 2), 75,8% (k = 5) e 72,3% (k = 10), a partir de referências de backprop de 99,1%, 89,8% e 87,8%.

Decomposição da soma dos quadrados de $`\log_2 I`$ entre as 24 células:

| Termo | Fração |
|---|---|
| Tarefa (T) | 88,5% |
| Algoritmo (G) | 2,8% |
| Inicialização × algoritmo (ZG) | 2,5% |
| Inicialização (Z) | 1,9% |
| Arquitetura × algoritmo (AG) | 1,6% |
| Arquitetura × inicialização (AZ) | 1,2% |
| Arquitetura (A) | 0,2% |
| Demais interações | 1,4% |
| **Todas as interações** | **6,7%** |

$`I`$ em bits (média geométrica de 3 sementes):

| Arquitetura, inicialização, algoritmo | k = 2 | k = 5 | k = 10 |
|---|---|---|---|
| prof, padrao, adam | 2.663 | 18.253 | 38.307 |
| prof, padrao, sgdm | 3.804 | 37.218 | 67.621 |
| prof, xavier, adam | 2.032 | 17.539 | 29.868 |
| prof, xavier, sgdm | 1.465 | 11.820 | 22.119 |
| larga, padrao, adam | 1.016 | 10.336 | 23.465 |
| larga, padrao, sgdm | 2.547 | 33.139 | 84.583 |
| larga, xavier, adam | 2.225 | 11.065 | 24.477 |
| larga, xavier, sgdm | 2.663 | 17.056 | 40.987 |

O que sai das tabelas:

- **$`I`$ cresce com o número de classes em todas as 8 combinações.** De k = 2 para k = 10, o fator vai de 11 a 33. O ajuste exploratório de $`\log_2 I`$ contra $`\log_2 k`$ tem inclinação 1,79.
- **$`I`$ não cresce com o tamanho da rede.** A rede larga tem cerca de 3,4 vezes mais pesos e usou, em média geométrica, 0,90 vez os bits da rede profunda.
- **A interação mais visível é entre inicialização e algoritmo.** O SGD com momento precisou de cerca de 2 vezes os bits do Adam com a inicialização padrão e de menos bits com a Xavier. Somadas, todas as interações ficaram em 6,7%, abaixo do limite de 20%.
- **Em k = 2 a medida é grossa.** O alvo cai entre 10 e 140 passos, e a avaliação é a cada 10 passos. O desvio de $`\log_2 I`$ entre sementes chegou a 0,94 nessas células, abaixo do limite de 1.

Limites que continuam: uma só família de tarefas (subconjuntos aninhados de Fashion-MNIST), em que k muda junto com quais classes entram; alvo relativo ao backprop de cada tarefa; protocolo fixo (r = 32, 2 bits por projeção), de modo que o $`I`$ medido é um limite superior para esse protocolo, não o mínimo sobre algoritmos.

## Como reproduzir

```bash
poetry install --extras torch
make e13
```

- Sementes: pesos `manual_seed(s)`, direções `Generator().manual_seed(1000 + s)`, canal `Generator().manual_seed(2000 + s)`, lotes `default_rng(s)`; s = 0, 1, 2.
- Ambiente: `poetry.lock`, Python 3.12, CPU; referências com 3 threads, depois as três sementes em processos paralelos com 2 threads cada.
