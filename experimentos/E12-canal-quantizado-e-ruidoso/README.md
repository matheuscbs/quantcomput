# E12: canal quantizado e ruidoso

> Linha de pesquisa: Hipótese central (H08, H10). Critérios registrados no git antes da execução.

## Pergunta

O E11 variou só o número de medidas, com cada medida em float32 e sem ruído. Aqui o canal tem capacidade de b bits por projeção. A acurácia passa a depender só do total de bits entregue ao otimizador, ou medidas e precisão têm papéis separados?

## Como foi feito

Modo aleatório do E11 (mesmas redes, sementes, lotes e direções), com as r projeções de cada passo passando por um canal:

| Canal | O que faz | Bits por passo |
|---|---|---|
| `quant` | quantização uniforme estocástica (sem viés) em 2^b níveis em [-s, s], s = max\|p\| enviado em float32 | r·b + 32 |
| `ruido` | ruído gaussiano com SNR = 2^(2b) - 1, ou seja, b bits de capacidade por projeção | r·b |
| `float` | sem degradação; controle que deve reproduzir o E11 | 32·r |

Grade: N em {13.002, 118.282}, r em {8, 32, 128}, b em {1, 2, 4, 8}, 2 sementes. 56 execuções por semente (2 backprop, 6 float, 48 com canal).

Limite que continua: a projeção sai do gradiente exato. Muda o canal, não a forma de medir.

## Critérios

Ajustes por N e por canal, nas 12 configurações (r, b), média das sementes:

- **B1 (total de bits governa):** acc ~ a + c·log2(bits_total) com R² >= 0,90.
- **B2 (medidas e precisão separadas):** acc ~ a + c1·log2(r) + c2·log2(b) reduz a soma dos quadrados dos resíduos de B1 em pelo menos 50%. Se B1 e B2 valerem juntos, vale B2.
- **B3 (sem penalidade em N):** em pelo menos 80% das configurações, acc(118.282) >= acc(13.002) - 1 ponto.
- **B4 (reprodução):** o canal `float` reproduz o E11 com diferença <= 0,5 ponto nas 6 células.
- **Inconclusivo:** desvio entre sementes acima de 2 pontos em mais de 1/4 das células.

## Resultado

**B1 reprovado; B2 aprovado no canal ruidoso e não no quantizado; B3 e B4 aprovados.** O total de bits não governa a acurácia. Quem governa é o número de medidas r. A precisão por medida satura cedo: acima de 1 a 2 bits por projeção, quase não muda nada.

| N, canal | R² de B1 (total de bits) | R² de B2 (r e b) | redução da SSE | coef. de log2 r | coef. de log2 b |
|---|---|---|---|---|---|
| 13.002, quant | 0,74 | 0,84 | 39% | 0,024 | 0,022 |
| 13.002, ruído | 0,70 | 0,97 | 89% | 0,023 | 0,002 |
| 118.282, quant | 0,84 | 0,90 | 39% | 0,016 | 0,015 |
| 118.282, ruído | 0,76 | 0,99 | 98% | 0,017 | 0,002 |

B1 não chegou a R² de 0,90 em nenhum dos quatro ajustes. B2 passou nos dois ajustes do canal ruidoso e ficou abaixo de 50% nos dois do canal quantizado. O script commitado antes da execução declara B2 aprovado se qualquer ajuste passar (`any`), e registra `B2_medidas_e_precisao_separadas: true`; o README não dizia se a regra era "qualquer" ou "todos", e a ressalva fica aqui. No canal quantizado, o efeito de b se concentra em b = 1 e não é linear em log2 b, o que penaliza os dois modelos lineares.

Acurácia de teste (%), média de 2 sementes, N = 118.282 (float: 74,2 / 77,6 / 80,7):

| canal | r | b = 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| ruído | 8 | 73,2 | 74,1 | 74,3 | 74,1 |
| ruído | 32 | 76,7 | 77,3 | 77,5 | 77,6 |
| ruído | 128 | 80,3 | 80,7 | 80,8 | 80,7 |
| quant | 8 | 69,3 | 73,4 | 74,3 | 74,1 |
| quant | 32 | 73,3 | 76,7 | 77,4 | 77,4 |
| quant | 128 | 75,8 | 79,8 | 80,6 | 80,8 |

N = 13.002 (float: 69,6 / 75,8 / 78,8):

| canal | r | b = 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| ruído | 8 | 68,7 | 69,8 | 69,6 | 69,6 |
| ruído | 32 | 75,1 | 75,5 | 75,9 | 75,8 |
| ruído | 128 | 78,7 | 79,0 | 78,9 | 78,8 |
| quant | 8 | 60,7 | 70,1 | 69,7 | 69,7 |
| quant | 32 | 68,8 | 74,7 | 75,8 | 75,8 |
| quant | 128 | 73,6 | 78,1 | 78,9 | 78,8 |

O que sai das tabelas:

- **Ruído gaussiano com 1 bit de capacidade (SNR = 3) custa no máximo 1 ponto** em relação ao float32.
- **Quantização com 1 bit custa de 4 a 9 pontos; com 2 bits, de 0,5 a 1,1; com 4 bits, empata com float32.** O limiar não cresceu de forma clara com r nessa grade, então a previsão do QSGD (b ≈ ½ log2 r) não foi confirmada nem refutada.
- **Com o mesmo total de bits, mais medidas de menor precisão ganham.** No canal ruidoso com N = 118.282: 96 mil bits dão 74,1% com r = 8, b = 8 e 77,3% com r = 32, b = 2; 384 mil bits dão 77,6% com r = 32, b = 8 e 80,7% com r = 128, b = 2. No canal quantizado a troca vale até b = 2; em b = 1 ela se inverte (192 mil bits: 77,4% com r = 32, b = 4 contra 75,8% com r = 128, b = 1).
- **B3:** em 24 de 24 configurações, a rede maior não perdeu mais de 1 ponto (fração 1,0).
- **B4:** o canal float reproduziu o E11 com diferença 0,0 nas 6 células (até a quarta casa decimal).
- **Inconclusivo:** não. A maior divergência entre sementes ficou em r = 8 com N = 13.002 (até 5,9 pontos), abaixo do limite de 1/4 das células.

**O que isso significa para a hipótese.** A forma $T \ge I/(B\log_2(1+\mathrm{SNR}))$ trata bits de medidas diferentes e bits de precisão como intercambiáveis. Nesse regime eles não são: um bit a mais de precisão vale quase nada, e uma medida a mais vale muito. A capacidade nominal do canal superestima a informação útil quando o SNR é alto. Uma explicação possível, ainda não testada, é que o ruído do minilote já domina o erro de cada projeção, de modo que ruído de canal abaixo dele é invisível. Fica registrada como H14 em [hipóteses](../../docs/hipoteses.md).

Limite que continua: as projeções saem do gradiente exato; o E12 muda o canal, não a forma de medir. A informação mútua efetiva entre medidas e gradiente segue sem medida.

## Como reproduzir

```bash
poetry install --extras torch
cd experimentos/E12-canal-quantizado-e-ruidoso
poetry run python canal_quantizado.py rodar ../../dados/fashion ../../saidas/e12-s0.json 0
poetry run python canal_quantizado.py rodar ../../dados/fashion ../../saidas/e12-s1.json 1
poetry run python canal_quantizado.py analisar ../../saidas/e12.json ../../saidas/e12-s0.json ../../saidas/e12-s1.json \
    --e11 ../E11-canais-de-gradiente/resultados/canais-de-gradiente.json
```

- Sementes: as do E11 (pesos `manual_seed(s)`, direções `Generator().manual_seed(1000 + s)`, lotes `default_rng(s)`), mais o canal `Generator().manual_seed(2000 + s)`; s = 0, 1.
- Ambiente: `poetry.lock` (torch 2.15.0, numpy 2.5.4), Python 3.12, CPU com 4 threads por processo; as duas sementes rodam em processos paralelos.
