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

Grade: N em {13.002, 118.282}, r em {8, 32, 128}, b em {1, 2, 4, 8}, 2 sementes. 102 execuções por semente.

Limite que continua: a projeção sai do gradiente exato. Muda o canal, não a forma de medir.

## Critérios

Ajustes por N e por canal, nas 12 configurações (r, b), média das sementes:

- **B1 (total de bits governa):** acc ~ a + c·log2(bits_total) com R² >= 0,90.
- **B2 (medidas e precisão separadas):** acc ~ a + c1·log2(r) + c2·log2(b) reduz a soma dos quadrados dos resíduos de B1 em pelo menos 50%. Se B1 e B2 valerem juntos, vale B2.
- **B3 (sem penalidade em N):** em pelo menos 80% das configurações, acc(118.282) >= acc(13.002) - 1 ponto.
- **B4 (reprodução):** o canal `float` reproduz o E11 com diferença <= 0,5 ponto nas 6 células.
- **Inconclusivo:** desvio entre sementes acima de 2 pontos em mais de 1/4 das células.

## Resultado

Em execução.

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
