# 0001. Adotar o limite de canal do treino físico como hipótese central

- Data: 2026-11-08
- Estado: aceita

## Contexto

A pesquisa começou buscando uma tese de fronteira ligada a treino em superposição, tempo e frequência, e energia. Três linhas caíram pelos próprios testes:

- Fourier fixo nos pesos (E01, E02): só ajuda onde há ordem real nos dados;
- memória em superposição no lugar da atenção (E03, E04): reprovada em energia e em acerto;
- atenção esparsa (E06 a E10): funciona (m ~ N^0,4 a 0,5), mas foi avaliada como incremental demais para uma tese.

A ponte entre o algoritmo de Jordan e o gradiente multiplexado do NIST mostrou que, nos dois, o gradiente é um coeficiente de Fourier da perda perturbada, e que o limite de Gabor ($T \ge N/B$) governa a leitura.

## Decisão

Adotar como hipótese central que o treino físico sem backpropagation é limitado pela informação que a tarefa exige, $T \ge I/(B \log_2(1+\mathrm{SNR}))$, e não pelo número de parâmetros.

## Consequências

- O primeiro problema passa a ser definir e medir $I$ (ver [questão 3](../03-questoes-abertas.md#3-como-definir-i)).
- A atenção esparsa fica como plano B (H06).
- O E11 vira o experimento de referência, e o E12 testa a distinção entre medidas e bits.
- A hipótese tem critérios de refutação explícitos (ver [pesquisa §7](../01-pesquisa.md#7-o-que-refutaria-a-hipótese)).
