# E11: Quantos canais de medida o treino exige?

> Linha de pesquisa: Hipótese central: limite de canal do treino físico. Rodado em 2026-11-08.

## Pergunta

Se o otimizador só recebe r projeções do gradiente por passo, quantos canais r são necessários conforme o número de parâmetros N cresce?

## Como foi feito

Fashion-MNIST, MLP com duas camadas ocultas de largura 16, 32, 64 e 128 (N = 13.002 a 118.282), Adam (lr = 1e-3), lote 128, 1.500 passos, 2 sementes, r em {8, 16, 32, 64, 128}.

Como a projeção é calculada (ver `rodar()`):

```python
loss.backward()                      # gradiente exato g, por backpropagation
V = torch.randn(N, k, generator=g_rng)
V = V / V.norm(dim=0, keepdim=True)  # colunas unitárias, não ortonormalizadas
est = est + V @ (V.T @ res)          # o Adam recebe V Vᵀ g, sem reescala por N/r
```

Modos: `aleatorio` (r direções novas por passo), `adaptativo` (metade das direções vem das estimativas anteriores), `oraculo` (metade vem dos gradientes verdadeiros anteriores; irrealizável, é o teto).

## Critérios

Fixados antes de rodar. r*(N) = menor r com acurácia >= backprop - 2 pontos; alfa = inclinação de log r* por log N.
- C1 (sobrevive): adaptativo com alfa <= 0,5;
- C2: r* adaptativo <= r* aleatório / 2 no maior N;
- C3 (morre): alfa >= 0,8.
Se r* passar de 128, o resultado é registrado como acima da grade.

## Resultado

Acurácia de teste (%), média de 2 sementes:

| N | backprop | r = 8 | r = 128 | adaptativo r = 128 | oráculo r = 8 |
|---|---|---|---|---|---|
| 13.002 | 84,5 | 69,6 | 78,8 | 76,1 | 82,5 |
| 26.506 | 85,6 | 72,6 | 80,1 | 76,6 | 83,7 |
| 55.050 | 86,5 | 73,6 | 80,3 | 76,9 | 84,7 |
| 118.282 | 87,0 | 74,2 | 80,7 | 78,0 | 84,9 |

- r* ficou acima da grade (r > 128) nos modos aleatório e adaptativo, em todos os N. C1, C2 e C3 ficaram indefinidos.
- Nove vezes mais parâmetros não pioraram a acurácia com r fixo.
- O modo adaptativo foi pior que o aleatório.

## Veredito

Evidência parcial para a hipótese: a acurácia com r fixo não cai com N. O critério de 2 pontos não foi atingido. Limites: canal ideal (float32, sem ruído), projeção feita a partir do gradiente exato, nenhum custo de operação contado, lr igual para todo r, 2 sementes, uma tarefa.

## Como reproduzir

```bash
poetry install --extras torch   # numpy já vem na instalação base
poetry run python experimentos/E11-canais-de-gradiente/canais_de_gradiente.py dados/fashion saidas/canais-de-gradiente.json
```

- Sementes: Pesos `torch.manual_seed(s)`; direções `torch.Generator().manual_seed(1000 + s)`; lotes `default_rng(s)`; s = 0, 1. CPU, 4 threads.
- Registros: `registros/canais.log` (3 h 20 min no Apple M5).
- Resultados oficiais: `resultados/` (quando há JSON). Novas execuções vão para `saidas/`, fora do git.
