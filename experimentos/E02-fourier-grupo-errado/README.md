# E02: Fourier no grupo errado

> Linha de pesquisa: Ideia descartada: base de Fourier fixa para os pesos de uma rede. Rodado em 2026-11-08.

## Pergunta

Uma base fixa (DCT) concentra a energia dos pesos treinados melhor que uma base aleatória?

## Como foi feito

MLP treinada em numpy, 12 sementes. Pelo lema de Schur, a ordem dos neurônios ocultos é arbitrária, então a lei dos pesos é invariante a permutação e nenhuma base fixa deveria ganhar da aleatória ali. No eixo de entrada, onde a ordem é real, a DCT pode ganhar se o dado for suave.

## Critérios

Predição teórica (Schur) escrita no cabeçalho do script antes de rodar.

## Resultado

- Eixo de entrada: metade dos modos DCT guarda 56,4% da energia (aleatória: 50,8%). A DCT vence a aleatória em 75% das sementes.
- Eixo oculto: 49,1% (DCT) contra 49,7% (aleatória). A DCT vence em 58% dos casos, perto do acaso.
- Restringir a camada oculta a uma base DCT não piorou o erro de teste (mediana 0,0503 contra 0,0487 da aleatória e 0,0517 da densa).
- A simetria de permutação vale pouco em bits: para n = 1000, log2(n!) = 8.529 bits, contra 32 milhões de bits da matriz em float32 (0,03%).

## Veredito

Predição confirmada. Fourier só ajuda onde há uma ordem real nos dados. Nas camadas ocultas, é a base do grupo errado.

## Como reproduzir

```bash
poetry install
poetry run python experimentos/E02-fourier-grupo-errado/fourier_grupo_errado.py
```

- Sementes: Dados: `default_rng(90000 + s)`; treino: `default_rng(s)`; bases: `default_rng(555 + s)`; s = 0..11.
- Registros: `registros/grupo-errado.log` (reexecução de 2026-11-09, 26 s).
- Resultados oficiais: `resultados/` (quando há JSON). Novas execuções vão para `saidas/`, fora do git.
