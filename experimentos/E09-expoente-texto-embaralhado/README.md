# E09: O expoente de m(N) vem da língua ou do modelo?

> Linha de pesquisa: Plano B: atenção esparsa. Rodado em 2026-11-08.

## Pergunta

O expoente de m(N) depende do discurso de longo alcance?

## Como foi feito

GPT-2 sem retreino, atenção cortada para as m maiores. Três textos: real, frases embaralhadas (mantém a frase, destrói o discurso) e tokens embaralhados (mantém só a frequência). IC 95% por bootstrap.

## Critérios

Fixados antes de rodar (tau = 0,1 nat):
- E1: se o IC 95% do expoente com frases embaralhadas não contiver o valor do texto real, o discurso participa do expoente;
- E2: se contiver, o expoente vem do modelo ou da estrutura local da língua;
- E3: tokens embaralhados são controle.

## Resultado

| Texto | expoente (tau = 0,1) | IC 95% |
|---|---|---|
| real | 0,41 | 0,35 a 0,51 |
| frases embaralhadas | 0,48 | 0,42 a 0,54 |
| tokens embaralhados | 0,69 | 0,38 a 0,88 |

## Veredito

Vale E2: o expoente (m ~ N^0,4 a 0,5) não depende do discurso. Vem do modelo ou da estrutura local.

## Como reproduzir

```bash
poetry install --extras nlp   # numpy já vem na instalação base
poetry run python experimentos/E09-expoente-texto-embaralhado/expoente_texto_embaralhado.py dados/moby.txt saidas/expoente-texto-embaralhado.json
```

- Sementes: Janelas `default_rng(11)`; embaralhamento `default_rng(3)`. Rodou em MPS.
- Registros: `registros/expoente.log`.
- Resultados oficiais: `resultados/` (quando há JSON). Novas execuções vão para `saidas/`, fora do git.
