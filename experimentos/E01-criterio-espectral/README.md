# E01: Critério espectral para truncar em frequência

> Linha de pesquisa: Ideia descartada: treinar só em poucos modos de Fourier. Rodado em 2026-11-08.

## Pergunta

A energia do gradiente inicial nos K modos de baixa frequência prevê se vale a pena treinar só nesses modos?

## Como foi feito

Regressão linear com p = 256 pesos e K = 21 modos de baixa frequência. Compara o erro em excesso do modelo truncado com o do modelo completo, no treino e sob mudança de distribuição, em alvos isotrópicos e anisotrópicos.

## Critérios

Teste exploratório, sem critério numérico fixado antes.

## Resultado

- Alvo isotrópico: a concentração de energia ordena bem os casos (Spearman -0,96). Um limiar C = 0,555 acerta 92,2% fora da amostra.
- Alvo anisotrópico: o mesmo limiar acerta 15,0% (erro no treino) e 23,9% (sob mudança de distribuição). Com a versão ponderada do critério, o Spearman cai para 0,08 e 0,00.

## Veredito

O critério só funciona quando o alvo é isotrópico. Não serve como regra geral.

## Como reproduzir

```bash
poetry install
poetry run python experimentos/E01-criterio-espectral/criterio_espectral.py
```

- Sementes: `default_rng(1000*i + int(10a) + 7*int(beta))` por repetição.
- Registros: `registros/criterio-espectral.log` (reexecução de 2026-11-09, 5 s).
- Resultados oficiais: `resultados/` (quando há JSON). Novas execuções vão para `saidas/`, fora do git.
