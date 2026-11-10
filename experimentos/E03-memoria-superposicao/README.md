# E03: Memória em superposição contra atenção (contagem de operações)

> Linha de pesquisa: Ideia descartada: memória HRR via FFT no lugar da atenção. Rodado em 2026-11-08.

## Pergunta

Uma memória associativa em superposição (HRR, ligação por convolução circular via FFT) recupera valores com menos energia que a atenção?

## Como foi feito

k pares chave-valor aleatórios. Mede a menor dimensão com 99% de acerto e estima a energia por consulta contando acessos à memória e operações. Só recuperação, sem aprendizado.

## Critérios

Teste exploratório. Os critérios formais vieram depois, em E04.

## Resultado

- Dimensão mínima para 99% de acerto: a atenção precisa de 32 em todos os k testados; a HRR precisa de 256 (k = 4) até 16.384 (k = 256).
- Energia estimada sem limpeza (DRAM): a razão atenção/HRR fica entre 0,34 e 0,76, ou seja, a HRR gasta mais. Com a limpeza por varredura, a razão cai para perto de zero em k grande.
- Com consulta ruidosa (rho = 0,7) a HRR precisa de 1.024 a 2.048 dimensões contra 32 a 64 da atenção. Em dois saltos, a HRR não atingiu 99% em nenhuma dimensão testada.

## Veredito

Reprovada já na contagem de operações. E04 refez a comparação com joules medidos.

## Como reproduzir

```bash
poetry install
poetry run python experimentos/E03-memoria-superposicao/memoria_superposicao.py
```

- Sementes: `default_rng(seed + d + k)` com seed = 7; MLP com sementes 0..4.
- Registros: `registros/memoria-superposicao.log` (reexecução de 2026-11-09, 44 s).
- Resultados oficiais: `resultados/` (quando há JSON). Novas execuções vão para `saidas/`, fora do git.
