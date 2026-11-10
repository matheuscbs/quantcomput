# E05: Custo de leitura conforme a memória cresce

> Linha de pesquisa: Ideia descartada: lei geométrica de energia por distância. Rodado em 2026-11-08.

## Pergunta

O custo de ler um dado cresce como potência do tamanho da memória usada, como prevê o argumento geométrico (fio mais longo, mais energia)?

## Como foi feito

Leitura aleatória (roteamento arbitrário, como a atenção de todos com todos) e sequencial (varredura com pré-busca), de 16 KB a 4 GB, num Apple M5. Com carga de fundo, a latência é a medida confiável.

## Critérios

Teste exploratório.

## Resultado

- Expoente do custo por leitura aleatória em função do tamanho: 0,10 na faixa de cache (16 KB a 16 MB), -0,02 na DRAM (32 MB a 4 GB), 0,14 no total.
- Latência aleatória: 3,5 ns (16 KB), 11 ns (64 MB), 15 ns (512 MB), 9 ns (4 GB).

## Veredito

A lei geométrica não apareceu no M5 até 4 GB. A hierarquia de memória domina, não a distância.

## Como reproduzir

```bash
poetry install --extras energia   # numpy já vem na instalação base
poetry run python experimentos/E05-lei-de-escala-memoria/lei_de_escala_memoria.py saidas/lei-de-escala.json
```

- Sementes: `default_rng(5)`.
- Registros: `registros/log3.txt`.
- Resultados oficiais: `resultados/` (quando há JSON). Novas execuções vão para `saidas/`, fora do git.
