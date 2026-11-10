# E04: Microteste de energia medida

> Linha de pesquisa: Ideia descartada: memória em superposição; lição de método sobre medir joules. Rodado em 2026-11-08.

## Pergunta

Com joules medidos no Apple M5, alguma variante em superposição bate a atenção?

## Como foi feito

Vocabulário GloVe-100. Medição com `zeus-apple-silicon` (CPU total + DRAM), um thread por consulta, janela ociosa logo antes de cada janela medida para subtrair o consumo basal. `microteste_confirmacao.py` repete k = 64 com janelas de 2 s, linha de base antes e depois, e registra a latência.

## Critérios

Fixados antes de rodar. A proposta só seria viável se:
- G1: alguma variante em superposição atinge 99% de acerto gastando menos energia por consulta que a atenção em 1536 dimensões, com decodificação nas duas;
- G2: com palavras parecidas como chaves, a dimensão necessária fica no máximo 2x a de palavras sorteadas;
- G3: a limpeza por índice HNSW mantém 99% e gasta pelo menos 2x menos que a varredura;
- G4: a estimativa óptica (só conversão, piso otimista) gasta menos que a limpeza eletrônica medida.

## Resultado

- G1 reprovado: melhor HRR 207 uJ por consulta contra 69 uJ da atenção 1536 com decodificação; a limpeza eletrônica custou 732 uJ.
- G2 reprovado: com chaves parecidas a HRR acertou 0,7% (k = 64) e 0,3% (k = 256), contra 100% da atenção.
- G3 e G4 passaram, mas não salvam a proposta sem G1 e G2.
- Confirmação: os joules líquidos tiveram intervalo interquartil cruzando zero em quase todas as variantes. A carga de fundo da máquina era maior que o sinal, e a latência virou a medida confiável (atenção 1536: 55 us; HRR com varredura: 361 us).

## Veredito

Memória em superposição como substituta da atenção: reprovada (G1 e G2). Lição de método: joules líquidos só com máquina ociosa ou medidor externo.

## Como reproduzir

```bash
poetry install --extras energia   # numpy já vem na instalação base
poetry run python experimentos/E04-microteste-energia/microteste_energia.py dados/glove-100-angular.hdf5 saidas/microteste-resultados.json
```

- Sementes: Episódios `default_rng(11 + k)` e `default_rng(99 + k)`; projeções `default_rng(seed)`; geral `default_rng(2026)`.
- Registros: `registros/log.txt` (microteste) e `registros/log2.txt` (confirmação). `auxiliares/probe.py` testou o medidor antes.
- Resultados oficiais: `resultados/` (quando há JSON). Novas execuções vão para `saidas/`, fora do git.
