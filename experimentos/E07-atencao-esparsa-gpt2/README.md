# E07: Atenção esparsa por conteúdo no GPT-2

> Linha de pesquisa: Plano B: atenção esparsa. Rodado em 2026-11-08.

## Pergunta

Quantas conexões por consulta o GPT-2 precisa, se escolhidas pelo conteúdo?

## Como foi feito

Mantém só as m maiores atenções por cabeça e por token (mais o primeiro token) e renormaliza. É um oráculo: os escores são calculados por inteiro antes da escolha.

## Critérios

Teste exploratório.

## Resultado

| m | perda extra (nat) | fração dos pares |
|---|---|---|
| 16 | 0,937 | 3,2% |
| 32 | 0,232 | 6,2% |
| 64 | 0,023 | 12,2% |
| 128 | 0,016 | 23,6% |

## Veredito

Com escolha por conteúdo, 12% dos pares bastam (contra 75% da janela fixa de E06).

## Como reproduzir

```bash
poetry install --extras nlp   # numpy já vem na instalação base
poetry run python experimentos/E07-atencao-esparsa-gpt2/atencao_esparsa_gpt2.py dados/moby.txt saidas/atencao-esparsa-gpt2.json
```

- Sementes: Inferência sem amostragem.
- Registros: `registros/log5.txt`. `auxiliares/insp.py` inspecionou a API de atenção do transformers 4.49.
- Resultados oficiais: `resultados/` (quando há JSON). Novas execuções vão para `saidas/`, fora do git.
