# E06: Comunicação de longa distância no GPT-2

> Linha de pesquisa: Plano B: atenção esparsa. Rodado em 2026-11-08.

## Pergunta

Quanta comunicação de longa distância um Transformer real usa?

## Como foi feito

GPT-2 em Moby Dick, janelas de 1.024 tokens. Observacional: massa de atenção por distância e expoente de Rent do fluxo. Causal: cada token só vê os r anteriores, com e sem o primeiro token, e mede a perda.

## Critérios

Atenção não é informação, então a parte causal decide.

## Resultado

- Perda completa: 3,893 nat.
- Janela local com o primeiro token: r = 128 (24% dos pares) ainda custa +0,58 nat; r = 512 (75% dos pares) custa +0,009 nat.
- Sem o primeiro token a perda explode (r = 512: +1,0 nat), o que mostra o papel de "sumidouro" do primeiro token.
- Expoente de Rent: 0,81 (real), 0,69 (sem o primeiro token), 0,87 (atenção uniforme).

## Veredito

Uma janela fixa por distância exige 75% dos pares no GPT-2. A comunicação precisa ser escolhida pelo conteúdo, o que motivou E07.

## Como reproduzir

```bash
poetry install --extras nlp   # numpy já vem na instalação base
poetry run python experimentos/E06-comunicacao-gpt2/comunicacao_gpt2.py dados/moby.txt saidas/comunicacao-gpt2.json
```

- Sementes: Inferência sem amostragem; `torch.set_num_threads(4)`.
- Registros: `registros/log4.txt`.
- Resultados oficiais: `resultados/` (quando há JSON). Novas execuções vão para `saidas/`, fora do git.
