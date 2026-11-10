# E10: Treinar com orçamento de conexões muda a lei de m(N)?

> Linha de pesquisa: Plano B: atenção esparsa. Rodado em 2026-11-08.

## Pergunta

Se a restrição de m conexões vale desde o primeiro passo de treino, o expoente de m(N) cai?

## Como foi feito

Transformer por caractere (6 camadas, d = 128, 4 cabeças) em Moby Dick, 85% treino, 2.000 passos, 2 sementes. Compara cortar depois do treino com treinar já cortado.

## Critérios

Fixados antes de rodar:
- T1 (poder): o contexto precisa ajudar, NLL(32) - NLL(256) >= 0,05 nat. Senão, inconclusivo;
- T2: expoente em treino <= expoente pós-treino - 0,2 e m_treino(256) <= m_pos(256)/2;
- T3: diferença de expoentes abaixo de 0,1 indica que treinar com orçamento não muda a lei.
Tamanho reduzido antes de ver resultados, por limitação térmica (cerca de 0,25 s por passo).

## Resultado

- T1 reprovado: o ganho de contexto foi 0,042 nat, abaixo de 0,05.
- m_treino ficou entre 4,0 e 6,3 (expoente 0,23); m_pos ficou indefinido para N >= 128.

## Veredito

Inconclusivo pelo critério T1: o modelo pequeno quase não usa o contexto longo.

## Como reproduzir

```bash
poetry install --extras torch   # numpy já vem na instalação base
poetry run python experimentos/E10-treino-com-orcamento/treino_com_orcamento.py dados/moby.txt saidas/treino-com-orcamento.json 2000 2
```

- Sementes: `torch.manual_seed(s)` e `default_rng(s)`, s = 0, 1. Rodou em MPS, que não é determinístico.
- Registros: `registros/treino.log`. `auxiliares/passo.py` e `passo2.py` mediram o tempo por passo para dimensionar o teste.
- Resultados oficiais: `resultados/` (quando há JSON). Novas execuções vão para `saidas/`, fora do git.
