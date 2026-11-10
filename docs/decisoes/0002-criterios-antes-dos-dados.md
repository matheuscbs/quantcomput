# 0002. Fixar critérios antes de rodar e registrar resultados negativos

- Data: 2026-11-08
- Estado: aceita

## Contexto

Com testes pequenos e muitas hipóteses, é fácil ajustar o critério depois de ver o número. Também é fácil esquecer as ideias que falharam e voltar a elas.

## Decisão

- Todo experimento declara na docstring a pergunta, os critérios de aprovação e reprovação e o que conta como inconclusivo, antes da execução oficial.
- O veredito aplica os critérios literalmente. Leituras posteriores são marcadas como exploratórias.
- Resultados negativos ficam no repositório, com o mesmo cuidado dos positivos, e o estado da hipótese vai para `docs/hipoteses.md`.
- A partir da versão 0.1.0, o commit do script com os critérios precede o commit dos resultados.

## Consequências

- O histórico do git passa a provar a ordem entre critério e resultado, a partir do E12.
- Para E01 a E11, a ordem é declarada nas docstrings, mas o git não a comprova: esses experimentos entraram no repositório já executados.
