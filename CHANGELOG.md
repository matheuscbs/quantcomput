# Changelog

Histórico de versões do conhecimento e do código. Convenção em [ARQUITETURA.md](ARQUITETURA.md#versionamento-do-conhecimento): maior quando a hipótese central muda; meio para experimento ou evidência nova; menor para correções sem efeito nos resultados.

## [0.2.0] - 2026-11-10

### Adicionado

- E12 (canal quantizado e ruidoso), com critérios commitados antes da execução (14e5011). Resultado: o total de bits não governa a acurácia (B1 reprovado); a acurácia segue o número de medidas r, e a precisão satura em 1 a 2 bits (ruído) ou 2 a 4 bits (quantização). B3 e B4 aprovados.
- Hipótese H14 (o ruído do minilote domina o erro de cada projeção) e experimento E13d para testá-la.
- `docs/07-mapa-ensaio.md`, `scripts/verificar_ensaio.py` (15 de 15), `docs/08-leituras.md` (20 artigos lidos no texto completo).
- Artigo em IEEEtran com a seção de resultados do E12.

### Mudado

- H10 refutada; H08 mantida aberta, com a forma $\log_2(1+\mathrm{SNR})$ a revisar no regime de alto SNR.
- A contribuição foi redefinida depois da leitura: Oripov et al. (2025) e Li et al. (2018) já observaram o custo quase independente de N; o que o projeto propõe é medi-lo em bits e ligá-lo a k, SNR e energia.

### Observações

- A reprodução do E11 com as versões fixadas foi feita: o controle float32 do E12 repetiu o E11 até a quarta casa decimal.
- O critério B2 do E12 passou no canal ruidoso e não no quantizado. O script declara aprovação se qualquer ajuste passar; o README pré-registrado não dizia se a regra era "qualquer" ou "todos".

## [0.1.0] - 2026-11-09

Primeira versão pública.

### Adicionado

- Experimentos E01 a E11, cada um com script, README, resultados e registros.
- Hipótese central H08 (limite de canal do treino físico) e registro de hipóteses H01 a H11.
- Documentação: pesquisa, resultados, questões abertas, base técnica, reprodutibilidade, roteiro.
- Decisões 0001 a 0003.
- `pyproject.toml` e `poetry.lock` com versões fixadas; `scripts/baixar_dados.py` com conferência de SHA-256.
- Artigo em LaTeX, duas colunas, derivado do ensaio.

### Observações

- E01 a E03 não tinham log da execução original. Os logs são de uma reexecução de 2026-11-09 com o mesmo código.
- As versões das dependências foram reconstruídas a partir do cache do uv. A reprodução do E11 com elas ainda não foi feita.
- Caminhos absolutos nos scripts foram trocados por caminhos relativos ao próprio arquivo; nomes de arquivo foram padronizados. Nenhuma lógica de experimento mudou.
