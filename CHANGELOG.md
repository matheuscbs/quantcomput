# Changelog

Histórico de versões do conhecimento e do código. Convenção em [ARQUITETURA.md](ARQUITETURA.md#versionamento-do-conhecimento): maior quando a hipótese central muda; meio para experimento ou evidência nova; menor para correções sem efeito nos resultados.

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
