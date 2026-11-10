# 0003. Publicar código, registros e textos em repositório aberto

- Data: 2026-11-09
- Estado: aceita

## Contexto

O ensaio ofereceu o código para reprodução, e a revisão técnica pediu código, instruções, versões, sementes e registros completos.

## Decisão

- Código, registros, documentação, ensaio e artigo ficam públicos sob a licença MIT.
- Currículo e correspondência ficam em `privado/`, fora do git.
- Dados não entram no git. Entram os hashes e o script que os baixa.
- Dependências fixadas com Poetry (`pyproject.toml` e `poetry.lock`).

## Consequências

- Qualquer pessoa pode reproduzir os números e auditar os critérios.
- As versões fixadas são uma reconstrução a partir do cache do uv; a equivalência com a execução original precisa ser confirmada (ver [reprodutibilidade](../05-reprodutibilidade.md)).
