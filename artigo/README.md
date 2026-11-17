# Artigo

Versão em duas colunas (IEEEtran) do trabalho, atualizada a cada experimento. O [ensaio](../ensaio/ensaio-instituto-ramanujan.md) é o texto original da candidatura e fica congelado; o artigo traz a formulação revisada, o método, os resultados de E11 a E17, a discussão e os critérios registrados.

```bash
make artigo        # na raiz: gera as tabelas e compila com tectonic
```

| Arquivo | Conteúdo |
|---|---|
| `main.tex` | texto do artigo |
| `resultados-gerados.tex` | tabelas, figuras e números de E13 a E17, gerados por `scripts/gerar_artigo.py` a partir dos JSONs oficiais; não editar à mão |
| `e12-resultados.tex` | tabela e texto do E12, gerados do JSON do E12 |
| `referencias.bib` | bibliografia em BibTeX |
| `main.pdf` | versão compilada |

Cada versão compilada acompanha uma tag do repositório.
