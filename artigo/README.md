# Artigo

Versão em duas colunas do [ensaio](../ensaio/ensaio-instituto-ramanujan.md), com uma seção de método que descreve como as projeções do E11 foram calculadas, a figura e a tabela do E11, e a discussão sobre backpropagation in situ.

```bash
make artigo        # na raiz; usa tectonic
# ou
cd artigo && tectonic main.tex
```

| Arquivo | Conteúdo |
|---|---|
| `main.tex` | texto, figura (pgfplots) e tabela |
| `referencias.bib` | bibliografia em BibTeX |
| `main.pdf` | versão compilada |

O ensaio em `ensaio/` é o texto enviado e não muda. O artigo evolui com o projeto, e cada versão compilada acompanha uma tag do repositório.
