# Dados

Os arquivos de dados não entram no git. Para recriá-los:

```bash
poetry install --extras dados
poetry run python scripts/baixar_dados.py          # baixa os três
poetry run python scripts/baixar_dados.py moby     # só um
```

| Arquivo | Origem | Usado em | Tamanho |
|---|---|---|---|
| `fashion/fashion.npz` | OpenML, `data_id=40996` (Fashion-MNIST, 70.000 imagens 28x28) | E11 | 55 MB |
| `moby.txt` | Project Gutenberg #2701, texto puro | E06 a E10 | 1,3 MB |
| `glove-100-angular.hdf5` | ann-benchmarks.com | E04 | 485 MB |

`SHA256SUMS` guarda os hashes dos arquivos usados em 2026-11-08. O script confere cada um ao final. Em 2026-11-09 o `moby.txt` do Gutenberg ainda tinha o mesmo hash.
