# Reprodutibilidade

## Ambiente original

| Item | Valor |
|---|---|
| Máquina | Apple M5, macOS 26.5.2 |
| Python | 3.12 (CPython 3.12.13 via uv) |
| Execução | `uv run --no-project --with <pacote>` sem versão fixada |
| Data | 2026-11-08 (E04 a E11); 2026-11-09 (reexecução de E01 a E03) |

## Versões

Os experimentos originais não fixaram versões. As versões abaixo são as que estavam no cache do uv quando os experimentos rodaram, e foram fixadas no `pyproject.toml` e no `poetry.lock`. São a melhor reconstrução disponível, não um registro da execução.

| Pacote | Versão | Usado em |
|---|---|---|
| numpy | 2.5.4 | todos |
| torch | 2.15.0 | E06 a E11 |
| transformers | 4.49.0 | E06 a E09 (`<4.50`, por mudança na API de atenção do GPT-2) |
| faiss-cpu | 1.15.1 | E04 |
| h5py | 3.16.0 | E04 |
| zeus-apple-silicon | 1.1.0 | E04, E05 (só macOS Apple Silicon) |
| scikit-learn | 1.9.1 | download do Fashion-MNIST |

Para confirmar que essa reconstrução reproduz os números, o passo seguinte é rodar `make e11` e comparar com `experimentos/E11-canais-de-gradiente/resultados/`. Até isso ser feito, a equivalência é uma suposição.

## Instalação

```bash
# Poetry
poetry install --all-extras
poetry run python experimentos/E01-criterio-espectral/criterio_espectral.py

# Alternativa sem Poetry, com uv e as mesmas versões
uv run --python 3.12 --no-project --with numpy==2.5.4 --with torch==2.15.0 \
    python experimentos/E11-canais-de-gradiente/canais_de_gradiente.py dados/fashion saidas/e11.json
```

## Sementes e determinismo

| ID | Sementes | Determinístico? |
|---|---|---|
| E01 | `default_rng(1000*i + int(10a) + 7*int(beta))` | sim |
| E02 | dados 90000+s, treino s, bases 555+s (s = 0..11) | sim |
| E03 | `default_rng(seed + d + k)`, seed = 7; MLP 0..4 | sim |
| E04 | episódios 11+k e 99+k; projeções por semente; geral 2026 | acerto sim; energia não (medição física) |
| E05 | `default_rng(5)` | latência e energia variam com a carga da máquina |
| E06, E07 | inferência sem amostragem | sim, em CPU com 4 threads |
| E08 | projeções `manual_seed(7)` | sim, em CPU |
| E09 | janelas `default_rng(11)`, embaralhamento `default_rng(3)` | não exatamente: rodou em MPS |
| E10 | `manual_seed(s)`, `default_rng(s)`, s = 0, 1 | não exatamente: rodou em MPS |
| E11 | pesos `manual_seed(s)`, direções `Generator().manual_seed(1000+s)`, lotes `default_rng(s)`, s = 0, 1 | sim, em CPU com 4 threads, salvo diferenças de ordem de soma entre versões do torch |

Em MPS (GPU da Apple), algumas operações não são determinísticas. Espere diferenças pequenas em E09 e E10.

## Dados

Ver [`dados/README.md`](../dados/README.md). Os três arquivos têm SHA-256 registrado. Em 2026-11-09 o `moby.txt` do Gutenberg ainda batia com o hash original e o GloVe do ann-benchmarks tinha o mesmo tamanho em bytes.

## Registros

Cada experimento guarda em `registros/` o stdout da execução oficial. E01 a E03 não tinham log salvo; os logs atuais são de uma reexecução em 2026-11-09 com o mesmo código, e os números citados no ensaio vêm de E04 e E11, que têm log original.

## Medições de energia (E04, E05)

As medições usaram os contadores internos do M5 (`zeus-apple-silicon`). A carga de fundo da máquina foi maior que o sinal medido, e os joules líquidos ficaram com intervalo interquartil cruzando zero. Conclusões sobre energia exigem máquina ociosa ou medidor externo. Com carga de fundo, a latência é a medida usada.

## Tempo de execução (Apple M5)

| ID | Tempo aproximado |
|---|---|
| E01 | 5 s |
| E02 | 26 s |
| E03 | 44 s |
| E11 | 3 h 20 min |

Os demais não registraram o tempo total.
