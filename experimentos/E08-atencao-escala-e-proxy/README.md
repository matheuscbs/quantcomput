# E08: Escala de m com o contexto e busca aproximada

> Linha de pesquisa: Plano B: atenção esparsa. Rodado em 2026-11-08.

## Pergunta

(A) As m conexões necessárias crescem com o contexto N? (B) Dá para achar essas m conexões lendo só uma projeção barata das chaves?

## Como foi feito

GPT-2 sem retreino. Em B, um escore aproximado (projeção em p de 64 dimensões) escolhe m candidatos e a atenção exata só é usada entre eles.

## Critérios

Teste exploratório.

## Resultado

- Escala: para perda extra de até cerca de 0,02 nat, m fica perto de 32 para N = 128 e 256, perto de 64 para N = 512, e entre 64 e 128 para N = 1024. A fração dos pares necessária cai de 44% (N = 128) para 12 a 24% (N = 1024).
- Proxy: com p = 32 e m = 256, a perda extra é 0,041 nat e o recall dos m verdadeiros é 84%. Com p = 4, o recall cai para 38% a 66%.

## Veredito

m cresce mais devagar que N (expoente medido em E09). A busca aproximada funciona só com projeções de pelo menos metade das dimensões.

## Como reproduzir

```bash
poetry install --extras nlp   # numpy já vem na instalação base
poetry run python experimentos/E08-atencao-escala-e-proxy/atencao_escala_e_proxy.py dados/moby.txt saidas/atencao-escala-e-proxy.json
```

- Sementes: `torch.Generator().manual_seed(7)` para as projeções.
- Registros: `registros/log6.txt`.
- Resultados oficiais: `resultados/` (quando há JSON). Novas execuções vão para `saidas/`, fora do git.
