# Changelog

Histórico de versões do conhecimento e do código. Convenção em [ARQUITETURA.md](ARQUITETURA.md#versionamento-do-conhecimento): maior quando a hipótese central muda; meio para experimento ou evidência nova; menor para correções sem efeito nos resultados.

## [Não lançado]

### Adicionado

- `docs/09-hipotese-final.md`: hipótese que junta os resultados do E11 ao E16 ao custo de gravar os pesos. A ineficiência de caminho $\rho$ é prevista pela SNR por coordenada, e um cache sináptico (Li e van Rossum, 2020) a recupera.
- E17 (cache sináptico), com critérios de H17, H18 e H19 commitados antes da execução; calibração de lr e modelos de brinquedo em `auxiliares/`.
- Resultado do E17: H17, H18 e H19 aprovadas. $\rho$ de 21 a 49 sem cache (backprop perto de 5), previsto com erro abaixo de 5%; regime de ruído, $\rho \approx \sqrt T$; $G(16)$ de 9,4 a 17,6.
- Artigo: seção de custo de gravação (formulação, método, resultado do E17, discussão e limites), tabela gerada a partir do JSON oficial e duas referências novas (Li e van Rossum, 2020; Girard, Jiang e van Rossum, 2023).
- E17, análises posteriores ao resultado: eventos de escrita por peso e expoente do κ ótimo contra o preço de manutenção.
- Roteiro renumerado: E18 consolidação por reforço, E19 bancada de energia; tarefas maiores, ruído de avaliação e limite inferior passam a E20, E21 e E22.

## [0.3.0] - 2026-11-16

### Adicionado

- E13 (desenho fatorial de I), E14 (perturbação real, sem backpropagation), E15 (ruído do minilote) e E16 (lote acima de B_simple), com critérios commitados antes da execução (c383af8, 88dca6c, b2dc840).
- `scripts/gerar_artigo.py`: gera as tabelas e figuras de E13 a E16 no artigo a partir dos JSONs oficiais.
- Artigo v0.3 completo, com a formulação revisada e as respostas às três perguntas do instituto.
- Hipóteses H10′, H15 e H16; leituras novas (escala de ruído do gradiente, limites inferiores de ordem zero, energia de conversores) em `docs/08-leituras.md`.

### Mudado

- H08 revisada: o denominador passa a ser a informação efetiva por medida, $\tfrac12\log_2(1+\mathrm{SNR}_{\text{ef}})$, com $\mathrm{SNR}_B = B/B_{\text{simple}}$ para o minilote.
- H11 e H12 aprovadas no E13; H14 parcial (E15) e sem veredito novo no E16 (inconclusivo).

### Resultados

- E13: tarefa 88,5% da soma dos quadrados de $\log_2 I$, interações 6,7%; $I$ cresce com k nas 8 combinações; rede 3,4 vezes maior usa 0,90 vez os bits.
- E14: P0 a P4 aprovados em 6 de 6 células; diferença central a no máximo 0,13 ponto da projeção exata; leitura absoluta exige 16 bits, diferencial 2.
- E15: $\mathrm{SNR}_B$ de 0,23 a 0,75; H10′ com $R^2 = 0{,}89$ (bits nominais: 0,02); H14b reprovado.
- E16: inconclusivo (mediana de $\mathrm{SNR}_B$ de 4,3 e 8,6, abaixo de 10); $B_{\text{simple}}$ cresceu de cerca de 565 para 5.079.

### Observações

- Os processos oficiais de E14 e E15 começaram alguns segundos antes do commit 88dca6c, por um erro na cadeia de comandos. O script executado é idêntico ao commitado, e nenhum resultado existia no momento do commit.
- Lição do E12 aplicada: todos os critérios novos dizem explicitamente "em todas", "em pelo menos n de m" ou "nas duas sementes".

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
