# Registro de hipóteses

Cada hipótese recebe um identificador permanente. O estado muda quando um experimento a testa. Hipóteses refutadas continuam aqui, com o motivo, para que não voltem sem evidência nova.

| ID | Hipótese | Estado | Testada em | Motivo do estado |
|---|---|---|---|---|
| H01 | A energia espectral do gradiente inicial prevê quando treinar em poucos modos de Fourier compensa | refutada como regra geral | E01 | funciona só com alvo isotrópico |
| H02 | Uma base de Fourier fixa comprime os pesos de uma rede treinada | refutada nas camadas ocultas | E02 | simetria de permutação: nenhuma base fixa ganha da aleatória |
| H03 | Memória em superposição (HRR via FFT) substitui a atenção com menos energia | refutada | E03, E04 | G1 e G2 reprovados; dimensão de 8x a 512x maior que a da atenção |
| H04 | O custo de ler um dado cresce geometricamente com o tamanho da memória | não observada | E05 | expoente perto de zero na DRAM até 4 GB |
| H05 | A comunicação necessária num Transformer é local | refutada | E06 | janela fixa exige 75% dos pares |
| H06 | A atenção esparsa por conteúdo tem custo sublinear em N, com expoente vindo do modelo | plano B | E07, E08, E09 | m ~ N^0,4 a 0,5, igual com frases embaralhadas; considerada insuficiente como tese de fronteira |
| H07 | Treinar já com orçamento de conexões reduz o expoente de m(N) | inconclusiva | E10 | modelo pequeno quase não usa contexto (T1) |
| H08 | O treino físico é limitado pela informação da tarefa, $T \ge I/(B\,b_{\text{ef}})$, com $b_{\text{ef}} = \tfrac12\log_2(1+\mathrm{SNR}_{\text{ef}})$ e $I$ sublinear em N | **aberta (central), forma revisada na v0.3** | E11 a E16 | a forma original com $\log_2(1+\mathrm{SNR})$ do canal foi substituída pelo SNR efetivo (E15: $R^2$ 0,89 contra 0,02); $I$ dominado pela tarefa e sem crescimento com N (E13); resultado mantido sem backpropagation (E14) |
| H09 | Escolher direções de medida a partir das medidas anteriores reduz os canais necessários | refutada no E11 | E11 | adaptativo pior que aleatório (78,0% contra 80,7% no maior N) |
| H10 | O total de bits $r \cdot b \cdot S$ governa a acurácia, e não $r$ ou $b$ separados | **refutada no E12** | E12 | R² de 0,70 a 0,84 contra 0,90 exigido; acurácia segue r, e b satura em 1 a 2 bits (ruído) ou 2 a 4 bits (quantização) |
| H14 | A precisão útil por medida é baixa porque o ruído do minilote já domina o erro de cada projeção; ruído de canal abaixo dele não muda o treino | **parcial** | E15, E16 | H14a aprovado ($\mathrm{SNR}_B$ de 0,23 a 0,75); H14b reprovado no E15 e inconclusivo no E16: nenhum lote passou de $B_{\text{simple}}$, que cresceu de cerca de 565 para 5.079 com o lote |
| H11 | $I$ é dominado pela tarefa, não pelas interações com arquitetura, inicialização e algoritmo | **aprovada no E13** | E13 | tarefa 88,5% da soma dos quadrados de $\log_2 I$; interações 6,7% |
| H12 | $I$ cresce com o número de classes k da tarefa (dimensão do subespaço de topo do gradiente), não com N | **aprovada no E13** | E13 | monotônico em k nas 8 combinações; inclinação exploratória 1,79 em log-log; rede 3,4x maior usou 0,90x os bits |
| H13 | Um modo adaptativo que acumula a estimativa do subespaço de topo ao longo de vários passos se aproxima do oráculo | a testar | — | explica o fracasso de H09; não testada na v0.3 |
| H10′ | A acurácia segue a informação efetiva por projeção, $\tfrac12\log_2(1+\mathrm{SNR}_{\text{ef}})$, com o ruído do minilote incluído | **aprovada no E15** | E15, E16 | $R^2$ = 0,89 nas 15 células do E15 (nominal: 0,02); 0,83 nas 23 células com o E16, em leitura exploratória |
| H15 | Com medida por diferença central, sem backpropagation, a acurácia do canal ideal se mantém e não cai com N | **aprovada no E14** | E14 | P1 e P4 em 6 de 6 células; diferença máxima de 0,13 ponto |
| H16 | Ler cada perda separadamente exige da ordem de 16 bits por leitura; ler a diferença exige 2 | **aprovada no E14** | E14 | P2 e P3 em 6 de 6 células; previsão $\log_2(4/(2\varepsilon\,\mathrm{med}\lvert p\rvert))$ de 14,8 a 16,5 bits |
| H17 | Sem cache, o treino por medidas ruidosas grava muito mais do que guarda: $\rho = \sum\lvert\Delta w\rvert / \lVert w_T-w_0\rVert_1 \ge 10$, e pelo menos 3 vezes o do backprop | **aprovada no E17** | E17 | $\rho$ de 21 a 49 nas 18 execuções; 5,2x e 6,9x o do backprop |
| H18 | $\rho$ é previsto pela SNR por coordenada, $q = (r/N)\,\mathrm{SNR}_{\text{ef}}/(1+\mathrm{SNR}_{\text{ef}})$, pela fórmula do passeio aleatório com deriva | **aprovada no E17** | E17 | mediana de $\rho/\rho_{\text{prev}}$ 0,98, Spearman 1,0; regime de ruído, $\rho \approx \sqrt T$ |
| H19 | Um cache transitório com consolidação por limiar (Li e van Rossum, 2020) recupera o desperdício: $G(16) \ge 5$ | **aprovada no E17** | E17 | $G(16)$ de 9,4 a 17,6, crescente em κ nas 6 células; decaimento do transitório fica para o E18 |

## Ideias descartadas sem experimento

| Ideia | Motivo |
|---|---|
| Algoritmo de Jordan para treinar redes clássicas | exige da ordem de $\sqrt{d}$ consultas no caso genérico; o backpropagation não depende de d (Gilyén, Arunachalam e Wiebe, 2019) |
| DQI com decodificador aprendido como tese | avaliada como de baixo retorno para quatro anos de doutorado |
