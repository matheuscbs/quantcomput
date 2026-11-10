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
| H08 | O treino físico é limitado pela informação da tarefa, $T \ge I/(B\log_2(1+\mathrm{SNR}))$, com $I$ sublinear em N | **aberta (central), forma a revisar** | E11, E12 | acurácia com r fixo não cai com N (também com canal de 1 a 8 bits); critério de 2 pontos não atingido; no E12 o total de bits não governa a acurácia, então o termo $\log_2(1+\mathrm{SNR})$ superestima a informação útil quando o SNR é alto |
| H09 | Escolher direções de medida a partir das medidas anteriores reduz os canais necessários | refutada no E11 | E11 | adaptativo pior que aleatório (78,0% contra 80,7% no maior N) |
| H10 | O total de bits $r \cdot b \cdot S$ governa a acurácia, e não $r$ ou $b$ separados | **refutada no E12** | E12 | R² de 0,70 a 0,84 contra 0,90 exigido; acurácia segue r, e b satura em 1 a 2 bits (ruído) ou 2 a 4 bits (quantização) |
| H14 | A precisão útil por medida é baixa porque o ruído do minilote já domina o erro de cada projeção; ruído de canal abaixo dele não muda o treino | a testar | E13d | sugerida pelo E12: SNR = 3 por projeção custa no máximo 1 ponto |
| H11 | $I$ é dominado pela tarefa, não pelas interações com arquitetura, inicialização e algoritmo | a testar | E13 | |
| H12 | $I$ cresce com o número de classes k da tarefa (dimensão do subespaço de topo do gradiente), não com N | a testar | E13a | sugerida por Gur-Ari et al. (2018) e Oripov et al. (2025); ver [leituras](08-leituras.md#3-o-subespaço-do-gradiente-gur-ari-et-al-2018-aghajanyan-et-al-2020) |
| H13 | Um modo adaptativo que acumula a estimativa do subespaço de topo ao longo de vários passos se aproxima do oráculo | a testar | E13b | explica o fracasso de H09 |

## Ideias descartadas sem experimento

| Ideia | Motivo |
|---|---|
| Algoritmo de Jordan para treinar redes clássicas | exige da ordem de $\sqrt{d}$ consultas no caso genérico; o backpropagation não depende de d (Gilyén, Arunachalam e Wiebe, 2019) |
| DQI com decodificador aprendido como tese | avaliada como de baixo retorno para quatro anos de doutorado |
