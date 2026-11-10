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
| H08 | O treino físico é limitado pela informação da tarefa, $T \ge I/(B\log_2(1+\mathrm{SNR}))$, com $I$ sublinear em N | **aberta (central)** | E11 | acurácia com r fixo não cai com N; canal ideal, critério de 2 pontos não atingido |
| H09 | Escolher direções de medida a partir das medidas anteriores reduz os canais necessários | refutada no E11 | E11 | adaptativo pior que aleatório (78,0% contra 80,7% no maior N) |
| H10 | O total de bits $r \cdot b \cdot S$ governa a acurácia, e não $r$ ou $b$ separados | a testar | E12 | |
| H11 | $I$ é dominado pela tarefa, não pelas interações com arquitetura, inicialização e algoritmo | a testar | E13 | |

## Ideias descartadas sem experimento

| Ideia | Motivo |
|---|---|
| Algoritmo de Jordan para treinar redes clássicas | exige da ordem de $\sqrt{d}$ consultas no caso genérico; o backpropagation não depende de d (Gilyén, Arunachalam e Wiebe, 2019) |
| DQI com decodificador aprendido como tese | avaliada como de baixo retorno para quatro anos de doutorado |
