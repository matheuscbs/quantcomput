# Quantos bits custa aprender?

Matheus Cardoso, engenheiro eletricista

*Ensaio para a seleção preliminar do Doutorado Profissional em Engenharia de Sistemas Inteligentes e Tecnologias Quânticas: quais conhecimentos poderão transformar a inteligência artificial e as tecnologias quânticas nos próximos dez anos?*

Computadores analógicos, ópticos e quânticos têm um problema em comum que quase não aparece nas conversas sobre eficiência em IA: nenhum deles faz backpropagation com naturalidade. Minha aposta para os próximos dez anos é uma lei de informação para o treinamento físico, que diga quantos bits de realimentação uma máquina precisa receber para aprender uma tarefa e quanto tempo e energia a física cobra por cada bit. Até onde pesquisei, essa conta ainda não foi feita. Sem ela, fica difícil separar o hardware promissor do hardware que só funciona no slide.

O backpropagation é um pequeno milagre digital. Baur e Strassen mostraram em 1983 que o gradiente completo custa poucas vezes o preço de avaliar a função, seja qual for o número de parâmetros. Para isso, a máquina guarda os estados intermediários e percorre o cálculo de trás para frente. Um circuito analógico tem componentes imperfeitos e não tem esse caminho de volta exato. Num processador quântico, medir destrói o estado e copiá-lo é proibido, e Abbas et al. mostraram o quanto isso encarece reaproveitar informação entre as medidas. Na prática, o método usual executa o circuito duas vezes por parâmetro, com muitas repetições em cada uma. Bowles et al. encontraram famílias de circuitos que escapam disso, ao preço de restringir o que o circuito pode fazer.

O prêmio é grande. Tkachenko estima que o treinamento numa rede física poderia dissipar mais de dez ordens de grandeza menos que o hardware atual, numa conta que deixa de fora conversão, controle e medição, custos que o próprio autor diz dominarem hoje. A medição é justamente onde o treino acontece.

Comecei querendo treinar uma rede em superposição. O algoritmo de Jordan faz algo bonito: coloca todas as perturbações dos parâmetros em superposição, acumula a perda como fase, e a Transformada de Fourier Quântica lê o gradiente. Para uma rede clássica, porém, ele perde a disputa. Gilyén, Arunachalam e Wiebe mostraram que o caso genérico exige da ordem de √d consultas para d parâmetros, enquanto o backpropagation não depende de d. Carregar dados clássicos em superposição também cobra caro, como argumentam Jaques e Rattew, e Chia et al. mostraram que boa parte do ganho prometido nesse modelo tem versão clássica. Concluí que a superposição interessa onde o backpropagation não existe.

Lá ela já tem uma versão clássica, velha conhecida de quem estudou engenharia elétrica. No gradiente multiplexado do NIST (McCaughan et al.), cada parâmetro oscila levemente na sua própria frequência, e um processo do tipo lock-in separa do sinal da perda a contribuição de cada um. Todos os parâmetros são perturbados ao mesmo tempo, numa superposição de tons. O algoritmo de Jordan e o lock-in fazem a mesma conta, porque nos dois o gradiente é um coeficiente de Fourier da perda perturbada.

Essa leitura traz um limite que a engenharia conhece bem. Numa janela de duração T, tons só se distinguem se estiverem separados por cerca de 1/T (Gabor). Com N parâmetros numa banda B, medir o gradiente inteiro exige T ≥ N/B, e o custo cresce com o tamanho da rede. Minha hipótese é que N é o número errado:

> **O treino físico é limitado pela informação que a tarefa exige, e essa informação não cresce proporcionalmente ao número de parâmetros: T ≥ I / (B · log₂(1 + SNR)), onde I é o menor número de bits de realimentação que leva a uma dada acurácia.**

A forma vem de Shannon. A novidade estaria em definir I, medir I e mostrar que ele governa o tempo e a energia de máquinas reais.

As transformadas entram com papéis diferentes. Fourier define quantos tons cabem. Laplace descreve o que acontece enquanto os pesos mudam: o sinal deixa de ser uma oscilação pura e ganha decaimento. O tom então vira uma linha de formato lorentziano, com largura proporcional à taxa de mudança. Daí sai uma previsão testável: quanto maior a taxa de aprendizado, mais largas as linhas e menos canais cabem na banda. A ponte tem limite claro, já que Shannon supõe um canal estável e a perda de uma rede em treino muda a cada passo. Parte da pesquisa é descobrir quanto da fórmula sobrevive a isso.

Já existe evidência parcial. Oripov et al., também do NIST, relatam que o tempo para atingir uma acurácia alvo não segue a escala linear em N e muitas vezes cai quando a rede cresce. Fiz um teste pequeno com Fashion-MNIST, em redes de 13.002 a 118.282 parâmetros, duas sementes e 1.500 passos, entregando ao otimizador só as projeções do gradiente em r direções aleatórias por passo. Com r = 8 a acurácia foi de 69,6% na rede menor e 74,2% na maior; com r = 128, de 78,8% e 80,7%. Nove vezes mais parâmetros não pioraram o resultado, e 12.000 medidas levaram a rede de 118 mil parâmetros a 74%, contra 10% do acaso. O backpropagation chegou a 84,5% e 87,0%, e nenhum valor de r até 128 ficou a menos de dois pontos dele, que era o critério que fixei antes de rodar. Tentei também escolher as direções a partir das medidas anteriores, e essa ideia foi pior que o acaso: 78,0% contra 80,7% na rede maior. O teste simula uma medida perfeita, sem ruído, em uma tarefa só. Serve para mostrar que a pergunta merece ser feita.

Cheguei aqui depois de descartar outras ideias. Uma memória em superposição baseada em Fourier, que eu pretendia usar no lugar da atenção do Transformer, saiu de 6 a 9 vezes pior quando somei a limpeza do ruído. Uma medição direta de energia no meu computador falhou porque a carga de fundo era oito vezes maior que o sinal, e desde então só confio em joules líquidos com medidor externo ou máquina ociosa.

No quântico, a mesma pergunta ganha uma forma precisa. Pelo limite de Holevo, n qubits entregam no máximo n bits por medida, então o número de execuções do circuito tem de ser ao menos I/n, antes mesmo de contar o ruído. Para que o treino seja viável, a conta precisa fechar com preparação de estado, leitura, comunicação e correção de erros incluídas. Considero vantagem verificável apenas o caso em que a acurácia, conferida classicamente em dados separados, é atingida com menor custo total que o do melhor treino clássico ou analógico. Espero encontrar isso primeiro em dados que já nascem quânticos, sem etapa de carga. Há também uma ponte a examinar: o método LCHS, de An, Liu e Lin, escreve um decaimento como soma de oscilações com peso lorentziano, o que permitiria aproximar em hardware unitário o decaimento de um gradiente. A cauda lenta dessa curva e a perda de amplitude ao longo do decaimento podem consumir o ganho, e não encontrei aplicação disso ao treinamento.

Mais computação não responde a essa pergunta. Com GPUs melhores treinamos modelos maiores, e I continua sem definição, porque ele depende de teoria da informação e da física do canal. O que a computação pode fazer é medir I em muitas tarefas, e eu começaria por aí.

Organizaria o doutorado em quatro etapas. Primeiro, definir I e medi-lo em tarefas de complexidade crescente, com realimentação quantizada e ruidosa. Segundo, construir uma bancada com amplificadores operacionais e FPGA, em que banda, ruído e energia sejam medidos com instrumento externo, para testar a fórmula e a previsão sobre a taxa de aprendizado. Terceiro, derivar a versão quântica, com ruído de disparo e sem cópia de estados. Quarto, transformar o resultado em regra de projeto: dada uma tarefa e um hardware, quanto custa treinar.

A hipótese estará refutada se I crescer proporcionalmente a N em tarefas realistas, se a fórmula errar por ordens de grandeza na bancada, ou se o treino digital seguir mais barato por joule depois de contadas todas as conversões. Considero essa última possibilidade séria.

Ainda não sei se I tem uma definição única, ou se muda tanto com o modelo e o otimizador que deixa de dizer algo sobre a tarefa. Meu teste não distingue essas situações. Esse é o primeiro problema que eu gostaria de resolver.

## Referências

- Baur e Strassen (1983), *The complexity of partial derivatives*, Theoretical Computer Science 22
- Abbas et al. (2023), *On quantum backpropagation, information reuse, and cheating measurement collapse*: https://arxiv.org/abs/2305.13362
- Bowles, Wierichs e Park (2025), *Backpropagation scaling in parameterised quantum circuits*, Quantum: https://quantum-journal.org/papers/q-2025-10-02-1873/
- Tkachenko (2025), *Thermodynamic cost of inference and learning in physical neural networks* (preprint): https://arxiv.org/abs/2503.09980
- Jordan (2005), *Fast quantum algorithm for numerical gradient estimation*, Physical Review Letters 95
- Gilyén, Arunachalam e Wiebe (2019), *Optimizing quantum optimization algorithms via faster quantum gradient computation*: https://arxiv.org/abs/1711.00465
- Jaques e Rattew (2023), *QRAM: A Survey and Critique*: https://arxiv.org/abs/2305.10310
- Chia et al. (2020), *Sampling-based sublinear low-rank matrix arithmetic framework for dequantizing quantum machine learning*: https://arxiv.org/abs/1910.06151
- McCaughan et al. (2023), *Multiplexed gradient descent*, APL Machine Learning: https://arxiv.org/abs/2303.03986
- Oripov et al. (2025), *Scaling of hardware-compatible perturbative training algorithms*: https://arxiv.org/abs/2501.15403
- Gabor (1946), *Theory of communication*, Journal of the IEE
- Shannon (1948), *A mathematical theory of communication*, Bell System Technical Journal
- Holevo (1973), *Bounds for the quantity of information transmitted by a quantum communication channel*, Problems of Information Transmission
- An, Liu e Lin (2023), *Linear combination of Hamiltonian simulation for nonunitary dynamics with optimal state preparation cost*: https://arxiv.org/abs/2303.01029
- Código dos testes: disponível mediante pedido.
