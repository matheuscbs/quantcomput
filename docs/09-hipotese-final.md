# 09. Hipótese final: quanto custa gravar o que se aprende

Este documento junta os resultados do E11 ao E16, a pesquisa sobre o custo físico de aprender e as leituras novas sobre plasticidade e memória numa hipótese única, e diz como testá-la. Cada afirmação vem marcada: **medido** (experimento deste repositório), **conferido** (lido no texto completo da fonte), **derivado** (conta feita aqui) ou **proposto** (ainda sem teste).

## 1. A hipótese em uma frase

Aprender custa os bits úteis que a tarefa exige, pagos ao menos uma vez quando são gravados de forma durável. O que afasta o custo real desse piso é quantas vezes o sistema grava o que depois desfaz. Em treino por medidas ruidosas esse fator é grande e previsível pela SNR de cada atualização, e um cache transitório que acumula antes de consolidar o recupera quase todo.

## 2. De onde ela vem

### O que os experimentos já mostraram (medido)

- A tarefa decide quantos bits chegam ao otimizador: 88,5% da variação de $`\log_2 I`$ no E13, e a rede 3,4 vezes maior usou 0,90 vez os bits.
- A informação útil por medida é baixa e limitada pelo ruído do minilote: de 0,1 a 0,4 bit por projeção no E15. A acurácia seguiu $`b_{\text{ef}}`$ ($`R^2`$ = 0,89) e não a capacidade do canal ($`R^2`$ = 0,02).
- A precisão de leitura exigida depende da forma de medir. No E14, a leitura absoluta precisou de 16 bits e a diferencial, de 2.

Esses resultados tratam do fluxo de aquisição. Não dizem nada sobre o custo de gravar os pesos.

### Por que gravar é o fluxo caro (conferido)

- Na mosca, a memória de longo prazo custa cerca de 10 mJ por bit (Girard, Jiang e van Rossum, 2023). Transmitir um bit num neurônio custa cerca de $`4\times10^{-13}`$ J (Laughlin et al., 1998). A diferença é de cerca de $`10^{10}`$ vezes (derivado).
- Tkachenko (2025, arXiv:2503.09980) mapeia uma rede direta num hamiltoniano físico. Pela análise dele, a inferência quase estática não exige trabalho. Já gravar os parâmetros custa "a few $`k_BT`$ per parameter", uma única vez, para os valores guardados no fim. Uma execução digital paga o apagamento a cada apresentação de exemplo. Segundo o autor, a dissipação residual tem uma correção "set by the variability of the parameter forces across training examples". O preço da rede, conclui ele, está na memória e não na aritmética.
- No chip Loihi, uma operação sináptica custa no mínimo 23,6 pJ e uma atualização de sinapse (STDP aos pares), 120 pJ (Davies et al., 2018).
- Uma célula de memória de mudança de fase (PCM) gasta de algumas centenas de fJ por bit até menos de 40 fJ nos dispositivos com nanotubos de carbono (revisão arXiv:2605.28336).

### Como a biologia lida com isso (conferido)

- Li e van Rossum (2020, eLife 9:e50804) medem o custo de plasticidade como $`M = \sum \lvert \Delta w\rvert`$. O perceptron que classifica 1.900 padrões com 1.000 sinapses gasta cerca de 900 vezes o mínimo. Uma rede de várias camadas em MNIST gasta pelo menos cerca de 20 vezes o mínimo. Guardar as mudanças primeiro numa forma transitória e consolidar só de vez em quando ("synaptic caching") reduz a energia em até 10 vezes.
- No modelo de Karbowski e Urban (arXiv:2404.14123), as taxas de ganho de informação e de gasto de energia têm o pico durante a indução do LTP e depois voltam à linha de base, enquanto o traço de memória dura muito mais. Os autores concluem que a fase de aprendizado gasta muito mais energia que a de memória.
- Pelo volume da cabeça das espinhas dendríticas, uma sinapse guarda de 4,1 a 4,59 bits, com 24 tamanhos distinguíveis (Samavat et al., 2024, *Neural Computation*). O armazenamento durável usa poucos bits por peso.
- Num modelo ajustado a cinco conjuntos de dados de plasticidade, a precisão sináptica segue $`\sigma^{-2} \propto E^5`$ (arXiv:2602.15787). A precisão é limitada pela energia disponível.
- O produto de precisão e velocidade num canal físico é limitado pela potência dissipada (Lahiri, Sohl-Dickstein e Ganguli, arXiv:1603.07758). Gravar devagar custa menos.

## 3. A conta (derivado)

Num treino em que cada atualização é ruidosa, cada peso faz um passeio aleatório com deriva $`\mu`$ e desvio $`s`$ por passo. Em T passos, o caminho soma $`T\,\mathbb{E}\lvert\mathcal N(\mu,s^2)\rvert`$ e o deslocamento líquido vale $`\mathbb{E}\lvert\mathcal N(T\mu, Ts^2)\rvert`$. A ineficiência de caminho é a razão entre os dois:

```math
\rho = \frac{M}{M_{\min}} \approx \frac{T\,\mathbb{E}\lvert\mathcal N(\sqrt q, 1)\rvert}{\mathbb{E}\lvert\mathcal N(T\sqrt q, T)\rvert}, \qquad q = \frac{r}{N}\,\frac{\mathrm{SNR}_{\text{ef}}}{1+\mathrm{SNR}_{\text{ef}}}.
```

$`q`$ é a SNR por coordenada quando o gradiente chega por $`r`$ projeções aleatórias, e $`\mathrm{SNR}_{\text{ef}}`$ é a do E15. Nos dois extremos:

- deriva dominante ($`Tq \gg 1`$): $`\rho \approx \sqrt{2/\pi}\,\sqrt{(N/r)(1+1/\mathrm{SNR}_{\text{ef}})}`$, que cresce com $`\sqrt N`$;
- ruído dominante ($`Tq \ll 1`$): $`\rho \approx \sqrt T`$, e o próprio deslocamento final é quase todo ruído.

Num modelo quadrático com treino projetado, a fórmula ficou a no máximo cerca de 20% do $`\rho`$ medido em 12 combinações de N, r e SNR (`experimentos/E17-cache-sinaptico/auxiliares/toy2.py`).

Com um cache de limiar $`\kappa\sigma`$, a simulação de passeio aleatório dá ganho $`G(\kappa)`$ de cerca de $`0{,}75\,\kappa`$ até saturar em $`\rho`$. A carga média do cache cresce em proporção a κ (`auxiliares/rw.py`). Se gravar custa $`e_{\text{esc}}`$ por unidade e manter o cache custa $`e_{\text{cache}}`$ por unidade e por passo, o custo de consolidação cai como $`1/\kappa`$ e o de manutenção sobe como $`\kappa`$. O ótimo fica em $`\kappa^* \propto \sqrt{e_{\text{esc}}/e_{\text{cache}}}`$, e o custo mínimo cresce como $`\sqrt{e_{\text{esc}}\,e_{\text{cache}}}`$. No E17 (medido, análise posterior), $`G`$ e a carga cresceram um pouco abaixo de linear em κ, e o expoente de κ* que essas inclinações preveem ficou entre 0,56 e 0,77. A grade de κ foi grossa demais para medir o expoente direto.

## 4. A forma completa (proposto)

```math
E_{\text{aprender}} \approx \underbrace{e_{\text{med}}\,M_{\text{med}}}_{\text{aquisição}} \;+\; \underbrace{e_{\text{esc}}\,\frac{\rho}{G(\kappa)}\,M_{\min}}_{\text{consolidação}} \;+\; \underbrace{e_{\text{cache}}\,L(\kappa)\,T}_{\text{manutenção do transitório}}
```

com o piso $`M_{\min} \ge`$ o deslocamento necessário para guardar os bits úteis, e cada bit útil gravado custando ao menos $`kT\ln 2`$.

| Parte da conta | No cérebro | Em hardware |
|---|---|---|
| aquisição com poucos bits úteis | córtex em escassez baixa a precisão e economiza 29% do ATP sináptico (Padamsey et al., 2022) | leitura diferencial de 2 bits no lugar de 16 (E14) |
| cache transitório $`u`$ | LTP de fase inicial, sem síntese de proteína | acumulador digital ou capacitor ao lado da célula |
| consolidação rara | LTP tardio; fluxo de energia sobe no corpo cogumelo só depois do treino espaçado (Plaçais et al., 2017) | escrita em PCM, RRAM ou flash só quando o acumulado passa do limiar |
| pouca precisão no que fica | 4,1 a 4,59 bits por sinapse | pesos de poucos níveis |

## 5. Então, quantos bits custa aprender

A resposta tem duas contas. Os bits que entram são fixados pela tarefa: cerca de 38 mil entregues e cerca de 3 mil úteis em Fashion-MNIST com 10 classes (medido). Os bits que se gravam são esses multiplicados por $`\rho`$. No E17 (medido), $`\rho`$ ficou entre 21 e 49 no treino projetado e perto de 5 no backprop, e a fórmula da seção 3 previu cada célula com erro abaixo de 5%. Sem cache, o sistema grava dezenas de vezes o que guarda. Um cache com limiar de 16 passos típicos reduz a gravação de 9 a 18 vezes, e com limiar maior chega ao mínimo. Contado em eventos, o cache com κ = 256 escreveu cada peso uma vez, contra 141 a 884 escritas por peso com κ = 1 (medido, análise posterior).

Em resumo: o custo de aprender, em bits, é o que a tarefa exige vezes $`\rho`$, e $`\rho`$ depende de quanto ruído cada atualização carrega, não do tamanho da rede. No regime medido, $`\rho \approx \sqrt T`$. Um sistema que lê pouco por passo precisa de mais passos e grava mais, a menos que acumule antes de consolidar, como faz a sinapse.

## 6. Testes

| Teste | O que testa | Estado |
|---|---|---|
| E17, cache sináptico | H17: $`\rho \ge 10`$ sem cache; H18: $`\rho`$ segue $`\rho_{\text{prev}}`$; H19: $`G(16) \ge 5`$ | H17, H18 e H19 aprovadas; ver [README](../experimentos/E17-cache-sinaptico/README.md) |
| E18, consolidação por reforço | um cache que só consolida o que se repete (marcação e captura sináptica), em vez de decair. O decaimento simples impediu o treino de chegar ao alvo no modelo de brinquedo. | a desenhar |
| E19, bancada de energia | P2 a P4 da proposta teórica: precisão ótima de leitura, curva do conversor e escrita em memória não volátil medidas em joules | depende de hardware |

## 7. Limites

- $`M = \sum\lvert\Delta w\rvert`$ é uma medida de custo escolhida por Li e van Rossum. Um dispositivo real pode cobrar por evento de escrita, por nível alterado ou por pulso. O E17 contou também os eventos, e o cache os reduziu ainda mais que $`M`$.
- No E17, o cache não muda a trajetória do treino. Ele mede o custo de gravar e não ganha nada em acurácia.
- A fórmula supõe que o sinal se espalha por igual pelas coordenadas. No E17 isso não pesou, porque o treino ficou no regime em que o ruído domina e nesse regime $`\rho \approx \sqrt T`$ quase não depende de como o sinal se distribui (derivado). No regime de deriva, em que $`\rho`$ dependeria de $`\sqrt{N/r}`$, a heterogeneidade pode pesar, e esse regime não foi alcançado.
- O cache do E17 não decai. Uma sinapse real perde o transitório se ele não for consolidado, e isso fica para o E18.
- Os trabalhos de 2025 e 2026 citados aqui (Tkachenko; arXiv:2602.15787; arXiv:2605.28336) ainda são pré-publicações.
