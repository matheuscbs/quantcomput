# E17: cache sináptico e o custo de gravar

> Linha de pesquisa: hipótese final (H17, H18, H19; ver [docs/09](../../docs/09-hipotese-final.md)). Critérios registrados no git antes da execução.

## Pergunta

O E13 contou os bits que chegam ao otimizador até o alvo. Um sistema físico também paga para gravar os pesos, e na biologia e no hardware de memória essa é a etapa cara. Quanto um treino por medidas ruidosas grava além do necessário, e um cache transitório, como a plasticidade de fase inicial do cérebro, recupera essa diferença?

## Medida

Custo de gravação de Li e van Rossum (2020): $M = \sum_t \sum_i \lvert \Delta w_i(t)\rvert$, o comprimento L1 do caminho dos pesos. O mínimo para chegar ao mesmo ponto é $M_{\min} = \lVert w_T - w_0\rVert_1$. Ineficiência de caminho: $\rho = M/M_{\min}$.

Cache: cada peso tem uma parte persistente e uma transitória $u_i$. A atualização entra em $u_i$, e quando $\lvert u_i\rvert \ge \kappa\sigma$ o valor é consolidado (custo $\lvert u_i\rvert$) e zerado. O peso usado no forward é a soma das duas partes, então o cache não muda a trajetória do treino. Ele muda só quanto é gravado. Ganho: $G(\kappa) = M/M_c(\kappa)$.

## Previsão

Com $r$ projeções aleatórias por um canal de 2 bits, a SNR por coordenada da atualização é $q = (r/N)\,\mathrm{SNR}_{\text{ef}}/(1+\mathrm{SNR}_{\text{ef}})$, com $\mathrm{SNR}_{\text{ef}}$ do E15. Para um passeio aleatório com deriva,

$$\rho_{\text{prev}} = \frac{T\,\mathbb{E}\lvert \mathcal N(\sqrt q, 1)\rvert}{\mathbb{E}\lvert \mathcal N(T\sqrt q, T)\rvert},$$

que vale $\sqrt{2/\pi}\sqrt{(N/r)(1+1/\mathrm{SNR}_{\text{ef}})}$ quando a deriva domina e $\sqrt T$ quando o ruído domina.

Conferência antes do registro, em `auxiliares/`:

- `rw.py`: num passeio aleatório, $\rho$ medido bateu com $\mathbb{E}\lvert\Delta\rvert/\lvert\mathbb{E}\Delta\rvert$ nos quatro níveis de SNR, e $G(\kappa)$ cresceu perto de $0{,}75\,\kappa$ até saturar em $\rho$.
- `toy2.py`: num modelo quadrático com treino projetado, a fórmula ficou a no máximo cerca de 20% do $\rho$ medido em 12 combinações de N, r e SNR.
- `toy.py`: um cache transitório que decai impediu o treino de chegar ao alvo em 20.000 passos, então o decaimento ficou fora deste teste.

## Protocolo

O protocolo do E13 é mantido, com SGD sem momento. Fashion-MNIST com k = 10, inicialização padrão, lote 128 e canal gaussiano de 2 bits por projeção. O alvo é o do E13 (72,26%), e cada execução para no alvo ou em 6.000 passos. A grade cruza as arquiteturas prof (N = 118.282) e larga (N = 407.050) com r em {16, 32, 64} e 3 sementes, o que dá 18 execuções. Há mais 6 execuções de backprop. Os valores de κ são {1, 4, 16, 64, 256}, medidos na mesma execução.

O lr foi calibrado com a semente 99, fora das sementes do teste (`auxiliares/calibracao.log`). O critério foi a maior acurácia em 1.500 passos, o mesmo do E13. O treino projetado ficou com lr = 10 (lr = 30 diverge) e o backprop com lr = 0,3. Duas leituras da calibração ficam registradas aqui porque podem pesar nos vereditos. No backprop, $\rho$ cresceu com o lr, de 5,4 em 0,01 para 36,3 em 0,3. No treino projetado, $\rho$ ficou entre 34,6 e 36,5 para lr de 1 a 10, perto de $\sqrt{1500} \approx 39$, o que sugere o regime dominado por ruído.

## Critérios

- **H17 (gravar é ineficiente sem cache):** aprovada se $\rho \ge 10$ em pelo menos 17 das 18 execuções projetadas e a mediana projetada for pelo menos 3 vezes a do backprop em cada arquitetura. Reprovada se $\rho < 10$ em mais de 6. Parcial nos demais casos.
- **H18 (a SNR por coordenada prevê ρ):** aprovada se a mediana de $\rho/\rho_{\text{prev}}$ estiver em [1/3; 3] e a correlação de Spearman entre os dois for de pelo menos 0,6. Reprovada se a mediana sair de [1/10; 10] ou se a correlação for ≤ 0. Parcial nos demais casos.
- **H19 (o cache recupera o desperdício):** aprovada se, nas 6 células, $G(\kappa)$ não decrescer com κ e $G(16) \ge 5$. Reprovada se $G(16) < 2$ em alguma célula ou se $G$ cair entre κ consecutivos em 2 ou mais células.
- **Inconclusivo:** mais de 1/4 das execuções projetadas sem atingir o alvo.
- **Exploratório:** carga do cache contra κ, energia $M_c + c\,L\,T$ para $c$ em {1e-4, 1e-3, 1e-2} (previsão: mínimo interior que cresce como $\sqrt c$), $M_{\min}$ contra N e o MQO de $\log\rho$ em $\log N$, $\log r$ e $\log T$.

## Como rodar

```bash
make e17
```

## Resultado

A preencher depois da execução oficial.
