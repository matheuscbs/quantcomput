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

**H17, H18 e H19 aprovadas.** As 18 execuções projetadas e as 6 de backprop chegaram ao alvo, sem nenhuma censurada.

| Célula | passos | $\rho$ medido | $\rho_{\text{prev}}$ | $G(1)$ | $G(4)$ | $G(16)$ | $G(64)$ | $G(256)$ | $M_{\min}$ |
|---|---|---|---|---|---|---|---|---|---|
| prof, r = 16 | 1.997 | 41,8 | 42,0 | 1,3 | 2,9 | 9,7 | 30,8 | 41,8 | 1.737 |
| prof, r = 32 | 1.060 | 30,7 | 30,4 | 1,3 | 2,9 | 9,6 | 26,7 | 30,7 | 1.810 |
| prof, r = 64 | 513 | 21,5 | 21,0 | 1,3 | 2,9 | 9,4 | 20,8 | 21,5 | 1.767 |
| larga, r = 16 | 2.753 | 49,3 | 51,5 | 1,7 | 4,8 | 17,1 | 45,4 | 49,3 | 2.872 |
| larga, r = 32 | 1.263 | 33,8 | 34,8 | 1,8 | 5,2 | 17,6 | 33,6 | 33,8 | 2.784 |
| larga, r = 64 | 693 | 25,1 | 25,7 | 1,8 | 5,2 | 16,6 | 25,1 | 25,1 | 2.873 |

Médias de 3 sementes.

- **H17:** $\rho \ge 10$ nas 18 execuções. A mediana projetada foi 30,1 contra 5,8 do backprop na rede prof (5,2 vezes) e 33,6 contra 4,8 na larga (6,9 vezes). O backprop chegou ao alvo em 30 a 60 passos.
- **H18:** a mediana de $\rho/\rho_{\text{prev}}$ foi 0,98, e a correlação de Spearman, 1,0. No MQO exploratório, $\log\rho$ = 0,49 $\log T$ + 0,008 $\log N$ − 0,002 $\log r$ ($R^2$ = 0,9999). O treino está no regime em que o ruído domina: $\rho \approx \sqrt T$, e N e r só entram pelo número de passos.
- **H19:** $G(\kappa)$ cresce com κ nas 6 células, e $G(16)$ vai de 9,4 a 17,6. Com κ = 256 o ganho chega a $\rho$, ou seja, o cache recupera todo o desperdício.
- **Exploratório, energia com preço de manutenção.** Nas duas células prof com r = 16 e r = 32, o mínimo ficou em κ = 16 com $c$ = 0,01 (custo de 4,1 a 5,1 vezes menor que sem cache), em κ = 64 a 256 com $c$ = 0,001 e no maior κ da grade com $c$ = 0,0001. Quando manter custa mais, o ótimo vai para κ menor, como previsto. A lei $\sqrt c$ não foi ajustada.
- **Exploratório, $M_{\min}$.** Cresceu 1,6 vezes da rede prof para a larga, com N 3,4 vezes maior.

**O que isso significa.** Sem cache, o treino por medidas ruidosas grava de 21 a 49 vezes o deslocamento que guarda, e o backprop, cerca de 5 vezes. A fórmula do passeio aleatório previu cada célula com erro abaixo de 5%. Como $\rho \approx \sqrt T$ e T não cresce com N (E13), o desperdício também não cresce com N nesse regime. Um cache com limiar de 16 passos típicos reduz a gravação de 9 a 18 vezes.

Limites: o cache não muda a trajetória, então o resultado mede a escrita e não a acurácia. O custo $\sum\lvert\Delta w\rvert$ é o de Li e van Rossum, e um dispositivo real pode cobrar por evento de escrita. O regime de deriva dominante ($\rho \propto \sqrt{N/r}$) não foi alcançado aqui.
