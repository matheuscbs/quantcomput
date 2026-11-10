# Quantos bits custa aprender?

Experimentos, notas e artigo sobre uma pergunta de pesquisa: quantos bits de realimentação uma máquina física (analógica, óptica ou quântica) precisa receber para aprender uma tarefa, e quanto tempo e energia a física cobra por esses bits.

O repositório nasceu de um ensaio para a seleção do Doutorado Profissional em Engenharia de Sistemas Inteligentes e Tecnologias Quânticas do Instituto Ramanujan. Ele guarda o ensaio, os onze experimentos que levaram até a hipótese (inclusive os que a derrubaram em versões anteriores), o teste complementar feito depois (E12), os registros completos das execuções e o artigo em LaTeX.

> **Estado:** v0.2.0, exploratório. Os experimentos rodaram em um notebook (Apple M5) com redes pequenas e canais simulados (sem ruído no E11; quantizados ou ruidosos no E12, sempre a partir do gradiente exato). Servem para mostrar que a pergunta merece ser feita, não para respondê-la.

## A hipótese

Hardware que não faz backpropagation mede o gradiente por perturbação. No gradiente multiplexado (McCaughan et al., NIST), cada parâmetro oscila na sua própria frequência e um processo do tipo lock-in separa a contribuição de cada um. Numa janela de duração $T$ os tons só se distinguem se estiverem a cerca de $1/T$ um do outro, então medir $N$ parâmetros numa banda $B$ exige $T \ge N/B$.

A hipótese troca $N$ pela informação que a tarefa exige:

$$
T \;\ge\; \frac{I}{B \,\log_2(1+\mathrm{SNR})}
$$

onde $I$ é o menor número de bits de realimentação que leva a uma dada acurácia. A forma vem de Shannon. O trabalho está em definir $I$, medi-lo e mostrar que ele governa o tempo e a energia de máquinas reais.

```mermaid
flowchart LR
    A["Hardware físico<br/>sem caminho de volta"] --> B["Gradiente medido<br/>por perturbação"]
    B --> C["Tons multiplexados<br/>(lock-in)"]
    C --> D["Limite de Gabor<br/>T ≥ N/B"]
    D --> E{"N é o número certo?"}
    E -- "hipótese" --> F["T ≥ I / (B·log₂(1+SNR))"]
    F --> G["E11: acurácia com r fixo<br/>não cai com N"]
    F --> H["Questões abertas:<br/>definir e medir I"]
```

## Resultado principal até aqui (E11)

O otimizador recebeu só $r$ projeções do gradiente por passo, em Fashion-MNIST, com redes de 13.002 a 118.282 parâmetros.

| N | backprop | r = 8 | r = 128 |
|---|---|---|---|
| 13.002 | 84,5% | 69,6% | 78,8% |
| 118.282 | 87,0% | 74,2% | 80,7% |

Nove vezes mais parâmetros não pioraram a acurácia com $r$ fixo. Nenhum $r$ até 128 chegou a menos de 2 pontos do backprop, que era o critério fixado antes de rodar. Escolher as direções a partir das medidas anteriores foi pior que sorteá-las.

## Canal com poucos bits (E12)

O E12 repetiu o E11 com um canal de b bits por projeção (quantização ou ruído gaussiano), para ver se a acurácia passa a depender só do total de bits. Não passa. Ela segue o número de medidas r, e a precisão satura cedo:

| N = 118.282, canal ruidoso | b = 1 (SNR = 3) | b = 8 | float32 |
|---|---|---|---|
| r = 8 | 73,2% | 74,1% | 74,2% |
| r = 128 | 80,3% | 80,7% | 80,7% |

Com o mesmo orçamento de bits, mais medidas de baixa precisão ganharam de poucas medidas precisas. Isso pede revisar a forma da hipótese: o termo $\log_2(1+\mathrm{SNR})$ superestima a informação útil quando o SNR é alto. Detalhes em [docs/02-resultados.md](docs/02-resultados.md#e12-canal-quantizado-e-ruidoso).

## Limites

O teste tem limites que precisam ficar claros. As projeções saem do gradiente exato, calculado por backpropagation; no E11 chegam ao otimizador em float32 sem ruído, e no E12 o canal é simulado. A contagem de "medidas" é só $r \times$ passos, sem custo de operação. A seção [Questões abertas](docs/03-questoes-abertas.md) trata disso.

## Trajetória

A hipótese atual veio depois de várias ideias reprovadas pelos próprios testes. Cada uma está registrada com critério, resultado e veredito.

```mermaid
flowchart TD
    subgraph S1["Fourier nos pesos"]
        E01["E01 critério espectral<br/>só vale no caso isotrópico"]
        E02["E02 Fourier no grupo errado<br/>sem ganho nas camadas ocultas"]
    end
    subgraph S2["Memória em superposição"]
        E03["E03 contagem de operações<br/>HRR gasta mais"]
        E04["E04 joules medidos<br/>reprovada em G1 e G2"]
        E05["E05 lei geométrica<br/>não aparece até 4 GB"]
    end
    subgraph S3["Atenção esparsa (plano B)"]
        E06["E06 janela fixa<br/>exige 75% dos pares"]
        E07["E07 escolha por conteúdo<br/>12% dos pares"]
        E08["E08 m cresce devagar com N"]
        E09["E09 expoente 0,4 a 0,5<br/>não vem do discurso"]
        E10["E10 treino com orçamento<br/>inconclusivo"]
    end
    subgraph S4["Limite de canal (hipótese atual)"]
        E11["E11 canais de gradiente<br/>evidência parcial"]
    end
    S1 --> S2 --> S3 --> S4
```

| ID | Pergunta | Veredito |
|---|---|---|
| [E01](experimentos/E01-criterio-espectral) | A energia do gradiente inicial prevê se truncar em frequência compensa? | Só no caso isotrópico |
| [E02](experimentos/E02-fourier-grupo-errado) | Uma base DCT fixa concentra a energia dos pesos treinados? | Só no eixo de entrada |
| [E03](experimentos/E03-memoria-superposicao) | Memória HRR recupera com menos operações que a atenção? | Não |
| [E04](experimentos/E04-microteste-energia) | Com joules medidos, a superposição bate a atenção? | Não (G1 e G2 reprovados) |
| [E05](experimentos/E05-lei-de-escala-memoria) | O custo de leitura cresce com o tamanho da memória? | Não até 4 GB |
| [E06](experimentos/E06-comunicacao-gpt2) | Quanta comunicação de longa distância o GPT-2 usa? | Janela fixa exige 75% dos pares |
| [E07](experimentos/E07-atencao-esparsa-gpt2) | Quantas conexões por conteúdo bastam? | 12% dos pares |
| [E08](experimentos/E08-atencao-escala-e-proxy) | m cresce com o contexto? Uma busca barata acha as conexões? | m sublinear; proxy exige p ≥ 32 |
| [E09](experimentos/E09-expoente-texto-embaralhado) | O expoente de m(N) vem do discurso? | Não |
| [E10](experimentos/E10-treino-com-orcamento) | Treinar já com orçamento muda a lei? | Inconclusivo |
| [E11](experimentos/E11-canais-de-gradiente) | Quantos canais de gradiente o treino exige conforme N cresce? | Evidência parcial |
| [E12](experimentos/E12-canal-quantizado-e-ruidoso) | Com b bits por medida, o total de bits governa a acurácia? | Não: governa r; b satura em 1 a 4 bits |

## Estrutura

```
.
├── README.md                  este arquivo
├── ARQUITETURA.md             como o repositório é organizado e versionado
├── CHANGELOG.md               histórico de versões do conhecimento
├── pyproject.toml             dependências (Poetry), versões fixadas
├── poetry.lock
├── Makefile                   atalhos: instalar, dados, e01..e12, artigo
├── docs/
│   ├── 01-pesquisa.md         pergunta, hipótese e o raciocínio por trás
│   ├── 02-resultados.md       todos os números, por experimento
│   ├── 03-questoes-abertas.md custo das projeções, bits, definição de I, fotônica
│   ├── 04-base-tecnica.md     literatura por eixo
│   ├── 05-reprodutibilidade.md ambiente, versões, sementes, dados
│   ├── 06-roteiro.md          próximos experimentos e etapas do doutorado
│   ├── 07-mapa-ensaio.md      cada número do ensaio → script → campo do JSON
│   ├── 08-leituras.md         síntese da leitura dos artigos-chave
│   ├── hipoteses.md           registro de hipóteses e estado de cada uma
│   └── decisoes/              registros de decisão (ADR)
├── experimentos/
│   └── E01..E12/              script, README, resultados/, registros/, auxiliares/
├── ensaio/                    ensaio original (markdown e PDF)
├── artigo/                    artigo em LaTeX, duas colunas
├── dados/                     SHA256SUMS e instruções; dados fora do git
└── scripts/baixar_dados.py
```

## Como começar

```bash
poetry install --all-extras        # Python 3.12
make dados                         # Fashion-MNIST, Moby Dick, GloVe (cerca de 540 MB)
make e11                           # cerca de 3 h 20 min num Apple M5
make artigo                        # compila artigo/main.tex com tectonic
```

Cada experimento também roda isolado. Os comandos estão no README de cada pasta. As novas execuções gravam em `saidas/`, e os resultados oficiais em `experimentos/*/resultados/` ficam intactos.

## Documentação

- [Pesquisa](docs/01-pesquisa.md): o problema, a ponte entre o algoritmo de Jordan e o lock-in, o papel de Fourier, Laplace e Holevo, e o que refutaria a hipótese.
- [Resultados](docs/02-resultados.md): tabelas completas de todos os experimentos.
- [Questões abertas](docs/03-questoes-abertas.md): o que o E11 mede e o que não mede.
- [Base técnica](docs/04-base-tecnica.md): referências por eixo.
- [Reprodutibilidade](docs/05-reprodutibilidade.md): versões, sementes, hardware, dados.
- [Roteiro](docs/06-roteiro.md): próximos experimentos e as quatro etapas propostas.
- [Mapa ensaio → script → resultado](docs/07-mapa-ensaio.md): cada número do ensaio, com verificação automática (`make verificar`).
- [Leituras aprofundadas](docs/08-leituras.md): o que 20 artigos-chave mostram e o que muda no projeto.
- [Arquitetura](ARQUITETURA.md): organização e versionamento.

## Como citar

Veja `CITATION.cff`. Código, textos e artigo estão sob a licença MIT (`LICENSE`).
