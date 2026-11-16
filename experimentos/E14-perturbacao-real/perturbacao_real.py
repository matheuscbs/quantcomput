"""O resultado do E11 sobrevive quando a projeção do gradiente é medida como um hardware sem backprop mediria?

E11 e E12 calculam o gradiente exato por backpropagation e só depois o projetam. Aqui não há backward.
Cada projeção v_i . g é estimada por diferença central da perda no mesmo minilote:
    p_i = (L(theta + eps v_i) - L(theta - eps v_i)) / (2 eps),  com ||v_i|| = 1 e eps = 1e-2.
Isso custa 2 avaliações da perda (passes de ida) por projeção, 2r por passo. O otimizador (Adam, lr 1e-3)
recebe V p, como no E11. Mesmas redes, sementes, lotes e direções do E11.

Leituras (como o número chega ao otimizador):
- exata:     V^T g por backprop (controle; deve repetir o E11). Custo: 1 ida + 1 volta por passo.
- dif:       diferença central em float32, sem degradação.
- abs-b:     cada perda L(theta +- eps v) é lida por um conversor de b bits em faixa fixa [0, 4]
             (arredondamento determinístico, sem dither), e a diferença é feita depois. b em {8, 12, 16, 20}.
             Bits por passo: 2 r b.
- difer-b:   a diferença L+ - L- é lida diretamente (leitura diferencial, como num lock-in) por um conversor
             de b bits com faixa adaptada a cada passo: quantização estocástica sem viés em [-s, s],
             s = max|L+ - L-| enviado em float32 (igual ao canal quant do E12). b em {1, 2, 4}.
             Bits por passo: r b + 32.

Previsão escrita antes de rodar: a leitura absoluta precisa resolver 2 eps |v.g| ~ 2e-2 * ||g|| / sqrt(N)
numa faixa de 4, o que pede da ordem de 16 bits ou mais; a leitura diferencial deve herdar o E12 (2 bits bastam).

Grade: N em {13.002, 118.282} (larguras 16 e 128), r em {8, 32, 128}, 1500 passos, lote 128, 2 sementes.

Critérios fixados ANTES de rodar (ver docs/decisoes/0002). Médias das 2 sementes; "célula" = (N, r).
- P0 (controle): exata repete o E11 com diferença absoluta <= 0,5 ponto em TODAS as 6 células.
- P1 (diferença finita basta): dif fica a no máximo 1 ponto de exata em PELO MENOS 5 das 6 células.
- P2 (leitura absoluta exige muitos bits): b_abs*(célula) = menor b em {8, 12, 16, 20} com
  acc(abs-b) >= acc(dif) - 1 ponto (infinito se nenhum). Aprovado se b_abs* >= 16 em PELO MENOS 5 das 6 células.
- P3 (leitura diferencial herda o E12): acc(difer-2) >= acc(dif) - 1,5 ponto em PELO MENOS 5 das 6 células.
- P4 (sem penalidade em N com medida real): acc(118.282) >= acc(13.002) - 1 ponto em PELO MENOS 5 das 6
  combinações (leitura em {dif, difer-2}) x (r em {8, 32, 128}).
Inconclusivo: desvio entre sementes acima de 2 pontos em mais de 1/4 das células (todas as leituras);
nesse caso P1 a P4 não são aplicados.

Custo registrado por execução: avaliações da perda (ida = 1; uma volta conta como 2 idas), leituras do
conversor e bits. Registro exploratório: mediana de |p_i| nas execuções dif, para comparar com a previsão de P2.

Uso:
  python perturbacao_real.py rodar dir_dados saida.json SEMENTES     (SEMENTES = "0", "1" ou "0,1")
  python perturbacao_real.py analisar final.json parcial_1.json [parcial_2.json ...] --e11 e11.json
Variável PASSOS no ambiente encurta o treino (só para teste de fumaça; nunca na execução oficial).
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, vmap

torch.set_num_threads(int(os.environ.get("THREADS", "3")))
LARGURAS = (16, 128)
RS = (8, 32, 128)
ABS_BITS = (8, 12, 16, 20)
DIFER_BITS = (1, 2, 4)
FAIXA_ABS = 4.0
EPS = 1e-2
PASSOS = int(os.environ.get("PASSOS", "1500"))
BATCH, LR = 128, 1e-3


def dados(raiz):
    d = np.load(os.path.join(raiz, "fashion.npz"))
    x = (torch.tensor(d["x"], dtype=torch.float32) / 255 - 0.286) / 0.353
    y = torch.tensor(d["y"], dtype=torch.long)
    return x[:60000], y[:60000], x[60000:], y[60000:]


def rede(w):
    return nn.Sequential(nn.Linear(784, w), nn.ReLU(), nn.Linear(w, w), nn.ReLU(), nn.Linear(w, 10))


def achatar(mod):
    return torch.cat([p.grad.view(-1) for p in mod.parameters()])


def devolver(mod, g):
    i = 0
    for p in mod.parameters():
        n = p.numel()
        p.grad.copy_(g[i:i + n].view_as(p))
        i += n


def perdas_perturbadas(mod, nomes, formas, theta, V, xb, yb):
    """L(theta + eps v_i) e L(theta - eps v_i) para as r colunas de V, sem backward."""
    deltas = torch.cat([V.T, -V.T]) * EPS                 # (2r, N)

    def perda(delta):
        params, i = {}, 0
        for nome, forma in zip(nomes, formas):
            n = int(np.prod(forma))
            params[nome] = (theta[i:i + n] + delta[i:i + n]).view(forma)
            i += n
        return F.cross_entropy(functional_call(mod, params, (xb,)), yb)

    with torch.no_grad():
        L = vmap(perda, chunk_size=64)(deltas)
    r = V.shape[1]
    return L[:r], L[r:]


def ler(Lmais, Lmenos, leitura, b, ger):
    """Devolve (p estimado, bits no passo, leituras do conversor no passo)."""
    r = Lmais.numel()
    if leitura == "dif":
        return (Lmais - Lmenos) / (2 * EPS), 2 * r * 32, 2 * r
    if leitura == "abs":
        passo = FAIXA_ABS / (2 ** b - 1)
        q = lambda L: torch.round(L.clamp(0, FAIXA_ABS) / passo) * passo
        return (q(Lmais) - q(Lmenos)) / (2 * EPS), 2 * r * b, 2 * r
    if leitura == "difer":
        D = Lmais - Lmenos
        s = D.abs().max().clamp_min(1e-30)
        niveis = 2 ** b - 1
        u = (D / s + 1) / 2 * niveis
        base = torch.floor(u)
        sobe = torch.rand(r, generator=ger) < (u - base)
        Dq = ((base + sobe.float()).clamp(0, niveis) / niveis * 2 - 1) * s
        return Dq / (2 * EPS), r * b + 32, r
    raise ValueError(leitura)


def rodar_um(dt, w, leitura, r, b, sem):
    xtr, ytr, xte, yte = dt
    torch.manual_seed(sem)
    g_rng = torch.Generator().manual_seed(1000 + sem)    # direções: igual ao E11
    c_rng = torch.Generator().manual_seed(2000 + sem)    # conversor estocástico
    idx_rng = np.random.default_rng(sem)
    mod = rede(w)
    opt = torch.optim.Adam(mod.parameters(), LR)
    nomes = [n for n, _ in mod.named_parameters()]
    formas = [tuple(p.shape) for _, p in mod.named_parameters()]
    N = sum(p.numel() for p in mod.parameters())
    bits = idas = conv = 0
    mods_p = []
    for _ in range(PASSOS):
        bt = idx_rng.integers(0, len(xtr), BATCH)
        xb, yb = xtr[bt], ytr[bt]
        V = torch.randn(N, r, generator=g_rng)
        V = V / V.norm(dim=0, keepdim=True)
        opt.zero_grad(set_to_none=False)
        if leitura == "exata":
            loss = F.cross_entropy(mod(xb), yb)
            loss.backward()
            p = V.T @ achatar(mod)
            idas += 3
            bits += 32 * r
            conv += r
        else:
            for q in mod.parameters():
                if q.grad is None:
                    q.grad = torch.zeros_like(q)
            theta = torch.cat([q.detach().view(-1) for q in mod.parameters()])
            Lmais, Lmenos = perdas_perturbadas(mod, nomes, formas, theta, V, xb, yb)
            p, nb, nc = ler(Lmais, Lmenos, leitura, b, c_rng)
            idas += 2 * r
            bits += nb
            conv += nc
            if leitura == "dif":
                mods_p.append(float(p.abs().median()))
        devolver(mod, V @ p)
        opt.step()
    with torch.no_grad():
        acc = float((mod(xte).argmax(1) == yte).float().mean())
    return dict(acc=acc, N=N, bits=bits, idas=idas, conversoes=conv,
                mediana_abs_p=float(np.median(mods_p)) if mods_p else None)


def configuracoes():
    yield "exata", None
    yield "dif", None
    for b in DIFER_BITS:
        yield "difer", b
    for b in ABS_BITS:
        yield "abs", b


def rodar(raiz, saida, sementes):
    dt = dados(raiz)
    res = dict(passos=PASSOS, eps=EPS, faixa_abs=FAIXA_ABS, linhas=[])
    t0 = time.time()
    for sem in sementes:
        for w in LARGURAS:
            for leitura, b in configuracoes():
                for r in RS:
                    out = rodar_um(dt, w, leitura, r, b, sem)
                    lin = dict(sem=sem, w=w, leitura=leitura, r=r, b=b, **out)
                    res["linhas"].append(lin)
                    print(lin, round(time.time() - t0), "s", flush=True)
                    json.dump(res, open(saida, "w"), indent=0)


def analisar(final, parciais, e11):
    L = [l for f in parciais for l in json.load(open(f))["linhas"]]
    Ns = sorted({l["N"] for l in L})
    sems = sorted({l["sem"] for l in L})

    def cel(N, leitura, r, b):
        v = [l["acc"] for l in L if l["N"] == N and l["leitura"] == leitura and l["r"] == r and l["b"] == b]
        return float(np.mean(v)), (float(np.std(v)) if len(v) > 1 else 0.0)

    an = dict(N=Ns, sementes=sems, celulas={})
    dps = []
    for N in Ns:
        for leitura, b in configuracoes():
            for r in RS:
                m, dp = cel(N, leitura, r, b)
                lin = [l for l in L if l["N"] == N and l["leitura"] == leitura and l["r"] == r and l["b"] == b]
                an["celulas"][f"{N}|{leitura}|{b}|{r}"] = dict(
                    acc=m, dp=dp, bits=lin[0]["bits"], idas=lin[0]["idas"], conversoes=lin[0]["conversoes"],
                    mediana_abs_p=(float(np.mean([l["mediana_abs_p"] for l in lin]))
                                   if lin[0]["mediana_abs_p"] is not None else None))
                dps.append(dp)
    an["inconclusivo"] = bool(len(sems) > 1 and np.mean(np.array(dps) > 0.02) > 0.25)
    A = lambda N, leitura, b, r: an["celulas"][f"{N}|{leitura}|{b}|{r}"]["acc"]
    celulas = [(N, r) for N in Ns for r in RS]

    ref = json.load(open(e11))["linhas"]
    difs = {}
    for N, r in celulas:
        v = [l["acc"] for l in ref if l["N"] == N and l["modo"] == "aleatorio" and l["r"] == r and l["sem"] in sems]
        difs[f"{N}|{r}"] = A(N, "exata", None, r) - float(np.mean(v))
    an["P0_diferencas_para_E11"] = difs
    an["P0_reproduz_E11"] = all(abs(d) <= 0.005 for d in difs.values())

    p1 = [A(N, "dif", None, r) >= A(N, "exata", None, r) - 0.01 for N, r in celulas]
    an["P1_contagem"] = int(sum(p1))
    an["P1_diferenca_finita_basta"] = sum(p1) >= 5

    babs = {}
    for N, r in celulas:
        ok = [b for b in ABS_BITS if A(N, "abs", b, r) >= A(N, "dif", None, r) - 0.01]
        babs[f"{N}|{r}"] = min(ok) if ok else None
    an["P2_b_abs_minimo"] = babs
    an["P2_contagem"] = int(sum(1 for v in babs.values() if v is None or v >= 16))
    an["P2_leitura_absoluta_exige_16_bits"] = an["P2_contagem"] >= 5

    p3 = [A(N, "difer", 2, r) >= A(N, "dif", None, r) - 0.015 for N, r in celulas]
    an["P3_contagem"] = int(sum(p3))
    an["P3_diferencial_2_bits_basta"] = sum(p3) >= 5

    p4 = [A(Ns[-1], le, b, r) >= A(Ns[0], le, b, r) - 0.01 for le, b in (("dif", None), ("difer", 2)) for r in RS]
    an["P4_contagem"] = int(sum(p4))
    an["P4_sem_penalidade_em_N"] = sum(p4) >= 5

    prev = {}
    for N, r in celulas:
        mp = an["celulas"][f"{N}|dif|None|{r}"]["mediana_abs_p"]
        prev[f"{N}|{r}"] = float(np.log2(FAIXA_ABS / (2 * EPS * mp))) if mp else None
    an["exploratorio_b_abs_previsto"] = prev
    if an["inconclusivo"]:
        for k in ("P1_diferenca_finita_basta", "P2_leitura_absoluta_exige_16_bits",
                  "P3_diferencial_2_bits_basta", "P4_sem_penalidade_em_N"):
            an[k] = None
    json.dump(dict(passos=PASSOS, eps=EPS, linhas=L, analise=an), open(final, "w"), indent=1)
    print(json.dumps({k: v for k, v in an.items() if k != "celulas"}, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "rodar":
        rodar(sys.argv[2], sys.argv[3], [int(s) for s in sys.argv[4].split(",")])
    elif sys.argv[1] == "analisar":
        args = sys.argv[2:]
        i = args.index("--e11")
        e11 = args[i + 1]
        args = args[:i] + args[i + 2:]
        analisar(args[0], args[1:], e11)
