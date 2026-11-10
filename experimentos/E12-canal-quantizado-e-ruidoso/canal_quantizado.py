"""O que governa a acurácia: o número de medidas, a precisão de cada uma ou o total de bits?

Repete o modo aleatório do E11 com o canal degradado. Por passo, o otimizador recebe r projeções do
gradiente, que agora passam por um canal com capacidade de b bits por projeção:
- quant: quantização uniforme estocástica (sem viés) em 2^b níveis no intervalo [-s, s], com s = max|p|
  enviado à parte em float32 (32 bits por passo, contados no total);
- ruido: soma ruído gaussiano com SNR = 2^(2b) - 1, de modo que (1/2) log2(1 + SNR) = b bits por projeção;
- float: sem degradação (b = 32). Mesmas sementes e mesma ordem de sorteios do E11: deve reproduzir os números
  do E11 para N = 13.002 e 118.282 se as versões fixadas equivalem às originais.

Bits por passo: quant = r*b + 32; ruido = r*b; float = 32*r. Total = bits por passo * PASSOS.
Limite que continua: a projeção sai do gradiente exato (como no E11); o que muda é o canal.

Critérios fixados ANTES de rodar (ver docs/decisoes/0002). Ajustes por N e por canal (quant, ruido),
nas 12 configurações (r em {8, 32, 128}, b em {1, 2, 4, 8}), média das sementes:
- B1 (total de bits governa): acc ~ a + c*log2(bits_total) tem R^2 >= 0,90.
- B2 (medidas e precisão separadas): acc ~ a + c1*log2(r) + c2*log2(b) reduz a soma dos quadrados dos resíduos
  de B1 em pelo menos 50%. Se B1 e B2 valerem juntos, vale B2 (o total explica muito, mas não tudo).
- B3 (sem penalidade em N): em pelo menos 80% das configurações (canal, r, b), acc(118.282) >= acc(13.002) - 1 ponto.
- B4 (controle de reprodução): float reproduz o E11 com diferença absoluta <= 0,5 ponto em todas as 6 células
  (2 N x 3 r), média das sementes.
Inconclusivo: se o desvio entre sementes passar de 2 pontos em mais de 1/4 das células, B1 a B3 não são aplicados.

Uso:
  python canal_quantizado.py rodar dir_dados saida.json SEMENTES     (SEMENTES = "0", "1" ou "0,1")
  python canal_quantizado.py analisar final.json parcial_1.json [parcial_2.json ...] [--e11 e11.json]
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)
LARGURAS = (16, 128)                 # N = 13.002 e 118.282
RS = (8, 32, 128)
BITS = (1, 2, 4, 8)
CANAIS = ("quant", "ruido")
PASSOS, BATCH, LR = 1500, 128, 1e-3


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


def canal(p, tipo, b, ger):
    """Passa o vetor de projeções p pelo canal. Devolve (p_recebido, bits_no_passo)."""
    r = p.numel()
    if tipo == "float":
        return p, 32 * r
    if tipo == "quant":
        s = p.abs().max().clamp_min(1e-30)
        niveis = 2 ** b - 1                      # intervalos entre 2^b níveis em [-s, s]
        u = (p / s + 1) / 2 * niveis             # em [0, niveis]
        base = torch.floor(u)
        sobe = torch.rand(r, generator=ger) < (u - base)
        q = (base + sobe.float()).clamp(0, niveis)
        return (q / niveis * 2 - 1) * s, r * b + 32
    if tipo == "ruido":
        snr = 2.0 ** (2 * b) - 1
        sigma = (p.pow(2).mean() / snr).sqrt()
        return p + sigma * torch.randn(r, generator=ger), r * b
    raise ValueError(tipo)


def rodar_um(dt, w, tipo, r, b, sem):
    xtr, ytr, xte, yte = dt
    torch.manual_seed(sem)
    g_rng = torch.Generator().manual_seed(1000 + sem)    # direções: igual ao E11
    c_rng = torch.Generator().manual_seed(2000 + sem)    # canal: gerador separado
    idx_rng = np.random.default_rng(sem)
    mod = rede(w)
    opt = torch.optim.Adam(mod.parameters(), LR)
    N = sum(p.numel() for p in mod.parameters())
    bits = 0
    for _ in range(PASSOS):
        bt = idx_rng.integers(0, len(xtr), BATCH)
        loss = F.cross_entropy(mod(xtr[bt]), ytr[bt])
        opt.zero_grad(set_to_none=False)
        loss.backward()
        if r is not None:
            g = achatar(mod)
            V = torch.randn(N, r, generator=g_rng)
            V = V / V.norm(dim=0, keepdim=True)
            p, nb = canal(V.T @ g, tipo, b, c_rng)
            bits += nb
            devolver(mod, V @ p)
        opt.step()
    with torch.no_grad():
        acc = float((mod(xte).argmax(1) == yte).float().mean())
    return acc, N, bits


def rodar(raiz, saida, sementes):
    dt = dados(raiz)
    res = dict(passos=PASSOS, linhas=[])
    t0 = time.time()

    def reg(**k):
        res["linhas"].append(k)
        print(k, round(time.time() - t0), "s", flush=True)
        json.dump(res, open(saida, "w"), indent=0)

    for sem in sementes:
        for w in LARGURAS:
            a, N, _ = rodar_um(dt, w, "float", None, None, sem)
            reg(sem=sem, w=w, N=N, canal="backprop", r=None, b=None, bits=None, acc=a)
            for r in RS:
                a, N, nb = rodar_um(dt, w, "float", r, 32, sem)
                reg(sem=sem, w=w, N=N, canal="float", r=r, b=32, bits=nb, acc=a)
            for tipo in CANAIS:
                for b in BITS:
                    for r in RS:
                        a, N, nb = rodar_um(dt, w, tipo, r, b, sem)
                        reg(sem=sem, w=w, N=N, canal=tipo, r=r, b=b, bits=nb, acc=a)


def ajuste(X, y):
    X = np.column_stack([np.ones(len(y))] + X)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    sse = float(((y - X @ coef) ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    return coef.tolist(), sse, 1 - sse / sst


def analisar(final, parciais, e11=None):
    L = [l for f in parciais for l in json.load(open(f))["linhas"]]
    Ns = sorted({l["N"] for l in L})
    sems = sorted({l["sem"] for l in L})

    def cel(N, canal, r, b):
        v = [l["acc"] for l in L if l["N"] == N and l["canal"] == canal and l["r"] == r and l["b"] == b]
        bits = [l["bits"] for l in L if l["N"] == N and l["canal"] == canal and l["r"] == r and l["b"] == b]
        return (float(np.mean(v)) if v else float("nan"), float(np.std(v)) if len(v) > 1 else 0.0,
                bits[0] if bits else None)

    an = dict(N=Ns, sementes=sems, backprop={str(N): cel(N, "backprop", None, None)[0] for N in Ns},
              celulas={}, ajustes={})
    dps = []
    for N in Ns:
        for canal in ("float",) + CANAIS:
            for r in RS:
                for b in ((32,) if canal == "float" else BITS):
                    m, dp, bits = cel(N, canal, r, b)
                    an["celulas"][f"{N}|{canal}|{r}|{b}"] = dict(acc=m, dp=dp, bits=bits)
                    dps.append(dp)
    an["inconclusivo"] = bool(len(sems) > 1 and np.mean(np.array(dps) > 0.02) > 0.25)

    b1, b2 = [], []
    for N in Ns:
        for canal in CANAIS:
            pts = [(r, b, an["celulas"][f"{N}|{canal}|{r}|{b}"]) for r in RS for b in BITS]
            y = np.array([c["acc"] for _, _, c in pts])
            lb = np.log2([c["bits"] for _, _, c in pts])
            c1, s1, r2_1 = ajuste([lb], y)
            c2, s2, r2_2 = ajuste([np.log2([r for r, _, _ in pts]), np.log2([b for _, b, _ in pts])], y)
            an["ajustes"][f"{N}|{canal}"] = dict(B1_coef=c1, B1_R2=r2_1, B1_sse=s1, B2_coef=c2, B2_R2=r2_2,
                                                 B2_sse=s2, B2_reducao=1 - s2 / s1 if s1 > 0 else 0.0)
            b1.append(r2_1 >= 0.90)
            b2.append(s2 <= 0.5 * s1)
    an["B1_total_de_bits"] = all(b1) and not any(b2)
    an["B2_medidas_e_precisao_separadas"] = any(b2)
    comp = [an["celulas"][f"{Ns[-1]}|{c}|{r}|{b}"]["acc"] >= an["celulas"][f"{Ns[0]}|{c}|{r}|{b}"]["acc"] - 0.01
            for c in CANAIS for r in RS for b in BITS]
    an["B3_fracao_sem_penalidade"] = float(np.mean(comp))
    an["B3_sem_penalidade_em_N"] = an["B3_fracao_sem_penalidade"] >= 0.80
    if e11:
        ref = json.load(open(e11))["linhas"]
        difs = {}
        for N in Ns:
            for r in RS:
                v = [l["acc"] for l in ref if l["N"] == N and l["modo"] == "aleatorio" and l["r"] == r
                     and l["sem"] in sems]
                difs[f"{N}|{r}"] = an["celulas"][f"{N}|float|{r}|32"]["acc"] - float(np.mean(v))
        an["B4_diferencas_para_E11"] = difs
        an["B4_reproduz_E11"] = all(abs(d) <= 0.005 for d in difs.values())
    if an["inconclusivo"]:
        for k in ("B1_total_de_bits", "B2_medidas_e_precisao_separadas", "B3_sem_penalidade_em_N"):
            an[k] = None
    json.dump(dict(passos=PASSOS, linhas=L, analise=an), open(final, "w"), indent=1)
    print(json.dumps({k: v for k, v in an.items() if k != "celulas"}, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "rodar":
        rodar(sys.argv[2], sys.argv[3], [int(s) for s in sys.argv[4].split(",")])
    elif sys.argv[1] == "analisar":
        args = sys.argv[2:]
        e11 = None
        if "--e11" in args:
            i = args.index("--e11")
            e11 = args[i + 1]
            args = args[:i] + args[i + 2:]
        analisar(args[0], args[1:], e11)
