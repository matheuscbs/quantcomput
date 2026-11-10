"""Quantos canais de medida o treino exige, conforme o número de parâmetros N cresce?

Hardware sem backpropagation (analógico, óptico, quântico) mede o gradiente por perturbação: cada canal
(tom de frequência, ou direção) devolve um número por medida. Aqui simulamos uma medida ideal e sem ruído
de r canais por passo: o otimizador só recebe as projeções do gradiente em r direções.

Modos:
- aleatorio: r direções aleatórias, novas a cada passo.
- adaptativo: metade das direções vem das ESTIMATIVAS anteriores (informação realmente disponível),
  metade é aleatória, para exploração.
- oraculo: metade vem dos gradientes verdadeiros anteriores. Irrealizável; é o teto do método.

Critérios fixados antes de rodar, em Fashion-MNIST, MLP de 2 camadas ocultas, N de cerca de 13 mil a 118 mil:
r*(N) = menor r com acurácia >= acurácia do backprop - 2 pontos. Expoente alfa = inclinação de log r* por log N.
- C1 (sobrevive): adaptativo com alfa <= 0,5.
- C2 (vale ser adaptativo): r* adaptativo <= r* aleatório / 2 no maior N.
- C3 (morre): alfa >= 0,8. O custo continua praticamente linear em N.
Se r* passar de 128 em algum N, o expoente fica indefinido e o resultado é registrado como acima da grade.
Uso: uv run --no-project --with torch --with numpy python -I canais_de_gradiente.py dir_dados saida.json
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
LARGURAS = (16, 32, 64, 128)
RS = (8, 16, 32, 64, 128)
MODOS = ("aleatorio", "adaptativo", "oraculo")
PASSOS, BATCH, LR, SEMENTES, FOLGA = 1500, 128, 1e-3, 2, 0.02


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


def rodar(dt, w, modo, r, sem):
    xtr, ytr, xte, yte = dt
    torch.manual_seed(sem)
    g_rng = torch.Generator().manual_seed(1000 + sem)
    idx_rng = np.random.default_rng(sem)
    mod = rede(w)
    opt = torch.optim.Adam(mod.parameters(), LR)
    N = sum(p.numel() for p in mod.parameters())
    h = 0 if r is None or modo == "aleatorio" else r // 2
    hist = []
    for _ in range(PASSOS):
        b = idx_rng.integers(0, len(xtr), BATCH)
        loss = F.cross_entropy(mod(xtr[b]), ytr[b])
        opt.zero_grad(set_to_none=False)
        loss.backward()
        if r is not None:
            g = achatar(mod)
            est = torch.zeros(N)
            res = g
            if hist:
                U, _ = torch.linalg.qr(torch.stack(hist, 1))
                c = U.T @ g
                est = U @ c
                res = g - est
            k = r - min(len(hist), h) if h else r
            V = torch.randn(N, k, generator=g_rng)
            V = V / V.norm(dim=0, keepdim=True)
            est = est + V @ (V.T @ res)
            if h:
                hist.append((g if modo == "oraculo" else est).clone())
                hist = hist[-h:]
            devolver(mod, est)
        opt.step()
    with torch.no_grad():
        acc = float((mod(xte).argmax(1) == yte).float().mean())
    return acc, N


def r_estrela(rs, accs, alvo):
    if accs[0] >= alvo:
        return float(rs[0])
    for i in range(1, len(rs)):
        if accs[i] >= alvo:
            f = (alvo - accs[i - 1]) / max(accs[i] - accs[i - 1], 1e-9)
            return float(np.exp(np.log(rs[i - 1]) + f * (np.log(rs[i]) - np.log(rs[i - 1]))))
    return float("nan")


def main():
    dt = dados(sys.argv[1])
    res = dict(passos=PASSOS, sementes=SEMENTES, folga=FOLGA, linhas=[])
    t0 = time.time()
    for sem in range(SEMENTES):
        for w in LARGURAS:
            a, N = rodar(dt, w, "backprop", None, sem)
            res["linhas"].append(dict(sem=sem, w=w, N=N, modo="backprop", r=None, acc=a))
            print(sem, w, N, "backprop", round(a, 4), round(time.time() - t0), "s", flush=True)
            for modo in MODOS:
                for r in RS:
                    a, _ = rodar(dt, w, modo, r, sem)
                    res["linhas"].append(dict(sem=sem, w=w, N=N, modo=modo, r=r, acc=a))
                    print(sem, w, modo, r, round(a, 4), round(time.time() - t0), "s", flush=True)
                json.dump(res, open(sys.argv[2], "w"), indent=0)
    L = res["linhas"]
    med = lambda w, modo, r: float(np.mean([l["acc"] for l in L if l["w"] == w and l["modo"] == modo and l["r"] == r]))
    Ns = [next(l["N"] for l in L if l["w"] == w) for w in LARGURAS]
    an = dict(N=Ns, backprop=[med(w, "backprop", None) for w in LARGURAS], acc={}, r_estrela={}, expoente={})
    for modo in MODOS:
        an["acc"][modo] = {str(w): [med(w, modo, r) for r in RS] for w in LARGURAS}
        y = [r_estrela(RS, an["acc"][modo][str(w)], med(w, "backprop", None) - FOLGA) for w in LARGURAS]
        an["r_estrela"][modo] = y
        an["expoente"][modo] = float(np.polyfit(np.log(Ns), np.log(y), 1)[0]) if not np.any(np.isnan(y)) else None
    al, ra, rr = an["expoente"]["adaptativo"], an["r_estrela"]["adaptativo"], an["r_estrela"]["aleatorio"]
    an["C1_sobrevive"] = None if al is None else al <= 0.5
    an["C3_morre"] = None if al is None else al >= 0.8
    an["C2_adaptativo_2x"] = None if math.isnan(ra[-1]) or math.isnan(rr[-1]) else ra[-1] <= rr[-1] / 2
    res["analise"] = an
    json.dump(res, open(sys.argv[2], "w"), indent=0)
    print(json.dumps(an, indent=1), flush=True)


if __name__ == "__main__":
    main()
