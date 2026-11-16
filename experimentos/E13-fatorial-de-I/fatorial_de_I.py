"""I é uma propriedade da tarefa ou depende de arquitetura, inicialização e algoritmo? (H11, H12)

Definição operacional (docs/03, questão 3): I_A(eps | tarefa, arquitetura, inicialização) = total de bits
entregues pelo canal até o algoritmo A atingir a acurácia alvo eps. Cada passo entrega r = 32 projeções
aleatórias do gradiente por um canal gaussiano com b = 2 bits de capacidade por projeção (canal "ruido" do E12):
64 bits por passo. I é um limite superior para o mínimo sobre protocolos; aqui ele é medido para um protocolo fixo.

Fatores (desenho completo 3 x 2 x 2 x 2 = 24 células, 3 sementes):
- tarefa: Fashion-MNIST restrito às classes 0..k-1, k em {2, 5, 10}.
- arquitetura: "prof" = 784-128-128-k (2 camadas ocultas); "larga" = 784-512-k (1 camada oculta, ~3,4x mais pesos).
- inicialização: "padrao" (PyTorch, Kaiming uniforme) ou "xavier" (Xavier normal nos pesos, viés zero).
- algoritmo: "adam" (lr 1e-3) ou "sgdm" (SGD com momento 0,9, lr 1,0, escolhido na calibração; ver auxiliares/).
Ambos recebem V V^T g ruidoso no lugar de g, como no E12.

Alvo por tarefa: eps_k = 1/k + F * (ref_k - 1/k), onde ref_k é a média da acurácia do backprop com Adam,
1500 passos, nas 4 combinações (arquitetura, inicialização) e 3 sementes, no mesmo subconjunto de avaliação
(as primeiras 2000 imagens de teste das classes da tarefa). F = 0,8, fixado na calibração (ver auxiliares/).
A acurácia é avaliada a cada 10 passos; I = 64 * (passos até a primeira avaliação >= eps_k).
Teto: PASSOS_MAX passos; quem não chega é censurado e entra com I = 64 * PASSOS_MAX (limite inferior).

Critérios fixados ANTES de rodar (ver docs/decisoes/0002). y = log2 I.
Decomposição da soma dos quadrados entre as 24 médias de célula (média de y nas sementes), com contrastes
ortogonais (desenho balanceado): efeitos principais T, A, Z (inicialização), G (algoritmo) e todas as interações.
- H11 (a tarefa domina): aprovada se SQ(T)/SQ_total >= 0,50 E SQ(todas as interações)/SQ_total <= 0,20.
  Reprovada se SQ(T) for menor que a SQ de algum outro efeito principal, OU se as interações somarem >= 0,50.
  Parcial nos demais casos.
- H12 (I cresce com k): em cada uma das 8 combinações (A, Z, G), média de y estritamente crescente em
  k = 2, 5, 10. Aprovada se isso vale em PELO MENOS 7 das 8; reprovada se em 4 ou menos; parcial com 5 ou 6.
Registro exploratório: razão geométrica I(larga)/I(prof) (N cresce ~3,4x); expoente de log2 I contra log2 k.
Inconclusivo (nenhum veredito aplicado): mais de 1/4 das 72 execuções censuradas, ou desvio-padrão de y entre
sementes acima de 1 (fator 2 em I) em mais de 1/4 das células.

Uso:
  python fatorial_de_I.py calibrar dir_dados saida.json
  python fatorial_de_I.py referencia dir_dados ref.json
  python fatorial_de_I.py rodar dir_dados ref.json saida.json SEMENTES
  python fatorial_de_I.py analisar final.json ref.json parcial_1.json [...]
"""
import itertools
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(int(os.environ.get("THREADS", "3")))
KS = (2, 5, 10)
ARQS = ("prof", "larga")
INITS = ("padrao", "xavier")
ALGS = ("adam", "sgdm")
SEMENTES = (0, 1, 2)
R, B_CANAL, BATCH = 32, 2, 128
BITS_PASSO = R * B_CANAL
LR_ADAM = 1e-3
LR_SGDM = 1.0       # calibração (auxiliares/calibracao.log): lr em {0,01..3}, melhor 1,0 (75,3%) em k=10, prof, padrao
FRAC = 0.8          # calibração: com Adam, alvo de k=10 atingido em 400 a 580 passos; deixa margem até PASSOS_MAX
PASSOS_MAX = 3000
PASSOS_REF, AVAL_CADA, N_AVAL = 1500, 10, 2000


def dados(raiz, k):
    d = np.load(os.path.join(raiz, "fashion.npz"))
    x = (torch.tensor(d["x"], dtype=torch.float32) / 255 - 0.286) / 0.353
    y = torch.tensor(d["y"], dtype=torch.long)
    xtr, ytr, xte, yte = x[:60000], y[:60000], x[60000:], y[60000:]
    mtr, mte = ytr < k, yte < k
    return xtr[mtr], ytr[mtr], xte[mte][:N_AVAL], yte[mte][:N_AVAL]


def rede(arq, init, k):
    if arq == "prof":
        mod = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, k))
    else:
        mod = nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, k))
    if init == "xavier":
        for m in mod:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    return mod


def achatar(mod):
    return torch.cat([p.grad.view(-1) for p in mod.parameters()])


def devolver(mod, g):
    i = 0
    for p in mod.parameters():
        n = p.numel()
        p.grad.copy_(g[i:i + n].view_as(p))
        i += n


def otimizador(mod, alg, lr_sgdm):
    if alg == "adam":
        return torch.optim.Adam(mod.parameters(), LR_ADAM)
    return torch.optim.SGD(mod.parameters(), lr_sgdm, momentum=0.9)


def acuracia(mod, xte, yte):
    with torch.no_grad():
        return float((mod(xte).argmax(1) == yte).float().mean())


def treinar(dt, k, arq, init, alg, sem, passos, projetar=True, alvo=None, lr_sgdm=LR_SGDM):
    """Devolve (curva de acurácia a cada AVAL_CADA passos, passos até o alvo ou None, N)."""
    xtr, ytr, xte, yte = dt
    torch.manual_seed(sem)
    g_rng = torch.Generator().manual_seed(1000 + sem)
    c_rng = torch.Generator().manual_seed(2000 + sem)
    idx_rng = np.random.default_rng(sem)
    mod = rede(arq, init, k)
    opt = otimizador(mod, alg, lr_sgdm)
    N = sum(p.numel() for p in mod.parameters())
    snr = 2.0 ** (2 * B_CANAL) - 1
    curva = []
    for t in range(1, passos + 1):
        bt = idx_rng.integers(0, len(xtr), BATCH)
        loss = F.cross_entropy(mod(xtr[bt]), ytr[bt])
        opt.zero_grad(set_to_none=False)
        loss.backward()
        if projetar:
            g = achatar(mod)
            V = torch.randn(N, R, generator=g_rng)
            V = V / V.norm(dim=0, keepdim=True)
            p = V.T @ g
            p = p + (p.pow(2).mean() / snr).sqrt() * torch.randn(R, generator=c_rng)
            devolver(mod, V @ p)
        opt.step()
        if t % AVAL_CADA == 0:
            a = acuracia(mod, xte, yte)
            curva.append(a)
            if alvo is not None and a >= alvo:
                return curva, t, N
            if not np.isfinite(loss.item()):
                return curva, None, N
    return curva, None, N


def calibrar(raiz, saida):
    """Antes dos critérios: lr do SGD com momento e fração F alcançável, em k = 10, prof, padrao, semente 0."""
    dt = dados(raiz, 10)
    res = dict()
    c, _, _ = treinar(dt, 10, "prof", "padrao", "adam", 0, PASSOS_REF, projetar=False)
    res["ref_backprop_adam"] = c[-1]
    for k in KS:
        dk = dados(raiz, k)
        ck, _, _ = treinar(dk, k, "prof", "padrao", "adam", 0, PASSOS_REF, projetar=False)
        res[f"ref_k{k}"] = ck[-1]
    for lr in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
        c, _, _ = treinar(dt, 10, "prof", "padrao", "sgdm", 0, 1500, lr_sgdm=lr)
        res[f"sgdm_lr{lr}"] = c[-1]
        print("sgdm", lr, c[-1], flush=True)
    for alg, arq in (("adam", "prof"), ("adam", "larga")):
        c, _, _ = treinar(dt, 10, arq, "padrao", alg, 0, PASSOS_MAX)
        res[f"curva_{alg}_{arq}"] = c
        print(alg, arq, c[149], c[-1], flush=True)
    json.dump(res, open(saida, "w"), indent=0)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("curva")}, indent=1))


def referencia(raiz, saida):
    res = dict(passos=PASSOS_REF, linhas=[])
    for k in KS:
        dt = dados(raiz, k)
        for arq, init, sem in itertools.product(ARQS, INITS, SEMENTES):
            c, _, N = treinar(dt, k, arq, init, "adam", sem, PASSOS_REF, projetar=False)
            res["linhas"].append(dict(k=k, arq=arq, init=init, sem=sem, N=N, acc=c[-1]))
            print(res["linhas"][-1], flush=True)
    res["ref"] = {str(k): float(np.mean([l["acc"] for l in res["linhas"] if l["k"] == k])) for k in KS}
    res["frac"] = FRAC
    res["alvo"] = {str(k): 1 / k + FRAC * (res["ref"][str(k)] - 1 / k) for k in KS}
    json.dump(res, open(saida, "w"), indent=1)
    print(res["ref"], res["alvo"])


def rodar(raiz, ref, saida, sementes):
    alvos = json.load(open(ref))["alvo"]
    res = dict(passos_max=PASSOS_MAX, lr_sgdm=LR_SGDM, linhas=[])
    t0 = time.time()
    for sem in sementes:
        for k in KS:
            dt = dados(raiz, k)
            for arq, init, alg in itertools.product(ARQS, INITS, ALGS):
                curva, t, N = treinar(dt, k, arq, init, alg, sem, PASSOS_MAX, alvo=alvos[str(k)])
                lin = dict(sem=sem, k=k, arq=arq, init=init, alg=alg, N=N, passos=t, censurado=t is None,
                           I=BITS_PASSO * (t if t is not None else PASSOS_MAX), acc_final=curva[-1], curva=curva)
                res["linhas"].append(lin)
                print({c: v for c, v in lin.items() if c != "curva"}, round(time.time() - t0), "s", flush=True)
                json.dump(res, open(saida, "w"), indent=0)


def contrastes(n):
    """Contrastes ortonormais (Helmert) para um fator de n níveis: matriz n x (n-1)."""
    H = np.zeros((n, n - 1))
    for j in range(1, n):
        H[:j, j - 1] = 1
        H[j, j - 1] = -j
        H[:, j - 1] /= np.linalg.norm(H[:, j - 1])
    return H


def analisar(final, ref, parciais):
    L = [l for f in parciais for l in json.load(open(f))["linhas"]]
    fatores = (("T", KS), ("A", ARQS), ("Z", INITS), ("G", ALGS))
    chave = lambda l: (l["k"], l["arq"], l["init"], l["alg"])
    celulas = list(itertools.product(KS, ARQS, INITS, ALGS))
    ys = {c: [np.log2(l["I"]) for l in L if chave(l) == c] for c in celulas}
    ybar = np.array([np.mean(ys[c]) for c in celulas])
    an = dict(execucoes=len(L), censuradas=int(sum(l["censurado"] for l in L)), celulas={})
    for c in celulas:
        an["celulas"]["|".join(map(str, c))] = dict(log2I=float(np.mean(ys[c])), dp=float(np.std(ys[c])),
                                                   I_geo=float(2 ** np.mean(ys[c])),
                                                   censuradas=int(sum(l["censurado"] for l in L if chave(l) == c)))
    an["inconclusivo"] = bool(an["censuradas"] > len(L) / 4 or
                              np.mean([np.std(ys[c]) > 1 for c in celulas]) > 0.25)

    # SQ por termo com contrastes ortonormais; a ordem das células segue itertools.product (T mais externo)
    Cs = [contrastes(len(niv)) for _, niv in fatores]
    Us = [np.ones((len(niv), 1)) / np.sqrt(len(niv)) for _, niv in fatores]
    yc = ybar - ybar.mean()
    sq_total = float((yc ** 2).sum())
    sq = {}
    for mascara in itertools.product((0, 1), repeat=4):
        if not any(mascara):
            continue
        X = np.ones((1, 1))
        for m, Cm, Um in zip(mascara, Cs, Us):
            X = np.kron(X, Cm if m else Um)
        nome = "".join(f for (f, _), m in zip(fatores, mascara) if m)
        sq[nome] = float(((X.T @ ybar) ** 2).sum())
    an["SQ_total"] = sq_total
    an["SQ_frac"] = {t: v / sq_total for t, v in sq.items()}
    principais = {t: an["SQ_frac"][t] for t in "TAZG"}
    inter = sum(v for t, v in an["SQ_frac"].items() if len(t) > 1)
    an["SQ_frac_interacoes"] = inter
    if principais["T"] >= 0.5 and inter <= 0.2:
        an["H11"] = "aprovada"
    elif any(principais[t] > principais["T"] for t in "AZG") or inter >= 0.5:
        an["H11"] = "reprovada"
    else:
        an["H11"] = "parcial"

    mono = {}
    for arq, init, alg in itertools.product(ARQS, INITS, ALGS):
        v = [np.mean(ys[(k, arq, init, alg)]) for k in KS]
        mono[f"{arq}|{init}|{alg}"] = bool(v[0] < v[1] < v[2])
    an["H12_monotono"] = mono
    n = sum(mono.values())
    an["H12_contagem"] = n
    an["H12"] = "aprovada" if n >= 7 else ("reprovada" if n <= 4 else "parcial")

    an["exploratorio_razao_larga_prof"] = float(2 ** np.mean(
        [np.mean(ys[(k, "larga", z, g)]) - np.mean(ys[(k, "prof", z, g)]) for k in KS for z in INITS for g in ALGS]))
    xk = np.log2(KS)
    yk = [np.mean([np.mean(ys[(k, a, z, g)]) for a in ARQS for z in INITS for g in ALGS]) for k in KS]
    an["exploratorio_expoente_k"] = float(np.polyfit(xk, yk, 1)[0])
    if an["inconclusivo"]:
        an["H11"] = an["H12"] = None
    an["referencia"] = json.load(open(ref))
    an["referencia"].pop("linhas", None)
    json.dump(dict(linhas=L, analise=an), open(final, "w"), indent=1)
    print(json.dumps({k: v for k, v in an.items() if k != "celulas"}, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "calibrar":
        calibrar(sys.argv[2], sys.argv[3])
    elif cmd == "referencia":
        referencia(sys.argv[2], sys.argv[3])
    elif cmd == "rodar":
        rodar(sys.argv[2], sys.argv[3], sys.argv[4], [int(s) for s in sys.argv[5].split(",")])
    elif cmd == "analisar":
        analisar(sys.argv[2], sys.argv[3], sys.argv[4:])
