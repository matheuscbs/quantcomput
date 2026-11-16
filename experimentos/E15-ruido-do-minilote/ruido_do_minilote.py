"""Por que 1 a 2 bits por projeção bastaram no E12? O ruído do minilote já domina o erro de cada medida? (H14)

E se o que governa a acurácia for a informação EFETIVA por medida, e não a capacidade nominal do canal?

Cada projeção recebida tem dois ruídos: o do minilote (a projeção de um lote de B exemplos difere da projeção
do gradiente da perda inteira) e o do canal (E12). Com sinal S = E[(v.g_verdadeiro)^2], variância do minilote
s2_B e canal gaussiano de SNR_c = 2^(2b) - 1 (aplicado sobre a potência recebida S + s2_B, como no E12):
    SNR_B   = S / s2_B
    SNR_ef  = S / (s2_B + (S + s2_B) / SNR_c) = 1 / (1/SNR_B + (1 + 1/SNR_B) / SNR_c)
    b_ef    = (1/2) log2(1 + SNR_ef)   bits úteis por projeção (aproximação gaussiana).
SNR_B é medido durante o treino: a cada 100 passos, as mesmas r direções do passo são aplicadas aos gradientes
de M = 8 minilotes independentes de tamanho B (gerador separado, sem mexer na trajetória); s2_B é a variância
entre minilotes e S = média(p_medio^2) - s2_B / M. Registra-se a mediana dos 15 pontos. A medida é feita nas
execuções float e usada para todas as execuções com o mesmo B e a mesma semente.

Grade: N = 118.282 (largura 128), r = 32, lote B em {128, 512, 2048}, 1500 passos, Adam lr 1e-3,
canal em {float, ruido com b em {1, 2, 4, 8}, quant com b em {1, 2, 4}} (canais idênticos aos do E12), 2 sementes.

Critérios fixados ANTES de rodar (ver docs/decisoes/0002). Médias das 2 sementes.
- H14a (o minilote domina em B = 128): mediana de SNR_B em B = 128 < 3 (o SNR do canal de 1 bit), nas DUAS sementes.
- H14b (a penalidade do canal aparece quando o minilote fica limpo): pen(B) = acc(float, B) - acc(ruido b=1, B).
  Aprovado se pen(2048) >= pen(128) + 1 ponto.
- H14: aprovada se H14a e H14b; reprovada se nenhum dos dois; parcial se só um.
- H10' (informação efetiva governa): nas 15 células (3 B x {float, ruido b = 1, 2, 4, 8}), o ajuste
  acc ~ a + c*log2(b_ef) tem R^2 >= 0,80. Para float, b_ef = (1/2) log2(1 + SNR_B).
  Comparação registrada: R^2 do ajuste com b nominal (float = 32) no lugar de b_ef.
Inconclusivo: (i) desvio entre sementes acima de 2 pontos em mais de 1/4 das células; ou (ii) para H10',
amplitude das acurácias das 15 células abaixo de 2 pontos (não há o que explicar); ou (iii) S estimado <= 0 em
mais da metade dos pontos de medida de algum B.

Uso:
  python ruido_do_minilote.py rodar dir_dados saida.json SEMENTES
  python ruido_do_minilote.py analisar final.json parcial_1.json [parcial_2.json ...]
Variável PASSOS no ambiente encurta o treino (só para teste de fumaça).
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(int(os.environ.get("THREADS", "3")))
W, R = 128, 32
LOTES = (128, 512, 2048)
CONFIGS = (("float", 32), ("ruido", 1), ("ruido", 2), ("ruido", 4), ("ruido", 8),
           ("quant", 1), ("quant", 2), ("quant", 4))
PASSOS = int(os.environ.get("PASSOS", "1500"))
LR, M_LOTES, A_CADA = 1e-3, 8, 100


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
    r = p.numel()
    if tipo == "float":
        return p, 32 * r
    if tipo == "quant":
        s = p.abs().max().clamp_min(1e-30)
        niveis = 2 ** b - 1
        u = (p / s + 1) / 2 * niveis
        base = torch.floor(u)
        sobe = torch.rand(r, generator=ger) < (u - base)
        q = (base + sobe.float()).clamp(0, niveis)
        return (q / niveis * 2 - 1) * s, r * b + 32
    if tipo == "ruido":
        snr = 2.0 ** (2 * b) - 1
        sigma = (p.pow(2).mean() / snr).sqrt()
        return p + sigma * torch.randn(r, generator=ger), r * b
    raise ValueError(tipo)


def medir_snr(mod, V, xtr, ytr, B, rng):
    """SNR do minilote nas direções V: sinal e variância entre M minilotes independentes."""
    params = list(mod.parameters())
    P = []
    for _ in range(M_LOTES):
        bt = rng.integers(0, len(xtr), B)
        gs = torch.autograd.grad(F.cross_entropy(mod(xtr[bt]), ytr[bt]), params)
        P.append(V.T @ torch.cat([g.reshape(-1) for g in gs]))
    P = torch.stack(P)                                   # (M, r)
    s2 = float(P.var(dim=0, unbiased=True).mean())
    S = float(P.mean(dim=0).pow(2).mean()) - s2 / M_LOTES
    return S, s2


def rodar_um(dt, tipo, b, B, sem, medir):
    xtr, ytr, xte, yte = dt
    torch.manual_seed(sem)
    g_rng = torch.Generator().manual_seed(1000 + sem)
    c_rng = torch.Generator().manual_seed(2000 + sem)
    idx_rng = np.random.default_rng(sem)
    m_rng = np.random.default_rng(3000 + sem)            # lotes extras da medida de SNR
    mod = rede(W)
    opt = torch.optim.Adam(mod.parameters(), LR)
    N = sum(p.numel() for p in mod.parameters())
    bits, medidas = 0, []
    for t in range(PASSOS):
        bt = idx_rng.integers(0, len(xtr), B)
        loss = F.cross_entropy(mod(xtr[bt]), ytr[bt])
        opt.zero_grad(set_to_none=False)
        loss.backward()
        g = achatar(mod)
        V = torch.randn(N, R, generator=g_rng)
        V = V / V.norm(dim=0, keepdim=True)
        if medir and t % A_CADA == 0:
            S, s2 = medir_snr(mod, V, xtr, ytr, B, m_rng)
            medidas.append(dict(passo=t, S=S, s2=s2))
        p, nb = canal(V.T @ g, tipo, b, c_rng)
        bits += nb
        devolver(mod, V @ p)
        opt.step()
    with torch.no_grad():
        acc = float((mod(xte).argmax(1) == yte).float().mean())
    return dict(acc=acc, N=N, bits=bits, medidas=medidas)


def rodar(raiz, saida, sementes):
    dt = dados(raiz)
    res = dict(passos=PASSOS, r=R, linhas=[])
    t0 = time.time()
    for sem in sementes:
        for B in LOTES:
            for tipo, b in CONFIGS:
                out = rodar_um(dt, tipo, b, B, sem, medir=(tipo == "float"))
                lin = dict(sem=sem, B=B, canal=tipo, b=b, **out)
                res["linhas"].append(lin)
                print({k: v for k, v in lin.items() if k != "medidas"}, round(time.time() - t0), "s", flush=True)
                json.dump(res, open(saida, "w"), indent=0)


def analisar(final, parciais):
    L = [l for f in parciais for l in json.load(open(f))["linhas"]]
    sems = sorted({l["sem"] for l in L})
    an = dict(sementes=sems, celulas={}, snr_minilote={})
    invalido = False
    for B in LOTES:
        por_sem = {}
        for s in sems:
            med = [m for l in L if l["B"] == B and l["sem"] == s and l["canal"] == "float" for m in l["medidas"]]
            snrs = [m["S"] / m["s2"] for m in med]
            if np.mean([m["S"] <= 0 for m in med]) > 0.5:
                invalido = True
            por_sem[str(s)] = float(np.median(snrs))
        an["snr_minilote"][str(B)] = dict(por_semente=por_sem, media=float(np.mean(list(por_sem.values()))))

    def b_ef(B, tipo, b):
        snr_b = an["snr_minilote"][str(B)]["media"]
        if tipo == "float":
            snr = snr_b
        else:
            snr_c = 2.0 ** (2 * b) - 1
            snr = 1 / (1 / snr_b + (1 + 1 / snr_b) / snr_c)
        return 0.5 * np.log2(1 + max(snr, 1e-12))

    dps = []
    for B in LOTES:
        for tipo, b in CONFIGS:
            v = [l["acc"] for l in L if l["B"] == B and l["canal"] == tipo and l["b"] == b]
            an["celulas"][f"{B}|{tipo}|{b}"] = dict(acc=float(np.mean(v)), dp=float(np.std(v)) if len(v) > 1 else 0.0,
                                                    b_ef=float(b_ef(B, tipo, b)) if tipo != "quant" else None)
            dps.append(an["celulas"][f"{B}|{tipo}|{b}"]["dp"])
    A = lambda B, t, b: an["celulas"][f"{B}|{t}|{b}"]["acc"]
    an["inconclusivo_sementes"] = bool(len(sems) > 1 and np.mean(np.array(dps) > 0.02) > 0.25)
    an["inconclusivo_sinal"] = invalido

    snr128 = an["snr_minilote"]["128"]["por_semente"]
    an["H14a_minilote_domina_em_128"] = all(v < 3 for v in snr128.values())
    pen = {str(B): A(B, "float", 32) - A(B, "ruido", 1) for B in LOTES}
    an["penalidade_ruido_b1"] = pen
    an["penalidade_quant_b1"] = {str(B): A(B, "float", 32) - A(B, "quant", 1) for B in LOTES}
    an["H14b_penalidade_cresce"] = pen["2048"] >= pen["128"] + 0.01
    a, bb = an["H14a_minilote_domina_em_128"], an["H14b_penalidade_cresce"]
    an["H14"] = "aprovada" if a and bb else ("reprovada" if not a and not bb else "parcial")

    cel = [(B, t, b) for B in LOTES for t, b in CONFIGS if t != "quant"]
    y = np.array([A(*c) for c in cel])

    def r2(x):
        X = np.column_stack([np.ones(len(y)), x])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return float(1 - ((y - X @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()), coef.tolist()

    an["H10l_R2_b_efetivo"], an["H10l_coef"] = r2(np.log2([an["celulas"][f"{B}|{t}|{b}"]["b_ef"] for B, t, b in cel]))
    an["comparacao_R2_b_nominal"], _ = r2(np.log2([b for _, _, b in cel]))
    an["amplitude_acc_15_celulas"] = float(y.max() - y.min())
    an["inconclusivo_H10l"] = an["amplitude_acc_15_celulas"] < 0.02
    an["H10l_informacao_efetiva_governa"] = an["H10l_R2_b_efetivo"] >= 0.80
    if an["inconclusivo_sementes"] or an["inconclusivo_sinal"]:
        for k in ("H14a_minilote_domina_em_128", "H14b_penalidade_cresce", "H14", "H10l_informacao_efetiva_governa"):
            an[k] = None
    elif an["inconclusivo_H10l"]:
        an["H10l_informacao_efetiva_governa"] = None
    json.dump(dict(passos=PASSOS, linhas=L, analise=an), open(final, "w"), indent=1)
    print(json.dumps({k: v for k, v in an.items() if k != "celulas"}, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "rodar":
        rodar(sys.argv[2], sys.argv[3], [int(s) for s in sys.argv[4].split(",")])
    elif sys.argv[1] == "analisar":
        analisar(sys.argv[2], sys.argv[3:])
