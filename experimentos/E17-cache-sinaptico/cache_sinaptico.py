"""Quanto se grava para aprender? Ineficiência de caminho e cache sináptico (H17, H18, H19)

Pergunta. O E13 mostrou que os bits entregues até o alvo (I) dependem da tarefa e não crescem com N. Mas um
sistema físico paga também para GRAVAR os pesos, e gravar costuma ser o fluxo caro (memória de longo prazo da
mosca ~1e18 kT ln 2 por bit; Loihi 120 pJ por atualização de sinapse contra 23,6 pJ por operação sináptica;
Tkachenko, 2025: o preço termodinâmico de uma rede está na memória, não na aritmética). Quanto se grava?

Medida de custo de gravação (Li e van Rossum, 2020, eLife 9:e50804): M = soma em t e em i de |dw_i(t)|, o
comprimento L1 do caminho dos pesos. O mínimo para chegar ao mesmo ponto é M_min = ||w_T - w_0||_1.
Ineficiência de caminho: rho = M / M_min.

Previsão (proposta em docs/09, conferida num modelo quadrático de brinquedo antes deste registro; ver
auxiliares/): com atualizações ruidosas, cada coordenada anda num passeio aleatório com deriva mu e desvio s
por passo. Em T passos, M soma T * E|N(mu, s^2)| e o deslocamento vale E|N(T mu, T s^2)|. Com r projeções
aleatórias do gradiente num canal ruidoso, a SNR por coordenada é q = (mu/s)^2 = (r/N) * SNR_ef / (1 + SNR_ef),
com SNR_ef = 1 / (1/SNR_B + (1 + 1/SNR_B)/SNR_c) (E15) e SNR_c = 2^(2b) - 1 = 15 para b = 2. Donde
    rho_prev = T * E|N(sqrt(q), 1)| / E|N(T sqrt(q), T)|,
que vale sqrt(2/pi) * sqrt((N/r) (1 + 1/SNR_ef)) quando T q >> 1 (a deriva domina) e sqrt(T) quando T q << 1
(o próprio deslocamento final é quase todo ruído). Nos dois regimes, o custo de gravar fica muito acima do
mínimo, embora I não cresça com N (E13), a menos que o sistema guarde as mudanças num cache transitório e
consolide só o que acumulou.

Cache sináptico (Li e van Rossum, 2020): cada peso tem uma parte persistente e uma transitória u_i; a
atualização entra em u_i; quando |u_i| >= kappa * sigma, u_i é consolidado (custo |u_i|) e zerado. sigma é a
raiz quadrática média de dw nos CAL primeiros passos de cada execução. O peso efetivo é persistente + u, então
o cache NÃO muda a trajetória: mede só quanto se grava. Custo com cache: M_c(kappa) = soma do que foi
consolidado + |u| restante no fim. Ganho G(kappa) = M / M_c(kappa). Carga média do cache:
L(kappa) = média em t de sum_i |u_i|.

Protocolo (E13 com SGD sem momento, para que dw seja a própria atualização): Fashion-MNIST, k = 10 classes,
inicialização padrão, lote 128; cada passo entrega r projeções aleatórias normalizadas do gradiente por um
canal gaussiano de b = 2 bits (como no E13); dw = -lr * V (V^T g + ruído). Alvo: o do E13 para k = 10
(1/k + 0,8 (ref_10 - 1/k), ref lido de E13/resultados/referencia.json). Para no alvo ou em PASSOS_MAX.
Referência sem projeção (backprop, dw = -lr_bp * g) nas duas arquiteturas.
SNR do minilote (SNR_B) medido a cada MED_CADA passos em 8 minilotes independentes, nas direções V do passo
(método do E15); entra só na previsão rho_prev.

Grade: arquitetura em {prof (N = 118.282), larga (N = 407.050)} x r em {16, 32, 64} x 3 sementes = 18
execuções projetadas, mais 2 x 3 = 6 de backprop. kappa em {1, 4, 16, 64, 256}, todos medidos na mesma
execução. lr e lr_bp fixados na calibração (subcomando calibrar, semente 99; ver auxiliares/).

Critérios fixados ANTES de rodar (ver docs/decisoes/0002).
- H17 (gravar é ineficiente sem cache): aprovada se rho >= 10 em pelo menos 17 das 18 execuções projetadas E
  mediana(rho projetado) >= 3 * mediana(rho backprop) em cada arquitetura. Reprovada se rho < 10 em mais de
  6 das 18. Parcial nos demais casos.
- H18 (a SNR por coordenada prevê rho): nas 18 execuções projetadas, com T = passos da execução e SNR_B =
  mediana das medidas da execução. Aprovada se a mediana de rho/rho_prev estiver em [1/3; 3] E a correlação
  de Spearman entre rho e rho_prev for >= 0,6. Reprovada se a mediana de rho/rho_prev ficar fora de
  [1/10; 10] ou se a correlação for <= 0. Parcial nos demais casos. Registra-se também (exploratório) o MQO
  de log rho em log N, log r e log T.
- H19 (o cache recupera o desperdício): nas 6 células projetadas (média das sementes), G(kappa) não
  decrescente em kappa E G(16) >= 5 em todas. Reprovada se G(16) < 2 em alguma célula ou se G for
  decrescente em algum par de kappas consecutivos em 2 ou mais células. Parcial nos demais casos.
Exploratório (sem veredito): L(kappa) contra kappa (previsão: proporcional); energia total
E(kappa) = M_c + c * L * T para razões c em {1e-4, 1e-3, 1e-2} (previsão: mínimo interior e
E_min proporcional a sqrt(c)); M_min contra N; rho de backprop contra a mesma fórmula com r = N.
Inconclusivo (nenhum veredito aplicado): mais de 1/4 das 18 execuções projetadas sem atingir o alvo.

Uso:
  python cache_sinaptico.py calibrar dir_dados saida.json
  python cache_sinaptico.py rodar dir_dados saida.json SEMENTES
  python cache_sinaptico.py analisar final.json parcial_1.json [...]
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "E13-fatorial-de-I"))
import fatorial_de_I as E13

K = 10
ARQS = ("prof", "larga")
RS = (16, 32, 64)
SEMENTES = (0, 1, 2)
KAPPAS = (1, 4, 16, 64, 256)
B_CANAL, BATCH = 2, 128
SNR_C = 2.0 ** (2 * B_CANAL) - 1
LR = 10.0       # calibração (auxiliares/calibracao.log), prof, r = 32, 1500 passos: lr em {1, 3, 10, 30}, maior acurácia 10 (75,8%); 30 diverge
LR_BP = 0.3     # calibração: lr em {0,01; 0,03; 0,1; 0,3}, maior acurácia 0,3 (87,4%; 0,1 deu 87,3%)
PASSOS_MAX = 6000
CAL = 10
MED_CADA, M_LOTES = 200, 8
PRECOS_CACHE = (1e-4, 1e-3, 1e-2)


def alvo_k10():
    ref = json.load(open(os.path.join(AQUI, "..", "E13-fatorial-de-I", "resultados", "referencia.json")))
    accs = [lin["acc"] for lin in ref["linhas"] if lin["k"] == K]
    ref10 = float(np.mean(accs))
    return 1 / K + E13.FRAC * (ref10 - 1 / K), ref10


def medir_snr(mod, V, xtr, ytr, rng):
    params = list(mod.parameters())
    P = []
    for _ in range(M_LOTES):
        bt = rng.integers(0, len(xtr), BATCH)
        gs = torch.autograd.grad(F.cross_entropy(mod(xtr[bt]), ytr[bt]), params)
        P.append(V.T @ torch.cat([g.reshape(-1) for g in gs]))
    P = torch.stack(P)
    s2 = float(P.var(dim=0, unbiased=True).mean())
    S = float(P.mean(dim=0).pow(2).mean()) - s2 / M_LOTES
    return S / s2 if s2 > 0 else float("nan")


def treinar(dt, arq, r, sem, lr, alvo, passos=PASSOS_MAX, projetar=True):
    xtr, ytr, xte, yte = dt
    torch.manual_seed(sem)
    g_rng = torch.Generator().manual_seed(1000 + sem)
    c_rng = torch.Generator().manual_seed(2000 + sem)
    idx_rng = np.random.default_rng(sem)
    m_rng = np.random.default_rng(3000 + sem)
    mod = E13.rede(arq, "padrao", K)
    params = list(mod.parameters())
    N = sum(p.numel() for p in params)
    w0 = torch.cat([p.detach().reshape(-1) for p in params]).clone()
    nk = len(KAPPAS)
    kap = torch.tensor(KAPPAS, dtype=torch.float32).view(-1, 1)
    U = torch.zeros(nk, N)
    Mc = torch.zeros(nk, dtype=torch.float64)
    Wc = torch.zeros(nk, dtype=torch.float64)
    Lc = torch.zeros(nk, dtype=torch.float64)
    M, rms, sigma, snrs, curva = 0.0, [], None, [], []
    t_alvo = None
    for t in range(1, passos + 1):
        bt = idx_rng.integers(0, len(xtr), BATCH)
        loss = F.cross_entropy(mod(xtr[bt]), ytr[bt])
        for p in params:
            p.grad = None
        loss.backward()
        g = torch.cat([p.grad.reshape(-1) for p in params])
        if projetar:
            V = torch.randn(N, r, generator=g_rng)
            V = V / V.norm(dim=0, keepdim=True)
            if t % MED_CADA == 1:
                snrs.append(medir_snr(mod, V, xtr, ytr, m_rng))
            pr = V.T @ g
            pr = pr + (pr.pow(2).mean() / SNR_C).sqrt() * torch.randn(r, generator=c_rng)
            d = -lr * (V @ pr)
        else:
            d = -lr * g
        with torch.no_grad():
            i = 0
            for p in params:
                n = p.numel()
                p.add_(d[i:i + n].view_as(p))
                i += n
        M += float(d.abs().sum())
        U += d
        if sigma is None:
            rms.append(float(d.pow(2).mean().sqrt()))
            if t == CAL:
                sigma = float(np.mean(rms))
        if sigma is not None:
            m = U.abs() >= kap * sigma
            Mc += (U.abs() * m).sum(1).double()
            Wc += m.sum(1).double()
            U[m] = 0
            Lc += U.abs().sum(1).double()
        if t % E13.AVAL_CADA == 0:
            a = E13.acuracia(mod, xte, yte)
            curva.append(a)
            if a >= alvo:
                t_alvo = t
                break
            if not math.isfinite(loss.item()):
                break
    Mc += U.abs().sum(1).double()
    Wc += (U != 0).sum(1).double()
    wT = torch.cat([p.detach().reshape(-1) for p in params])
    Mmin = float((wT - w0).abs().sum())
    return dict(N=N, passos=t, atingiu=t_alvo is not None, acc_final=curva[-1] if curva else None,
                M=M, Mmin=Mmin, rho=M / Mmin, sigma=sigma,
                Mc=[float(x) for x in Mc], Wc=[float(x) for x in Wc], L=[float(x) / t for x in Lc],
                snr_b=snrs, curva=curva)


def snr_ef(snr_b):
    return 1.0 / (1.0 / snr_b + (1.0 + 1.0 / snr_b) / SNR_C)


def media_abs(m, s):
    """E|X| para X ~ N(m, s^2)."""
    return s * math.sqrt(2 / math.pi) * math.exp(-m * m / (2 * s * s)) + m * math.erf(m / (s * math.sqrt(2)))


def rho_prev(N, r, snr_b, T):
    s = snr_ef(snr_b)
    q = (r / N) * s / (1 + s)
    return T * media_abs(math.sqrt(q), 1.0) / media_abs(T * math.sqrt(q), math.sqrt(T))


def calibrar(raiz, saida):
    dt = E13.dados(raiz, K)
    alvo, ref10 = alvo_k10()
    out = dict(alvo=alvo, ref10=ref10, proj=[], bp=[])
    for lr in (1.0, 3.0, 10.0, 30.0):
        o = treinar(dt, "prof", 32, 99, lr, 2.0, passos=1500)
        out["proj"].append(dict(lr=lr, acc=o["acc_final"], rho=o["rho"]))
        print("proj", lr, o["acc_final"], round(o["rho"], 1), flush=True)
    for lr in (0.01, 0.03, 0.1, 0.3):
        o = treinar(dt, "prof", 0, 99, lr, 2.0, passos=1500, projetar=False)
        out["bp"].append(dict(lr=lr, acc=o["acc_final"], rho=o["rho"]))
        print("bp", lr, o["acc_final"], round(o["rho"], 1), flush=True)
    json.dump(out, open(saida, "w"), indent=1)


def rodar(raiz, saida, sementes):
    assert LR is not None and LR_BP is not None, "fixe LR e LR_BP pela calibração"
    dt = E13.dados(raiz, K)
    alvo, ref10 = alvo_k10()
    res = dict(alvo=alvo, ref10=ref10, lr=LR, lr_bp=LR_BP, kappas=KAPPAS, linhas=[])
    t0 = time.time()
    for sem in sementes:
        for arq in ARQS:
            for r in RS + (0,):
                o = treinar(dt, arq, r, sem, LR if r else LR_BP, alvo, projetar=bool(r))
                lin = dict(sem=sem, arq=arq, r=r, **o)
                res["linhas"].append(lin)
                print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in lin.items()
                       if k not in ("curva", "snr_b", "Mc", "Wc", "L")}, round(time.time() - t0), "s", flush=True)
                json.dump(res, open(saida, "w"), indent=0)


def analisar(final, parciais):
    linhas, base = [], None
    for p in parciais:
        d = json.load(open(p))
        base = base or d
        linhas += d["linhas"]
    proj = [lin for lin in linhas if lin["r"] > 0]
    bp = [lin for lin in linhas if lin["r"] == 0]
    out = dict(alvo=base["alvo"], lr=base["lr"], lr_bp=base["lr_bp"], kappas=list(KAPPAS), n_proj=len(proj))
    censuradas = sum(not lin["atingiu"] for lin in proj)
    out["censuradas"] = censuradas
    out["inconclusivo"] = censuradas > len(proj) / 4
    for lin in proj:
        sb = float(np.nanmedian(lin["snr_b"]))
        lin["snr_b_med"] = sb
        lin["rho_prev"] = rho_prev(lin["N"], lin["r"], sb, lin["passos"])
    # H17
    n10 = sum(lin["rho"] >= 10 for lin in proj)
    razoes = {}
    for arq in ARQS:
        rp = np.median([lin["rho"] for lin in proj if lin["arq"] == arq])
        rb = np.median([lin["rho"] for lin in bp if lin["arq"] == arq])
        razoes[arq] = dict(rho_proj=float(rp), rho_bp=float(rb), razao=float(rp / rb))
    h17 = "aprovada" if n10 >= 17 and all(v["razao"] >= 3 for v in razoes.values()) else (
        "reprovada" if len(proj) - n10 > 6 else "parcial")
    out["H17"] = dict(veredito=h17, rho_maior_que_10=n10, por_arq=razoes)
    # H18
    X = np.array([[1.0, math.log(lin["N"]), math.log(lin["r"]), math.log(lin["passos"])] for lin in proj])
    y = np.array([math.log(lin["rho"]) for lin in proj])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    razao_prev = float(np.median([lin["rho"] / lin["rho_prev"] for lin in proj]))
    rk = lambda v: np.argsort(np.argsort(v))  # noqa: E731
    sp = float(np.corrcoef(rk([lin["rho"] for lin in proj]), rk([lin["rho_prev"] for lin in proj]))[0, 1])
    if 1 / 3 <= razao_prev <= 3 and sp >= 0.6:
        h18 = "aprovada"
    elif not (0.1 <= razao_prev <= 10) or sp <= 0:
        h18 = "reprovada"
    else:
        h18 = "parcial"
    out["H18"] = dict(veredito=h18, mediana_rho_sobre_prev=razao_prev, spearman=sp,
                      exploratorio_mqo=dict(logN=float(coef[1]), logr=float(coef[2]), logT=float(coef[3]),
                                            r2=float(r2)))
    # H19
    celulas = {}
    for arq in ARQS:
        for r in RS:
            ls = [lin for lin in proj if lin["arq"] == arq and lin["r"] == r]
            G = np.mean([[lin["M"] / mc for mc in lin["Mc"]] for lin in ls], axis=0)
            L = np.mean([lin["L"] for lin in ls], axis=0)
            celulas[f"{arq}-r{r}"] = dict(G=[float(x) for x in G], L=[float(x) for x in L],
                                         rho=float(np.mean([lin["rho"] for lin in ls])),
                                         rho_prev=float(np.mean([lin["rho_prev"] for lin in ls])),
                                         passos=float(np.mean([lin["passos"] for lin in ls])),
                                         Mmin=float(np.mean([lin["Mmin"] for lin in ls])))
    i16 = KAPPAS.index(16)
    mono = [all(c["G"][j + 1] >= c["G"][j] for j in range(len(KAPPAS) - 1)) for c in celulas.values()]
    g16 = [c["G"][i16] for c in celulas.values()]
    if all(mono) and min(g16) >= 5:
        h19 = "aprovada"
    elif min(g16) < 2 or sum(not m for m in mono) >= 2:
        h19 = "reprovada"
    else:
        h19 = "parcial"
    out["H19"] = dict(veredito=h19, G16=g16, monotono=mono)
    out["celulas"] = celulas
    # exploratório: energia com preço de manutenção do cache
    expl = {}
    for nome, c in celulas.items():
        ls = [lin for lin in proj if f"{lin['arq']}-r{lin['r']}" == nome]
        Mc = np.mean([lin["Mc"] for lin in ls], axis=0)
        T = c["passos"]
        expl[nome] = {str(pc): [float(Mc[j] + pc * c["L"][j] * T) for j in range(len(KAPPAS))]
                      for pc in PRECOS_CACHE}
    out["exploratorio_energia"] = expl
    out["linhas"] = [{k: v for k, v in lin.items() if k != "curva"} for lin in linhas]
    if out["inconclusivo"]:
        for h in ("H17", "H18", "H19"):
            out[h]["veredito"] = "inconclusivo"
    json.dump(out, open(final, "w"), indent=1)
    print(json.dumps({h: out[h] for h in ("H17", "H18", "H19")}, indent=1))
    print("censuradas", censuradas)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "calibrar":
        calibrar(sys.argv[2], sys.argv[3])
    elif cmd == "rodar":
        rodar(sys.argv[2], sys.argv[3], [int(s) for s in sys.argv[4].split(",")])
    elif cmd == "analisar":
        analisar(sys.argv[2], sys.argv[3:])
