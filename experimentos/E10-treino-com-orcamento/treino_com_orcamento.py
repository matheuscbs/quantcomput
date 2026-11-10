"""Treinar com poucas conexões muda a lei de crescimento de m(N)?

Transformer pequeno, por caractere, treinado em Moby Dick (85% treino, 15% teste). Cada consulta só
pode ler as m maiores atenções (mais o primeiro token), e a restrição vale desde o primeiro passo.
Compara dois modos de medir o orçamento necessário:
- pós-treino: treina sem limite e corta depois;
- em treino: um modelo separado para cada m_treino, comparado ao modelo sem limite.

m(N) = menor m com perda extra <= EPS, em janelas de contexto N. Expoente = inclinação log m por log N.

Critérios fixados antes de rodar:
- T1 (poder): o contexto precisa ajudar. NLL(32) - NLL(256) >= 0,05 nat no modelo sem limite. Senão, inconclusivo.
- T2 (hábito de treino): expoente em treino <= expoente pós-treino - 0,2 E m_treino(256) <= m_pos(256)/2.
- T3 (piso): se a diferença de expoentes ficar abaixo de 0,1, treinar com orçamento não muda a lei
  neste modelo e nestes dados.
EPS = 0,02 nat, salvo se o desvio entre sementes do modelo sem limite passar disso. Nesse caso, EPS = 2*dp.
Tamanho reduzido antes de ver resultados: a máquina estava em limitação térmica (cerca de 0,25 s por passo).
Uso: uv run --no-project --with torch --with numpy python -I treino_com_orcamento.py texto.txt saida.json [passos] [sementes]
"""
import json
import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
NMAX, D, CAB, CAM = 256, 128, 4, 6
BATCH = 32
NS = (32, 64, 128, 256)
MTREINO = (None, 4, 8, 16, 32, 64)
MPOS = (4, 8, 16, 32, 64)
EPS0 = 0.02
JANELAS = 192


class Bloco(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.saida = nn.Linear(D, D, bias=False)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, x, m):
        B, T, _ = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, -1)
        q, k, v = [a.view(B, T, CAB, D // CAB).transpose(1, 2) for a in (q, k, v)]
        s = q @ k.transpose(-1, -2) / math.sqrt(D // CAB)
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        w = s.masked_fill(~causal, float("-inf")).softmax(-1)
        if m is not None:
            with torch.no_grad():
                idx = w.topk(min(m, T), dim=-1).indices
                mask = torch.zeros_like(w, dtype=torch.bool).scatter(-1, idx, True)
                mask[..., 0] = True
                mask = mask & causal
            w = w * mask
            w = w / w.sum(-1, keepdim=True).clamp_min(1e-12)
        y = (w @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.saida(y)
        return x + self.mlp(self.n2(x))


class Modelo(nn.Module):
    def __init__(self, voc):
        super().__init__()
        self.emb, self.pos = nn.Embedding(voc, D), nn.Embedding(NMAX, D)
        self.blocos = nn.ModuleList(Bloco() for _ in range(CAM))
        self.fim, self.cab = nn.LayerNorm(D), nn.Linear(D, voc, bias=False)

    def forward(self, x, m):
        h = self.emb(x) + self.pos(torch.arange(x.shape[1], device=x.device))
        for b in self.blocos:
            h = b(h, m)
        return self.cab(self.fim(h))


def treinar(treino, voc, m, sem, passos):
    torch.manual_seed(sem)
    g = np.random.default_rng(sem)
    mod = Modelo(voc).to(DEV)
    opt = torch.optim.AdamW(mod.parameters(), lr=1e-3, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=passos, pct_start=0.05)
    t0 = time.time()
    for i in range(passos):
        st = g.integers(0, len(treino) - NMAX - 1, BATCH)
        lote = torch.stack([treino[s:s + NMAX + 1] for s in st]).to(DEV)
        lg = mod(lote[:, :-1], m)
        loss = F.cross_entropy(lg.reshape(-1, voc), lote[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(mod.parameters(), 1.0)
        opt.step()
        sch.step()
        if i % 500 == 0:
            print("  passo", i, round(loss.item(), 3), round(time.time() - t0), "s", flush=True)
    return mod.eval()


@torch.no_grad()
def avaliar(mod, teste, N, m):
    st = np.linspace(0, len(teste) - N - 1, JANELAS).astype(int)
    out = []
    for a in range(0, JANELAS, 32):
        x = torch.stack([teste[s:s + N + 1] for s in st[a:a + 32]]).to(DEV)
        lg = mod(x[:, :-1], m)[:, N // 2:]
        y = x[:, 1:][:, N // 2:]
        out.append(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1)).item())
    return float(np.mean(out))


def m_necessario(ms, deltas, eps):
    if deltas[0] <= eps:
        return float(ms[0])
    for i in range(1, len(ms)):
        if deltas[i] <= eps:
            f = (deltas[i - 1] - eps) / max(deltas[i - 1] - deltas[i], 1e-9)
            return float(np.exp(np.log(ms[i - 1]) + f * (np.log(ms[i]) - np.log(ms[i - 1]))))
    return float("nan")


def inclinacao(y):
    if np.any(np.isnan(y)):
        return float("nan")
    return float(np.polyfit(np.log(NS), np.log(y), 1)[0])


def main():
    passos = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    sementes = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    txt = open(sys.argv[1], encoding="utf-8", errors="ignore").read()
    txt = txt[txt.find("CHAPTER 1."):]
    voc = sorted(set(txt))
    mapa = {c: i for i, c in enumerate(voc)}
    ids = torch.tensor([mapa[c] for c in txt], dtype=torch.long)
    corte = int(0.85 * len(ids))
    treino, teste = ids[:corte], ids[corte:]
    res = dict(device=DEV, passos=passos, sementes=sementes, modelos={})
    for sem in range(sementes):
        res["modelos"][str(sem)] = {}
        for m in MTREINO:
            print("semente", sem, "m_treino", m, flush=True)
            mod = treinar(treino, len(voc), m, sem, passos)
            linha = dict(proprio={str(N): avaliar(mod, teste, N, m) for N in NS})
            if m is None:
                linha["pos"] = {str(N): {str(p): avaliar(mod, teste, N, p) for p in MPOS} for N in NS}
            res["modelos"][str(sem)][str(m)] = linha
            print("  ", linha["proprio"], flush=True)
            json.dump(res, open(sys.argv[2], "w"), indent=1)
            del mod
            torch.mps.empty_cache() if DEV == "mps" else None
    # análise
    M = res["modelos"]
    media = lambda f: float(np.mean([f(str(s)) for s in range(sementes)]))
    full = {N: media(lambda s: M[s]["None"]["proprio"][str(N)]) for N in NS}
    dp = {N: float(np.std([M[str(s)]["None"]["proprio"][str(N)] for s in range(sementes)])) for N in NS} if sementes > 1 else {N: 0.0 for N in NS}
    eps = max(EPS0, 2 * max(dp.values()))
    pos = {N: [media(lambda s: M[s]["None"]["pos"][str(N)][str(p)] - M[s]["None"]["proprio"][str(N)]) for p in MPOS] for N in NS}
    treinado = {N: [media(lambda s: M[s][str(m)]["proprio"][str(N)] - M[s]["None"]["proprio"][str(N)]) for m in MTREINO[1:]] for N in NS}
    m_pos = [m_necessario(MPOS, pos[N], eps) for N in NS]
    m_tr = [m_necessario(MPOS, treinado[N], eps) for N in NS]
    a = dict(eps=eps, nll_sem_limite=full, dp_entre_sementes=dp, ganho_contexto=full[32] - full[256],
             delta_pos=pos, delta_treinado=treinado, m_pos=m_pos, m_treino=m_tr,
             expoente_pos=inclinacao(m_pos), expoente_treino=inclinacao(m_tr))
    a["T1_poder"] = a["ganho_contexto"] >= 0.05
    d = a["expoente_pos"] - a["expoente_treino"]
    a["T2_habito"] = bool(d >= 0.2 and m_tr[-1] <= m_pos[-1] / 2) if not math.isnan(d) else None
    a["T3_piso"] = bool(abs(d) < 0.1) if not math.isnan(d) else None
    res["analise"] = a
    json.dump(res, open(sys.argv[2], "w"), indent=1)
    print(json.dumps(a, indent=1), flush=True)


if __name__ == "__main__":
    main()
