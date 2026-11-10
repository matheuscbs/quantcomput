"""O expoente de m(N) vem da língua ou do modelo?

GPT-2 sem retreino, atenção cortada para as m maiores por consulta (mais o primeiro token).
Três textos: real, frases embaralhadas (destrói o discurso, mantém a frase) e tokens embaralhados
(mantém só a frequência das palavras). Em cada um, m(N) = menor m com perda extra <= tau.

Critérios fixados antes de rodar (tau = 0,1 nat):
- E1: se o IC 95% do expoente em frases embaralhadas não contiver o valor do texto real,
  o discurso de longo alcance participa do expoente.
- E2: se frases embaralhadas derem expoente dentro do IC do texto real, o expoente não depende
  do discurso, e a explicação fica no modelo ou na estrutura local da língua.
- E3: tokens embaralhados servem só de controle. Se o corte não custar nada ali, não há expoente.
Uso: uv run --no-project --with torch --with 'transformers<4.50' --with numpy python -I expoente_texto_embaralhado.py texto.txt saida.json
"""
import json
import re
import sys

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers.models.gpt2 import modeling_gpt2 as mg

WINDOWS = 32
NS = (128, 256, 512, 1024)
MS = (2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128)
TAUS = (0.05, 0.1, 0.2)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
S = dict(m=None, calls=0)
orig = mg.eager_attention_forward


def patched(module, query, key, value, attention_mask=None, head_mask=None, **kw):
    out, w = orig(module, query, key, value, attention_mask, head_mask=head_mask, **kw)
    S["calls"] += 1
    m = S["m"]
    if m is None:
        return out, w
    idx = w.topk(min(m, w.shape[-1]), dim=-1).indices
    mask = torch.zeros_like(w, dtype=torch.bool).scatter(-1, idx, True)
    mask[..., 0] = True
    mask = mask & (w > 0)
    w2 = w * mask
    w2 = w2 / w2.sum(-1, keepdim=True).clamp_min(1e-12)
    return torch.matmul(w2, value).transpose(1, 2), w2


mg.eager_attention_forward = patched


@torch.no_grad()
def nll(model, ids):
    n = len(ids)
    x = ids.to(DEV)
    lg = model(x[None], use_cache=False).logits[0, n // 2:-1]
    return float(torch.nn.functional.cross_entropy(lg, x[n // 2 + 1:]).item())


def m_needed(ms, deltas, tau):
    """Menor m com perda <= tau, interpolando em log m. NaN se nenhum m da grade cumprir."""
    if deltas[0] <= tau:
        return float(ms[0])
    for i in range(1, len(ms)):
        if deltas[i] <= tau:
            f = (deltas[i - 1] - tau) / max(deltas[i - 1] - deltas[i], 1e-9)
            return float(np.exp(np.log(ms[i - 1]) + f * (np.log(ms[i]) - np.log(ms[i - 1]))))
    return float("nan")


def expo(mat, tau, rng, B=2000):
    """mat[N][m] = vetor por janela. Devolve m(N), expoente e IC por bootstrap de janelas."""
    ms = list(MS)
    def fit(sel):
        y = [m_needed(ms, [mat[n][m][sel].mean() for m in ms], tau) for n in NS]
        if np.any(np.isnan(y)):
            return float("nan"), y
        return float(np.polyfit(np.log(NS), np.log(y), 1)[0]), y
    b, y = fit(np.arange(WINDOWS))
    boot = [fit(rng.integers(0, WINDOWS, WINDOWS))[0] for _ in range(B)]
    falhas = float(np.mean(np.isnan(boot)))
    boot = np.array(boot)
    boot = boot[~np.isnan(boot)]
    ic = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))] if len(boot) > 100 else None
    return dict(m_N=y, expoente=b, ic95=ic, bootstraps_sem_cruzamento=falhas)


def textos(tok, raw):
    rng = np.random.default_rng(11)
    raw = raw[raw.find("CHAPTER 1."):][:900000]
    real = tok(raw)["input_ids"]
    frases = [f.strip() for f in re.split(r"(?<=[.!?])\s+", raw) if f.strip()]
    rng.shuffle(frases)
    embar = tok(" ".join(frases))["input_ids"]
    toks = np.array(real)
    rng.shuffle(toks)
    return dict(real=real, frases_embaralhadas=embar, tokens_embaralhados=toks.tolist())


def main():
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.model_max_length = 10 ** 9
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager", use_safetensors=True).eval().to(DEV)
    raw = open(sys.argv[1], encoding="utf-8", errors="ignore").read()
    rng = np.random.default_rng(3)
    res = dict(device=DEV, janelas=WINDOWS)
    for nome, lista in textos(tok, raw).items():
        ids = torch.tensor(lista)
        mat, base_nll = {}, {}
        for N in NS:
            starts = np.linspace(0, len(ids) - N - 1, WINDOWS).astype(int)
            S["m"] = None
            base = np.array([nll(model, ids[s:s + N]) for s in starts])
            base_nll[N] = float(base.mean())
            mat[N] = {}
            for m in MS:
                S["m"] = m
                mat[N][m] = np.array([nll(model, ids[s:s + N]) for s in starts]) - base
            print(nome, N, round(base_nll[N], 3), {m: round(float(mat[N][m].mean()), 3) for m in MS}, flush=True)
        res[nome] = dict(nll=base_nll, delta={str(N): {str(m): float(mat[N][m].mean()) for m in MS} for N in NS},
                         por_tau={str(t): expo(mat, t, rng) for t in TAUS})
        print(nome, json.dumps(res[nome]["por_tau"]), flush=True)
    assert S["calls"] > 0
    json.dump(res, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
