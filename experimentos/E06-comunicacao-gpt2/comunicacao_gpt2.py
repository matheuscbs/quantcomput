"""Quanta comunicação de longa distância um Transformer real usa?

GPT-2 em texto real (1.024 tokens). Duas medidas:
1. Observacional: massa de atenção por distância e expoente de Rent do fluxo de atenção
   (fluxo que sai de blocos de tamanho B ~ B^p), comparado com atenção uniforme.
2. Causal: restringe cada token a olhar só para os r anteriores (com e sem o primeiro token)
   e mede a perda no texto. Atenção não é igual a informação, por isso a parte causal manda.
Uso: uv run --no-project --with 'torch' --with 'transformers<4.50' --with numpy python -I comunicacao_gpt2.py texto.txt saida.json
"""
import json
import sys

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

N, WINDOWS = 1024, 16
RADII = [2, 8, 32, 128, 512, 1024]
BLOCKS = [2, 4, 8, 16, 32, 64, 128, 256, 512]
torch.set_num_threads(4)


def rent(A, sizes):
    """A: N x N linhas somam ~1. Fluxo médio que sai de um bloco contíguo de tamanho B."""
    out = []
    n = A.shape[0]
    for B in sizes:
        T = []
        for s in range(0, n - B + 1, B):
            inside = A[s:s + B, s:s + B].sum()
            T.append(A[s:s + B].sum() - inside)
        out.append(float(np.mean(T)))
    return out


def fit(sizes, T, lo=8, hi=256):
    x, y = np.log2(sizes), np.log2(np.maximum(T, 1e-12))
    s = (np.array(sizes) >= lo) & (np.array(sizes) <= hi)
    return float(np.polyfit(x[s], y[s], 1)[0])


def band(M, r, sink):
    m = torch.zeros(M, M, dtype=torch.bool)
    for i in range(M):
        m[i, max(0, i - r + 1):i + 1] = True
        if sink:
            m[i, 0] = True
    return m[None, None]


def set_bias(model, b):
    for h in model.transformer.h:
        h.attn.bias = b.clone()


@torch.no_grad()
def nll(model, ids):
    out = model(ids[None], use_cache=False)
    lg = out.logits[0, 200:-1]  # avalia só tokens com contexto longo
    return float(torch.nn.functional.cross_entropy(lg, ids[201:]).item())


def main():
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager", use_safetensors=True).eval()
    txt = open(sys.argv[1], encoding="utf-8", errors="ignore").read()
    txt = txt[txt.find("CHAPTER 1."):]
    ids = torch.tensor(tok(txt[:900000])["input_ids"])
    starts = np.linspace(0, len(ids) - N - 1, WINDOWS).astype(int)
    full = model.transformer.h[0].attn.bias.clone()
    M = full.shape[-1]

    dist = np.zeros(N)
    rents, rents_ns = [], []
    with torch.no_grad():
        for s in starts:
            at = model(ids[s:s + N][None], output_attentions=True, use_cache=False).attentions
            for L in at:
                A = L[0].mean(0).numpy()
                for d in range(N):
                    dist[d] += float(np.trace(A, offset=-d)) / N
                rents.append(rent(A, BLOCKS))
                B = A.copy()
                B[:, 0] = 0
                B = B / np.maximum(B.sum(1, keepdims=True), 1e-9)
                rents_ns.append(rent(B, BLOCKS))
    dist = dist / (WINDOWS * 12)
    cum = np.cumsum(dist)
    U = np.tril(np.ones((N, N)))
    U = U / U.sum(1, keepdims=True)
    T, Tn, Tu = np.mean(rents, 0), np.mean(rents_ns, 0), rent(U, BLOCKS)

    res = dict(massa_dist0_tokens_proximos={str(d): float(cum[d - 1]) for d in (1, 4, 16, 64, 256, 1024)},
               massa_primeiro_token=None,
               rent=dict(blocos=BLOCKS, fluxo_real=list(map(float, T)), fluxo_sem_primeiro=list(map(float, Tn)), fluxo_uniforme=list(map(float, Tu)),
                         p_real=fit(BLOCKS, T), p_sem_primeiro=fit(BLOCKS, Tn), p_uniforme=fit(BLOCKS, Tu)))

    base = [nll(model, ids[s:s + N]) for s in starts]
    res["nll_completo"] = float(np.mean(base))
    res["ablacao"] = {}
    total = N * (N + 1) / 2
    for r in RADII:
        for sink in (False, True):
            set_bias(model, band(M, r, sink))
            v = [nll(model, ids[s:s + N]) for s in starts]
            pairs = float(band(N, r, sink).sum())
            res["ablacao"]["r%d_%s" % (r, "com1" if sink else "sem1")] = dict(
                nll=float(np.mean(v)), delta=float(np.mean(np.array(v) - np.array(base))),
                delta_dp=float(np.std(np.array(v) - np.array(base))), fracao_pares=pairs / total)
            print(r, sink, res["ablacao"]["r%d_%s" % (r, "com1" if sink else "sem1")], flush=True)
    set_bias(model, full)
    assert abs(float(np.mean([nll(model, ids[s:s + N]) for s in starts[:2]])) - float(np.mean(base[:2]))) < 1e-6
    assert res["ablacao"]["r2_sem1"]["delta"] > 0.05, "a máscara não teve efeito"
    json.dump(res, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res["rent"], indent=0))


if __name__ == "__main__":
    main()
