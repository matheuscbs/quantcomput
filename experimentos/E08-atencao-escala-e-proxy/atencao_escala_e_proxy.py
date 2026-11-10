"""Dois testes em GPT-2 real, sem retreino:

A. Escala: as m conexões necessárias por consulta crescem com o contexto N?
B. Proxy: dá para achar as m conexões lendo só uma fração das chaves (projeção em p de 64 dimensões)?
   Em B, o escore aproximado escolhe os m candidatos; a atenção exata só é usada entre eles.
Uso: uv run --no-project --with torch --with 'transformers<4.50' --with numpy python -I atencao_escala_e_proxy.py texto.txt saida.json
"""
import json
import sys

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers.models.gpt2 import modeling_gpt2 as mg

WINDOWS = 16
torch.set_num_threads(4)
S = dict(m=None, p=None, P=None, kept=0.0, total=0.0, hit=0.0, n=0.0, calls=0)
orig = mg.eager_attention_forward


def patched(module, query, key, value, attention_mask=None, head_mask=None, **kw):
    out, w = orig(module, query, key, value, attention_mask, head_mask=head_mask, **kw)
    S["calls"] += 1
    m = S["m"]
    if m is None:
        return out, w
    allowed = w > 0
    k = min(m, w.shape[-1])
    if S["P"] is None:
        idx = w.topk(k, dim=-1).indices
    else:
        P = S["P"]
        proxy = torch.matmul(query @ P, (key @ P).transpose(-1, -2))
        proxy = proxy.masked_fill(~allowed, float("-inf"))
        idx = proxy.topk(k, dim=-1).indices
        true = torch.zeros_like(w, dtype=torch.bool).scatter(-1, w.topk(k, dim=-1).indices, True) & allowed
        got = torch.zeros_like(w, dtype=torch.bool).scatter(-1, idx, True) & allowed
        S["hit"] += float((true & got).sum())
        S["n"] += float(true.sum())
    mask = torch.zeros_like(w, dtype=torch.bool).scatter(-1, idx, True)
    mask[..., 0] = True
    mask = mask & allowed
    S["kept"] += float(mask.sum())
    S["total"] += float(allowed.sum())
    w2 = w * mask
    w2 = w2 / w2.sum(-1, keepdim=True).clamp_min(1e-12)
    return torch.matmul(w2, value).transpose(1, 2), w2


mg.eager_attention_forward = patched


@torch.no_grad()
def nll(model, ids):
    n = len(ids)
    lg = model(ids[None], use_cache=False).logits[0, n // 2:-1]
    return float(torch.nn.functional.cross_entropy(lg, ids[n // 2 + 1:]).item())


def batch(model, ids, N, starts):
    return np.array([nll(model, ids[s:s + N]) for s in starts])


def main():
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager", use_safetensors=True).eval()
    txt = open(sys.argv[1], encoding="utf-8", errors="ignore").read()
    txt = txt[txt.find("CHAPTER 1."):]
    ids = torch.tensor(tok(txt[:900000])["input_ids"])
    res = dict(escala={}, proxy={})
    for N in (128, 256, 512, 1024):
        starts = np.linspace(0, len(ids) - N - 1, WINDOWS).astype(int)
        S.update(m=None, P=None)
        base = batch(model, ids, N, starts)
        row = dict(nll=float(base.mean()))
        for m in (4, 8, 16, 32, 64, 128):
            S.update(m=m, P=None, kept=0.0, total=0.0)
            d = batch(model, ids, N, starts) - base
            row["m%d" % m] = dict(delta=float(d.mean()), dp=float(d.std()), fracao=S["kept"] / S["total"])
        res["escala"]["N%d" % N] = row
        print(N, row, flush=True)
    N = 1024
    starts = np.linspace(0, len(ids) - N - 1, WINDOWS).astype(int)
    S.update(m=None, P=None)
    base = batch(model, ids, N, starts)
    g = torch.Generator().manual_seed(7)
    for p in (4, 8, 16, 32):
        Q, _ = torch.linalg.qr(torch.randn(64, p, generator=g))
        for m in (64, 128, 256):
            S.update(m=m, P=Q, kept=0.0, total=0.0, hit=0.0, n=0.0)
            d = batch(model, ids, N, starts) - base
            res["proxy"]["p%d_m%d" % (p, m)] = dict(delta=float(d.mean()), dp=float(d.std()), fracao=S["kept"] / S["total"], recall_topm=S["hit"] / S["n"])
            print(p, m, res["proxy"]["p%d_m%d" % (p, m)], flush=True)
    assert S["calls"] > 0
    json.dump(res, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
