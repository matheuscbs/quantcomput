"""Quantas conexões por consulta o GPT-2 precisa, escolhidas pelo conteúdo?

Mantém, em cada cabeça e cada token, apenas as m maiores atenções (mais o primeiro token) e
renormaliza. É um oráculo: os escores são calculados por inteiro antes de escolher, portanto mede
o quanto de comunicação de VALORES é necessária. Uma busca aproximada precisaria achar esses m.
Uso: uv run --no-project --with torch --with 'transformers<4.50' --with numpy python -I atencao_esparsa_gpt2.py texto.txt saida.json
"""
import json
import sys

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers.models.gpt2 import modeling_gpt2 as mg

N, WINDOWS = 1024, 16
MS = [1, 2, 4, 8, 16, 32, 64, 128]
torch.set_num_threads(4)
STATE = {"m": None, "kept": 0.0, "total": 0.0}
orig = mg.eager_attention_forward


def patched(module, query, key, value, attention_mask=None, head_mask=None, **kw):
    out, w = orig(module, query, key, value, attention_mask, head_mask=head_mask, **kw)
    STATE["calls"] = STATE.get("calls", 0) + 1
    m = STATE["m"]
    if m is None:
        return out, w
    idx = w.topk(min(m, w.shape[-1]), dim=-1).indices
    mask = torch.zeros_like(w, dtype=torch.bool).scatter(-1, idx, True)
    mask[..., 0] = True
    allowed = w > 0
    mask = mask & allowed
    STATE["kept"] += float(mask.sum())
    STATE["total"] += float(allowed.sum())
    w2 = w * mask
    w2 = w2 / w2.sum(-1, keepdim=True).clamp_min(1e-12)
    return torch.matmul(w2, value).transpose(1, 2), w2


mg.eager_attention_forward = patched


@torch.no_grad()
def nll(model, ids):
    lg = model(ids[None], use_cache=False).logits[0, 200:-1]
    return float(torch.nn.functional.cross_entropy(lg, ids[201:]).item())


def main():
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager", use_safetensors=True).eval()
    txt = open(sys.argv[1], encoding="utf-8", errors="ignore").read()
    txt = txt[txt.find("CHAPTER 1."):]
    ids = torch.tensor(tok(txt[:900000])["input_ids"])
    starts = np.linspace(0, len(ids) - N - 1, WINDOWS).astype(int)
    base = [nll(model, ids[s:s + N]) for s in starts]
    res = dict(nll_completo=float(np.mean(base)), esparsa={})
    for m in MS:
        STATE.update(m=m, kept=0.0, total=0.0)
        v = [nll(model, ids[s:s + N]) for s in starts]
        d = np.array(v) - np.array(base)
        res["esparsa"]["m%d" % m] = dict(nll=float(np.mean(v)), delta=float(d.mean()), delta_dp=float(d.std()),
                                         fracao_pares=STATE["kept"] / STATE["total"])
        print(m, res["esparsa"]["m%d" % m], flush=True)
    assert STATE.get("calls", 0) > 0 and res["esparsa"]["m1"]["delta"] > 0.01, "patch sem efeito"
    json.dump(res, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
