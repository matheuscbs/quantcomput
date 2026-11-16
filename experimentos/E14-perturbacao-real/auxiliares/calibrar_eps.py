"""Calibração feita antes de fixar os critérios: correlação entre a diferença central e a projeção exata,
no início do treino e depois de 300 passos de backprop, para eps em {1e-1, 1e-2, 1e-3}, e tempo por passo."""
import os, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import perturbacao_real as P

xtr, ytr, xte, yte = P.dados(sys.argv[1])
for w in P.LARGURAS:
    torch.manual_seed(0)
    mod = P.rede(w); opt = torch.optim.Adam(mod.parameters(), 1e-3)
    nomes = [n for n, _ in mod.named_parameters()]; formas = [tuple(p.shape) for _, p in mod.named_parameters()]
    N = sum(p.numel() for p in mod.parameters()); rng = np.random.default_rng(0)
    for fase in ("inicio", "300 passos"):
        if fase != "inicio":
            for _ in range(300):
                bt = rng.integers(0, 60000, 128); opt.zero_grad(); F.cross_entropy(mod(xtr[bt]), ytr[bt]).backward(); opt.step()
        bt = rng.integers(0, 60000, 128); xb, yb = xtr[bt], ytr[bt]
        opt.zero_grad(); F.cross_entropy(mod(xb), yb).backward(); g = P.achatar(mod)
        V = torch.randn(N, 128); V /= V.norm(dim=0)
        ex = V.T @ g
        theta = torch.cat([q.detach().view(-1) for q in mod.parameters()])
        for eps in (1e-1, 1e-2, 1e-3):
            P.EPS = eps
            t = time.time(); Lm, Ln = P.perdas_perturbadas(mod, nomes, formas, theta, V, xb, yb); dt = time.time() - t
            est = (Lm - Ln) / (2 * eps)
            c = float(np.corrcoef(ex.numpy(), est.numpy())[0, 1]); rel = float((est - ex).norm() / ex.norm())
            print(f"N={N} {fase} eps={eps:g} corr={c:.4f} erro_rel={rel:.3f} |p|med={float(ex.abs().median()):.2e} t(r=128)={dt:.3f}s")
