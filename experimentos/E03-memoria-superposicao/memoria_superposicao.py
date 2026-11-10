"""Atenção como busca vetorial contra memória em superposição (HRR via FFT).

Tarefa: k pares chave-valor aleatórios, consultar um valor. Mede a dimensão mínima para 99%
de acerto, e estima energia por consulta contando acessos a memória e operações.
É inferência de recuperação pura: não há aprendizado aqui.
Uso: uv run --no-project --with numpy python memoria_superposicao.py
"""
import math
import numpy as np

V, EPISODES, BETA = 1024, 200, 50.0
DRAM_PJ, SRAM_PJ, MUL_PJ, ADD_PJ = 640.0, 10.0, 3.7, 0.9  # 32 bits, 45 nm (Horowitz, ISSCC 2014)


def vocab(rng, d):
    return (rng.standard_normal((V, d)) / np.sqrt(d)).astype(np.float32)


def noisy(rng, q, rho):
    if rho >= 1:
        return q
    n = (rng.standard_normal(q.shape) / np.sqrt(q.size)).astype(np.float32)
    return rho * q + math.sqrt(1 - rho ** 2) * n


def decode(voc, r):
    return int(np.argmax(voc @ (r / (np.linalg.norm(r) + 1e-12))))


def attend(K, Val, q):
    s = K @ q
    w = np.exp(BETA * (s - s.max()))
    return (w / w.sum()) @ Val


def hrr_recall(S, q):
    return np.fft.irfft(S * np.conj(np.fft.rfft(q)), n=q.size).astype(np.float32)


def accuracy(d, k, seed, method, rho=1.0, hops=1):
    rng = np.random.default_rng(seed + d + k)
    voc = vocab(rng, d)
    hit = 0
    for _ in range(EPISODES):
        ids = rng.choice(V, 2 * k, replace=False)
        ki, vi = ids[:k], ids[k:].copy()
        if hops == 2:
            vi[0] = ki[1]
            target = vi[1]
        else:
            target = vi[0]
        K, Val = voc[ki], voc[vi]
        q = noisy(rng, voc[ki[0]], rho)
        perm = np.random.default_rng(d).permutation(d)  # ligação não comutativa: chave permutada
        if method in ("hrr", "hrrp"):
            Kb = K[:, perm] if method == "hrrp" else K
            S = np.sum(np.fft.rfft(Kb, axis=1) * np.fft.rfft(Val, axis=1), axis=0)
        for h in range(hops):
            r = attend(K, Val, q) if method == "att" else hrr_recall(S, q[perm] if method == "hrrp" else q)
            out = decode(voc, r)
            q = voc[out]
        hit += out == target
    return hit / EPISODES


def min_dim(k, method, seed=7, rho=1.0, hops=1):
    for p in range(5, 17):
        d = 2 ** p
        if 2 * k > V:
            return None
        if accuracy(d, k, seed, method, rho, hops) >= 0.99:
            return d
    return None


def fft_cost(d):
    butterflies = (d / 4) * math.log2(d / 2)
    return 4 * butterflies, 6 * butterflies, d * math.log2(d)  # mult, soma, tráfego interno


def energy(kind, k, d, mem):
    if kind == "att":
        reads, mul, add, internal = 2 * k * d + d, 2 * k * d, 2 * k * d, 0
    else:
        m, a, t = fft_cost(d)
        reads, mul, add, internal = 2 * d, 3 * m + 2 * d, 3 * a, 3 * t
    return (reads * mem + internal * SRAM_PJ + mul * MUL_PJ + add * ADD_PJ) / 1e6  # microjoules


def main():
    print("k   d_att  d_hrr | sem limpeza: E_att  E_hrr (uJ, DRAM) razão | SRAM: razão | com limpeza V*d: razão(DRAM)")
    for k in [4, 8, 16, 32, 64, 128, 256]:
        da, dh = min_dim(k, "att"), min_dim(k, "hrr")
        if dh is None:
            print(k, da, "hrr>65536")
            continue
        ea, eh = energy("att", k, da, DRAM_PJ), energy("hrr", k, dh, DRAM_PJ)
        ea2, eh2 = energy("att", k, da, SRAM_PJ), energy("hrr", k, dh, SRAM_PJ)
        ca = V * da * (DRAM_PJ + MUL_PJ + ADD_PJ) / 1e6
        ch = V * dh * (DRAM_PJ + MUL_PJ + ADD_PJ) / 1e6
        print("%3d %6d %6d | %10.3f %10.3f %6.2f | %6.2f | %6.2f" % (k, da, dh, ea, eh, ea / eh, ea2 / eh2, (ea + ca) / (eh + ch)))
    print("armazenamento em bytes (float32), k pares: atenção 2*k*d_att; HRR d_hrr")
    for rho, hops, name in [(0.7, 1, "consulta ruidosa rho=0.7"), (1.0, 2, "dois saltos")]:
        for k in [8, 32]:
            print("%s k=%d: d_att=%s d_hrr=%s" % (name, k, min_dim(k, "att", rho=rho, hops=hops), min_dim(k, "hrr", rho=rho, hops=hops)))
    for d, k in [(1024, 32), (4096, 128)]:
        print("d=%d k=%d acerto HRR %.3f atenção %.3f" % (d, k, accuracy(d, k, 3, "hrr"), accuracy(d, k, 3, "att")))
    n, dim = 1e9, 1536
    print("1 bilhão de vetores de 1536 dimensões em float32: %.1f TB; com PQ de 64 bytes: %.3f TB" % (n * dim * 4 / 1e12, n * 64 / 1e12))


def train_mlp_ops(k, d=256, h=256, seed=0, max_epochs=20000):
    """Decora k pares com MLP e entropia cruzada; devolve épocas e operações até 100% de acerto."""
    rng = np.random.default_rng(seed + k)
    X = rng.standard_normal((k, d)) / np.sqrt(d) * 8
    y = rng.integers(0, V, k)
    W = rng.standard_normal((d, h)) / np.sqrt(d)
    U = rng.standard_normal((h, V)) / np.sqrt(h)
    ps = [W, U]
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    for ep in range(1, max_epochs + 1):
        H = np.maximum(X @ W, 0)
        L = H @ U
        L = L - L.max(1, keepdims=True)
        P = np.exp(L)
        P = P / P.sum(1, keepdims=True)
        if np.all(P.argmax(1) == y):
            return ep, 2 * ep * k * (d * h + h * V)
        E = P.copy()
        E[np.arange(k), y] -= 1
        gU = H.T @ E / k
        gW = X.T @ ((E @ U.T) * (H > 0)) / k
        for i, g in enumerate([gW, gU]):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            ps[i] -= 2e-2 * m[i] / (np.sqrt(v[i]) + 1e-8) * 0.3
    return None, None


def total_uJ(kind, k, d, mem, with_decode=True):
    e = energy(kind, k, d, mem)
    return e + (V * d * (mem + MUL_PJ + ADD_PJ) / 1e6 if with_decode else 0)


def extra():
    for k in [8, 32]:
        print("dois saltos, ligação não comutativa, k=%d: d_hrrp=%s" % (k, min_dim(k, "hrrp", hops=2)))
    print("k=64, d_hrr=4096 medido. Razão E_att/E_hrr (>1: superposição gasta menos)")
    for da in [32, 128, 768, 1536]:
        print("  d_att=%4d | DRAM sem decodificação %7.2f | SRAM sem decodificação %6.2f | DRAM com decodificação nas duas %6.2f | armazenamento att/hrr %6.1f"
              % (da, energy("att", 64, da, DRAM_PJ) / energy("hrr", 64, 4096, DRAM_PJ),
                 energy("att", 64, da, SRAM_PJ) / energy("hrr", 64, 4096, SRAM_PJ),
                 total_uJ("att", 64, da, DRAM_PJ) / total_uJ("hrr", 64, 4096, DRAM_PJ), 2 * 64 * da / 4096))
    print("Escrever contra treinar (mesmas contagens de multiplicação)")
    for k in [16, 64, 256]:
        rs = [train_mlp_ops(k, seed=s) for s in range(5)]
        eps = [r[0] for r in rs if r[0]]
        ops = np.median([r[1] for r in rs if r[1]]) if eps else float("nan")
        d = 64 * k
        m, _, _ = fft_cost(d)
        write = k * (2 * m + 2 * d)
        print("k=%d: épocas medianas %s (%d/5 convergiram), operações MLP %.3g; escrita HRR d=%d %.3g; razão %.1f"
              % (k, np.median(eps) if eps else None, len(eps), ops, d, write, ops / write))
    gj = 1287e6 * 3600
    print("GPT-3: %.3g J / 3.14e23 ops = %.1f pJ por operação" % (gj, gj / 3.14e23 * 1e12))


if __name__ == "__main__":
    import sys
    extra() if "extra" in sys.argv else main()
