"""Microteste de viabilidade: joules medidos em banco vetorial, atenção e memória em superposição.

Critérios fixados ANTES de rodar. A proposta só é considerada viável neste teste se:
  G1  alguma variante em superposição atinge 99% de acerto e gasta menos energia medida por
      consulta que a atenção em 1536 dimensões, com decodificação nas duas;
  G2  com palavras parecidas como chaves, a dimensão necessária é no máximo 2x a de palavras sorteadas;
  G3  a limpeza por índice HNSW mantém 99% de acerto e gasta pelo menos 2x menos que a limpeza por varredura;
  G4  a estimativa óptica (só conversão, piso otimista) gasta menos que a limpeza eletrônica medida.

Medição: Apple M5, zeus-apple-silicon (CPU total + DRAM), um thread por consulta, janelas pareadas
com janela ociosa imediatamente antes para subtrair o consumo basal da máquina.
Uso: uv run --no-project --with numpy --with faiss-cpu --with zeus-apple-silicon --with h5py \
     python -I microteste_energia.py glove-100-angular.hdf5
"""
import json
import sys
import time

import faiss
import h5py
import numpy as np
from zeus_apple_silicon import AppleEnergyMonitor

T, ROUNDS, BETA, EPISODES, V = 0.8, 7, 50.0, 300, 1024
mon = AppleEnergyMonitor()
DAC_PJ, ADC_PJ = 1.0, 2.0  # conversão eletro-óptica típica citada em ONNs, sem laser nem modulador


def window(fn):
    mon.begin_window("w")
    t0, n = time.perf_counter(), 0
    while time.perf_counter() - t0 < T:
        fn(n)
        n += 1
    m = mon.end_window("w")
    return m.cpu_total_mj + (m.dram_mj or 0), n, time.perf_counter() - t0, (m.dram_mj or 0)


def idle():
    mon.begin_window("i")
    t0 = time.perf_counter()
    time.sleep(T)
    m = mon.end_window("i")
    return (m.cpu_total_mj + (m.dram_mj or 0)) / (time.perf_counter() - t0)


def measure(configs, rng):
    """configs: nome -> fn(i). Devolve nome -> (mediana, q25, q75) em microjoules por consulta."""
    out = {k: [] for k in configs}
    gross = {k: [] for k in configs}
    names = list(configs)
    for _ in range(ROUNDS):
        rng.shuffle(names)
        for k in names:
            base = idle()
            e, n, dt, _ = window(configs[k])
            out[k].append((e - base * dt) * 1e3 / n)
            gross[k].append(e * 1e3 / n)
    res = {}
    for k, v in out.items():
        a = np.array(v)
        res[k] = dict(uJ=float(np.median(a)), q25=float(np.percentile(a, 25)), q75=float(np.percentile(a, 75)),
                      bruto_uJ=float(np.median(gross[k])))
    return res


def normal(a):
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)


def recall(idx, Q, gt, k=10):
    _, I = idx.search(Q, k)
    return float(np.mean([len(set(I[i]) & set(gt[i, :k])) / k for i in range(len(Q))]))


def tune(idx, setter, values, Q, gt, target=0.95):
    best = None
    for v in values:
        setter(v)
        r = recall(idx, Q, gt)
        best = (v, r)
        if r >= target:
            break
    setter(best[0])
    return best


def part_a(X, Q, gt, rng):
    d = X.shape[1]
    Qs = Q[:2000]
    gts = gt[:2000]
    idx = {}
    info = {}
    faiss.omp_set_num_threads(4)
    t = time.time()
    flat = faiss.IndexFlatIP(d)
    flat.add(X)
    idx["flat"] = flat
    info["flat"] = dict(recall=1.0, param="-", bytes=X.nbytes, build_s=time.time() - t)
    t = time.time()
    h = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    h.hnsw.efConstruction = 100
    h.add(X)
    bh = time.time() - t
    t = time.time()
    quant = faiss.IndexFlatIP(d)
    ivf = faiss.IndexIVFFlat(quant, d, 2048, faiss.METRIC_INNER_PRODUCT)
    ivf.train(X[rng.choice(len(X), 150000, replace=False)])
    ivf.add(X)
    bi = time.time() - t
    t = time.time()
    quant2 = faiss.IndexFlatIP(d)
    pq = faiss.IndexIVFPQ(quant2, d, 2048, 25, 8, faiss.METRIC_INNER_PRODUCT)
    pq.train(X[rng.choice(len(X), 150000, replace=False)])
    pq.add(X)
    bp = time.time() - t
    faiss.omp_set_num_threads(1)

    def setter_h(v):
        h.hnsw.efSearch = v
    v, r = tune(h, setter_h, [16, 32, 64, 128, 256, 512, 1024], Qs, gts)
    idx["hnsw"] = h
    info["hnsw"] = dict(recall=r, param=v, build_s=bh, bytes=X.nbytes + 2 * 32 * 4 * len(X))

    def setter_i(v):
        ivf.nprobe = v
    v, r = tune(ivf, setter_i, [4, 8, 16, 32, 64, 128, 256], Qs, gts)
    idx["ivf_flat"] = ivf
    info["ivf_flat"] = dict(recall=r, param=v, build_s=bi, bytes=X.nbytes)

    def setter_p(v):
        pq.nprobe = v
    v, r = tune(pq, setter_p, [8, 32, 128, 512], Qs, gts)
    idx["ivf_pq"] = pq
    info["ivf_pq"] = dict(recall=r, param=v, build_s=bp, bytes=25 * len(X) + 8 * len(X))

    cfg = {k: (lambda i, ix=ix: ix.search(Qs[i % len(Qs)][None], 10)) for k, ix in idx.items()}
    res = measure(cfg, rng)
    for k in res:
        res[k].update(info[k])
    res["_n"] = len(X)
    res["_d"] = d
    res["_flat_pJ_por_vetor_dim"] = res["flat"]["uJ"] * 1e6 / (len(X) * d)
    res["_flat_1bi_1536_J"] = res["_flat_pJ_por_vetor_dim"] * 1e9 * 1536 * 1e-12
    return res


def project(a, d, seed):
    if d == a.shape[1]:
        return a.astype(np.float32)
    P = np.random.default_rng(seed).standard_normal((a.shape[1], d), dtype=np.float32) / np.sqrt(d)
    return normal(a @ P).astype(np.float32)


def make_vocab(X, mode, seed):
    rng = np.random.default_rng(seed)
    if mode == "sorteadas":
        return X[rng.choice(len(X), V, replace=False)].copy()
    s = X[rng.integers(len(X))]
    return X[np.argsort(-(X @ s))[:V]].copy()


def episodes(rng, k, n):
    out = []
    for _ in range(n):
        ids = rng.choice(V, 2 * k, replace=False)
        out.append((ids[:k], ids[k:], int(rng.integers(k))))
    return out


def attend(K, Val, q):
    s = K @ q
    w = np.exp(BETA * (s - s.max()))
    return (w / w.sum()) @ Val


def build_hrr(voc, eps, d):
    perm = np.random.default_rng(d).permutation(d)
    Fv = np.fft.rfft(voc, axis=1).astype(np.complex64)
    Fk = np.fft.rfft(voc[:, perm], axis=1).astype(np.complex64)
    return perm, Fk, [np.sum(Fk[ki] * Fv[vi], axis=0) for ki, vi, _ in eps]


def acc_att(voc, eps):
    hit = 0
    for ki, vi, t in eps:
        hit += int(np.argmax(voc @ attend(voc[ki], voc[vi], voc[ki[t]]))) == vi[t]
    return hit / len(eps)


def acc_hrr(voc, eps, d):
    _, Fk, Ss = build_hrr(voc, eps, d)
    hit = 0
    for (ki, vi, t), S in zip(eps, Ss):
        r = np.fft.irfft(S * np.conj(Fk[ki[t]]), n=d).astype(np.float32)
        hit += int(np.argmax(voc @ r)) == vi[t]
    return hit / len(eps)


def min_dim(X0, mode, k, seed):
    rng = np.random.default_rng(seed + k)
    eps = episodes(rng, k, EPISODES)
    base = make_vocab(X0, mode, seed)
    for p in range(8, 17):
        d = 2 ** p
        a = acc_hrr(project(base, d, seed), eps, d)
        if a >= 0.99:
            return d, a
    return None, a


def part_b(X0, rng):
    res = {"min_dim": {}, "energia": {}}
    Xc = normal(X0 - X0.mean(0))
    for mode in ("sorteadas", "parecidas"):
        for k in (64, 256):
            d, a = min_dim(Xc, mode, k, 11)
            base = make_vocab(Xc, mode, 11)
            eps = episodes(np.random.default_rng(11 + k), k, EPISODES)
            res["min_dim"]["%s_k%d" % (mode, k)] = dict(d_hrr=d, acc=a, att100=acc_att(project(base, 100, 11), eps),
                                                         att1536=acc_att(project(base, 1536, 11), eps))
    for k in (64, 256):
        d = res["min_dim"]["sorteadas_k%d" % k]["d_hrr"]
        if d is None:
            continue
        base = make_vocab(Xc, "sorteadas", 11)
        eps = episodes(np.random.default_rng(99 + k), k, 256)
        vocs = {n: project(base, n, 11) for n in (100, 1536, d)}
        vh = vocs[d]
        perm, Fk, Ss = build_hrr(vh, eps, d)
        faiss.omp_set_num_threads(1)
        best = None
        for ef in (64, 128, 256, 512):
            hn = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
            hn.hnsw.efConstruction = 200
            hn.add(vh)
            hn.hnsw.efSearch = ef
            hit = 0
            for (ki, vi, t), S in zip(eps, Ss):
                r = normal(np.fft.irfft(S * np.conj(Fk[ki[t]]), n=d).astype(np.float32))
                hit += int(hn.search(r[None], 1)[1][0, 0]) == vi[t]
            best = (ef, hit / len(eps))
            if best[1] >= 0.99:
                break
        res["hnsw_limpeza_k%d" % k] = dict(ef=best[0], acc=best[1])
        A = {n: ([vocs[n][ki] for ki, _, _ in eps], [vocs[n][vi] for _, vi, _ in eps]) for n in (100, 1536)}
        qs = [(ki[t]) for ki, _, t in eps]

        def att(n, dec):
            def f(i):
                j = i % 256
                o = attend(A[n][0][j], A[n][1][j], vocs[n][qs[j]])
                if dec:
                    np.argmax(vocs[n] @ o)
            return f

        def rec(i):
            j = i % 256
            return np.fft.irfft(Ss[j] * np.conj(Fk[qs[j]]), n=d).astype(np.float32)

        cfg = {
            "att100_sem": att(100, False), "att100_dec": att(100, True),
            "att1536_sem": att(1536, False), "att1536_dec": att(1536, True),
            "hrr_sem": lambda i: rec(i),
            "hrr_varredura": lambda i: np.argmax(vh @ rec(i)),
            "hrr_hnsw": lambda i: hn.search(normal(rec(i))[None], 1),
        }
        res["energia"]["k%d" % k] = dict(d_hrr=d, medicoes=measure(cfg, rng))
    return res


def gates(b):
    g = {}
    for k in (64, 256):
        key = "k%d" % k
        if key not in b["energia"]:
            continue
        m = b["energia"][key]["medicoes"]
        d = b["energia"][key]["d_hrr"]
        hn = b["hnsw_limpeza_k%d" % k]
        best = min(m["hrr_varredura"]["uJ"], m["hrr_hnsw"]["uJ"] if hn["acc"] >= 0.99 else 1e18)
        clean_el = m["hrr_varredura"]["uJ"] - m["hrr_sem"]["uJ"]
        opt = (d * DAC_PJ + V * ADC_PJ) * 1e-6
        g[key] = dict(
            G1=bool(best < m["att1536_dec"]["uJ"]),
            G3=bool(hn["acc"] >= 0.99 and m["hrr_hnsw"]["uJ"] * 2 <= m["hrr_varredura"]["uJ"]),
            G4=bool(opt < clean_el), limpeza_eletronica_uJ=clean_el, optico_piso_uJ=opt,
            melhor_hrr_uJ=best, att1536_dec_uJ=m["att1536_dec"]["uJ"])
    for k in (64, 256):
        r, s = b["min_dim"]["sorteadas_k%d" % k], b["min_dim"]["parecidas_k%d" % k]
        g["G2_k%d" % k] = bool(r["d_hrr"] and s["d_hrr"] and s["d_hrr"] <= 2 * r["d_hrr"])
    return g


def main():
    rng = np.random.default_rng(2026)
    with h5py.File(sys.argv[1], "r") as f:
        X, Q, gt = np.array(f["train"], dtype=np.float32), np.array(f["test"], dtype=np.float32), np.array(f["neighbors"])
    X, Q = normal(X), normal(Q)
    t0 = time.time()
    a = part_a(X, Q, gt, rng)
    print("A pronto", round(time.time() - t0), flush=True)
    b = part_b(X, rng)
    out = dict(banco_vetorial=a, memoria=b, criterios=gates(b), rodadas=ROUNDS, janela_s=T)
    json.dump(out, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
