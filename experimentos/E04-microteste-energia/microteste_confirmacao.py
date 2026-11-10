"""Confirmação do microteste (k=64) com janelas de 2 s, linha de base antes e depois, e latência.
Parte exploratória, posterior aos critérios: branquear o vocabulário corrige a falha de chaves parecidas?
Uso: igual ao microteste; argumentos: glove.hdf5 saida.json
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import faiss
import h5py
import numpy as np

spec = importlib.util.spec_from_file_location("mt", Path(__file__).with_name("microteste_energia.py"))
mt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mt)
T, ROUNDS, K = 2.0, 15, 64


def base():
    mt.mon.begin_window("i")
    t0 = time.perf_counter()
    time.sleep(T)
    m = mt.mon.end_window("i")
    return (m.cpu_total_mj + (m.dram_mj or 0)) / (time.perf_counter() - t0)


def run(fn):
    mt.mon.begin_window("w")
    t0, n = time.perf_counter(), 0
    while time.perf_counter() - t0 < T:
        fn(n)
        n += 1
    m = mt.mon.end_window("w")
    return m.cpu_total_mj + (m.dram_mj or 0), n, time.perf_counter() - t0


def measure(cfg, rng):
    names = list(cfg)
    rec = {k: dict(net=[], gross=[], lat=[]) for k in names}
    for r in range(ROUNDS):
        rng.shuffle(names)
        for k in names:
            b0 = base()
            e, n, dt = run(cfg[k])
            b1 = base()
            rec[k]["net"].append((e - 0.5 * (b0 + b1) * dt) * 1e3 / n)
            rec[k]["gross"].append(e * 1e3 / n)
            rec[k]["lat"].append(dt / n * 1e6)
    out = {}
    for k, v in rec.items():
        a = np.array(v["net"])
        out[k] = dict(liquido_uJ=float(np.median(a)), q25=float(np.percentile(a, 25)), q75=float(np.percentile(a, 75)),
                      bruto_uJ=float(np.median(v["gross"])), latencia_us=float(np.median(v["lat"])))
    return out


def whiten(a):
    a = a - a.mean(0)
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    return mt.normal((a @ vt.T) / (s + 1e-6) * np.sqrt(len(a)) @ vt)


def min_dim(base_voc, k, seed):
    eps = mt.episodes(np.random.default_rng(seed + k), k, mt.EPISODES)
    last = 0.0
    for p in range(8, 17):
        d = 2 ** p
        last = mt.acc_hrr(mt.project(base_voc, d, seed), eps, d)
        if last >= 0.99:
            return d, last
    return None, last


def main():
    rng = np.random.default_rng(77)
    with h5py.File(sys.argv[1], "r") as f:
        X = np.array(f["train"], dtype=np.float32)
    X = mt.normal(X)
    Xc = mt.normal(X - X.mean(0))
    base_voc = mt.make_vocab(Xc, "sorteadas", 11)
    d = 4096
    eps = mt.episodes(np.random.default_rng(99 + K), K, 256)
    vocs = {n: mt.project(base_voc, n, 11) for n in (100, 1536, d)}
    vh = vocs[d]
    perm, Fk, Ss = mt.build_hrr(vh, eps, d)
    faiss.omp_set_num_threads(1)
    hn = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    hn.hnsw.efConstruction = 200
    hn.add(vh)
    hn.hnsw.efSearch = 64
    A = {n: ([vocs[n][ki] for ki, _, _ in eps], [vocs[n][vi] for _, vi, _ in eps]) for n in (100, 1536)}
    qs = [ki[t] for ki, _, t in eps]

    def att(n, dec):
        def f(i):
            j = i % 256
            o = mt.attend(A[n][0][j], A[n][1][j], vocs[n][qs[j]])
            if dec:
                np.argmax(vocs[n] @ o)
        return f

    def rec(i):
        j = i % 256
        return np.fft.irfft(Ss[j] * np.conj(Fk[qs[j]]), n=d).astype(np.float32)

    cfg = {"att100_sem": att(100, False), "att100_dec": att(100, True), "att1536_dec": att(1536, True),
           "hrr_sem": lambda i: rec(i), "hrr_varredura": lambda i: np.argmax(vh @ rec(i)),
           "hrr_hnsw": lambda i: hn.search(mt.normal(rec(i))[None], 1)}
    energia = measure(cfg, rng)

    explor = {}
    for mode, k in (("parecidas", 64), ("sorteadas", 256)):
        v = mt.make_vocab(Xc, mode, 11)
        explor["%s_k%d_branqueado" % (mode, k)] = min_dim(whiten(v), k, 11)
    out = dict(energia=energia, exploratorio=explor, janela_s=T, rodadas=ROUNDS)
    json.dump(out, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
