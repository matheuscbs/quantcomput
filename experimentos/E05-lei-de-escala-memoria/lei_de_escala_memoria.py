"""Lei de escala: quanto custa ler um dado conforme a memória usada cresce.

Mede latência por leitura aleatória (roteamento arbitrário, como a atenção de todos com todos)
e por leitura sequencial (varredura com pré-busca), de 16 KB a 4 GB. Com carga de fundo, latência
é a medida confiável; a potência bruta serve só para converter em energia aproximada.
Uso: uv run --no-project --with numpy --with zeus-apple-silicon python -I lei_de_escala_memoria.py saida.json
"""
import json
import sys
import time

import numpy as np
from zeus_apple_silicon import AppleEnergyMonitor

mon = AppleEnergyMonitor()
READS, REPS = 2_000_000, 5


def timed(fn):
    mon.begin_window("x")
    t0 = time.perf_counter()
    fn()
    dt = time.perf_counter() - t0
    m = mon.end_window("x")
    return dt, (m.cpu_total_mj + (m.dram_mj or 0)) / 1e3 / dt


def main():
    rng = np.random.default_rng(5)
    rows = []
    for p in range(11, 30):  # 2^p elementos de 8 bytes: 16 KB a 4 GB
        n = 2 ** p
        a = rng.integers(0, 1 << 40, n, dtype=np.int64)
        idx = rng.integers(0, n, READS)
        rand, seq, watts = [], [], []
        for _ in range(REPS):
            dt, w = timed(lambda: np.take(a, idx).sum())
            rand.append(dt / READS * 1e9)
            watts.append(w)
            m = min(n, READS)
            dt2, _ = timed(lambda: np.sum(a[:m]))
            seq.append(dt2 / m * 1e9)
        rows.append(dict(bytes=n * 8, aleatoria_ns=float(np.median(rand)), sequencial_ns=float(np.median(seq)), watts=float(np.median(watts))))
        print(rows[-1], flush=True)
        del a, idx
    x = np.log2([r["bytes"] for r in rows])
    y = np.log2([r["aleatoria_ns"] for r in rows])
    fits = {}
    for name, lo, hi in (("cache_16KB_16MB", 14, 24), ("dram_32MB_4GB", 25, 32), ("tudo", 14, 32)):
        s = (x >= lo) & (x <= hi)
        fits[name] = float(np.polyfit(x[s], y[s], 1)[0])
    json.dump(dict(linhas=rows, expoente_custo_vs_memoria=fits), open(sys.argv[1], "w"), indent=1)
    print(fits)


if __name__ == "__main__":
    main()
