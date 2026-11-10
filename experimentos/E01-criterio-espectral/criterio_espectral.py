"""Teste de brinquedo: a energia do gradiente inicial prevê se truncar em frequência compensa?

Regressão linear, p=256 pesos, K=21 modos de baixa frequência. Compara o erro em excesso
do modelo truncado com o do modelo completo, no treino e sob mudança de distribuição.
Uso: uv run --no-project --with numpy python criterio_espectral.py
"""
import numpy as np

P, K, N_TRAIN, N_EARLY, SIGMA, SEEDS = 256, 21, 600, 320, 0.2, 30
ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]


def fourier_basis(p):
    t = np.arange(p)
    cols, freqs = [np.ones(p) / np.sqrt(p)], [0]
    for k in range(1, p // 2):
        cols += [np.sqrt(2 / p) * np.cos(2 * np.pi * k * t / p), np.sqrt(2 / p) * np.sin(2 * np.pi * k * t / p)]
        freqs += [k, k]
    cols.append(np.cos(np.pi * t) / np.sqrt(p))
    freqs.append(p // 2)
    return np.array(cols).T, np.array(freqs)


def fit(X, y, B=None, lam=1e-3):
    A = X if B is None else X @ B
    c = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)
    return c if B is None else B @ c


def excess(w, w_true, B, s):
    d = B.T @ (w - w_true)
    return float(np.sum(s * d ** 2))


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def run(rng, Phi, freqs, alpha, beta, target="fourier"):
    amp = (freqs + 1.0) ** (-alpha / 2)
    if target == "fourier":
        w = Phi @ (rng.standard_normal(P) * amp)
    else:  # estrutura real, mas numa base aleatória
        R, _ = np.linalg.qr(rng.standard_normal((P, P)))
        w = R[:, :K] @ rng.standard_normal(K)
    w /= np.linalg.norm(w)
    s = (freqs + 1.0) ** (-beta)
    s = s / s.mean()
    Z = rng.standard_normal((N_TRAIN, P))
    X = (Z * np.sqrt(s)) @ Phi.T
    y = X @ w + SIGMA * rng.standard_normal(N_TRAIN)

    g = X[:N_EARLY].T @ y[:N_EARLY] / N_EARLY
    c = Phi.T @ g
    conc = float(np.sum(c[:K] ** 2) / np.sum(c ** 2))
    s_hat = np.mean((X[:N_EARLY] @ Phi) ** 2, axis=0)
    cw = c / s_hat
    conc_w = float(np.sum(cw[:K] ** 2) / np.sum(cw ** 2))

    full = fit(X, y)
    trunc = fit(X, y, Phi[:, :K])
    R, _ = np.linalg.qr(rng.standard_normal((P, P)))
    rand = fit(X, y, R[:, :K])
    white = np.ones(P)
    return dict(
        conc=conc, conc_w=conc_w,
        ratio_in=excess(trunc, w, Phi, s) / excess(full, w, Phi, s),
        ratio_shift=excess(trunc, w, Phi, white) / excess(full, w, Phi, white),
        rand_in=excess(rand, w, Phi, s) / excess(full, w, Phi, s),
        energy=float(np.sum((Phi.T @ w)[:K] ** 2)),
    )


def main():
    Phi, freqs = fourier_basis(P)
    print("base nula: base aleatória captura em média K/p = %.3f" % (K / P))
    scen = [("iso", 0.0, a, "fourier") for a in ALPHAS] + [("aniso", 1.0, a, "fourier") for a in ALPHAS]
    scen += [("iso-alvo-base-aleatoria", 0.0, 0.0, "random")]
    rows = {}
    for name, beta, a, tgt in scen:
        rs = [run(np.random.default_rng(1000 * i + int(a * 10) + int(beta) * 7), Phi, freqs, a, beta, tgt) for i in range(SEEDS)]
        rows[(name, a)] = rs
        m = lambda k: np.median([r[k] for r in rs])
        print("%-24s a=%.1f energia_K=%.3f C=%.3f Cw=%.3f trunc/full treino=%9.3f shift=%9.3f aleat/full=%8.3f"
              % (name, a, m("energy"), m("conc"), m("conc_w"), m("ratio_in"), m("ratio_shift"), m("rand_in")))

    def collect(prefix, key):
        rs = [r for (n, _), v in rows.items() if n == prefix for r in v]
        return np.array([r[key] for r in rs])

    for label, name in (("isotrópico", "iso"), ("anisotrópico", "aniso")):
        for crit in ("conc", "conc_w"):
            print("%s %s: Spearman treino=%.2f shift=%.2f" % (
                label, crit,
                spearman(collect(name, crit), np.log(collect(name, "ratio_in"))),
                spearman(collect(name, crit), np.log(collect(name, "ratio_shift")))))

    iso_c, iso_y = collect("iso", "conc"), collect("iso", "ratio_in") < 1
    grid = np.linspace(0, 1, 201)
    idx = np.arange(len(iso_c)) % 2 == 0
    th = grid[np.argmax([np.mean((iso_c[idx] > t) == iso_y[idx]) for t in grid])]
    print("limiar C=%.3f, acurácia fora da amostra (iso): %.3f; prevalência truncar vence: %.2f"
          % (th, np.mean((iso_c[~idx] > th) == iso_y[~idx]), iso_y.mean()))
    for key in ("ratio_in", "ratio_shift"):
        an_c, an_y = collect("aniso", "conc"), collect("aniso", key) < 1
        print("mesmo limiar no aniso (%s): acurácia %.3f; prevalência %.2f" % (key, np.mean((an_c > th) == an_y), an_y.mean()))


if __name__ == "__main__":
    main()
