"""Fourier no grupo errado: testes de brinquedo em MLP treinada em numpy.

Predição (lema de Schur): em camadas ocultas, a ordem dos neurônios é arbitrária. A lei dos
pesos treinados é invariante a permutações, então nenhuma base fixa ganha da aleatória em
média. No eixo de entrada, onde a ordem é real, a DCT pode ganhar se o dado for suave.
Uso: uv run --no-project --with numpy python fourier_grupo_errado.py
"""
import math
import numpy as np

D, H, N_TR, N_TE, STEPS, LR, SEEDS = 64, 64, 1500, 3000, 1200, 3e-3, 12


def dct_basis(n):
    t = (np.arange(n) + 0.5) / n
    B = np.array([np.cos(np.pi * k * t) for k in range(n)]).T
    return B / np.linalg.norm(B, axis=0)


def rand_basis(rng, n):
    return np.linalg.qr(rng.standard_normal((n, n)))[0]


def make_data(seed):
    rng = np.random.default_rng(90000 + seed)
    C = dct_basis(D)
    s = (np.arange(D) + 1.0) ** (-2.0)
    s = s / s.mean()
    sample = lambda n: (rng.standard_normal((n, D)) * np.sqrt(s)) @ C.T
    Xtr, Xte = sample(N_TR), sample(N_TE)
    F = C[:, :8] @ rng.standard_normal((8, 8))  # professor olha modos baixos
    g = rng.standard_normal(8)
    teach = lambda X: np.tanh(X @ F) @ g
    ytr, yte = teach(Xtr), teach(Xte)
    sd = ytr.std()
    return Xtr, ytr / sd + 0.05 * rng.standard_normal(N_TR), Xte, yte / sd


class Adam:
    def __init__(self, ps):
        self.m = [np.zeros_like(p) for p in ps]
        self.v = [np.zeros_like(p) for p in ps]
        self.t = 0

    def step(self, ps, gs):
        self.t += 1
        for i, (p, g) in enumerate(zip(ps, gs)):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            p -= LR * (self.m[i] / (1 - 0.9 ** self.t)) / (np.sqrt(self.v[i] / (1 - 0.999 ** self.t)) + 1e-8)


def train(seed, data, w1_par=None, w2_par=None):
    """w_par=(A, B, M): W = A (M*C) B^T. None = peso denso."""
    Xtr, ytr, Xte, yte = data
    rng = np.random.default_rng(seed)
    init = lambda a, b: rng.standard_normal((a, b)) / np.sqrt(a)
    W1, W2, a = init(D, H), init(H, H), rng.standard_normal(H) / np.sqrt(H)
    b1, b2 = np.zeros(H), np.zeros(H)

    def enc(W, par):
        if par is None:
            return W.copy()
        A, B, M = par
        return M * (A.T @ W @ B)

    def dec(C, par):
        if par is None:
            return C
        A, B, M = par
        return A @ (M * C) @ B.T

    C1, C2 = enc(W1, w1_par), enc(W2, w2_par)
    ps = [C1, C2, a, b1, b2]
    opt = Adam(ps)

    def forward(X, W1_, W2_):
        h1 = np.maximum(X @ W1_ + b1, 0)
        h2 = np.maximum(h1 @ W2_ + b2, 0)
        return h1, h2, h2 @ a

    for _ in range(STEPS):
        idx = rng.integers(0, N_TR, 128)
        X, y = Xtr[idx], ytr[idx]
        W1_, W2_ = dec(C1, w1_par), dec(C2, w2_par)
        h1, h2, p = forward(X, W1_, W2_)
        e = (p - y)[:, None]
        m1, m2 = (h1 > 0).astype(float), (h2 > 0).astype(float)
        ga = (h2.T @ e)[:, 0] / len(y)
        d2 = e * a[None, :] * m2
        gb2 = d2.mean(0)
        gW2 = h1.T @ d2 / len(y)
        d1 = (d2 @ W2_.T) * m1
        gb1 = d1.mean(0)
        gW1 = X.T @ d1 / len(y)
        g1 = gW1 if w1_par is None else w1_par[2] * (w1_par[0].T @ gW1 @ w1_par[1])
        g2 = gW2 if w2_par is None else w2_par[2] * (w2_par[0].T @ gW2 @ w2_par[1])
        opt.step(ps, [g1, g2, ga, gb1, gb2])
    W1_, W2_ = dec(C1, w1_par), dec(C2, w2_par)
    mse = float(np.mean((forward(Xte, W1_, W2_)[2] - yte) ** 2))
    return mse, W1_, W2_


def low_energy(W, axis, Basis, hidden_axes=(1,)):
    """Remove a média nos eixos ocultos (componente trivial de S_n), como exige o lema de Schur.
    Depois disso o modo DC da DCT fica vazio: a referência da DCT é 31/63 = 0.492 num eixo oculto."""
    for ax in hidden_axes:
        W = W - W.mean(axis=ax, keepdims=True)
    c = Basis.T @ W if axis == 0 else W @ Basis
    half = W.shape[axis] // 2
    e = (c[:half, :] ** 2).sum() if axis == 0 else (c[:, :half] ** 2).sum()
    return float(e / (c ** 2).sum())


def seriate(W):
    S = W @ W.T
    L = np.diag(np.abs(S).sum(1)) - np.abs(S)
    return np.argsort(np.linalg.eigh(L)[1][:, 1])


def main():
    C = dct_basis(D)
    Id = np.eye(D)
    mask_in = np.zeros((D, H)); mask_in[:D // 2, :] = 1
    i, j = np.indices((H, H))
    mask_2d = ((i + j) < np.sort((i + j).ravel())[H * H // 2]).astype(float)
    print("metade da energia seria 0.500 sem estrutura. Sementes: %d" % SEEDS)
    res = {k: [] for k in ["W1_in_dct", "W1_in_rand", "W1_hid_dct", "W1_hid_rand", "W2_a_dct", "W2_a_rand", "W2_b_dct",
                           "ser_trained", "ser_ctrl", "dense", "in_dct", "in_rand", "h_dct", "h_rand"]}
    for s in range(SEEDS):
        rng = np.random.default_rng(555 + s)
        data = make_data(s)
        mse, W1, W2 = train(s, data)
        res["dense"].append(mse)
        R1, R2 = rand_basis(rng, D), rand_basis(rng, H)
        res["W1_in_dct"].append(low_energy(W1, 0, C)); res["W1_in_rand"].append(low_energy(W1, 0, R1))
        res["W1_hid_dct"].append(low_energy(W1, 1, C)); res["W1_hid_rand"].append(low_energy(W1, 1, R2))
        both = (0, 1)
        res["W2_a_dct"].append(low_energy(W2, 0, C, both)); res["W2_a_rand"].append(low_energy(W2, 0, R2, both))
        res["W2_b_dct"].append(low_energy(W2, 1, C, both))
        p = seriate(W2)
        res["ser_trained"].append(low_energy(W2[p], 0, C, both))
        U, S, Vt = np.linalg.svd(W2)
        ctrl = rand_basis(rng, H) @ np.diag(S) @ rand_basis(rng, H)
        res["ser_ctrl"].append(low_energy(ctrl[seriate(ctrl)], 0, C, both))
        for name, w1, w2 in [
            ("in_dct", (C, Id, mask_in), None),
            ("in_rand", (R1, Id, mask_in), None),
            ("h_dct", None, (C, C, mask_2d)),
            ("h_rand", None, (R2, rand_basis(rng, H), mask_2d)),
        ]:
            res[name].append(train(s, data, w1, w2)[0])

    for k in list(res)[:9]:
        v = np.array(res[k])
        print("%-12s energia em metade dos modos: média %.3f dp %.3f" % (k, v.mean(), v.std()))
    for k in ["dense", "in_dct", "in_rand", "h_dct", "h_rand"]:
        v = np.array(res[k])
        print("%-8s MSE teste mediana %.4f (q25 %.4f q75 %.4f)" % (k, np.median(v), *np.percentile(v, [25, 75])))
    wins = np.mean(np.array(res["in_dct"]) < np.array(res["in_rand"]))
    winh = np.mean(np.array(res["h_dct"]) < np.array(res["h_rand"]))
    print("DCT vence aleatória: eixo de entrada %.2f; oculto-oculto %.2f" % (wins, winh))
    for n in (64, 1000, 100000):
        print("n=%d: log2(n!)=%.0f bits; matriz n x n a 32 bits=%.0f bits; razão %.4f%%"
              % (n, math.lgamma(n + 1) / math.log(2), 32.0 * n * n, 100 * (math.lgamma(n + 1) / math.log(2)) / (32.0 * n * n)))


if __name__ == "__main__":
    main()
