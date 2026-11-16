"""Com o minilote limpo, o ruído do canal passa a pesar? (fecha H14b, que o E15 não alcançou)

O E15 mediu SNR do minilote de 0,23 a 0,75 por projeção, ou seja, B_simple = B / SNR_B de cerca de 565 a 2.730
(McCandlish et al., 2018). Todos os lotes do E15 ficaram abaixo de B_simple, e a H14b (a penalidade do canal de
1 bit cresce com o lote) reprovou sem que o teste chegasse ao regime em que deveria aparecer.
Aqui o lote passa de B_simple: B em {8192, 32768} (o treino tem 60.000 imagens; sorteio com reposição, como no E15).

Mecânica idêntica à do E15 (importa as funções de ruido_do_minilote.py sem alterá-las): N = 118.282, r = 32,
1500 passos, Adam lr 1e-3, canais float, ruido (b = 1, 2, 4) e quant (b = 1, 2); SNR do minilote medido a cada
100 passos com 8 minilotes independentes nas execuções float; 2 sementes.

Previsão (H14 aplicada a este regime): com SNR_B acima do SNR do canal de 1 bit (3), o canal domina o erro de
cada projeção, b_ef cai para perto de 1 bit no canal de 1 bit, e a penalidade em relação ao float cresce.

Critérios fixados ANTES de rodar (ver docs/decisoes/0002). Médias das 2 sementes.
- R0 (regime alcançado): mediana de SNR_B em B = 32768 >= 10 nas DUAS sementes. Se falhar, E16 é inconclusivo
  e nenhum outro critério é aplicado.
- H14b' (a penalidade aparece): pen(B) = acc(float, B) - acc(ruido b=1, B). Aprovado se
  pen(32768) >= pen_E15(128) + 1 ponto, com pen_E15(128) lido do resultado oficial do E15 (0,91 ponto).
- H14 (veredito combinado com o E15): aprovada se R0 e H14b'; reprovada se R0 vale e H14b' não.
- H10'' (informação efetiva governa em toda a faixa de lotes): nas 23 células ruido/float do E15 (15) e do
  E16 (2 B x {float, ruido b = 1, 2, 4} = 8), acc ~ a + c*log2(b_ef) tem R^2 >= 0,80.
  Registra-se também o R^2 com b nominal (float = 32).
Inconclusivo adicional: desvio entre sementes acima de 2 pontos em mais de 1/4 das células do E16.

Uso:
  python lote_grande.py rodar dir_dados saida.json SEMENTES
  python lote_grande.py analisar final.json e15_oficial.json parcial_1.json [parcial_2.json ...]
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "E15-ruido-do-minilote"))
import ruido_do_minilote as E15  # noqa: E402

LOTES = (8192, 32768)
CONFIGS = (("float", 32), ("ruido", 1), ("ruido", 2), ("ruido", 4), ("quant", 1), ("quant", 2))


def rodar(raiz, saida, sementes):
    dt = E15.dados(raiz)
    res = dict(passos=E15.PASSOS, r=E15.R, linhas=[])
    t0 = time.time()
    for sem in sementes:
        for B in LOTES:
            for tipo, b in CONFIGS:
                out = E15.rodar_um(dt, tipo, b, B, sem, medir=(tipo == "float"))
                lin = dict(sem=sem, B=B, canal=tipo, b=b, **out)
                res["linhas"].append(lin)
                print({k: v for k, v in lin.items() if k != "medidas"}, round(time.time() - t0), "s", flush=True)
                json.dump(res, open(saida, "w"), indent=0)


def b_ef(snr_b, tipo, b):
    if tipo == "float":
        snr = snr_b
    else:
        snr_c = 2.0 ** (2 * b) - 1
        snr = 1 / (1 / snr_b + (1 + 1 / snr_b) / snr_c)
    return 0.5 * np.log2(1 + max(snr, 1e-12))


def analisar(final, e15, parciais):
    L = [l for f in parciais for l in json.load(open(f))["linhas"]]
    sems = sorted({l["sem"] for l in L})
    an = dict(sementes=sems, snr_minilote={}, celulas={})
    for B in LOTES:
        por_sem = {}
        for s in sems:
            med = [m for l in L if l["B"] == B and l["sem"] == s and l["canal"] == "float" for m in l["medidas"]]
            por_sem[str(s)] = float(np.median([m["S"] / m["s2"] for m in med]))
        an["snr_minilote"][str(B)] = dict(por_semente=por_sem, media=float(np.mean(list(por_sem.values()))),
                                          B_simple=float(B / np.mean(list(por_sem.values()))))
    dps = []
    for B in LOTES:
        for tipo, b in CONFIGS:
            v = [l["acc"] for l in L if l["B"] == B and l["canal"] == tipo and l["b"] == b]
            snr_b = an["snr_minilote"][str(B)]["media"]
            an["celulas"][f"{B}|{tipo}|{b}"] = dict(acc=float(np.mean(v)), dp=float(np.std(v)) if len(v) > 1 else 0.0,
                                                    b_ef=float(b_ef(snr_b, tipo, b)) if tipo != "quant" else None)
            dps.append(an["celulas"][f"{B}|{tipo}|{b}"]["dp"])
    A = lambda B, t, b: an["celulas"][f"{B}|{t}|{b}"]["acc"]
    an["inconclusivo_sementes"] = bool(len(sems) > 1 and np.mean(np.array(dps) > 0.02) > 0.25)
    an["R0_regime_alcancado"] = all(v >= 10 for v in an["snr_minilote"]["32768"]["por_semente"].values())

    e = json.load(open(e15))["analise"]
    pen_e15 = e["penalidade_ruido_b1"]["128"]
    an["penalidade_E15_128"] = pen_e15
    an["penalidade_ruido_b1"] = {str(B): A(B, "float", 32) - A(B, "ruido", 1) for B in LOTES}
    an["penalidade_quant_b1"] = {str(B): A(B, "float", 32) - A(B, "quant", 1) for B in LOTES}
    an["H14b_linha_penalidade_aparece"] = an["penalidade_ruido_b1"]["32768"] >= pen_e15 + 0.01

    cel = [(c["acc"], c["b_ef"], k.split("|")) for k, c in e["celulas"].items() if k.split("|")[1] != "quant"]
    cel += [(c["acc"], c["b_ef"], k.split("|")) for k, c in an["celulas"].items() if k.split("|")[1] != "quant"]
    y = np.array([a for a, _, _ in cel])

    def r2(x):
        X = np.column_stack([np.ones(len(y)), x])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return float(1 - ((y - X @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()), coef.tolist()

    an["H10ll_n_celulas"] = len(cel)
    an["H10ll_R2_b_efetivo"], an["H10ll_coef"] = r2(np.log2([be for _, be, _ in cel]))
    an["comparacao_R2_b_nominal"], _ = r2(np.log2([int(k[2]) for _, _, k in cel]))
    an["H10ll_informacao_efetiva_governa"] = an["H10ll_R2_b_efetivo"] >= 0.80

    if not an["R0_regime_alcancado"] or an["inconclusivo_sementes"]:
        an["H14b_linha_penalidade_aparece"] = an["H14"] = an["H10ll_informacao_efetiva_governa"] = None
    else:
        an["H14"] = "aprovada" if an["H14b_linha_penalidade_aparece"] else "reprovada"
    json.dump(dict(passos=E15.PASSOS, linhas=L, analise=an), open(final, "w"), indent=1)
    print(json.dumps({k: v for k, v in an.items() if k != "celulas"}, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "rodar":
        rodar(sys.argv[2], sys.argv[3], [int(s) for s in sys.argv[4].split(",")])
    elif sys.argv[1] == "analisar":
        analisar(sys.argv[2], sys.argv[3], sys.argv[4:])
