"""Recalcula cada número citado no ensaio a partir dos JSONs oficiais e diz se bate.

Uso: python scripts/verificar_ensaio.py
Saída: uma linha por afirmação (OK ou DIVERGE), com o arquivo e o campo de origem. Sai com código 1 se algo divergir.
O mapa comentado está em docs/07-mapa-ensaio.md.
"""
import json
import math
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "experimentos"
E04 = R / "E04-microteste-energia/resultados"
E11 = R / "E11-canais-de-gradiente/resultados/canais-de-gradiente.json"


def j(p):
    return json.loads(Path(p).read_text())


e11 = j(E11)
an = e11["analise"]
acc = an["acc"]
conf = j(E04 / "microteste-confirmacao.json")["energia"]
micro = j(E04 / "microteste-resultados.json")

# (afirmação no ensaio, valor citado, valor recalculado, tolerância, origem)
CHECAGENS = [
    ("redes de 13.002 a 118.282 parâmetros", (13002, 118282), (an["N"][0], an["N"][-1]), 0,
     "E11 analise.N"),
    ("duas sementes e 1.500 passos", (2, 1500), (e11["sementes"], e11["passos"]), 0, "E11 sementes, passos"),
    ("r = 8: 69,6% na rede menor", 69.6, 100 * acc["aleatorio"]["16"][0], 0.05, "E11 analise.acc.aleatorio['16'][0]"),
    ("r = 8: 74,2% na rede maior", 74.2, 100 * acc["aleatorio"]["128"][0], 0.05, "E11 analise.acc.aleatorio['128'][0]"),
    ("r = 128: 78,8% na rede menor", 78.8, 100 * acc["aleatorio"]["16"][4], 0.05, "E11 analise.acc.aleatorio['16'][4]"),
    ("r = 128: 80,7% na rede maior", 80.7, 100 * acc["aleatorio"]["128"][4], 0.05, "E11 analise.acc.aleatorio['128'][4]"),
    ("12.000 medidas (r = 8 x 1.500 passos)", 12000, 8 * e11["passos"], 0, "8 x E11 passos"),
    ("backprop 84,5% (rede menor)", 84.5, 100 * an["backprop"][0], 0.05, "E11 analise.backprop[0]"),
    ("backprop 87,0% (rede maior)", 87.0, 100 * an["backprop"][-1], 0.05, "E11 analise.backprop[-1]"),
    ("nenhum r até 128 a menos de 2 pontos", True, all(math.isnan(x) for x in an["r_estrela"]["aleatorio"]), 0,
     "E11 analise.r_estrela.aleatorio (todos NaN)"),
    ("adaptativo 78,0% na rede maior", 78.0, 100 * acc["adaptativo"]["128"][4], 0.05,
     "E11 analise.acc.adaptativo['128'][4]"),
    ("aleatório 80,7% na rede maior (comparação)", 80.7, 100 * acc["aleatorio"]["128"][4], 0.05,
     "E11 analise.acc.aleatorio['128'][4]"),
    ("limpeza: 6x pior (latência, varredura)", 6.0,
     conf["hrr_varredura"]["latencia_us"] / conf["att1536_dec"]["latencia_us"], 0.6,
     "E04 confirmacao energia.hrr_varredura.latencia_us / att1536_dec.latencia_us"),
    ("limpeza: 9x pior (latência, HNSW)", 9.0,
     conf["hrr_hnsw"]["latencia_us"] / conf["att1536_dec"]["latencia_us"], 0.6,
     "E04 confirmacao energia.hrr_hnsw.latencia_us / att1536_dec.latencia_us"),
    ("carga de fundo 8x maior que o sinal (banco flat)", 8.0,
     micro["banco_vetorial"]["flat"]["bruto_uJ"] / micro["banco_vetorial"]["flat"]["uJ"], 0.5,
     "E04 resultados banco_vetorial.flat.bruto_uJ / uJ"),
]


def igual(citado, real, tol):
    if isinstance(citado, tuple):
        return all(abs(a - b) <= tol for a, b in zip(citado, real))
    if isinstance(citado, bool):
        return citado == real
    return abs(citado - real) <= tol


def fmt(v):
    if isinstance(v, tuple):
        return ", ".join(fmt(x) for x in v)
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


falhas = 0
for texto, citado, real, tol, origem in CHECAGENS:
    ok = igual(citado, real, tol)
    falhas += not ok
    print(f"{'OK     ' if ok else 'DIVERGE'} | {texto:45s} | citado {fmt(citado):>16s} | recalculado {fmt(real):>16s} | {origem}")
print(f"\n{len(CHECAGENS) - falhas}/{len(CHECAGENS)} afirmações conferem.")
sys.exit(1 if falhas else 0)
