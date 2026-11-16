"""Gera artigo/resultados-gerados.tex a partir dos JSONs oficiais dos experimentos E13 a E16.

Nenhum número das tabelas e figuras desses experimentos é digitado à mão no artigo: tudo sai daqui.
Uso: python3 scripts/gerar_artigo.py   (na raiz do repositório)
"""
import itertools
import json
import os

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(RAIZ, "experimentos")
SAIDA = os.path.join(RAIZ, "artigo", "resultados-gerados.tex")


def ler(rel):
    p = os.path.join(EXP, rel)
    return json.load(open(p)) if os.path.exists(p) else None


def pt(x, casas=1):
    """Número com vírgula decimal."""
    return f"{x:.{casas}f}".replace(".", ",")


def milhar(x):
    return f"{int(round(x)):,}".replace(",", ".")


def macro(nome, valor):
    return f"\\newcommand{{\\{nome}}}{{{valor}}}\n"


out = ["% Gerado por scripts/gerar_artigo.py a partir dos JSONs em experimentos/*/resultados. Não editar à mão.\n"]

# ---------------- E13 ----------------
e13 = ler("E13-fatorial-de-I/resultados/fatorial-de-I.json")
if e13:
    a = e13["analise"]
    f = a["SQ_frac"]
    out.append(macro("EtrezeTarefa", pt(100 * f["T"])))
    out.append(macro("EtrezeInter", pt(100 * a["SQ_frac_interacoes"])))
    out.append(macro("EtrezeAlg", pt(100 * f["G"])))
    out.append(macro("EtrezeInit", pt(100 * f["Z"])))
    out.append(macro("EtrezeArq", pt(100 * f["A"])))
    out.append(macro("EtrezeZG", pt(100 * f["ZG"])))
    out.append(macro("EtrezeExpK", pt(a["exploratorio_expoente_k"], 2)))
    out.append(macro("EtrezeRazaoLarga", pt(a["exploratorio_razao_larga_prof"], 2)))
    out.append(macro("EtrezeExec", str(a["execucoes"])))
    out.append(macro("EtrezeCens", str(a["censuradas"])))
    out.append(macro("EtrezeHdoze", str(a["H12_contagem"])))
    ref = a["referencia"]
    for k in ("2", "5", "10"):
        nome = {"2": "Dois", "5": "Cinco", "10": "Dez"}[k]
        out.append(macro(f"EtrezeAlvo{nome}", pt(100 * ref["alvo"][k])))
        out.append(macro(f"EtrezeRef{nome}", pt(100 * ref["ref"][k])))
    linhas = []
    nomes_arq = {"prof": "prof.", "larga": "larga"}
    for arq, init, alg in itertools.product(("prof", "larga"), ("padrao", "xavier"), ("adam", "sgdm")):
        vals = [a["celulas"][f"{k}|{arq}|{init}|{alg}"]["I_geo"] for k in (2, 5, 10)]
        linhas.append(f"{nomes_arq[arq]} & {'padrão' if init == 'padrao' else 'Xavier'} & "
                      f"{'Adam' if alg == 'adam' else 'SGDM'} & " + " & ".join(milhar(v) for v in vals) + r" \\")
    out.append("\\newcommand{\\EtrezeTabela}{%\n\\begin{table}[t]\n\\centering\n"
               "\\caption{E13: bits entregues até o alvo, $I$ (média geométrica de 3 sementes)}\n\\label{tab:e13}\n"
               "\\begin{tabular}{lllrrr}\n\\toprule\nArq. & Inic. & Alg. & $k=2$ & $k=5$ & $k=10$ \\\\\n\\midrule\n"
               + "\n".join(linhas) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}}\n")
    # figura: log2 I contra log2 k, uma linha por combinação
    plots = []
    for arq, init, alg in itertools.product(("prof", "larga"), ("padrao", "xavier"), ("adam", "sgdm")):
        cor = "blue" if alg == "adam" else "orange"
        estilo = "solid" if arq == "prof" else "dashed"
        marca = "*" if init == "padrao" else "square*"
        pts = " ".join(f"({k},{a['celulas'][f'{k}|{arq}|{init}|{alg}']['I_geo'] / 1000:.3f})" for k in (2, 5, 10))
        plots.append(f"\\addplot[{cor}, {estilo}, mark={marca}, mark size=1.5pt] coordinates {{{pts}}};")
    out.append("\\newcommand{\\EtrezeFigura}{%\n\\begin{figure}[t]\n\\centering\n\\begin{tikzpicture}\n"
               "\\begin{axis}[width=\\columnwidth, height=5.2cm, xmode=log, ymode=log, log basis x=2,"
               " xtick={2,5,10}, xticklabels={2,5,10}, xlabel={número de classes $k$},"
               " ylabel={$I$ (mil bits)}, grid=major, grid style={gray!20}, every axis plot/.append style={thick}]\n"
               + "\n".join(plots) +
               "\n\\end{axis}\n\\end{tikzpicture}\n\\caption{E13: $I$ em função de $k$ nas oito combinações. "
               "Azul: Adam; laranja: SGD com momento; linha cheia: rede profunda; tracejada: rede larga; "
               "círculo: inicialização padrão; quadrado: Xavier.}\n\\label{fig:e13}\n\\end{figure}}\n")

# ---------------- E15 e E16 ----------------
e15 = ler("E15-ruido-do-minilote/resultados/ruido-do-minilote.json")
e16 = ler("E16-lote-grande/resultados/lote-grande.json")
if e15:
    a = e15["analise"]
    out.append(macro("EquinzeRdois", pt(a["H10l_R2_b_efetivo"], 2)))
    out.append(macro("EquinzeRdoisNominal", pt(a["comparacao_R2_b_nominal"], 2)))
    linhas = []
    fontes = [("E15", a)] + ([("E16", e16["analise"])] if e16 else [])
    for nome, an in fontes:
        for B, v in an["snr_minilote"].items():
            snr = v["media"]
            pen = an["penalidade_ruido_b1"][B]
            penq = an["penalidade_quant_b1"][B]
            fl = an["celulas"][f"{B}|float|32"]["acc"]
            linhas.append(f"{nome} & {milhar(int(B))} & {pt(snr, 2)} & {milhar(int(B) / snr)} & {pt(100 * fl)} & "
                          f"{pt(100 * pen)} & {pt(100 * penq)} \\\\")
    out.append("\\newcommand{\\EquinzeTabela}{%\n\\begin{table}[t]\n\\centering\n"
               "\\caption{E15 e E16: SNR do minilote por projeção, $B_{\\text{simple}} = B/\\SNR_B$, acurácia do canal float "
               "e penalidade (pontos) dos canais de 1 bit}\n\\label{tab:e15}\n"
               "\\begin{tabular}{lrrrrrr}\n\\toprule\n & $B$ & $\\SNR_B$ & $B_{\\text{simple}}$ & float (\\%) & ruído 1 b & quant. 1 b \\\\\n"
               "\\midrule\n" + "\n".join(linhas) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}}\n")
    pts = []
    for nome, an in fontes:
        for c in an["celulas"].values():
            if c.get("b_ef") is not None:
                pts.append((c["b_ef"], 100 * c["acc"], nome))
    p15 = " ".join(f"({x:.4f},{y:.2f})" for x, y, n in pts if n == "E15")
    p16 = " ".join(f"({x:.4f},{y:.2f})" for x, y, n in pts if n == "E16")
    out.append("\\newcommand{\\EquinzeFigura}{%\n\\begin{figure}[t]\n\\centering\n\\begin{tikzpicture}\n"
               "\\begin{axis}[width=\\columnwidth, height=5cm, xmode=log, log basis x=2,"
               " xlabel={$b_{\\text{ef}}$ (bits úteis por projeção)}, ylabel={acurácia (\\%)},"
               " grid=major, grid style={gray!20}, legend style={font=\\scriptsize, at={(0.03,0.97)}, anchor=north west}]\n"
               f"\\addplot[only marks, blue, mark=*] coordinates {{{p15}}};\n\\addlegendentry{{E15 ($B \\le 2048$)}}\n"
               + (f"\\addplot[only marks, orange, mark=square*] coordinates {{{p16}}};\n\\addlegendentry{{E16 ($B \\ge 8192$)}}\n" if p16 else "")
               + "\\end{axis}\n\\end{tikzpicture}\n\\caption{E15 e E16: acurácia contra a informação efetiva por projeção, "
               "nos canais ruidoso e float.}\n\\label{fig:e15}\n\\end{figure}}\n")
if e16:
    a = e16["analise"]
    out.append(macro("EdezesseisRdois", pt(a["H10ll_R2_b_efetivo"], 2)))
    out.append(macro("EdezesseisRdoisNominal", pt(a["comparacao_R2_b_nominal"], 2)))
    out.append(macro("EdezesseisCelulas", str(a["H10ll_n_celulas"])))

# ---------------- E14 ----------------
e14 = ler("E14-perturbacao-real/resultados/perturbacao-real.json")
if e14:
    a = e14["analise"]
    cols = [("exata", None), ("dif", None), ("abs", 8), ("abs", 12), ("abs", 16), ("abs", 20),
            ("difer", 1), ("difer", 2), ("difer", 4)]
    linhas = []
    for N in a["N"]:
        for r in (8, 32, 128):
            vals = [a["celulas"][f"{N}|{le}|{b}|{r}"]["acc"] for le, b in cols]
            linhas.append(f"{milhar(N)} & {r} & " + " & ".join(pt(100 * v) for v in vals) + r" \\")
    out.append("\\newcommand{\\EquatorzeTabela}{%\n\\begin{table*}[t]\n\\centering\n"
               "\\caption{E14: acurácia de teste (\\%) com projeções medidas por diferença central da perda, sem backpropagation "
               "(média de 2 sementes). \\emph{abs}: cada perda lida por um conversor de $b$ bits em faixa fixa; "
               "\\emph{difer}: a diferença lida diretamente, com faixa adaptada}\n\\label{tab:e14}\n"
               "\\begin{tabular}{rrccccccccc}\n\\toprule\n"
               "$N$ & $r$ & exata & dif. & abs 8 & abs 12 & abs 16 & abs 20 & difer 1 & difer 2 & difer 4 \\\\\n"
               "\\midrule\n" + "\n".join(linhas) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}}\n")
    for k, nome in (("P1_contagem", "EquatorzePum"), ("P2_contagem", "EquatorzePdois"),
                    ("P3_contagem", "EquatorzePtres"), ("P4_contagem", "EquatorzePquatro")):
        out.append(macro(nome, str(a[k])))

open(SAIDA, "w").write("".join(out))
print("gerado:", SAIDA, "| E13", bool(e13), "E14", bool(e14), "E15", bool(e15), "E16", bool(e16))
