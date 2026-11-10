"""Baixa os três conjuntos de dados usados nos experimentos e confere os hashes em dados/SHA256SUMS.

Uso: python scripts/baixar_dados.py [fashion] [moby] [glove]   (sem argumento, baixa os três)

- fashion: Fashion-MNIST pelo OpenML (data_id 40996), salvo como dados/fashion/fashion.npz (E11).
- moby: Moby Dick, Project Gutenberg #2701, texto puro, salvo como dados/moby.txt (E06 a E10).
- glove: glove-100-angular do ann-benchmarks, salvo como dados/glove-100-angular.hdf5 (E04).

O Gutenberg revisa seus arquivos de tempos em tempos. Se o hash de moby.txt não bater, o script avisa
e mantém o arquivo; os números de E06 a E10 podem então variar um pouco.
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1] / "dados"
URLS = {
    "moby": ("https://www.gutenberg.org/cache/epub/2701/pg2701.txt", RAIZ / "moby.txt"),
    "glove": ("http://ann-benchmarks.com/glove-100-angular.hdf5", RAIZ / "glove-100-angular.hdf5"),
}


def esperado():
    tabela = {}
    for linha in (RAIZ / "SHA256SUMS").read_text().splitlines():
        if linha and not linha.startswith("#"):
            h, nome = linha.split(None, 1)
            tabela[nome.split()[0]] = h
    return tabela


def sha256(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def conferir(caminho):
    alvo = esperado().get(caminho.name)
    real = sha256(caminho)
    estado = "ok" if real == alvo else f"DIFERENTE (esperado {alvo})"
    print(f"{caminho.name}: sha256 {real[:16]}... {estado}")


def fashion():
    import numpy as np
    from sklearn.datasets import fetch_openml

    destino = RAIZ / "fashion"
    destino.mkdir(parents=True, exist_ok=True)
    x, y = fetch_openml(data_id=40996, as_frame=False, return_X_y=True,
                        data_home=str(destino / "openml"), parser="liac-arff")
    np.savez(destino / "fashion.npz", x=x.astype(np.uint8), y=y.astype(np.int64))
    conferir(destino / "fashion.npz")


def baixar(nome):
    url, caminho = URLS[nome]
    caminho.parent.mkdir(parents=True, exist_ok=True)
    if not caminho.exists():
        print("baixando", url)
        urllib.request.urlretrieve(url, caminho)
    conferir(caminho)


if __name__ == "__main__":
    pedidos = sys.argv[1:] or ["fashion", "moby", "glove"]
    for p in pedidos:
        fashion() if p == "fashion" else baixar(p)
