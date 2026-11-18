"""Carga de datos del proyecto (features_v2.joblib del repo) y metadatos.

El archivo contiene los descriptores YA extraídos (1835 variables) de:
  train (360 filas) = 192 copias aumentadas + 168 imágenes originales
  val   (36 filas)  = 36 originales
  test  (36 filas)  = 36 originales
Dentro de cada bloque de 90 filas de `train` (una clase), las primeras 48 son
copias aumentadas (aug_000..aug_047) y las 42 restantes son originales, porque
`aumentar_datos.py` nombra las copias `aug_XXX.jpg` y `load_split` ordena los
archivos por nombre. La copia aug_i proviene del original `i mod 42`.
"""
import hashlib, os, urllib.request
from pathlib import Path
import numpy as np
import joblib
import mys2_paths as P

CLASSES = ["cuarzo", "pirita", "malaquita", "hematita"]

# Familias de variables (orden de concatenación de clasificador.py::extract_features)
FAMILIES = [
    ("HOG",        0,    1764),
    ("Color",      1764, 1782),
    ("Gabor",      1782, 1798),
    ("LBP",        1798, 1808),
    ("GLCM",       1808, 1828),
    ("Hu",         1828, 1835),
]
FAM_SLICE = {n: slice(a, b) for n, a, b in FAMILIES}
N_VARS = 1835


def feature_names():
    names = []
    # HOG: 7x7 bloques x (2x2 celdas) x 9 orientaciones
    for i in range(1764):
        names.append(f"HOG_{i:04d}")
    for sp in ("HSV", "LAB"):
        for ch in (("H", "S", "V") if sp == "HSV" else ("L", "a", "b")):
            for st in ("media", "desv", "asim"):
                names.append(f"Color_{sp}-{ch}_{st}")
    for th in (0, 45, 90, 135):
        for lam in (8, 16):
            for st in ("media", "desv"):
                names.append(f"Gabor_th{th}_lam{lam}_{st}")
    for b in range(10):
        names.append(f"LBP_bin{b}")
    for prop in ("contraste", "disimilitud", "homogeneidad", "energia", "correlacion"):
        for th in (0, 45, 90, 135):
            names.append(f"GLCM_{prop}_{th}")
    for k in range(1, 8):
        names.append(f"Hu_{k}")
    assert len(names) == N_VARS
    return names


def family_of_index():
    fam = np.empty(N_VARS, dtype=object)
    for n, a, b in FAMILIES:
        fam[a:b] = n
    return fam


DATA_URL = ("https://raw.githubusercontent.com/Mista299/clasificador-minerales-pdi-udea-2026-2/main/"
            "data/dataset/features_v2.joblib")
DATA_SHA256 = "a173186b082a7ab13aea726e7303e6b8f5678eaecc4100f3b48b93a9c888e908"


def ensure_data(path=None):
    """Ruta de features_v2.joblib; si falta, lo descarga del repositorio original y verifica su SHA-256."""
    path = Path(path or P.DATA)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Descargando {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, path)
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    if h != DATA_SHA256:
        raise RuntimeError(f"{path} no coincide con el archivo del repositorio original (sha256 {h}).")
    return path


def load(path=None):
    """Devuelve dict con X, y, is_aug, group, role y nombres."""
    d = joblib.load(ensure_data(path))
    Xtr, ytr = d["train"]; Xva, yva = d["val"]; Xte, yte = d["test"]
    X = np.vstack([Xtr, Xva, Xte]).astype(np.float64)
    y = np.concatenate([ytr, yva, yte]).astype(int)
    n = len(y)
    is_aug = np.zeros(n, bool)
    group = np.zeros(n, int)
    role = np.empty(n, dtype=object)
    # train
    for r in range(360):
        c, pos = divmod(r, 90)
        if pos < 48:                       # copia aumentada aug_pos
            is_aug[r] = True
            group[r] = c * 60 + (pos % 42)  # original de origen
        else:
            group[r] = c * 60 + (pos - 48)
        role[r] = "train"
    for t in range(36):                    # val: 9 por clase
        c = t // 9
        group[360 + t] = c * 60 + 42 + (t % 9); role[360 + t] = "val"
    for t in range(36):                    # test: 9 por clase
        c = t // 9
        group[396 + t] = c * 60 + 51 + (t % 9); role[396 + t] = "test"
    return dict(X=X, y=y, is_aug=is_aug, group=group, role=role,
                names=feature_names(), family=family_of_index())
