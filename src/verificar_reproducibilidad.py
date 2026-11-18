"""Verifica la reproducibilidad de los resultados.

Recalcula DESDE CERO (en una carpeta temporal, sin tocar results/cache) y compara con lo guardado en results/cache:
  * `grids`  : mallas completas de regresión logística y k-NN, y la mejor configuración de RF, MLP y SVM (1 de 120 configs. por modelo caro);
  * `final`  : evaluación final de los cinco modelos (validación sin fuga, CV con fuga, sin aumentación, prueba);
  * `sel`, `pca`, `umap` : las reducciones de dimensión (las más lentas: ≈ 7, 5 y 7 min en 2 núcleos).

Uso (desde la raíz del repositorio):
    python src/verificar_reproducibilidad.py                 # grids + final (≈ 3 min en 2 núcleos)
    python src/verificar_reproducibilidad.py final sel pca   # lo que se indique

Termina con código 0 si todas las cifras coinciden (tolerancia 1e-6) y con código 1 si alguna difiere.
"""
import sys, tempfile, time
from pathlib import Path
import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mys2_paths as P, mys2_data as D, mys2_core as C, mys2_exp as E

TOL = 1e-6


def compare(a, b, path="", out=None):
    """Compara recursivamente dos estructuras (dict, list, tuple, ndarray, escalares); devuelve la mayor diferencia absoluta."""
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a.keys() & b.keys():
            compare(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{path}[{i}]", out)
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray) or isinstance(a, (float, np.floating)) or isinstance(b, (float, np.floating)):
        try:
            x, y = np.asarray(a, float), np.asarray(b, float)
            if x.shape == y.shape:
                out.append((path, float(np.nanmax(np.abs(x - y))) if x.size else 0.0))
        except (TypeError, ValueError):
            pass
    return out


def report(nombre, pairs):
    worst = max((d for _, d in pairs), default=0.0)
    ok = worst <= TOL
    print(f"  {nombre:28s} {len(pairs):5d} cantidades comparadas | mayor diferencia absoluta = {worst:.2e}  {'OK' if ok else 'DIFIERE'}", flush=True)
    if not ok:
        for p, d in sorted(pairs, key=lambda t: -t[1])[:5]:
            print("      ", p, f"{d:.2e}")
    return ok


def main(stages):
    d = D.load(P.DATA)
    stored = P.CACHE
    ok_all, tmp = True, Path(tempfile.mkdtemp(prefix="mys2_repro_"))
    print(f"Resultados guardados: {stored}\nRecálculo en carpeta temporal: {tmp}\n", flush=True)
    grids_stored = joblib.load(stored / "stage1_final.joblib")
    for stage in stages:
        t0 = time.time()
        print(f"[{stage}]", flush=True)
        if stage == "grids":
            splits = C.make_splits(d); sd = C.scaled_split_data(d["X"], d["y"], splits)
            pairs = []
            for name, grid in C.GRIDS.items():
                pts = C.expand_grid(grid)
                if name not in ("Regresión logística", "k-NN"):
                    pts = [C.best_of(grids_stored[name])["params"]]          # solo la mejor configuración
                new, _ = C.run_grid_cached(name, pts, sd, splits, {}, n_jobs=2)
                old = {C._key(r["params"]): r for r in grids_stored[name]}
                for p, r in zip(pts, new):
                    pairs += compare(dict(f1=r["f1"], acc=r["acc"]), dict(f1=old[C._key(p)]["f1"], acc=old[C._key(p)]["acc"]), name)
                ok_all &= report(f"mallas: {name} ({len(pts)} config.)", [x for x in pairs if x[0].startswith(name)])
        elif stage == "final":
            new = E.run_final(d, grids_stored, str(tmp), force=True, verbose=False)
            ok_all &= report("evaluación final (5 modelos)", compare(new, joblib.load(stored / "stage3_final.joblib")))
        elif stage in ("sel", "pca", "umap"):
            s3 = joblib.load(stored / "stage3_final.joblib")
            new = E.run_reduction(d, stage, s3, str(tmp), force=True, verbose=False)
            ok_all &= report(f"reducción: {stage}", compare(new["summary"], joblib.load(stored / f"stage4_{stage}.joblib")["summary"]))
        else:
            raise SystemExit(f"etapa desconocida: {stage}")
        print(f"  ({time.time() - t0:.0f} s)\n", flush=True)
    print("TODO COINCIDE" if ok_all else "HAY DIFERENCIAS", flush=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["grids", "final"]))
