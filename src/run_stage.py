"""Ejecuta una etapa con el código de mys2_exp (mismo que usa el notebook).
Uso: python run_stage.py {grids|final|individual|sel|pca|umap} [--force] [--cache DIR]
(por defecto los resultados se guardan en results/cache; con --force se recalcula la etapa pedida)"""
import sys
import mys2_data as D, mys2_exp as E, mys2_paths as P

args = sys.argv[1:]
stage = args[0]
force = "--force" in args
cache = args[args.index("--cache") + 1] if "--cache" in args else str(P.CACHE)
d = D.load()
grids = E.run_grids(d, cache, force=(force and stage == "grids"))
if stage == "grids":
    sys.exit()
if stage == "individual":
    E.run_individual(d, cache, force=force); sys.exit()
s3 = E.run_final(d, grids, cache, force=(force and stage == "final"))
if stage == "final":
    print("Ranking:", s3["ranking"]); sys.exit()
E.run_reduction(d, stage, s3, cache, force=force)
