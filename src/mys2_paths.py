"""Rutas del proyecto. Por defecto, ROOT es la carpeta que contiene a `src/`; se puede cambiar con la variable de
entorno MYS2_ROOT (y la carpeta de resultados costosos con MYS2_CACHE)."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("MYS2_ROOT", Path(__file__).resolve().parent.parent)).resolve()
DATA = ROOT / "data" / "features_v2.joblib"                          # 1835 descriptores ya extraídos (3 MB)
RESULTS = ROOT / "results"                                           # CSV/JSON con los resultados
CACHE = Path(os.environ.get("MYS2_CACHE", RESULTS / "cache"))        # resultados costosos (joblib) reutilizables
REPORT = ROOT / "informe" / "latex"                                  # proyecto LaTeX del informe
FIGS = REPORT / "figs"                                               # figuras del informe (PDF + PNG)
