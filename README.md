# Clasificación automática de minerales con descriptores de imagen

**Validación sin fuga de información y reducción de dimensión**

Proyecto de aula — *Modelos y Simulación de Sistemas II* (Universidad de Antioquia, 2026-2), aprendizaje supervisado.

Integrantes: Michael Stiven Tabares Tobón, María Fernanda Atencia Oliva, Valeria Vásquez Barragán y Jhon Sebastián Martínez Soto.

Este repositorio contiene el cuaderno de Jupyter que reproduce **todas** las cifras, tablas y figuras de los resultados, el código que usa el cuaderno y, como parte del **Entregable I**, el informe en formato IEEE Transactions (PDF y código LaTeX).

| Contenido | Archivo |
|---|---|
| Cuaderno que reproduce los resultados | [`notebooks/proyecto_minerales.ipynb`](notebooks/proyecto_minerales.ipynb) |
| Informe del Entregable I (PDF, plantilla IEEE Transactions, 8 páginas) | [`informe/informe_ieee.pdf`](informe/informe_ieee.pdf) |
| Fuente LaTeX del informe (Overleaf) | [`informe/latex/`](informe/latex) |

## Informe (Entregable I)

El informe en formato IEEE Transactions hizo parte del **Entregable I** y se conserva en este repositorio como referencia: el PDF está en [`informe/informe_ieee.pdf`](informe/informe_ieee.pdf) y su fuente LaTeX en [`informe/latex/`](informe/latex). El cuaderno reproduce todas sus cifras, tablas y figuras (la correspondencia sección por sección está en «Correspondencia con la guía del proyecto»), y el propio informe remite a este repositorio en su apartado de disponibilidad.

## El problema en una línea

Clasificar fotografías de cuatro minerales (cuarzo, pirita, malaquita y hematita) a partir de **1835 descriptores clásicos de imagen** (HOG, color, Gabor, LBP, GLCM y momentos de Hu) calculados sobre 432 imágenes (240 fotografías originales y 192 copias aumentadas). Se compararon cinco modelos con una validación cruzada repetida **sin fuga de información** y se estudió cuánto se puede reducir la dimensión con selección de variables, PCA y UMAP.

## Resultados principales

F1 macro de validación: media de 15 particiones (5 pliegues × 3 repeticiones) sobre 204 imágenes originales, con IC del 95 % de Nadeau–Bengio. La prueba (36 imágenes, usadas una sola vez) se resume con la exactitud y su IC de Wilson.

| Modelo | F1 de validación | Exactitud en prueba [IC 95 %] |
|---|---|---|
| **Random Forest** | **0.832 ± 0.063** | **0.778 [0.62, 0.88]** |
| MLP | 0.622 ± 0.106 | 0.667 [0.50, 0.80] |
| Regresión logística | 0.604 ± 0.104 | 0.639 [0.48, 0.78] |
| SVM | 0.586 ± 0.108 | 0.528 [0.37, 0.68] |
| k-NN | 0.512 ± 0.074 | 0.472 [0.32, 0.63] |

Reducción de dimensión (misma validación; el tamaño *k\** se elige con la regla del error estándar):

| Representación | Random Forest: *k\** (reducción) → F1 de validación | MLP: *k\** (reducción) → F1 de validación |
|---|---|---|
| Las 1835 variables | 1835 → 0.832 | 1835 → 0.622 |
| Selección por η² (supervisada) | 30 (98.4 %) → 0.833 | 10 (99.5 %) → 0.782 |
| PCA | 32 (98.3 %) → 0.523 | 32 (98.3 %) → 0.552 |
| UMAP | 8 (99.6 %) → 0.468 | 3 (99.8 %) → 0.467 |

Hallazgos: (1) las variables más discriminantes están en las 71 que no son HOG (3.9 % del total); (2) una selección supervisada de ~30 variables iguala al Random Forest con las 1835 variables y mejora al MLP, mientras que PCA y UMAP (no supervisados) degradan el desempeño, porque la varianza y la vecindad euclídea están dominadas por las 1764 variables HOG; (3) la validación con fuga (copias aumentadas repartidas entre entrenamiento y validación) sobreestima el F1 hasta 0.16 puntos; (4) la prueba tiene solo 36 imágenes, por lo que sus intervalos son anchos (≈ ±0.13).

## Estructura del repositorio

```
.
├── README.md
├── requirements.txt
├── data/
│   └── features_v2.joblib          # 1835 descriptores ya extraídos (copia del repositorio original)
├── notebooks/
│   └── proyecto_minerales.ipynb    # cuaderno con TODAS las secciones del informe, ejecutado
├── src/
│   ├── mys2_paths.py               # rutas del proyecto
│   ├── mys2_data.py                # carga de datos, familias de variables, verificación del archivo (SHA-256)
│   ├── mys2_core.py                # particiones sin fuga, modelos, mallas, IC (Nadeau-Bengio, Wilson, bootstrap)
│   ├── mys2_exp.py                 # experimentos: mallas, evaluación final, análisis individual, reducción
│   ├── umap_np.py                  # implementación propia de UMAP en NumPy (ver «Notas»)
│   ├── make_report_assets.py       # genera cifras (numeros.tex), tablas LaTeX y CSV desde results/cache
│   ├── fig_*.py, figstyle.py       # figuras del informe
│   ├── run_stage.py                # ejecuta una etapa desde la línea de comandos
│   └── verificar_reproducibilidad.py  # recalcula desde cero y compara con lo guardado
├── results/
│   ├── cache/                      # resultados costosos ya calculados (joblib)
│   ├── resultados_modelos.csv      # tabla de resultados de los cinco modelos
│   ├── variables_individual.csv    # análisis individual de las 1835 variables (η², q, información mutua, redundancia)
│   ├── reduccion_curvas.csv        # F1 de validación por técnica, modelo y k
│   └── cifras_informe.json         # todas las cifras citadas en el informe
└── informe/
    ├── informe_ieee.pdf
    └── latex/                      # informe.tex, numeros.tex, integrantes.tex, tablas/, figs/
```

## Correspondencia con la guía del proyecto

| Requisito de la guía | Informe (PDF) | Cuaderno |
|---|---|---|
| Descripción del problema: contexto y utilidad, composición de la base (n, variables, significado, faltantes, imputación, codificación), paradigma y justificación, análisis exploratorio | Sec. I | 1.A, 1.B, 1.C |
| Estado del arte (≥ 4 artículos, ≤ 1 página, definiciones de métricas) | Sec. II | 2 |
| Metodología de validación (incl. sub/sobremuestreo), tabla de hiperparámetros y mallas, ≥ 5 modelos, métricas y justificación | Sec. III | 3, 4 (tabla de mallas) |
| Entrenamiento y evaluación: efecto de los hiperparámetros, resultados en entrenamiento/validación/prueba con IC | Sec. IV | 4.1, 4.2 |
| Reducción de dimensión 2.1: análisis individual y candidatas a eliminar | Sec. V-A | 5.1 |
| Reducción 2.2: PCA con criterio propio y re-evaluación de los dos mejores modelos | Sec. V-B, V-C | 5.2, 5.3, 5.4 |
| Reducción 2.3: UMAP y discusión comparativa | Sec. V-D, V-E | 5.5, 5.6 |
| Conclusiones | Sec. VI | 6 |
| Reproducibilidad | — | 7 |

## Cómo reproducir

Los paquetes están en `requirements.txt`. Los resultados se generaron con Python 3.13, numpy 2.5, scipy 1.18, pandas 3.0, scikit-learn 1.9 y matplotlib 3.11; con otras versiones pueden aparecer diferencias numéricas mínimas al recalcular, pero los resultados guardados en `results/cache/` permiten ejecutar el cuaderno sin recalcular.

```bash
python -m venv .venv && source .venv/bin/activate      # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab notebooks/proyecto_minerales.ipynb          # o: jupyter notebook ...
```

**Ejecutar el cuaderno completo** («Run all») toma unos 20–30 s: por defecto (`FORCE = False`, primera celda de código) reutiliza los resultados costosos guardados en `results/cache/` y vuelve a generar todas las tablas y figuras, así como `informe/latex/numeros.tex` y `informe/latex/tablas/*.tex`. Con `FORCE = True` recalcula todo desde cero con las mismas semillas (del orden de 35 min con 2 núcleos).

**Etapas por línea de comandos** (con `--force` se recalcula la etapa pedida):

```bash
python src/run_stage.py grids      # 120 configuraciones × 15 particiones
python src/run_stage.py final      # evaluación final de los cinco modelos
python src/run_stage.py individual # análisis individual de las variables
python src/run_stage.py sel        # selección por η²   (también: pca, umap)
python src/verificar_reproducibilidad.py grids final sel pca umap   # recalcula en una carpeta temporal y compara con lo guardado
```

**Verificación de reproducibilidad.** `verificar_reproducibilidad.py` recalcula desde cero las mallas de la regresión logística y de k-NN, la mejor configuración de los otros tres modelos, la evaluación final de los cinco modelos y las tres reducciones, y las compara con `results/cache/`. En la ejecución con la que se entregó el proyecto todas las cantidades comparadas coincidieron exactamente (diferencia máxima 0.0). Evaluación final ≈ 2 min; selección, PCA y UMAP ≈ 7, 5 y 7 min en 2 núcleos.

**Compilar el informe.** En Overleaf: *New project → Upload project* con la carpeta `informe/latex/` comprimida; documento principal `informe.tex`; compilador pdfLaTeX (usa la clase `IEEEtran`, incluida en TeX Live). Localmente: `cd informe/latex && pdflatex informe.tex && pdflatex informe.tex`. Los autores y el enlace del repositorio se editan en `informe/latex/integrantes.tex`. Las cifras y tablas del texto no se escriben a mano: `numeros.tex` y `tablas/*.tex` los genera `src/make_report_assets.py` (o la última celda del cuaderno) a partir de los resultados.

## Datos y créditos

* Los 1835 descriptores (`data/features_v2.joblib`, SHA-256 `a173186b…e908`) provienen del trabajo previo de Procesamiento Digital de Imágenes del mismo equipo: [`Mista299/clasificador-minerales-pdi-udea-2026-2`](https://github.com/Mista299/clasificador-minerales-pdi-udea-2026-2). Si falta el archivo, `mys2_data.ensure_data()` lo descarga de ese repositorio y verifica su huella.
* Las fotografías originales provienen del banco `tedqc/mineral-dataset` de Hugging Face; las condiciones de uso son las de esa página.
* Este proyecto parte de los mismos descriptores, pero cambia el protocolo de evaluación: el repositorio original estima la validación cruzada con copias aumentadas repartidas entre entrenamiento y validación (fuga de información), lo que da exactitudes cercanas a 0.92 que no se sostienen en la prueba (≈ 0.78).

## Notas y limitaciones

* **UMAP propio.** `umap-learn` requiere compilar `numba` y no estaba disponible en el entorno de ejecución; se escribió una implementación propia en NumPy (`src/umap_np.py`) y se verificó con el conjunto *digits* de scikit-learn (5-NN en el plano 2-D: 0.98 con UMAP frente a 0.61 con PCA). Con `MYS2_UMAP=learn` se usa `umap-learn` si está instalado; los resultados guardados corresponden a la implementación propia.
* **Prueba pequeña.** 36 imágenes de la misma fuente que el entrenamiento; las decisiones se apoyan en la validación (204 imágenes × 15 particiones) y no se midió el cambio de dominio.
* **Optimismo de la selección.** Hiperparámetros y *k\** se eligen con la misma validación con la que se reportan; el efecto es similar para las tres técnicas de reducción.
