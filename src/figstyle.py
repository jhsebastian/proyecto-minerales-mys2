"""Estilo común de las figuras del informe (vectoriales, tipografía tipo Times, paleta validada).

Paleta categórica: valores documentados (azul, naranja, aguamarina, amarillo, magenta, verde, violeta, rojo) en el
orden fijo recomendado; los 4 colores de clase (azul, amarillo, magenta, verde) fueron validados con
`validate_palette.js --pairs all` sobre fondo blanco (sin FAIL; el contraste < 3:1 de amarillo/magenta se compensa
con marcadores distintos y leyenda).  Las figuras usan marcadores + color: la identidad nunca depende solo del color.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CLASS_NAMES = ["cuarzo", "pirita", "malaquita", "hematita"]
CLASS_COLOR = [BLUE, YELLOW, GREEN, MAGENTA]
CLASS_MARK = ["o", "s", "^", "D"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

COL1, COL2 = 3.5, 7.25          # anchos de columna simple y doble de IEEE (pulgadas)


def setup():
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "legend.frameon": False,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.4, "axes.axisbelow": True, "lines.linewidth": 1.0, "lines.markersize": 3.2,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42, "figure.facecolor": "white", "axes.facecolor": "white",
    })


def panel_label(ax, s, dx=-0.02, dy=1.02):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right", color=INK)


import mys2_paths as _P
OUTDIR = str(_P.FIGS)                          # informe/latex/figs (el notebook y los scripts escriben aquí)


def save(fig, name, outdir=None, close=True):
    """Guarda la figura en PDF (vectorial, para el informe) y PNG; con close=False sigue abierta (plt.show())."""
    import os
    outdir = outdir or OUTDIR
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, name + ".pdf"))
    fig.savefig(os.path.join(outdir, name + ".png"), dpi=200)
    if close:
        plt.close(fig)
