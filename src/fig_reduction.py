"""Figura: F1 de validación frente al tamaño de la representación (selección, PCA, UMAP) para los dos mejores modelos."""
import joblib, os, numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import figstyle as F

import mys2_paths as P
OUT = str(P.CACHE) + "/"
TICKS = {"sel": [5, 10, 20, 40, 80, 200, 800], "pca": [2, 5, 12, 32, 64, 128], "umap": [2, 3, 5, 8, 16, 32]}
TITLES = {"sel": "(a) Selección por $\\eta^2$", "pca": "(b) PCA", "umap": "(c) UMAP"}
XLAB = {"sel": "variables conservadas $k$", "pca": "componentes principales $k$", "umap": "dimensiones de UMAP $k$"}


def make(S, s3):
    F.setup()
    techs = [t for t in ("sel", "pca", "umap") if t in S]
    models = S[techs[0]]["models"]
    col = {models[0]: F.BLUE, models[1]: F.ORANGE}
    mk = {models[0]: "o", models[1]: "s"}
    fig, axs = plt.subplots(1, len(techs), figsize=(F.COL2, 2.35))
    axs = np.atleast_1d(axs)
    for ax, t in zip(axs, techs):
        for m in models:
            sm = S[t]["summary"][m]
            per = [x for x in sm["per_k"] if x["k"] > 0]
            ks = np.array([x["k"] for x in per]); mu = np.array([x["f1_mean"] for x in per]); ci = np.array([x["f1_ci"] for x in per])
            ax.fill_between(ks, mu - ci, mu + ci, color=col[m], alpha=0.12, lw=0)
            ax.plot(ks, mu, color=col[m], marker=mk[m], lw=1.0)
            ax.axhline(s3[m]["val"]["f1_m"][0], color=col[m], ls=":", lw=1.0)
            k1 = sm["k_1se"]
            y1 = [x for x in per if x["k"] == k1][0]["f1_mean"]
            ax.scatter([k1], [y1], s=55, facecolors="none", edgecolors=F.INK, linewidths=0.9, zorder=6)
        ax.set_xscale("log")
        ax.set_xticks(TICKS[t]); ax.set_xticklabels([str(v) for v in TICKS[t]])
        ax.minorticks_off()
        ax.set_xlabel(XLAB[t]); ax.set_title(TITLES[t], loc="left", pad=3)
        ax.set_ylim(0.2, 1.0); ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_ylabel("F1 macro de validación")
    handles = [Line2D([], [], color=col[m], marker=mk[m], lw=1.0, label=m) for m in models]
    handles += [Line2D([], [], color=F.INK2, ls=":", lw=1.0, label="todas las variables (1835)"),
                Line2D([], [], color="none", marker="o", mfc="none", mec=F.INK, ms=6, label="$k^*$ (regla de 1 error estándar)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.085), handlelength=1.8, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0.0, 1, 1))
    return fig


if __name__ == "__main__":
    s3 = joblib.load(OUT + "stage3_final.joblib")
    S = {t: joblib.load(OUT + f"stage4_{t}.joblib") for t in ("sel", "pca", "umap") if os.path.exists(OUT + f"stage4_{t}.joblib")}
    F.save(make(S, s3), "fig_reduction")
