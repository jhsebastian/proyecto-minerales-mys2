"""Figura: índice de discriminación individual (eta^2) de las 1835 variables, por familia de descriptores."""
import joblib, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import figstyle as F
import mys2_data as D



def make(df):
    F.setup()
    fams = [f for f, _, _ in D.FAMILIES]
    rng = np.random.default_rng(0)
    labels = []

    fig, ax = plt.subplots(figsize=(F.COL1, 2.35))
    for i, f in enumerate(fams):
        sub = df[df.familia == f]
        x = i + rng.uniform(-0.28, 0.28, len(sub))
        sig = ~sub.no_significativa.values
        ax.scatter(x[~sig], sub.eta2.values[~sig], s=2.2, color=F.MUTED, alpha=0.55, linewidths=0, zorder=2)
        ax.scatter(x[sig], sub.eta2.values[sig], s=2.6 if f == "HOG" else 7, color=F.BLUE, alpha=0.55 if f == "HOG" else 0.9,
                   linewidths=0, zorder=3)
        med = sub.eta2.median()
        ax.plot([i - 0.36, i + 0.36], [med, med], color=F.INK, lw=1.1, zorder=4, solid_capstyle="butt")
        labels.append(f"{f}\n$n$={len(sub)}\n{100 * sig.mean():.0f} %")
    ax.set_xticks(range(len(fams))); ax.set_xticklabels(labels)
    ax.set_xlim(-0.6, len(fams) - 0.4); ax.set_ylim(0, 0.56)
    ax.set_ylabel(r"Índice de discriminación $\eta^2$")
    ax.set_xlabel("Familia de descriptores ($n$ variables y % significativas)")
    ax.grid(axis="x", visible=False)
    # leyenda manual (identidad por color + etiqueta)
    ax.scatter([], [], s=8, color=F.BLUE, label=r"significativa (BH, $q<0.05$)")
    ax.scatter([], [], s=8, color=F.MUTED, label="no significativa")
    ax.plot([], [], color=F.INK, lw=1.1, label="mediana de la familia")
    ax.legend(loc="upper right", handletextpad=0.4, borderaxespad=0.1)
    return fig


if __name__ == '__main__':
    import mys2_paths as P
    F.save(make(joblib.load(P.CACHE / 'stage2_individual.joblib')['df']), 'fig_individual')
