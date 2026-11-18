"""Figura: efecto de los hiperparámetros sobre el F1 macro de validación (CV sin fuga, IC95 % Nadeau-Bengio)."""
import joblib, numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import figstyle as F
import mys2_core as C



def make(R):
    F.setup()
    YL = (0.30, 1.04)
    SER = [F.BLUE, F.ORANGE, F.AQUA, F.YELLOW]          # orden fijo de la paleta (pares adyacentes validados)
    fig, axs = plt.subplots(2, 3, figsize=(F.COL2, 3.4), gridspec_kw=dict(hspace=0.62, wspace=0.34))


    def band(ax, x, res, color, label, ls="-", marker="o", lw=1.0, ms=2.6, alpha=0.13):
        m = np.array([r["f1_mean"] for r in res]); h = np.array([r["f1_ci"] for r in res])
        ax.fill_between(x, m - h, m + h, color=color, alpha=alpha, lw=0)
        ax.plot(x, m, color=color, ls=ls, marker=marker, lw=lw, ms=ms, label=label)


    def train_line(ax, x, res):
        ax.plot(x, [r["train_f1"] for r in res], color=F.MUTED, lw=0.7, ls=(0, (1.5, 1.5)), label="entrenamiento")


    # ---- (a) Regresión logística
    ax = axs[0, 0]; res = R["Regresión logística"]; x = np.array([r["params"]["C"] for r in res])
    band(ax, x, res, F.BLUE, "validación"); train_line(ax, x, res)
    ax.set_xscale("log"); ax.set_xlabel("$C$ (inverso de la regularización)"); ax.set_title("(a) Regresión logística", loc="left")
    # ---- (b) k-NN
    ax = axs[0, 1]
    for w, col, mk in (("uniform", F.BLUE, "o"), ("distance", F.ORANGE, "s")):
        rr = [r for r in R["k-NN"] if r["params"]["weights"] == w]; x = [r["params"]["n_neighbors"] for r in rr]
        band(ax, x, rr, col, "validación, pesos " + ("uniformes" if w == "uniform" else "por distancia"), marker=mk)
    ax.set_xlabel("vecinos $k$"); ax.set_title("(b) $k$-NN", loc="left")
    # ---- (c) Random Forest: x = variables por división; una línea por profundidad (100 árboles)
    ax = axs[0, 2]
    mfs = ["log2", "sqrt", 0.1, 0.3, 0.6, 1.0]; mf_lab = [r"$\log_2p$", r"$\sqrt{p}$", "0.1", "0.3", "0.6", "1"]
    for d_, col, mk in zip([3, 5, 10, None], SER, ["o", "s", "^", "D"]):
        xs, rr = [], []
        for j, mf in enumerate(mfs):
            hit = [r for r in R["Random Forest"] if r["params"]["n_estimators"] == 100 and r["params"]["max_depth"] == d_
                   and r["params"]["max_features"] == mf]
            if hit: xs.append(j); rr.append(hit[0])
        band(ax, xs, rr, col, "profundidad " + ("sin límite" if d_ is None else str(d_)), marker=mk)
    ax.set_xticks(range(len(mfs))); ax.set_xticklabels(mf_lab); ax.set_xlabel("variables por división (fracción de $p$)")
    ax.set_title("(c) Random Forest (100 árboles)", loc="left")
    # ---- (d) MLP
    ax = axs[1, 0]
    for arch, col, mk in zip([(32,), (64,), (128,), (64, 32)], SER, ["o", "s", "^", "D"]):
        rr = [r for r in R["MLP"] if tuple(r["params"]["hidden_layer_sizes"]) == arch]; rr.sort(key=lambda r: r["params"]["alpha"])
        x = [r["params"]["alpha"] for r in rr]
        band(ax, x, rr, col, "capas " + str(arch).replace(",)", ")").replace(" ", ""), marker=mk, alpha=0.10)
    ax.set_xscale("log"); ax.set_xlabel(r"$\alpha$ (penalización L2)"); ax.set_title("(d) Red neuronal (MLP)", loc="left")
    # ---- (e) SVM RBF: mapa de calor gamma x C
    ax = axs[1, 1]
    gam = sorted({r["params"]["gamma"] for r in R["SVM"] if r["params"]["kernel"] == "rbf"})
    Cs = sorted({r["params"]["C"] for r in R["SVM"] if r["params"]["kernel"] == "rbf"})
    M = np.full((len(Cs), len(gam)), np.nan)
    for r in R["SVM"]:
        if r["params"]["kernel"] == "rbf":
            M[Cs.index(r["params"]["C"]), gam.index(r["params"]["gamma"])] = r["f1_mean"]
    im = ax.imshow(M, origin="lower", cmap=F.SEQ, vmin=0.0, vmax=0.8, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.6, color="white" if M[i, j] > 0.45 else F.INK)
    ax.set_xticks(range(len(gam))); ax.set_xticklabels([("$10^{%d}$" % int(np.round(np.log10(g)))) if abs(np.log10(g) - np.round(np.log10(g))) < 1e-9
                                                        else ("$3\\cdot10^{%d}$" % int(np.floor(np.log10(g)))) for g in gam], fontsize=5.8)
    ax.set_yticks(range(len(Cs))); ax.set_yticklabels([str(int(c)) for c in Cs])
    ax.set_xlabel(r"$\gamma$ (núcleo RBF)"); ax.set_ylabel("$C$"); ax.set_title("(e) SVM RBF: F1 de validación", loc="left")
    ax.grid(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02); cb.ax.tick_params(labelsize=5.8, width=0.4, length=2)
    cb.outline.set_visible(False)
    # ---- (f) SVM lineal
    ax = axs[1, 2]; rr = sorted([r for r in R["SVM"] if r["params"]["kernel"] == "linear"], key=lambda r: r["params"]["C"])
    x = np.array([r["params"]["C"] for r in rr]); band(ax, x, rr, F.BLUE, "validación"); train_line(ax, x, rr)
    ax.set_xscale("log"); ax.set_xlabel("$C$"); ax.set_title("(f) SVM lineal", loc="left")

    for ax in (axs[0, 0], axs[0, 1], axs[0, 2], axs[1, 0], axs[1, 2]):
        ax.set_ylim(*YL); ax.set_ylabel("F1 macro")
        ax.set_yticks([0.4, 0.6, 0.8, 1.0])
        ax.legend(loc="lower right" if ax is axs[0, 2] else "upper right", handlelength=1.6, borderaxespad=0.2, labelspacing=0.25)
    return fig


if __name__ == '__main__':
    import mys2_paths as P
    F.save(make(joblib.load(P.CACHE / 'stage1_final.joblib')), 'fig_hyper')
