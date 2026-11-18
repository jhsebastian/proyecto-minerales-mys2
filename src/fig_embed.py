"""Figura: proyecciones 2-D (PCA y UMAP) del conjunto de desarrollo + test, coloreadas por clase."""
import numpy as np, joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import figstyle as F
import mys2_data as D
from umap_np import get_umap



def make(d):
    F.setup()
    X, y, role, aug = d["X"], d["y"], d["role"], d["is_aug"]
    dev = role != "test"
    sc = StandardScaler().fit(X[dev]); Xs = sc.transform(X)
    pca = PCA(n_components=2).fit(Xs[dev]); P = pca.transform(Xs)
    um = get_umap()(n_components=2, n_neighbors=15, min_dist=0.1, random_state=2026)
    U = np.zeros((len(X), 2)); U[dev] = um.fit_transform(Xs[dev]); U[~dev] = um.transform(Xs[~dev])

    fig, axs = plt.subplots(1, 2, figsize=(F.COL2, 2.4), gridspec_kw=dict(wspace=0.18))
    for ax, Z, title, xl, yl in ((axs[0], P, "(a) PCA (2 componentes)",
                                  f"PC1 ({100 * pca.explained_variance_ratio_[0]:.1f} % de la varianza)",
                                  f"PC2 ({100 * pca.explained_variance_ratio_[1]:.1f} %)"),
                                 (axs[1], U, "(b) UMAP (2 dimensiones, 15 vecinos)", "UMAP 1", "UMAP 2")):
        for c in range(4):
            m_aug = dev & aug & (y == c); m_orig = dev & ~aug & (y == c); m_te = ~dev & (y == c)
            ax.scatter(Z[m_aug, 0], Z[m_aug, 1], s=9, marker=F.CLASS_MARK[c], facecolors="none", edgecolors=F.CLASS_COLOR[c],
                       linewidths=0.5, alpha=0.55, zorder=2)
            ax.scatter(Z[m_orig, 0], Z[m_orig, 1], s=10, marker=F.CLASS_MARK[c], color=F.CLASS_COLOR[c], alpha=0.9,
                       linewidths=0, zorder=3)
            ax.scatter(Z[m_te, 0], Z[m_te, 1], s=17, marker=F.CLASS_MARK[c], color=F.CLASS_COLOR[c], edgecolors=F.INK,
                       linewidths=0.7, zorder=4)
        ax.set_title(title, loc="left"); ax.set_xlabel(xl); ax.set_ylabel(yl)
    # leyenda: clases (color + marcador) y papel de cada punto
    h = [plt.Line2D([], [], marker=F.CLASS_MARK[c], ls="", color=F.CLASS_COLOR[c], ms=4, label=F.CLASS_NAMES[c]) for c in range(4)]
    h += [plt.Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=F.INK2, ms=3.5, label="copia aumentada"),
          plt.Line2D([], [], marker="o", ls="", color=F.INK2, ms=3.5, label="original (desarrollo)"),
          plt.Line2D([], [], marker="o", ls="", color=F.INK2, markeredgecolor=F.INK, ms=4.5, label="original (test)")]
    fig.legend(handles=h, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.135), handletextpad=0.3, columnspacing=1.0)
    return fig


if __name__ == '__main__':
    F.save(make(D.load()), 'fig_embed')
