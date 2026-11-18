"""Figura: matrices de confusión del mejor modelo (validación fuera de muestra y prueba)."""
import joblib, numpy as np
import matplotlib.pyplot as plt
import figstyle as F


def make(s3, model=None):
    F.setup()
    model = model or s3["ranking"][0]
    r = s3[model]
    cms = [("(a) Validación, 204 originales", r["cm_val"]), ("(b) Prueba, 36 imágenes", r["cm_test"].astype(float))]
    fig, axs = plt.subplots(1, 2, figsize=(F.COL1, 1.95), sharey=True)
    for ax, (title, cm) in zip(axs, cms):
        P = cm / cm.sum(1, keepdims=True)
        im = ax.imshow(P, cmap=F.SEQ, vmin=0, vmax=1, aspect="equal")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center", fontsize=6.3,
                        color="white" if P[i, j] > 0.55 else F.INK)
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(F.CLASS_NAMES, rotation=40, ha="right", rotation_mode="anchor")
        ax.set_yticklabels(F.CLASS_NAMES)
        ax.set_xlabel("clase predicha")
        ax.set_title(title, fontsize=7, loc="center", pad=3)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    axs[0].set_ylabel("clase verdadera")
    cb = fig.colorbar(im, ax=axs, fraction=0.035, pad=0.02, aspect=18)
    cb.set_label("proporción de la clase", fontsize=6.5)
    cb.outline.set_visible(False); cb.ax.tick_params(length=0, labelsize=6)
    return fig


if __name__ == "__main__":
    import mys2_paths as P
    F.save(make(joblib.load(P.CACHE / "stage3_final.joblib")), "fig_confusion")
