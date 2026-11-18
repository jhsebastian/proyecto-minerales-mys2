"""Experimentos del proyecto: mallas (etapa 1), evaluación final (etapa 3) y reducción de dimensión (etapa 4).

Lo usan tanto los scripts de línea de comandos (stage*.py) como el notebook, de modo que las cifras del informe y las
del notebook salen del mismo código y de las mismas semillas.  Todo resultado costoso se guarda en `cache_dir`
(joblib) y se reutiliza; con force=True se recalcula.
"""
import os, time, warnings, joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import mys2_core as C
from umap_np import get_umap

warnings.filterwarnings("ignore")


def _p(cache_dir, name):
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, name)


# =============================================================================================== etapa 1: mallas
def run_grids(d, cache_dir, n_jobs=2, verbose=True, force=False):
    """Evalúa la malla FINAL de cada modelo (C.GRIDS) con la CV sin fuga; reutiliza lo ya calculado
    (con force=True lo recalcula todo). Devuelve {modelo: [resultado por configuración]} en el orden de C.expand_grid."""
    path_final, path_cache = _p(cache_dir, "stage1_final.joblib"), _p(cache_dir, "grid_cache.joblib")
    cache = joblib.load(path_cache) if (os.path.exists(path_cache) and not force) else {}
    if os.path.exists(path_final) and not force:        # resultados ya ordenados de ejecuciones anteriores
        for n, res in joblib.load(path_final).items():
            cache.setdefault(n, {}).update({C._key(r["params"]): r for r in res})
    X, y = d["X"], d["y"]
    splits = C.make_splits(d)
    sd = None
    final, t0 = {}, time.time()
    for name, grid in C.GRIDS.items():
        pts = C.expand_grid(grid)
        missing = [p for p in pts if C._key(p) not in cache.get(name, {})]
        if missing and sd is None:
            sd = C.scaled_split_data(X, y, splits)
        res, n_new = C.run_grid_cached(name, pts, sd, splits, cache, n_jobs=n_jobs)
        final[name] = res
        if verbose:
            b = C.best_of(res)
            print(f"[{time.time() - t0:5.0f}s] {name:20s} {len(pts):3d} configs ({n_new:2d} nuevas)  "
                  f"mejor F1={b['f1_mean']:.3f}±{b['f1_ci']:.3f} acc={b['acc_mean']:.3f} {b['params']}", flush=True)
    joblib.dump(cache, path_cache)
    joblib.dump(final, path_final)
    return final


# =============================================================================================== etapa 2: análisis individual
def run_individual(d, cache_dir, force=False, rho_max=0.95):
    """Análisis individual de las 1835 variables con SOLO las 204 originales de desarrollo (sin test ni copias, que no son
    independientes): eta^2 (SSB/SST), F de ANOVA, q de Benjamini-Hochberg, información mutua, redundancia (|rho| >= rho_max con
    una variable más discriminante ya conservada) y candidatas a eliminar. Devuelve dict(df, tab, kept, n_pairs_hi, spearman)."""
    import pandas as pd
    from scipy import stats
    from sklearn.feature_selection import f_classif, mutual_info_classif
    import mys2_data as D
    path = _p(cache_dir, "stage2_individual.joblib")
    if os.path.exists(path) and not force:
        return joblib.load(path)
    X, y, role, aug = d["X"], d["y"], d["role"], d["is_aug"]
    dev = (role != "test") & (~aug)
    Xd, yd = X[dev], y[dev]
    n, p = Xd.shape
    k = len(np.unique(yd))
    names, fam = np.array(d["names"]), d["family"]
    F, pval = f_classif(Xd, yd)
    eta2 = F * (k - 1) / (F * (k - 1) + (n - k))                 # razón de correlación
    q = C.bh_qvalues(pval)                                         # FDR (Benjamini-Hochberg)
    mi = mutual_info_classif(Xd, yd, n_neighbors=3, random_state=0)
    Z = (Xd - Xd.mean(0)) / (Xd.std(0) + 1e-12)
    Cm = np.abs(Z.T @ Z) / n
    np.fill_diagonal(Cm, 0.0)
    kept, redundant = [], np.zeros(p, bool)
    for v in np.argsort(-eta2):
        if kept and Cm[v, kept].max() >= rho_max:
            redundant[v] = True
        else:
            kept.append(v)
    n_pairs_hi = int((np.triu(Cm, 1) >= rho_max).sum())
    nonsig = q >= 0.05
    cand = nonsig | redundant
    df = pd.DataFrame(dict(variable=names, familia=fam, eta2=eta2, F=F, p=pval, q_BH=q, MI=mi,
                           no_significativa=nonsig, redundante=redundant, candidata=cand))
    rows = []
    for fam_name, a, b in D.FAMILIES:
        m = (fam == fam_name)
        rows.append(dict(familia=fam_name, n_vars=int(m.sum()), eta2_mediana=float(np.median(eta2[m])), eta2_max=float(eta2[m].max()),
                         MI_mediana=float(np.median(mi[m])), pct_signif=float(100 * (~nonsig[m]).mean()),
                         n_no_signif=int(nonsig[m].sum()), n_redundantes=int((redundant & ~nonsig)[m].sum()),
                         n_candidatas=int(cand[m].sum()), n_conservadas=int((~cand[m]).sum())))
    out = dict(df=df, tab=pd.DataFrame(rows), kept=np.array(kept), n_pairs_hi=n_pairs_hi,
               spearman=float(stats.spearmanr(eta2, mi)[0]))
    joblib.dump(out, path)
    return out


# =============================================================================================== etapa 3: evaluación final
def _cv_preds(name, p, sdata, n_jobs=2):
    return Parallel(n_jobs=n_jobs)(delayed(C._fit_eval)(name, p, *s) for s in sdata)


def _summarize(res, spl):
    acc = np.array([r[0] for r in res]); f1 = np.array([r[1] for r in res])
    n_tr = np.mean([s["n_tr_orig"] for s in spl]); n_va = np.mean([len(s["val"]) for s in spl])
    return dict(acc=acc, f1=f1, acc_m=C.nb_ci(acc, n_tr, n_va), f1_m=C.nb_ci(f1, n_tr, n_va),
                train_acc=float(np.mean([r[4] for r in res])), train_f1=float(np.mean([r[5] for r in res])))


def run_final(d, grids, cache_dir, force=False, n_jobs=2, verbose=True):
    """Para cada modelo (mejor configuración de su malla): validación sin fuga (+ matriz de confusión fuera de muestra),
    CV 'ingenua' con fuga, ablación sin aumentación, ajuste final sobre el desarrollo y evaluación en test."""
    path = _p(cache_dir, "stage3_final.joblib")
    if os.path.exists(path) and not force:
        return joblib.load(path)
    X, y, role = d["X"], d["y"], d["role"]
    best = {n: C.best_of(r) for n, r in grids.items()}
    names = list(C.GRIDS)
    splits = C.make_splits(d); sd = C.scaled_split_data(X, y, splits)
    naive_splits = C.make_naive_splits(d); naive_sd = C.scaled_split_data(X, y, naive_splits)
    noaug_splits = C.make_splits(d, use_aug=False); noaug_sd = C.scaled_split_data(X, y, noaug_splits)
    dev = np.flatnonzero(role != "test"); te = np.flatnonzero(role == "test")
    sc = StandardScaler().fit(X[dev])
    Xd, yd, Xt, yt = sc.transform(X[dev]), y[dev], sc.transform(X[te]), y[te]
    out, t0 = {}, time.time()
    for name in names:
        p = best[name]["params"]
        r = dict(params=p)
        res = _cv_preds(name, p, sd, n_jobs)
        r["val"] = _summarize(res, splits)
        cm = np.zeros((4, 4))
        for s, rr in zip(splits, res):
            cm += confusion_matrix(y[s["val"]], rr[3], labels=range(4))
        r["cm_val"] = cm / C.N_REPEATS                      # una pasada completa sobre las 204 originales
        r["naive"] = _summarize(_cv_preds(name, p, naive_sd, n_jobs), naive_splits)
        r["noaug"] = _summarize(_cv_preds(name, p, noaug_sd, n_jobs), noaug_splits)
        clf = C.build_clf(name, p).fit(Xd, yd)
        ptr, pte = clf.predict(Xd), clf.predict(Xt)
        r["train_acc"] = accuracy_score(yd, ptr); r["train_f1"] = C.macro_f1(yd, ptr)
        k = int((pte == yt).sum()); n = len(yt)
        r["test_acc"] = k / n; r["test_acc_ci"] = C.wilson_ci(k, n)
        r["test_f1"] = C.macro_f1(yt, pte)
        r["test_f1_ci"] = tuple(C.boot_ci(yt, pte, C.macro_f1, B=2000, seed=0))
        r["cm_test"] = confusion_matrix(yt, pte, labels=range(4))
        pr, rc, f1c, _ = precision_recall_fscore_support(yt, pte, labels=range(4), zero_division=0)
        r["test_per_class"] = dict(precision=pr, recall=rc, f1=f1c)
        r["pred_test"] = pte
        out[name] = r
        if verbose:
            print(f"[{time.time() - t0:5.0f}s] {name:20s} val F1={r['val']['f1_m'][0]:.3f}±{r['val']['f1_m'][1]:.3f} "
                  f"acc={r['val']['acc_m'][0]:.3f} | train F1={r['train_f1']:.3f} | test acc={r['test_acc']:.3f} "
                  f"[{r['test_acc_ci'][0]:.2f},{r['test_acc_ci'][1]:.2f}] F1={r['test_f1']:.3f} "
                  f"[{r['test_f1_ci'][0]:.2f},{r['test_f1_ci'][1]:.2f}] | CV-ingenua F1={r['naive']['f1_m'][0]:.3f} "
                  f"| sin-aug F1={r['noaug']['f1_m'][0]:.3f}", flush=True)
    out["ranking"] = sorted(names, key=lambda n: -out[n]["val"]["f1_m"][0])
    joblib.dump(out, path)
    return out


# =============================================================================================== etapa 4: reducción
KS = dict(sel=[5, 10, 15, 20, 30, 40, 60, 80, 120, 200, 400, 800],
          pca=[2, 3, 5, 8, 12, 16, 24, 32, 48, 64, 96, 128],
          umap=[2, 3, 5, 8, 12, 16, 24, 32])
UMAP_NN, UMAP_MD = 15, 0.1


def retune(name, k):
    """Pocas configuraciones por modelo para re-ajustar en el espacio reducido (k = tamaño de la representación)."""
    if name == "Random Forest":
        mfs = ["sqrt", 0.3] + ([1.0] if k <= 100 else [])
        return [dict(n_estimators=300, max_depth=None, max_features=m) for m in mfs]
    if name == "MLP":
        return [dict(hidden_layer_sizes=(128,), alpha=a) for a in (1e-2, 1.0, 10.0)]
    if name == "Regresión logística":
        return [dict(C=c) for c in (1e-2, 1.0, 100.0)]
    if name == "SVM":
        return ([dict(kernel="rbf", C=c, gamma=g) for c in (1.0, 10.0, 100.0) for g in ("scale", 1e-2 / max(k, 1) * 5)]
                + [dict(kernel="linear", C=c) for c in (1e-3, 1.0)])
    if name == "k-NN":
        return [dict(n_neighbors=n, weights="uniform") for n in (3, 5, 9, 15)]
    raise ValueError(name)


class Reducer:
    """Ajusta la reducción SOLO con el entrenamiento y entrega representaciones de tamaño k."""

    def __init__(self, tech, ks, seed):
        self.tech, self.ks, self.seed = tech, ks, seed

    def fit(self, Xtr, ytr, aug_tr):
        t = self.tech
        if t == "pca":
            kmax = min(max(self.ks), Xtr.shape[0] - 1)
            self.pca = PCA(n_components=kmax, svd_solver="full").fit(Xtr)
            self.cumvar = np.cumsum(self.pca.explained_variance_ratio_)
        elif t == "sel":
            o = ~aug_tr                                          # estadísticos solo con imágenes originales
            rk = C.variable_ranking(Xtr[o], ytr[o])
            self.order = rk["order"]
            self.n_sig = int(((rk["q"] < 0.05) & ~rk["redundant"]).sum())
            self.rk = rk
        return self

    def views(self, Xtr, Xva):
        t = self.tech
        if t == "pca":
            Ttr, Tva = self.pca.transform(Xtr), self.pca.transform(Xva)
            for k in self.ks:
                yield k, Ttr[:, :k], Tva[:, :k], dict(var=float(self.cumvar[k - 1]))
        elif t == "sel":
            for k in self.ks:
                idx = self.order[:k]
                yield k, Xtr[:, idx], Xva[:, idx], {}
            idx = self.order[: self.n_sig]
            yield -1, Xtr[:, idx], Xva[:, idx], dict(n=self.n_sig)    # k=-1: todas las significativas no redundantes
        elif t == "umap":
            for k in self.ks:
                u = get_umap()(n_components=k, n_neighbors=UMAP_NN, min_dist=UMAP_MD, random_state=self.seed)
                Ztr = u.fit_transform(Xtr)
                yield k, Ztr, u.transform(Xva), {}


def _fold_job(j, spl, X, y, is_aug, tech, ks, models):
    tr, va = spl["train"], spl["val"]
    sc = StandardScaler().fit(X[tr])
    Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])
    red = Reducer(tech, ks, seed=C.SEED + j).fit(Xtr, y[tr], is_aug[tr])
    out = {}
    for k, Ztr, Zva, extra in red.views(Xtr, Xva):
        s2 = StandardScaler().fit(Ztr)                         # componentes/variables en escala unitaria
        Ztr, Zva = s2.transform(Ztr), s2.transform(Zva)
        for m in models:
            for ci, p in enumerate(retune(m, Ztr.shape[1])):
                a, f, tf, _, _, _ = C._fit_eval(m, p, Ztr, y[tr], Zva, y[va])
                out[(m, k, ci)] = (a, f)
                out[("params", m, k, ci)] = p
        out[("extra", k)] = extra
    return out


def run_reduction(d, tech, s3, cache_dir, models=None, force=False, n_jobs=2, verbose=True):
    path = _p(cache_dir, f"stage4_{tech}.joblib")
    if os.path.exists(path) and not force:
        return joblib.load(path)
    X, y, role, is_aug = d["X"], d["y"], d["role"], d["is_aug"]
    models = models or s3["ranking"][:2]
    ks = KS[tech]
    t0 = time.time()
    splits = C.make_splits(d)
    path_folds = _p(cache_dir, f"stage4_{tech}_folds.joblib")
    key_folds = (tuple(models), tuple(ks))
    if os.path.exists(path_folds) and not force:
        saved = joblib.load(path_folds)
        res = saved["res"] if saved["key"] == key_folds else None
    else:
        res = None
    if res is None:
        res = Parallel(n_jobs=n_jobs)(delayed(_fold_job)(j, s, X, y, is_aug, tech, ks, models) for j, s in enumerate(splits))
        joblib.dump(dict(key=key_folds, res=res), path_folds)
    n_tr = np.mean([s["n_tr_orig"] for s in splits]); n_va = np.mean([len(s["val"]) for s in splits])
    if verbose:
        print(f"[{time.time() - t0:5.0f}s] {tech}: CV hecha", flush=True)
    ks_all = ks + ([-1] if tech == "sel" else [])
    summary = {}
    for m in models:
        per_k = []
        for k in ks_all:
            cis = sorted(key[2] for key in res[0] if len(key) == 3 and key[0] == m and key[1] == k)
            best = None
            for ci in cis:
                acc = np.array([r[(m, k, ci)][0] for r in res]); f1 = np.array([r[(m, k, ci)][1] for r in res])
                fm, fh = C.nb_ci(f1, n_tr, n_va); am, ah = C.nb_ci(acc, n_tr, n_va)
                cand = dict(k=k, ci=ci, params=res[0][("params", m, k, ci)], f1=f1, acc=acc,
                            f1_mean=fm, f1_ci=fh, acc_mean=am, acc_ci=ah)
                if best is None or cand["f1_mean"] > best["f1_mean"]:
                    best = cand
            ex = [r[("extra", k)] for r in res]
            best["extra"] = {key: float(np.mean([e[key] for e in ex])) for key in ex[0]} if ex and ex[0] else {}
            per_k.append(best)
        grid = [r for r in per_k if r["k"] > 0]
        i1 = C.one_se_choice([r["k"] for r in grid], grid, key="f1")         # regla de 1 error estándar
        ib = int(np.argmax([r["f1_mean"] for r in grid]))
        summary[m] = dict(per_k=per_k, k_1se=grid[i1]["k"], k_best=grid[ib]["k"])
        if verbose:
            print(f"  {m:20s} mejor k={grid[ib]['k']:4d} F1={grid[ib]['f1_mean']:.3f}±{grid[ib]['f1_ci']:.3f} | "
                  f"k*(1-SE)={grid[i1]['k']:4d} F1={grid[i1]['f1_mean']:.3f}±{grid[i1]['f1_ci']:.3f} acc={grid[i1]['acc_mean']:.3f}", flush=True)
            for r in per_k:
                print(f"      k={r['k']:4d} F1={r['f1_mean']:.3f}±{r['f1_ci']:.3f} acc={r['acc_mean']:.3f} {r['params']} {r['extra']}")
    # ---- re-entrenamiento sobre todo el desarrollo (396 filas) y evaluación en test con k = k*
    dev = np.flatnonzero(role != "test"); te = np.flatnonzero(role == "test")
    sc = StandardScaler().fit(X[dev]); Xd, Xt = sc.transform(X[dev]), sc.transform(X[te]); yd, yt = y[dev], y[te]
    for m in models:
        S = summary[m]
        kstar = S["k_1se"]
        pbest = [r for r in S["per_k"] if r["k"] == kstar][0]["params"]
        red = Reducer(tech, [kstar], seed=C.SEED).fit(Xd, yd, is_aug[dev])
        (k, Ztr, Zte, ex), = [v for v in red.views(Xd, Xt) if v[0] == kstar]
        s2 = StandardScaler().fit(Ztr)
        Ztr, Zte = s2.transform(Ztr), s2.transform(Zte)
        clf = C.build_clf(m, pbest).fit(Ztr, yd)
        ptr, pte = clf.predict(Ztr), clf.predict(Zte)
        kk = int((pte == yt).sum()); n = len(yt)
        S["final"] = dict(k=kstar, params=pbest, train_acc=accuracy_score(yd, ptr), train_f1=C.macro_f1(yd, ptr),
                          test_acc=kk / n, test_acc_ci=C.wilson_ci(kk, n), test_f1=C.macro_f1(yt, pte),
                          test_f1_ci=tuple(C.boot_ci(yt, pte, C.macro_f1, B=2000, seed=0)), extra=ex, pred_test=pte)
        if tech == "pca":
            S["final"]["cumvar"] = float(red.cumvar[kstar - 1])
        if verbose:
            print(f"  TEST {m:20s} k*={kstar}: acc={kk / n:.3f} {tuple(round(v, 2) for v in S['final']['test_acc_ci'])} "
                  f"F1={S['final']['test_f1']:.3f} {tuple(round(v, 2) for v in S['final']['test_f1_ci'])} "
                  f"train F1={S['final']['train_f1']:.3f}", flush=True)
    out = dict(tech=tech, ks=ks, models=models, summary=summary, umap_nn=UMAP_NN, umap_md=UMAP_MD)
    joblib.dump(out, path)
    return out
