"""Núcleo experimental: validación cruzada sin fuga, IC y mallas de hiperparámetros."""
import itertools, time, warnings
import numpy as np
from joblib import Parallel, delayed
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

SEED = 2026
N_SPLITS = 5
N_REPEATS = 3


# ---------------------------------------------------------------- particiones
def make_splits(d, n_splits=N_SPLITS, n_repeats=N_REPEATS, seed=SEED, use_aug=True):
    """K-fold estratificado repetido sobre las imágenes ORIGINALES de desarrollo.

    - desarrollo = originales de train + val (204 imágenes); el test (36) no se toca.
    - Las copias aumentadas se agregan SOLO al entrenamiento de cada fold, y
      únicamente las de originales que están en ese entrenamiento (sin fuga).
    - La validación de cada fold contiene solo imágenes originales.
    """
    role, is_aug, grp, y = d["role"], d["is_aug"], d["group"], d["y"]
    dev_orig = np.flatnonzero((role != "test") & (~is_aug))
    aug_idx = np.flatnonzero(is_aug)
    splits = []
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed + r)
        for tr, va in skf.split(dev_orig, y[dev_orig]):
            tr_orig, va_orig = dev_orig[tr], dev_orig[va]
            if use_aug:
                aug_tr = aug_idx[np.isin(grp[aug_idx], grp[tr_orig])]
                train_rows = np.concatenate([tr_orig, aug_tr])
            else:
                train_rows = tr_orig
            splits.append(dict(train=train_rows, val=va_orig, n_tr_orig=len(tr_orig), rep=r))
    return splits


def make_naive_splits(d, n_splits=N_SPLITS, n_repeats=N_REPEATS, seed=SEED):
    """CV 'ingenuo' (con fuga): K-fold estratificado sobre las 360 filas de train
    (originales + copias aumentadas mezcladas), como en el trabajo previo."""
    role, y = d["role"], d["y"]
    rows = np.flatnonzero(role == "train")
    out = []
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed + r)
        for tr, va in skf.split(rows, y[rows]):
            out.append(dict(train=rows[tr], val=rows[va], n_tr_orig=len(tr), rep=r))
    return out


# ---------------------------------------------------------------- intervalos de confianza
def nb_ci(scores, n_tr, n_va, alpha=0.05):
    """IC (1-alpha) de la media de J puntajes de CV repetida, con la corrección de
    varianza de Nadeau & Bengio (2003): Var = (1/J + n_va/n_tr) * s^2 ; t con J-1 g.l."""
    s = np.asarray(scores, float)
    J = len(s)
    m = s.mean()
    var = s.var(ddof=1) * (1.0 / J + float(n_va) / float(n_tr))
    half = stats.t.ppf(1 - alpha / 2, J - 1) * np.sqrt(var)
    return m, half


def wilson_ci(k, n, alpha=0.05):
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def boot_ci(y_true, y_pred, metric, B=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        vals[b] = metric(y_true[i], y_pred[i])
    return np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def macro_f1(a, b):
    return f1_score(a, b, average="macro", zero_division=0)


# ---------------------------------------------------------------- modelos y mallas
GRIDS = {
    "Regresión logística": dict(C=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]),
    "k-NN": dict(n_neighbors=[1, 3, 5, 7, 9, 11, 15, 21], weights=["uniform", "distance"]),
    "Random Forest": [dict(n_estimators=[100, 300], max_depth=[3, 5, 10, None],
                           max_features=["log2", "sqrt", 0.1, 0.3]),
                      dict(n_estimators=[100], max_depth=[3, 5, 10, None], max_features=[0.6, 1.0])],
    "MLP": dict(hidden_layer_sizes=[(32,), (64,), (128,), (64, 32)],
                alpha=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]),
    "SVM": [dict(kernel=["rbf"], C=[1.0, 10.0, 100.0, 1000.0],
                 gamma=[1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1e-2]),
            dict(kernel=["linear"], C=[1e-4, 1e-3, 1e-2, 1e-1, 1.0])],
}


def _key(params):
    import json
    return json.dumps(params, sort_keys=True, default=str)


def run_grid_cached(name, pts, split_data, splits, cache, n_jobs=2):
    """Evalúa solo los puntos de la malla que no están en `cache` (name -> {key: resultado})."""
    store = cache.setdefault(name, {})
    todo = [p for p in pts if _key(p) not in store]
    if todo:
        for p, r in zip(todo, run_grid(name, todo, split_data, splits, n_jobs=n_jobs)):
            store[_key(p)] = r
    return [store[_key(p)] for p in pts], len(todo)


def expand_grid(grid):
    if isinstance(grid, dict):
        grid = [grid]
    out = []
    for g in grid:
        keys = list(g)
        for vals in itertools.product(*[g[k] for k in keys]):
            out.append(dict(zip(keys, vals)))
    return out


def build_clf(name, p, seed=SEED):
    if name == "Regresión logística":
        return LogisticRegression(C=p["C"], max_iter=3000, random_state=seed)
    if name == "k-NN":
        return KNeighborsClassifier(n_neighbors=p["n_neighbors"], weights=p["weights"])
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                                      max_features=p["max_features"], random_state=seed, n_jobs=1)
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=p["hidden_layer_sizes"], alpha=p["alpha"],
                             solver="adam", learning_rate_init=1e-3, max_iter=500,
                             random_state=seed)
    if name == "SVM":
        return SVC(kernel=p["kernel"], C=p["C"], gamma=p.get("gamma", "scale"), random_state=seed)
    raise ValueError(name)


def _fit_eval(name, p, Xtr, ytr, Xva, yva):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.time()
        clf = build_clf(name, p).fit(Xtr, ytr)
        t_fit = time.time() - t0
        pred = clf.predict(Xva)
        ptr = clf.predict(Xtr)
    return (accuracy_score(yva, pred), macro_f1(yva, pred), t_fit, pred,
            accuracy_score(ytr, ptr), macro_f1(ytr, ptr))


def scaled_split_data(X, y, splits):
    """Estandariza con media/desv. del ENTRENAMIENTO de cada fold (sin fuga)."""
    out = []
    for s in splits:
        sc = StandardScaler().fit(X[s["train"]])
        out.append((sc.transform(X[s["train"]]), y[s["train"]], sc.transform(X[s["val"]]), y[s["val"]]))
    return out


def run_grid(name, grid_points, split_data, splits, n_jobs=2):
    """Evalúa todos los puntos de la malla en todos los splits. Devuelve lista de dicts."""
    tasks = [(i, j) for i in range(len(grid_points)) for j in range(len(split_data))]
    res = Parallel(n_jobs=n_jobs)(
        delayed(_fit_eval)(name, grid_points[i], *split_data[j]) for i, j in tasks)
    n_s = len(split_data)
    out = []
    for i, p in enumerate(grid_points):
        acc = np.array([res[i * n_s + j][0] for j in range(n_s)])
        f1 = np.array([res[i * n_s + j][1] for j in range(n_s)])
        tfit = np.array([res[i * n_s + j][2] for j in range(n_s)])
        tracc = np.array([res[i * n_s + j][4] for j in range(n_s)])
        trf1 = np.array([res[i * n_s + j][5] for j in range(n_s)])
        n_tr = np.mean([s["n_tr_orig"] for s in splits])
        n_va = np.mean([len(s["val"]) for s in splits])
        a_m, a_h = nb_ci(acc, n_tr, n_va)
        f_m, f_h = nb_ci(f1, n_tr, n_va)
        out.append(dict(params=p, acc=acc, f1=f1, acc_mean=a_m, acc_ci=a_h, f1_mean=f_m, f1_ci=f_h,
                        fit_time=float(tfit.mean()), train_acc=float(tracc.mean()),
                        train_f1=float(trf1.mean())))
    return out


def best_of(results):
    """Mejor configuración: mayor F1-macro medio de validación (desempata por exactitud)."""
    return max(results, key=lambda r: (round(r["f1_mean"], 12), r["acc_mean"]))


def one_se_choice(ks, results, key="f1"):
    """Regla de 1 error estándar: el menor k cuyo F1 medio de validación quede dentro de
    un error estándar (half-IC / t) del mejor. Devuelve el índice elegido."""
    means = np.array([r[key + "_mean"] for r in results])
    halves = np.array([r[key + "_ci"] for r in results])
    ib = int(np.argmax(means))
    J = len(results[ib][key])
    se = halves[ib] / stats.t.ppf(0.975, J - 1)
    ok = np.flatnonzero(means >= means[ib] - se)
    return int(min(ok, key=lambda i: ks[i]))


# ---------------------------------------------------------------- análisis individual de variables
def bh_qvalues(p):
    """Valores q de Benjamini-Hochberg (control de la tasa de falsos descubrimientos)."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(m)
    q[order] = np.minimum(q_sorted, 1.0)
    return q


def variable_ranking(Xs, y, rho_max=0.95):
    """Índice de discriminación por variable (razón de correlación eta^2 = SSB/SST, ANOVA),
    valor q (BH) y marca de redundancia (|rho| >= rho_max con una variable MÁS discriminante ya
    conservada, recorrido voraz por eta^2 decreciente). Devuelve dict con eta2, q, redundant, order."""
    from sklearn.feature_selection import f_classif
    n, p = Xs.shape
    k = len(np.unique(y))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        F, pv = f_classif(Xs, y)
    F, pv = np.nan_to_num(F, nan=0.0), np.nan_to_num(pv, nan=1.0)
    eta2 = F * (k - 1) / (F * (k - 1) + (n - k))
    q = bh_qvalues(pv)
    Z = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-12)
    Cm = np.abs(Z.T @ Z) / n
    np.fill_diagonal(Cm, 0.0)
    kept, redundant = [], np.zeros(p, bool)
    order_eta = np.argsort(-eta2)
    for v in order_eta:
        if kept and Cm[v, kept].max() >= rho_max:
            redundant[v] = True
        else:
            kept.append(v)
    order = np.array([v for v in order_eta if not redundant[v]])      # no redundantes, por eta^2
    return dict(eta2=eta2, F=F, p=pv, q=q, redundant=redundant, order=order)
