"""UMAP en NumPy/SciPy/scikit-learn (sin numba).

Implementación del algoritmo de McInnes, Healy y Melville (2018),
"UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction",
siguiendo los pasos y los valores por defecto de la librería de referencia
`umap-learn` (n_neighbors=15, min_dist=0.1, spread=1.0, local_connectivity=1,
negative_sample_rate=5, init espectral, métrica euclidiana):

  1. Grafo de k vecinos más cercanos con distancias a escala local
     (rho_i = distancia al vecino más cercano, sigma_i por bisección tal que
     sum_j exp(-(d_ij - rho_i)/sigma_i) = log2(k)).
  2. Conjunto simplicial difuso simétrico (unión difusa: A + A^T - A∘A^T).
  3. Inicialización espectral (autovectores del Laplaciano normalizado).
  4. Optimización por descenso de gradiente estocástico de la entropía cruzada
     difusa con muestreo negativo. DIFERENCIA con la referencia: las
     actualizaciones se hacen en mini-lotes aleatorios vectorizados (la
     referencia actualiza arista por arista con numba).
  5. `transform` para puntos nuevos: inicialización por promedio ponderado de
     los vecinos de entrenamiento y refinamiento con el embedding de
     entrenamiento fijo (n_epochs // 3 épocas, tasa inicial / 4).

Esta clase existe porque `umap-learn` (y numba) no estaban disponibles en el
entorno donde se generaron los resultados. Para usar la librería oficial: `pip install umap-learn` y definir la variable de entorno
MYS2_UMAP=learn (ver `get_umap`); la API usada (fit_transform / transform) es la misma.
"""
import numpy as np
from scipy import sparse
from scipy.optimize import curve_fit
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3


def find_ab_params(spread=1.0, min_dist=0.1):
    """Ajusta a, b de 1/(1 + a x^(2b)) a la función de pertenencia objetivo."""
    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))
    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, _ = curve_fit(curve, xv, yv)
    return float(params[0]), float(params[1])


def smooth_knn_dist(distances, k, local_connectivity=1.0, n_iter=64, bandwidth=1.0):
    """rho y sigma por punto. `distances` (n, k) ascendentes; la columna 0 se omite en la suma
    (es el propio punto en el ajuste), igual que en umap-learn."""
    target = np.log2(k) * bandwidth
    n = distances.shape[0]
    rho = np.zeros(n)
    for i in range(n):
        nzd = distances[i][distances[i] > 0.0]
        if nzd.shape[0] >= local_connectivity:
            idx = int(np.floor(local_connectivity))
            interp = local_connectivity - idx
            if idx > 0:
                rho[i] = nzd[idx - 1]
                if interp > SMOOTH_K_TOLERANCE:
                    rho[i] += interp * (nzd[idx] - nzd[idx - 1])
            else:
                rho[i] = interp * nzd[0]
        elif nzd.shape[0] > 0:
            rho[i] = nzd.max()
    lo = np.zeros(n)
    hi = np.full(n, np.inf)
    mid = np.ones(n)
    for _ in range(n_iter):
        d = distances[:, 1:] - rho[:, None]
        psum = np.where(d > 0, np.exp(-d / mid[:, None]), 1.0).sum(1)
        ok = np.abs(psum - target) < SMOOTH_K_TOLERANCE
        gt = psum > target
        hi = np.where(~ok & gt, mid, hi)
        lo = np.where(~ok & ~gt, mid, lo)
        new_mid = np.where(gt, (lo + hi) / 2.0,
                           np.where(np.isinf(hi), mid * 2.0, (lo + hi) / 2.0))
        mid = np.where(ok, mid, new_mid)
    sigma = mid.copy()
    mean_i = distances.mean(1)
    mean_all = distances.mean()
    pos = rho > 0.0
    sigma = np.where(pos & (sigma < MIN_K_DIST_SCALE * mean_i), MIN_K_DIST_SCALE * mean_i, sigma)
    sigma = np.where(~pos & (sigma < MIN_K_DIST_SCALE * mean_all), MIN_K_DIST_SCALE * mean_all, sigma)
    return sigma, rho


def membership_strengths(knn_idx, knn_dist, sigma, rho, exclude_self=True):
    n, k = knn_idx.shape
    rows = np.repeat(np.arange(n), k)
    cols = knn_idx.ravel()
    d = knn_dist.ravel()
    r = np.repeat(rho, k)
    s = np.repeat(sigma, k)
    safe_s = np.where(s == 0.0, 1.0, s)
    vals = np.where((d - r <= 0.0) | (s == 0.0), 1.0, np.exp(-(d - r) / safe_s))
    if exclude_self:
        vals[cols == rows] = 0.0
    return rows, cols, vals


def make_epochs_per_sample(weights, n_epochs):
    result = -1.0 * np.ones(weights.shape[0], dtype=np.float64)
    n_samples = n_epochs * (weights / weights.max())
    result[n_samples > 0] = float(n_epochs) / n_samples[n_samples > 0]
    return result


def _optimize_layout(embedding, other, head, tail, n_epochs, epochs_per_sample, a, b,
                     gamma, initial_alpha, neg_rate, rng, move_other, batch=256):
    n_other = other.shape[0]
    alpha = initial_alpha
    eps_neg = epochs_per_sample / neg_rate
    next_sample = epochs_per_sample.copy()
    next_neg = eps_neg.copy()
    for n in range(n_epochs):
        due = np.flatnonzero(next_sample <= n)
        if due.size:
            rng.shuffle(due)
            for s in range(0, due.size, batch):
                ch = due[s:s + batch]
                j, k = head[ch], tail[ch]
                diff = embedding[j] - other[k]
                dsq = (diff ** 2).sum(1)
                coeff = np.zeros_like(dsq)
                m = dsq > 0
                coeff[m] = -2.0 * a * b * dsq[m] ** (b - 1.0) / (a * dsq[m] ** b + 1.0)
                grad = np.clip(coeff[:, None] * diff, -4.0, 4.0) * alpha
                np.add.at(embedding, j, grad)
                if move_other:
                    np.add.at(other, k, -grad)
                n_neg = ((n - next_neg[ch]) / eps_neg[ch]).astype(int)
                next_sample[ch] += epochs_per_sample[ch]
                next_neg[ch] += n_neg * eps_neg[ch]
                for p in range(int(n_neg.max()) if n_neg.size else 0):
                    sel = n_neg > p
                    jj = j[sel]
                    kk = rng.integers(0, n_other, size=jj.size)
                    diff = embedding[jj] - other[kk]
                    dsq = (diff ** 2).sum(1)
                    pos = dsq > 0
                    coeff = np.zeros_like(dsq)
                    coeff[pos] = 2.0 * gamma * b / ((0.001 + dsq[pos]) * (a * dsq[pos] ** b + 1.0))
                    g = np.clip(coeff[:, None] * diff, -4.0, 4.0)
                    g[~pos] = 4.0
                    g[(~pos) & (jj == kk)] = 0.0
                    np.add.at(embedding, jj, g * alpha)
        alpha = initial_alpha * (1.0 - float(n) / float(n_epochs))
    return embedding


class UMAP:
    def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1, spread=1.0,
                 local_connectivity=1.0, negative_sample_rate=5, repulsion_strength=1.0,
                 learning_rate=1.0, n_epochs=None, random_state=0):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.spread = spread
        self.local_connectivity = local_connectivity
        self.negative_sample_rate = negative_sample_rate
        self.repulsion_strength = repulsion_strength
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state

    # ---- sklearn API --------------------------------------------------
    def get_params(self, deep=True):
        keys = ["n_components", "n_neighbors", "min_dist", "spread", "local_connectivity",
                "negative_sample_rate", "repulsion_strength", "learning_rate", "n_epochs",
                "random_state"]
        return {k: getattr(self, k) for k in keys}

    def set_params(self, **p):
        for k, v in p.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        rng = np.random.default_rng(self.random_state)
        k = min(self.n_neighbors, n - 1)
        self._k = k
        self.n_epochs_ = self.n_epochs if self.n_epochs is not None else (500 if n <= 10000 else 200)
        self.a_, self.b_ = find_ab_params(self.spread, self.min_dist)
        self.nn_ = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(X)
        dist, idx = self.nn_.kneighbors(X, n_neighbors=k)
        # asegurar que el propio punto sea la primera columna
        for i in range(n):
            if idx[i, 0] != i:
                w = np.where(idx[i] == i)[0]
                if w.size:
                    j = w[0]
                    idx[i, [0, j]] = idx[i, [j, 0]]
                    dist[i, [0, j]] = dist[i, [j, 0]]
                else:
                    idx[i, 1:] = idx[i, :-1]
                    dist[i, 1:] = dist[i, :-1]
                    idx[i, 0] = i
                    dist[i, 0] = 0.0
        sigma, rho = smooth_knn_dist(dist, float(k), self.local_connectivity)
        rows, cols, vals = membership_strengths(idx, dist, sigma, rho, exclude_self=True)
        G = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        G = G + G.T - G.multiply(G.T)            # unión difusa
        G.eliminate_zeros()
        self.graph_ = G
        # --- inicialización espectral
        n_comp_graph, _ = connected_components(G)
        d = self.n_components
        if n_comp_graph == 1 and n > d + 1:
            A = G.toarray()
            deg = A.sum(1)
            Dm = 1.0 / np.sqrt(deg)
            L = np.eye(n) - (Dm[:, None] * A * Dm[None, :])
            evals, evecs = np.linalg.eigh(L)
            init = evecs[:, 1:d + 1]
            expansion = 10.0 / np.abs(init).max()
            emb = init * expansion + rng.normal(scale=0.0001, size=init.shape)
        else:
            emb = rng.uniform(-10.0, 10.0, size=(n, d))
        # --- optimización
        Gc = G.tocoo()
        data = Gc.data.copy()
        data[data < (data.max() / float(self.n_epochs_))] = 0.0
        keep = data > 0
        head, tail, w = Gc.row[keep], Gc.col[keep], data[keep]
        eps = make_epochs_per_sample(w, self.n_epochs_)
        emb = emb.astype(np.float64)
        emb = _optimize_layout(emb, emb, head, tail, self.n_epochs_, eps, self.a_, self.b_,
                               self.repulsion_strength, self.learning_rate,
                               self.negative_sample_rate, rng, move_other=True)
        self.embedding_ = emb
        self._train_X = X
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X).embedding_

    def transform(self, Xnew):
        Xnew = np.asarray(Xnew, dtype=np.float64)
        rng = np.random.default_rng(self.random_state + 1)
        k = self._k
        dist, idx = self.nn_.kneighbors(Xnew, n_neighbors=k)
        adj_lc = max(0.0, self.local_connectivity - 1.0)
        sigma, rho = smooth_knn_dist(dist, float(k), adj_lc)
        rows, cols, vals = membership_strengths(idx, dist, sigma, rho, exclude_self=False)
        m = Xnew.shape[0]
        W = sparse.coo_matrix((vals, (rows, cols)), shape=(m, self.embedding_.shape[0])).tocsr()
        rs = np.asarray(W.sum(1)).ravel()
        rs[rs == 0] = 1.0
        init = (sparse.diags(1.0 / rs) @ W) @ self.embedding_
        n_epochs = max(1, int(self.n_epochs_ // 3))
        Wc = W.tocoo()
        data = Wc.data.copy()
        data[data < (data.max() / float(n_epochs))] = 0.0
        keep = data > 0
        head, tail, w = Wc.row[keep], Wc.col[keep], data[keep]
        eps = make_epochs_per_sample(w, n_epochs)
        emb = _optimize_layout(init.astype(np.float64), self.embedding_.copy(), head, tail, n_epochs,
                               eps, self.a_, self.b_, self.repulsion_strength,
                               self.learning_rate / 4.0, self.negative_sample_rate, rng,
                               move_other=False)
        return emb


def get_umap():
    """Clase UMAP que usa el proyecto: la implementación propia (por defecto) o `umap-learn` si MYS2_UMAP=learn."""
    import os
    if os.environ.get("MYS2_UMAP", "").lower() == "learn":
        from umap import UMAP as _UMAP            # pip install umap-learn
        return _UMAP
    return UMAP
