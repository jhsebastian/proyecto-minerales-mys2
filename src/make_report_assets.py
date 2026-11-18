"""Genera tablas LaTeX (report/tablas/*.tex), cifras del texto (report/numeros.tex) y CSV de resultados.

TODAS las cifras del informe salen de aquí: el .tex solo usa \\N{clave} (numeros.tex) y \\input de tablas/*.tex,
de modo que informe y notebook no pueden desalinearse.  Uso: python make_report_assets.py
"""
import os, joblib, json
import numpy as np
import pandas as pd
import mys2_core as C
import mys2_data as D

import mys2_paths as P
OUT = str(P.CACHE) + "/"
REP = str(P.REPORT) + "/"
RES = str(P.RESULTS) + "/"
os.makedirs(REP + "tablas", exist_ok=True)
os.makedirs(RES, exist_ok=True)

MODELS = ["Random Forest", "MLP", "Regresión logística", "SVM", "k-NN"]
SHORT = {"Random Forest": "rf", "MLP": "mlp", "Regresión logística": "lr", "SVM": "svm", "k-NN": "knn"}
TECH_NAME = {"sel": "Selección por $\\eta^2$", "pca": "PCA", "umap": "UMAP"}
NUM = {}


def put(key, val):
    NUM[key] = val


def f3(x, nd=3):
    return f"{x:.{nd}f}"


def pm(m, h, nd=3):
    return f"{m:.{nd}f}$\\pm${h:.{nd}f}"


def br(lo, hi, nd=2):
    return f"[{lo:.{nd}f}, {hi:.{nd}f}]"


def load_all():
    S = dict(g=joblib.load(OUT + "stage1_final.joblib"), s2=joblib.load(OUT + "stage2_individual.joblib"),
             s3=joblib.load(OUT + "stage3_final.joblib"))
    for t in ("sel", "pca", "umap"):
        if os.path.exists(OUT + f"stage4_{t}.joblib"):
            S[t] = joblib.load(OUT + f"stage4_{t}.joblib")
    return S


def nb_sizes():
    spl = C.make_splits(D.load())
    return float(np.mean([x["n_tr_orig"] for x in spl])), float(np.mean([len(x["val"]) for x in spl]))


def hp_text(m, p):
    if m == "Random Forest":
        mf = p["max_features"]
        mf_s = {"log2": "$\\log_2 p$", "sqrt": "$\\sqrt p$"}.get(mf, f"{mf}$p$")
        depth = "sin límite" if p["max_depth"] is None else f"prof. {p['max_depth']}"
        return f"{p['n_estimators']} árboles, {depth}, {mf_s} var./div."
    if m == "MLP":
        return "capas (" + ",".join(str(h) for h in p["hidden_layer_sizes"]) + f"), $\\alpha={p['alpha']:g}$"
    if m == "Regresión logística":
        return f"$C={p['C']:g}$"
    if m == "SVM":
        if p["kernel"] == "rbf":
            g = f"{p['gamma']:g}"
            if "e-" in g:
                a, b = g.split("e-")
                g = f"{a}\\cdot10^{{-{int(b)}}}" if a != "1" else f"10^{{-{int(b)}}}"
            return f"RBF, $C={p['C']:g}$, $\\gamma={g}$"
        return f"lineal, $C={p['C']:g}$"
    return f"$k={p['n_neighbors']}$, pesos {'uniformes' if p['weights'] == 'uniform' else 'por distancia'}"


# ---------------------------------------------------------------------------------------------- Tabla de mallas
def _logseq(vals):
    """Texto compacto de una malla geométrica (razón 10) o lista explícita."""
    v = [float(x) for x in vals]
    e = [np.log10(x) for x in v]
    if len(v) >= 4 and np.allclose(np.diff(e), 1.0):
        pw = lambda x: "1" if int(round(x)) == 0 else f"10^{{{int(round(x))}}}"
        return f"${pw(e[0])},{pw(e[1])},\\dots,{pw(e[-1])}$"
    return ", ".join(f"{x:g}" for x in v)


def tab_grids():
    G = C.GRIDS
    n_cfg = {m: len(C.expand_grid(g)) for m, g in G.items()}
    g_svm_rbf, g_svm_lin = G["SVM"]
    rows = [
        ("Reg. logística", [("$C$ (inverso de la regularización L2)", _logseq(G["Regresión logística"]["C"]))]),
        ("$k$-NN", [("vecinos $k$", ", ".join(str(k) for k in G["k-NN"]["n_neighbors"])), ("pesos", "uniformes, por distancia")]),
        ("Random Forest", [("n.º de árboles", "100, 300$^{\\dagger}$"), ("profundidad máxima", "3, 5, 10, sin límite"),
                           ("variables por división", "$\\log_2 p$, $\\sqrt p$, $0.1p$, $0.3p$, $0.6p$, $p$")]),
        ("MLP", [("capas ocultas", "(32), (64), (128), (64,32)"), ("penalización L2 $\\alpha$", _logseq(G["MLP"]["alpha"]))]),
        ("SVM", [("RBF: $C$", ", ".join(f"{int(c)}" for c in g_svm_rbf["C"])),
                 ("RBF: $\\gamma$", "$10^{-5}$, $3\\cdot10^{-5}$, $10^{-4}$, $3\\cdot10^{-4}$, $10^{-3}$, $10^{-2}$"),
                 ("lineal: $C$", _logseq(g_svm_lin["C"]))]),
    ]
    L = []
    L.append("\\begin{table}[!t]")
    L.append("\\caption{Hiperparámetros analizados y valores de la malla de cada modelo (%d configuraciones $\\times$ 15 pliegues)}" % sum(n_cfg.values()))
    L.append("\\label{tab:grids}\n\\centering\\footnotesize\n\\setlength{\\tabcolsep}{2.5pt}\\renewcommand{\\arraystretch}{1.05}")
    L.append("\\begin{tabular}{@{}p{1.95cm} p{2.85cm} p{3.45cm}@{}}\n\\toprule\nModelo & Hiperparámetro & Valores\\\\\n\\midrule")
    for k, (model, hp) in enumerate(rows):
        for j, (h, v) in enumerate(hp):
            L.append(f"{model if j == 0 else ''} & {h} & {v}\\\\")
        if k < len(rows) - 1:
            L.append("\\addlinespace[1pt]")
    L.append("\\bottomrule\n\\end{tabular}")
    L.append("\\par\\smallskip\\noindent{\\scriptsize Configuraciones por modelo: reg. logística %d, $k$-NN %d, Random Forest %d, MLP %d, SVM %d. $p=1835$. "
             "$^{\\dagger}$300 árboles solo con $\\le0.3p$ variables por división (costo). "
             "Fijos: LogReg multinomial con L2 y máx. 3000 iteraciones; MLP con ReLU, Adam, tasa $10^{-3}$ y máx. 500 épocas; semilla 2026.}"
             % (n_cfg["Regresión logística"], n_cfg["k-NN"], n_cfg["Random Forest"], n_cfg["MLP"], n_cfg["SVM"]))
    L.append("\\end{table}")
    open(REP + "tablas/tab_grids.tex", "w").write("\n".join(L) + "\n")


# ---------------------------------------------------------------------------------------------- Resultados (etapa 3)
def tab_resultados(S):
    s3 = S["s3"]
    L = ["\\begin{table*}[!t]",
         "\\caption{Desempeño de los cinco modelos con su mejor configuración. Validación: media $\\pm$ IC del 95\\,\\% "
         "(Nadeau--Bengio, 15 particiones sin fuga). Prueba: 36 imágenes nunca usadas; IC de Wilson (exactitud) y "
         "\\emph{bootstrap} (F1). Las dos últimas columnas repiten la validación con fuga (copias aumentadas repartidas al azar) y sin aumentación}",
         "\\label{tab:res}", "\\centering\\footnotesize", "\\setlength{\\tabcolsep}{2.7pt}\\renewcommand{\\arraystretch}{1.08}",
         "\\begin{tabular}{@{}l >{\\raggedright\\arraybackslash}p{3.0cm} c c c c c c c c@{}}", "\\toprule",
         " & & \\multicolumn{2}{c}{Entrenamiento} & \\multicolumn{2}{c}{Validación (sin fuga)} & \\multicolumn{2}{c}{Prueba ($N=36$)} & \\multicolumn{2}{c}{F1 de validación} \\\\",
         "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(l){9-10}",
         "Modelo & Hiperparámetros óptimos & Exactitud & F1 & Exactitud & F1 macro & Exactitud [IC 95\\,\\%] & F1 macro [IC 95\\,\\%] & con fuga & sin aument. \\\\",
         "\\midrule"]
    for m in s3["ranking"]:
        r = s3[m]
        v = r["val"]
        L.append(" & ".join([
            m, hp_text(m, r["params"]), f3(r["train_acc"], 2), f3(r["train_f1"], 2),
            pm(*v["acc_m"]), pm(*v["f1_m"]),
            f"{f3(r['test_acc'])} {br(*r['test_acc_ci'])}", f"{f3(r['test_f1'])} {br(*r['test_f1_ci'])}",
            f3(r["naive"]["f1_m"][0]), f3(r["noaug"]["f1_m"][0])]) + " \\\\")
        s = SHORT[m]
        put(f"{s}-val-f1", f3(v["f1_m"][0])); put(f"{s}-val-f1-ci", f3(v["f1_m"][1]))
        put(f"{s}-val-acc", f3(v["acc_m"][0])); put(f"{s}-val-acc-ci", f3(v["acc_m"][1]))
        put(f"{s}-train-f1", f3(r["train_f1"], 2)); put(f"{s}-train-acc", f3(r["train_acc"], 2))
        put(f"{s}-test-acc", f3(r["test_acc"])); put(f"{s}-test-acc-lo", f3(r["test_acc_ci"][0], 2)); put(f"{s}-test-acc-hi", f3(r["test_acc_ci"][1], 2))
        put(f"{s}-test-f1", f3(r["test_f1"])); put(f"{s}-test-f1-lo", f3(r["test_f1_ci"][0], 2)); put(f"{s}-test-f1-hi", f3(r["test_f1_ci"][1], 2))
        put(f"{s}-naive-f1", f3(r["naive"]["f1_m"][0])); put(f"{s}-noaug-f1", f3(r["noaug"]["f1_m"][0]))
        put(f"{s}-leak-gap", f3(r["naive"]["f1_m"][0] - v["f1_m"][0]))
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    open(REP + "tablas/tab_resultados.tex", "w").write("\n".join(L) + "\n")
    # --- diferencias pareadas (mismos pliegues) con el IC de Nadeau-Bengio
    n_tr, n_va = nb_sizes()
    for m in s3["ranking"]:
        dif = s3[m]["val"]["f1"] - s3[m]["noaug"]["f1"]
        mm, hh = C.nb_ci(dif, n_tr, n_va)
        put(f"{SHORT[m]}-aug-gain", f3(mm)); put(f"{SHORT[m]}-aug-gain-ci", f3(hh))
        put(f"{SHORT[m]}-aug-sig", "1" if abs(mm) > hh else "0")
    b, s2_ = s3["ranking"][0], s3["ranking"][1]
    mm, hh = C.nb_ci(s3[b]["val"]["f1"] - s3[s2_]["val"]["f1"], n_tr, n_va)
    put("rf-vs-mlp", f3(mm)); put("rf-vs-mlp-ci", f3(hh))
    mm, hh = C.nb_ci(s3[b]["val"]["acc"] - s3[s2_]["val"]["acc"], n_tr, n_va)
    put("rf-vs-mlp-acc", f3(mm)); put("rf-vs-mlp-acc-ci", f3(hh))
    # --- cifras de la matriz de confusión del mejor modelo
    best = s3["ranking"][0]
    cmv = s3[best]["cm_val"]; cmv = cmv / cmv.sum(1, keepdims=True)
    cmt = s3[best]["cm_test"] / s3[best]["cm_test"].sum(1, keepdims=True)
    for i in range(4):
        for j in range(4):
            put(f"cmv-{i}-{j}", f3(cmv[i, j], 2)); put(f"cmt-{i}-{j}", f3(cmt[i, j], 2))
    pc = s3[best]["test_per_class"]
    for i in range(4):
        put(f"tpc-p-{i}", f3(pc["precision"][i], 2)); put(f"tpc-r-{i}", f3(pc["recall"][i], 2)); put(f"tpc-f-{i}", f3(pc["f1"][i], 2))
    # CSV
    rows = []
    for m in s3["ranking"]:
        r = s3[m]
        rows.append(dict(modelo=m, hiperparametros=json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in r["params"].items()}),
                         f1_train=r["train_f1"], acc_val=r["val"]["acc_m"][0], acc_val_ic=r["val"]["acc_m"][1],
                         f1_val=r["val"]["f1_m"][0], f1_val_ic=r["val"]["f1_m"][1], acc_test=r["test_acc"],
                         acc_test_lo=r["test_acc_ci"][0], acc_test_hi=r["test_acc_ci"][1], f1_test=r["test_f1"],
                         f1_test_lo=r["test_f1_ci"][0], f1_test_hi=r["test_f1_ci"][1], f1_val_con_fuga=r["naive"]["f1_m"][0],
                         f1_val_sin_aumentacion=r["noaug"]["f1_m"][0]))
    pd.DataFrame(rows).to_csv(RES + "resultados_modelos.csv", index=False)


# ---------------------------------------------------------------------------------------------- Efecto de hiperparámetros
def hyper_numbers(S):
    g = S["g"]

    def f1_of(m, **kw):
        for r in g[m]:
            if all(r["params"].get(k) == v for k, v in kw.items()):
                return r["f1_mean"]
        raise KeyError((m, kw))

    def best_over(m, **kw):
        return max(r["f1_mean"] for r in g[m] if all(r["params"].get(k) == v for k, v in kw.items()))

    lr = {r["params"]["C"]: r for r in g["Regresión logística"]}
    put("lr-f1-cmin", f3(lr[1e-4]["f1_mean"])); put("lr-f1-c2", f3(lr[1e-2]["f1_mean"])); put("lr-f1-best", f3(max(r["f1_mean"] for r in lr.values())))
    put("lr-train-cmin", f3(lr[1e-4]["train_f1"], 2))
    kn = [r["f1_mean"] for r in g["k-NN"]]
    put("knn-f1-min", f3(min(kn))); put("knn-f1-max", f3(max(kn)))
    put("rf-f1-log2", f3(best_over("Random Forest", max_features="log2"))); put("rf-f1-sqrt", f3(best_over("Random Forest", max_features="sqrt")))
    put("rf-f1-m01", f3(best_over("Random Forest", max_features=0.1))); put("rf-f1-m03", f3(best_over("Random Forest", max_features=0.3)))
    put("rf-f1-m06", f3(best_over("Random Forest", max_features=0.6))); put("rf-f1-m10", f3(best_over("Random Forest", max_features=1.0)))
    put("rf-f1-d3", f3(best_over("Random Forest", max_depth=3))); put("rf-f1-dinf", f3(best_over("Random Forest", max_depth=None)))
    put("rf-f1-n100", f3(f1_of("Random Forest", n_estimators=100, max_depth=None, max_features=0.3)))
    put("rf-f1-n300", f3(f1_of("Random Forest", n_estimators=300, max_depth=None, max_features=0.3)))
    ml = g["MLP"]
    put("mlp-f1-a4", f3(max(r["f1_mean"] for r in ml if r["params"]["alpha"] == 1e-4)))
    put("mlp-f1-a1", f3(max(r["f1_mean"] for r in ml if r["params"]["alpha"] == 1.0)))
    put("mlp-f1-a10", f3(max(r["f1_mean"] for r in ml if r["params"]["alpha"] == 10.0)))
    put("mlp-f1-a100", f3(max(r["f1_mean"] for r in ml if r["params"]["alpha"] == 100.0)))
    put("mlp-f1-a100-min", f3(min(r["f1_mean"] for r in ml if r["params"]["alpha"] == 100.0)))
    sv = g["SVM"]
    rbf = [r for r in sv if r["params"]["kernel"] == "rbf"]; lin = [r for r in sv if r["params"]["kernel"] == "linear"]
    put("svm-rbf-best", f3(max(r["f1_mean"] for r in rbf))); put("svm-lin-best", f3(max(r["f1_mean"] for r in lin)))
    put("svm-g1e2", f3(max(r["f1_mean"] for r in rbf if r["params"]["gamma"] == 1e-2)))
    put("svm-c1-g1e5", f3(f1_of("SVM", kernel="rbf", C=1.0, gamma=1e-5)))
    put("svm-lin-c4", f3(f1_of("SVM", kernel="linear", C=1e-4)))
    put("svm-lin-c3", f3(f1_of("SVM", kernel="linear", C=1e-3)))
    pl = [r["f1_mean"] for r in rbf if r["params"]["gamma"] <= 1e-3 and r["params"]["C"] >= 10]
    put("svm-rbf-plateau-lo", f3(min(pl))); put("svm-rbf-plateau-hi", f3(max(pl)))
    from scipy.stats import hypergeom
    p_all, p_cmp = 1835, 71
    for key, m in (("log2", int(np.log2(p_all))), ("sqrt", int(np.sqrt(p_all))), ("m03", int(0.3 * p_all))):
        put(f"rf-see-{key}", f3(1 - hypergeom.pmf(0, p_all, p_cmp, m), 2))
        put(f"rf-mtry-{key}", str(m))
    put("ncfg-total", str(sum(len(v) for v in g.values())))
    for m in MODELS:
        put(f"{SHORT[m]}-ncfg", str(len(g[m])))


# ---------------------------------------------------------------------------------------------- Análisis individual
def tab_individual(S):
    s2 = S["s2"]; tab = s2["tab"]; df = s2["df"]
    L = ["\\begin{table}[!t]",
         "\\caption{Análisis individual de las variables por familia: índice de discriminación $\\eta^2$ (204 originales), "
         "porcentaje de variables significativas ($q<0.05$, Benjamini--Hochberg) y candidatas a eliminar (no significativas o redundantes, $|\\rho|\\ge0.95$)}",
         "\\label{tab:ind}", "\\centering\\footnotesize", "\\setlength{\\tabcolsep}{2.6pt}\\renewcommand{\\arraystretch}{1.05}",
         "\\begin{tabular}{@{}l r c c r r r r@{}}", "\\toprule",
         "Familia & $n$ & $\\tilde\\eta^2$ & $\\eta^2_{\\max}$ & \\% sig. & Redund. & Candid. & Conserv.\\\\", "\\midrule"]
    for _, r in tab.iterrows():
        L.append(f"{r['familia']} & {int(r['n_vars'])} & {r['eta2_mediana']:.3f} & {r['eta2_max']:.3f} & "
                 f"{r['pct_signif']:.0f} & {int(r['n_redundantes'])} & {int(r['n_candidatas'])} & {int(r['n_conservadas'])}\\\\")
    tot = dict(n=int(tab["n_vars"].sum()), red=int(tab["n_redundantes"].sum()), cand=int(tab["n_candidatas"].sum()),
               cons=int(tab["n_conservadas"].sum()), nosig=int(tab["n_no_signif"].sum()))
    L.append("\\midrule")
    L.append(f"Total & {tot['n']} & {df['eta2'].median():.3f} & {df['eta2'].max():.3f} & "
             f"{100 * (1 - tot['nosig'] / tot['n']):.0f} & {tot['red']} & {tot['cand']} & {tot['cons']}\\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(REP + "tablas/tab_individual.tex", "w").write("\n".join(L) + "\n")
    put("ind-nonsig", str(tot["nosig"])); put("ind-red", str(tot["red"])); put("ind-cand", str(tot["cand"])); put("ind-cons", str(tot["cons"]))
    put("ind-pct-cand", f"{100 * tot['cand'] / tot['n']:.0f}")
    put("ind-pairs", str(s2["n_pairs_hi"])); put("ind-spearman", f3(s2["spearman"], 2))
    t = tab.set_index("familia")
    put("ind-hog-pct", f"{t.loc['HOG', 'pct_signif']:.0f}"); put("ind-hog-eta", f3(t.loc["HOG", "eta2_mediana"]))
    put("ind-hog-cand", str(int(t.loc["HOG", "n_candidatas"]))); put("ind-hog-cons", str(int(t.loc["HOG", "n_conservadas"])))
    put("ind-hog-etamax", f3(t.loc["HOG", "eta2_max"]))
    for fam in ("Color", "Gabor", "LBP", "GLCM", "Hu"):
        put(f"ind-{fam.lower()}-eta", f3(t.loc[fam, "eta2_mediana"])); put(f"ind-{fam.lower()}-etamax", f3(t.loc[fam, "eta2_max"]))
        put(f"ind-{fam.lower()}-cons", str(int(t.loc[fam, "n_conservadas"]))); put(f"ind-{fam.lower()}-cand", str(int(t.loc[fam, "n_candidatas"])))
    compact = tab[tab["familia"] != "HOG"]
    put("ind-compact-n", str(int(compact["n_vars"].sum()))); put("ind-compact-pct", f"{100 * compact['n_vars'].sum() / tot['n']:.1f}")
    put("ind-compact-cons", str(int(compact["n_conservadas"].sum())))
    srt = df.sort_values("eta2", ascending=False)
    put("ind-top-eta", f3(srt.iloc[0]["eta2"]))
    put("ind-n-eta02", str(int((df["eta2"] >= 0.2).sum()))); put("ind-n-eta03", str(int((df["eta2"] >= 0.3).sum())))
    put("ind-top50-hog", str(int((srt.head(50)["familia"] == "HOG").sum())))
    df.to_csv(RES + "variables_individual.csv", index=False)


# ---------------------------------------------------------------------------------------------- Reducción de dimensión
def reduction_rows(S):
    """Filas (modelo, técnica) con k*, % de reducción y medidas de desempeño; incluye la línea base (todas las variables)."""
    s3 = S["s3"]
    P = 1835
    rows = []
    ref = next(S[t] for t in ("sel", "pca", "umap") if t in S)
    for m in ref["models"]:
        r = s3[m]
        rows.append(dict(model=m, tech="all", k=P, red=0.0, var=None, f1=r["val"]["f1_m"], acc=r["val"]["acc_m"],
                         tr_f1=r["train_f1"], tr_acc=r["train_acc"], te_acc=r["test_acc"], te_acc_ci=r["test_acc_ci"], te_f1=r["test_f1"], te_f1_ci=r["test_f1_ci"]))
        for t in ("sel", "pca", "umap"):
            if t not in S:
                continue
            sm = S[t]["summary"][m]
            ks = sm["k_1se"]
            pk = [x for x in sm["per_k"] if x["k"] == ks][0]
            fin = sm["final"]
            rows.append(dict(model=m, tech=t, k=ks, red=100 * (P - ks) / P, var=fin.get("cumvar"), f1=(pk["f1_mean"], pk["f1_ci"]),
                             acc=(pk["acc_mean"], pk["acc_ci"]), tr_f1=fin["train_f1"], tr_acc=fin["train_acc"], te_acc=fin["test_acc"], te_acc_ci=fin["test_acc_ci"],
                             te_f1=fin["test_f1"], te_f1_ci=fin["test_f1_ci"], k_best=sm["k_best"]))
    return rows


def tab_reduccion(S):
    rows = reduction_rows(S)
    L = ["\\begin{table*}[!t]",
         "\\caption{Efecto de la reducción de dimensión sobre los dos mejores modelos. $k^*$: n.º de variables o componentes elegido con la regla de "
         "un error estándar sobre el F1 de validación (15 particiones sin fuga, media $\\pm$ IC 95\\,\\%); reducción $=(1835-k^*)/1835$; "
         "var.~expl.: varianza explicada acumulada (solo PCA). Con $k^*$ se reajusta con todo el desarrollo y se evalúa en prueba ($N=36$)}",
         "\\label{tab:red}", "\\centering\\footnotesize", "\\setlength{\\tabcolsep}{2.8pt}\\renewcommand{\\arraystretch}{1.08}",
         "\\begin{tabular}{@{}l l r r r c c c c c c@{}}", "\\toprule",
         " & & & & & \\multicolumn{2}{c}{Entrenamiento} & \\multicolumn{2}{c}{Validación (sin fuga)} & \\multicolumn{2}{c}{Prueba ($N=36$)}\\\\",
         "\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\\cmidrule(l){10-11}",
         "Modelo & Representación & $k^*$ & Reducc. (\\%) & Var.~expl. (\\%) & Exact. & F1 & Exactitud & F1 macro & Exactitud [IC 95\\,\\%] & F1 macro [IC 95\\,\\%]\\\\",
         "\\midrule"]
    last = None
    for r in rows:
        if last is not None and r["model"] != last:
            L.append("\\midrule")
        name = r["model"] if r["model"] != last else ""
        last = r["model"]
        rep = "Todas las variables" if r["tech"] == "all" else TECH_NAME[r["tech"]]
        L.append(" & ".join([name, rep, str(r["k"]), f"{r['red']:.1f}", f"{100 * r['var']:.1f}" if r["var"] is not None else "--",
                             f3(r["tr_acc"], 2), f3(r["tr_f1"], 2), pm(*r["acc"]), pm(*r["f1"]),
                             f"{f3(r['te_acc'])} {br(*r['te_acc_ci'])}", f"{f3(r['te_f1'])} {br(*r['te_f1_ci'])}"]) + " \\\\")
        s = SHORT[r["model"]]
        put(f"{r['tech']}-{s}-kstar", str(r["k"])); put(f"{r['tech']}-{s}-red", f"{r['red']:.1f}")
        put(f"{r['tech']}-{s}-val-f1", f3(r["f1"][0])); put(f"{r['tech']}-{s}-val-f1-ci", f3(r["f1"][1]))
        put(f"{r['tech']}-{s}-val-acc", f3(r["acc"][0])); put(f"{r['tech']}-{s}-trainf1", f3(r["tr_f1"], 2))
        put(f"{r['tech']}-{s}-test-acc", f3(r["te_acc"])); put(f"{r['tech']}-{s}-test-f1", f3(r["te_f1"]))
        put(f"{r['tech']}-{s}-test-acc-lo", f3(r["te_acc_ci"][0], 2)); put(f"{r['tech']}-{s}-test-acc-hi", f3(r["te_acc_ci"][1], 2))
        put(f"{r['tech']}-{s}-test-f1-lo", f3(r["te_f1_ci"][0], 2)); put(f"{r['tech']}-{s}-test-f1-hi", f3(r["te_f1_ci"][1], 2))
        if r["tech"] != "all":
            sm = S[r["tech"]]["summary"][r["model"]]
            pk = [x for x in sm["per_k"] if x["k"] == r["k"]][0]
            n_tr, n_va = nb_sizes()
            dm, dh = C.nb_ci(pk["f1"] - S["s3"][r["model"]]["val"]["f1"], n_tr, n_va)
            put(f"{r['tech']}-{s}-dif", f3(dm)); put(f"{r['tech']}-{s}-dif-ci", f3(dh))
            put(f"{r['tech']}-{s}-dif-sig", "1" if abs(dm) > dh else "0")
            bi = max((x for x in sm["per_k"] if x["k"] > 0), key=lambda x: x["f1_mean"])
            put(f"{r['tech']}-{s}-f1best", f3(bi["f1_mean"])); put(f"{r['tech']}-{s}-f1best-ci", f3(bi["f1_ci"]))
            if r["tech"] == "sel":
                ka = [x for x in sm["per_k"] if x["k"] == -1][0]
                put(f"sel-{s}-allsig-f1", f3(ka["f1_mean"])); put(f"sel-{s}-allsig-n", f"{ka['extra']['n']:.0f}")
        if r["var"] is not None:
            put(f"{r['tech']}-{s}-var", f"{100 * r['var']:.1f}")
        if "k_best" in r:
            put(f"{r['tech']}-{s}-kbest", str(r["k_best"]))
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    open(REP + "tablas/tab_reduccion.tex", "w").write("\n".join(L) + "\n")
    # CSV completo de las curvas (todas las k)
    cur = []
    for t in ("sel", "pca", "umap"):
        if t not in S:
            continue
        for m, sm in S[t]["summary"].items():
            for x in sm["per_k"]:
                cur.append(dict(tecnica=t, modelo=m, k=x["k"], f1_val=x["f1_mean"], f1_val_ic=x["f1_ci"], acc_val=x["acc_mean"],
                                acc_val_ic=x["acc_ci"], hiperparametros=json.dumps({a: (list(b) if isinstance(b, tuple) else b) for a, b in x["params"].items()}),
                                **{f"extra_{a}": b for a, b in x["extra"].items()}))
    if cur:
        pd.DataFrame(cur).to_csv(RES + "reduccion_curvas.csv", index=False)


def extra_reduction_numbers(S):
    """Composición por familia de las k* variables elegidas y k necesarios para 80/90/95 % de varianza (PCA)."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    d = D.load()
    X, y, role, aug, fam, names = d["X"], d["y"], d["role"], d["is_aug"], np.array(d["family"]), np.array(d["names"])
    dev = np.flatnonzero(role != "test")
    sc = StandardScaler().fit(X[dev]); Xd, yd = sc.transform(X[dev]), y[dev]
    o = ~aug[dev]
    rk = C.variable_ranking(Xd[o], yd[o])
    if "sel" in S:
        for m in S["sel"]["models"]:
            k = S["sel"]["summary"][m]["k_1se"]
            fams = fam[rk["order"][:k]]
            comp = {f: int((fams == f).sum()) for f in ("HOG", "Color", "Gabor", "LBP", "GLCM", "Hu")}
            put(f"sel-{SHORT[m]}-comp", ", ".join(f"{v} {f}" for f, v in comp.items() if v))
            for f, v in comp.items():
                put(f"sel-{SHORT[m]}-n{f.lower()}", str(v))
        put("sel-top5", ", ".join(str(n).replace("_", "\\_") for n in names[rk["order"][:5]]))
        put("sel-nsig-fold", f"{np.mean([x['n'] for x in [S['sel']['summary'][S['sel']['models'][0]]['per_k'][-1]['extra']]]):.0f}")
    p = PCA(svd_solver="full").fit(Xd)
    cv = np.cumsum(p.explained_variance_ratio_)
    for q in (80, 90, 95):
        put(f"pca-k{q}", str(int(np.searchsorted(cv, q / 100) + 1)))
    put("pca-var2", f"{100 * cv[1]:.1f}"); put("pca-var2-pc1", f"{100 * p.explained_variance_ratio_[0]:.1f}")
    put("pca-var32", f"{100 * cv[31]:.1f}"); put("pca-var128", f"{100 * cv[127]:.1f}")
    # --- diagnóstico descriptivo: ¿en qué direcciones está la varianza y en cuáles la información sobre la clase?
    # (PCA ajustada con las 396 filas de desarrollo, como en el ajuste final; eta^2 de cada componente sobre las 204 originales)
    Z = p.transform(Xd)[:, :128]
    rkz = C.variable_ranking(Z[o], yd[o])
    e2 = rkz["eta2"]
    put("pcadiag-max-eta", f3(e2.max())); put("pcadiag-argmax", str(int(np.argmax(e2)) + 1))
    put("pcadiag-n-eta02", str(int((e2 >= 0.2).sum()))); put("pcadiag-n-eta01", str(int((e2 >= 0.1).sum())))
    put("pcadiag-first-eta02", str(int(np.flatnonzero(e2 >= 0.2)[0]) + 1) if (e2 >= 0.2).any() else "--")
    put("pcadiag-eta-first32-max", f3(e2[:32].max()))
    # varianza (de las 1835 variables estandarizadas) que corresponde a las 71 no HOG, y norma de las cargas en HOG en las primeras 32 componentes
    hog = fam == "HOG"
    share = (p.components_[:128][:, hog] ** 2).sum(1)
    put("pcadiag-hog-share32", f"{100 * share[:32].mean():.0f}"); put("pcadiag-hog-share128", f"{100 * share.mean():.0f}")
    put("pcadiag-hog-expected", f"{100 * hog.mean():.0f}")
    put("pcadiag-compact-var", f"{100 * (~hog).mean():.1f}")
    # mejor componente: composición de sus cargas (familia con más norma)
    j = int(np.argmax(e2))
    fam_share = {f: float((p.components_[j][fam == f] ** 2).sum()) for f in ("HOG", "Color", "Gabor", "LBP", "GLCM", "Hu")}
    put("pcadiag-bestpc-hog", f"{100 * fam_share['HOG']:.0f}"); put("pcadiag-bestpc-color", f"{100 * fam_share['Color']:.0f}")
    # --- diagnóstico descriptivo de la vecindad (la geometría que usa UMAP): proporción de los 15 vecinos más cercanos de la misma clase
    from sklearn.neighbors import NearestNeighbors

    def purity(cols, k=15):
        Zc, yy = Xd[o][:, cols], yd[o]
        idx = NearestNeighbors(n_neighbors=k + 1).fit(Zc).kneighbors(Zc, return_distance=False)[:, 1:]     # sin el propio punto
        return float((yy[idx] == yy[:, None]).mean())
    put("nbr-pur-all", f3(purity(np.arange(Xd.shape[1])), 2)); put("nbr-pur-compact", f3(purity(np.flatnonzero(~hog)), 2))
    n_o = int(o.sum()); put("nbr-pur-chance", f3((n_o / 4 - 1) / (n_o - 1), 2))
    # --- diferencias pareadas (mismos pliegues) entre técnicas de reducción, en el k* de cada una (IC de Nadeau-Bengio)
    n_tr, n_va = nb_sizes()

    def f1_at_kstar(t, m):
        sm = S[t]["summary"][m]
        return [x for x in sm["per_k"] if x["k"] == sm["k_1se"]][0]["f1"]
    for m in S["sel"]["models"]:
        for a, b in (("sel", "pca"), ("sel", "umap"), ("pca", "umap")):
            if a in S and b in S:
                dm, dh = C.nb_ci(f1_at_kstar(a, m) - f1_at_kstar(b, m), n_tr, n_va)
                put(f"{a}-{b}-{SHORT[m]}", f3(dm)); put(f"{a}-{b}-{SHORT[m]}-ci", f3(dh))
                put(f"{a}-{b}-{SHORT[m]}-sig", "1" if abs(dm) > dh else "0")


def eda_numbers():
    """Cifras del análisis exploratorio (204 originales de desarrollo): rangos, escalas, asimetría y correlación."""
    from scipy import stats
    d = D.load()
    X, role, aug, fam = d["X"], d["role"], d["is_aug"], np.array(d["family"])
    Xo = X[(role != "test") & (~aug)]
    sd = Xo.std(0)
    put("eda-n", str(X.shape[0])); put("eda-p", str(X.shape[1]))
    put("eda-color-max", f"{X[:, fam == 'Color'].max():.0f}")
    put("eda-gabor-max", f"{X[:, fam == 'Gabor'].max():,.0f}".replace(",", "\\,"))
    put("eda-hu-max", f"{X[:, fam == 'Hu'].max():.0f}")
    ratio = sd.max() / sd[sd > 0].min()
    e = int(np.floor(np.log10(ratio)))
    put("eda-sd-ratio", f"{ratio / 10 ** e:.0f}\\times10^{{{e}}}")
    sk = np.abs(stats.skew(Xo, axis=0))
    put("eda-skew-n", str(int((sk > 1).sum()))); put("eda-skew-pct", f"{100 * (sk > 1).mean():.0f}")
    put("eda-hu-skew-pct", f"{100 * (sk[fam == 'Hu'] > 1).mean():.0f}")
    for nm, m in (("compact", fam != "HOG"), ("hog", fam == "HOG")):
        R = np.corrcoef(Xo[:, m].T); iu = np.triu_indices(R.shape[0], 1)
        put(f"eda-pairs-{nm}", str(int((np.abs(R[iu]) >= 0.95).sum())))


def write_numeros():
    L = ["% Generado por make_report_assets.py -- NO editar a mano.",
         "\\makeatletter",
         "\\providecommand{\\N}[1]{\\@ifundefined{N:#1}{\\textbf{??#1??}\\PackageWarning{numeros}{Falta la cifra #1}}{\\csname N:#1\\endcsname}}"]
    for k in sorted(NUM):
        L.append("\\expandafter\\gdef\\csname N:%s\\endcsname{%s}" % (k, NUM[k]))
    L.append("\\makeatother")
    open(REP + "numeros.tex", "w").write("\n".join(L) + "\n")
    json.dump(NUM, open(RES + "cifras_informe.json", "w"), ensure_ascii=False, indent=1, sort_keys=True)


def main():
    """Genera informe/latex/{numeros.tex, tablas/*.tex} y results/*.csv|json a partir de los resultados guardados en results/cache."""
    NUM.clear()
    S = load_all()
    tab_grids()
    tab_resultados(S)
    hyper_numbers(S)
    tab_individual(S)
    eda_numbers()
    if any(t in S for t in ("sel", "pca", "umap")):
        tab_reduccion(S)
        extra_reduction_numbers(S)
    write_numeros()
    print("listo:", len(NUM), "cifras;", sorted(S))


if __name__ == "__main__":
    main()
