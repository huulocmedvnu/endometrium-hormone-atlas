"""S15 (R15 to R18, post hoc): clock rigour.
R15 mid-range validation on Wang 2020 10x donors with rank-space snRNA-seq clocks. R16 linear calibration of each
clock on natural leave-one-donor-out predictions, H1 and residual tests repeated. R17 bootstrap of natural training
donors (200 resamples, stratified by stage), out-of-bag scoring. R18 model comparison by leave-one-donor-out.
Writes outputs/s15_clock_rigor.json and outputs/s15_bootstrap_donors.csv. Usage: s15_clock_rigor.py"""
import json, os
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, linregress
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV, LogisticRegression, RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from s2_clock import fit, lodo, stage_value, treatment_group, ROOT, W, MIN_N, N_HVG, ALPHAS
from s4_sensitivity import residuals
from s8_effects import hodges_lehmann
from s10_rank_bulk import rank_quantiles

OUT = os.path.join(ROOT, "outputs")
COMPS = ("stroma", "epithelium")
N_BOOT = 200
WANG = {"Proliferative": 0.0, "Secretory Early": 1.0, "Secretory Early-Mid": 1.5, "Secretory Mid": 2.0}


def load(s):
    P = pd.read_parquet(os.path.join(W, f"profiles_{s}.parquet")); M = pd.read_csv(os.path.join(W, f"profiles_{s}_meta.csv"), index_col=0).loc[P.index]
    M["group"] = M["Hormonal treatment"].map(treatment_group); M["y"] = M.Stage.map(stage_value); M.loc[M.group != "natural", "y"] = np.nan
    return P, M


def p1(a, b, alt):
    return float(f"{mannwhitneyu(a, b, alternative=alt).pvalue:.3g}")


# ---------- model factories: each returns a function (Xtr, ytr) -> predictor(X) ----------
def hv(X, n=N_HVG):
    return np.argsort(X.var(0))[::-1][:n]


def m_ridge(X, y):
    return fit(X, y)


def m_enet(X, y):
    h = hv(X); m = make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5, max_iter=5000, n_jobs=1)).fit(X[:, h], y)
    return lambda Z: m.predict(Z[:, h])


def m_logit(X, y):
    h = hv(X); m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)).fit(X[:, h], y.astype(int))
    return lambda Z: m.predict_proba(Z[:, h]) @ m.classes_.astype(float)


def m_knn(X, y):
    h = hv(X); m = make_pipeline(StandardScaler(), PCA(10), KNeighborsRegressor(5)).fit(X[:, h], y)
    return lambda Z: m.predict(Z[:, h])


def m_marker(X, y, n=50):
    d = X[y == 2].mean(0) - X[y == 0].mean(0); up, dn = np.argsort(d)[::-1][:n], np.argsort(d)[:n]
    sc = lambda Z: Z[:, up].mean(1) - Z[:, dn].mean(1)
    b = np.polyfit(sc(X), y, 1)
    return lambda Z: np.polyval(b, sc(Z))


MODELS = {"ridge": m_ridge, "elastic_net": m_enet, "multinomial_logistic": m_logit, "knn": m_knn, "marker_score": m_marker}


def clocks(P, M, model, tg):
    """Leave-one-donor-out for staged natural donors, full fit for the rest. Returns wide table with y."""
    out = {}
    for comp in COMPS:
        m = (M.compartment == comp) & (M.n >= MIN_N); Mc, X = M[m], P[m].to_numpy(); tr = Mc.y.notna().to_numpy(); y = Mc.y.to_numpy()
        pred = np.full(len(Mc), np.nan)
        for i in np.flatnonzero(tr):
            k = tr.copy(); k[i] = False
            pred[i] = model(X[k], y[k])(X[i:i + 1])[0]
        pred[~tr] = model(X[tr], y[tr])(X[~tr])
        out[comp] = pd.Series(pred, index=Mc.donor.values)
    D = pd.concat(out, axis=1)
    info = M.drop_duplicates("donor").set_index("donor")
    D = D.join(info[["group", "Stage"]]); D["y"] = np.where(D.group == "natural", D.Stage.map(stage_value), np.nan)
    return D[D.group.isin(["natural", tg])]


def metrics(D, comp):
    x = D[D.y.notna()].dropna(subset=[comp]); e = x[comp] - x.y
    return dict(n=len(x), mae=round(float(e.abs().mean()), 3), rmse=round(float(np.sqrt((e ** 2).mean())), 3),
                rho=round(float(spearmanr(x[comp], x.y)[0]), 3), calibration_slope=round(float(linregress(x.y, x[comp]).slope), 3),
                early_secretory_median=round(float(x[comp][x.y == 1].median()), 3) if (x.y == 1).any() else None)


def h1(D, tg):
    x = D.dropna(subset=list(COMPS)); idx = x.stroma - x.epithelium
    t, n = idx[x.group == tg], idx[x.group == "natural"]
    nat = x[x.group == "natural"]; tr = x[x.group == tg]
    rn, rt = residuals(nat.stroma, nat.epithelium, (tr.stroma, tr.epithelium))
    return dict(n_exposed=int(len(t)), index_shift=round(hodges_lehmann(t, n), 3), p_index=p1(t, n, "greater"), p_residual=p1(rt, rn, "less"))


def r15(P, M):
    Pc, Mc = load("cells")
    wang = Mc[(Mc.dataset == "Wang") & (Mc.n >= MIN_N)]
    U = sorted(set(P.columns) & set(Pc.columns))
    Re, Rw = rank_quantiles(P[U]), rank_quantiles(Pc.loc[wang.index, U])
    out = {"genes": len(U), "donors": int(wang.donor.nunique())}
    for comp in COMPS:
        m = (M.compartment == comp) & (M.n >= MIN_N) & M.y.notna()
        f = fit(Re.loc[m].to_numpy(), M.y[m].to_numpy())
        w = wang[wang.compartment == comp]; pred = pd.Series(f(Rw.loc[w.index].to_numpy()), index=w.donor.values)
        ph = w.Stage.map(WANG); ph.index = w.donor.values
        ok = ph.notna()
        out[comp] = dict(n=int(ok.sum()), rho=round(float(spearmanr(pred[ok], ph[ok])[0]), 3), mae=round(float((pred[ok] - ph[ok]).abs().mean()), 3),
                         by_phase={k: round(float(pred[(ph == v).values].median()), 3) for k, v in WANG.items() if (ph == v).any()},
                         late_secretory=[round(float(v), 3) for v in pred[(w.Stage == "Secretory Late").values]])
    return out


def r16(D, tg):
    C = D.copy(); out = {}
    for comp in COMPS:
        x = C[C.y.notna()].dropna(subset=[comp]); b = np.polyfit(x[comp], x.y, 1)
        C[comp] = np.polyval(b, C[comp]); out[f"{comp}_map"] = [round(float(b[0]), 3), round(float(b[1]), 3)]
        out[f"{comp}_early_secretory_after"] = round(float(C[comp][C.y == 1].median()), 3)
    out.update(h1(C, tg)); out["exposed_medians"] = {c: round(float(C[c][C.group == tg].median()), 3) for c in COMPS}
    return out


def r17(P, M, tg, seed=0):
    rng = np.random.default_rng(seed); info = M.drop_duplicates("donor").set_index("donor")
    data = {}
    for comp in COMPS:
        m = (M.compartment == comp) & (M.n >= MIN_N); data[comp] = (M[m], P[m].to_numpy())
    staged = info.index[info.group.eq("natural") & info.Stage.map(stage_value).notna()]
    yst = info.Stage.map(stage_value)[staged]
    reps, per = [], []
    for b in range(N_BOOT):
        bag = np.concatenate([rng.choice(yst.index[yst == s], (yst == s).sum(), replace=True) for s in sorted(yst.unique())])
        pr = {}
        for comp, (Mc, X) in data.items():
            idx = [np.flatnonzero(Mc.donor.values == d2)[0] for d2 in bag if (Mc.donor.values == d2).any()]
            f = fit(X[idx], Mc.y.to_numpy()[idx]); pred = pd.Series(f(X), index=Mc.donor.values)
            pred[pred.index.isin(set(bag))] = np.nan
            pr[comp] = pred
        D = pd.concat(pr, axis=1).join(info[["group"]]); D = D[D.group.isin(["natural", tg])].dropna(subset=list(COMPS))
        idx = D.stroma - D.epithelium; t, n = idx[D.group == tg], idx[D.group == "natural"]
        reps.append(dict(shift=hodges_lehmann(t, n), p=mannwhitneyu(t, n, alternative="greater").pvalue))
        per.append(idx[D.group == tg].rename(b))
    R = pd.DataFrame(reps); Pd = pd.concat(per, axis=1)
    donors = pd.DataFrame({"median": Pd.median(1), "lo": Pd.quantile(0.025, axis=1), "hi": Pd.quantile(0.975, axis=1)})
    return dict(replicates=N_BOOT, shift_median=round(float(R["shift"].median()), 3),
                shift_interval=[round(float(R["shift"].quantile(0.025)), 3), round(float(R["shift"].quantile(0.975)), 3)],
                p_median=float(f"{R.p.median():.3g}"), share_p_below_005=round(float((R.p < 0.05).mean()), 3),
                exposed_with_interval_above_0=f"{int((donors.lo > 0).sum())}/{len(donors)}"), donors


if __name__ == "__main__":
    summ = {}
    for s, tg in (("nuclei", "progestin_only"), ("cells", "combined_pill")):
        P, M = load(s); P = P.loc[:, (P > 0).mean(0) > 0.5]
        res = {}
        models = {}
        for name, mdl in MODELS.items():
            D = clocks(P, M, mdl, tg); models[name] = D
            res[name] = {c: metrics(D, c) for c in COMPS} | {"h1": h1(D, tg)}
        summ[s] = dict(r18_models=res, r16_calibration=r16(models["ridge"], tg))
        boot, donors = r17(P, M, tg); summ[s]["r17_bootstrap"] = boot
        donors.round(4).rename_axis("donor").to_csv(os.path.join(OUT, f"s15_bootstrap_donors_{s}.csv"))
        if s == "nuclei":
            summ["r15_wang"] = r15(P, M)
        print(s, "done", flush=True)
    json.dump(summ, open(os.path.join(OUT, "s15_clock_rigor.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))
