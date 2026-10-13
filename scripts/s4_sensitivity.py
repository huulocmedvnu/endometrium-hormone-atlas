"""S4: paper sensitivity analyses S1 to S5 (PLAN.md, paper extension).
S1 position-matched dyssynchrony: epithelium clock regressed on stroma clock in natural donors (leave-one-donor-out
residuals), treated donors scored by the fit on all natural donors. S2 without endometriosis. S3 per progestin and
leave-one-drug-out. S4 label permutations. S5 clock variants (variable genes, minimum nuclei, compartment-specific
genes). Writes outputs/s4_<set>_summary.json and outputs/s4_<set>_donors.csv.
Usage: s4_sensitivity.py nuclei|cells [treated group, default progestin_only]"""
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
from s2_clock import fit, lodo, stage_value, treatment_group, ROOT, W, N_WOI

COMPS = ("stroma", "epithelium")
N_PERM = 10000


def drug(h):
    h = str(h)
    if h.startswith("Mirena"):
        return "LNG-IUS"
    if h.startswith("Implant"):
        return "implant"
    if h.startswith(("POP", "Desogestrel")):
        return "desogestrel"
    return h


def residuals(x_nat, y_nat, x_new):
    """Leave-one-out residuals of y on x (linear) for the natural donors, and residuals of new donors from the fit on
    all natural donors."""
    x_nat, y_nat = np.asarray(x_nat, float), np.asarray(y_nat, float)
    r = np.empty(len(x_nat))
    for i in range(len(x_nat)):
        m = np.arange(len(x_nat)) != i
        b = np.polyfit(x_nat[m], y_nat[m], 1); r[i] = y_nat[i] - np.polyval(b, x_nat[i])
    b = np.polyfit(x_nat, y_nat, 1)
    return r, np.asarray(x_new[1], float) - np.polyval(b, np.asarray(x_new[0], float))


def perm_p(a, b, n=N_PERM, seed=0, less=False):
    """One-sided permutation p for mean(a) > mean(b) (or < with less=True), (k + 1) / (n + 1)."""
    a, b = np.asarray(a, float), np.asarray(b, float); x = np.concatenate([a, b]); rng = np.random.default_rng(seed)
    obs = a.mean() - b.mean(); k = 0
    for _ in range(n):
        rng.shuffle(x); d = x[:len(a)].mean() - x[len(a):].mean()
        k += (d <= obs) if less else (d >= obs)
    return (k + 1) / (n + 1)


def specific_genes(P, M, frac=0.1):
    """Top decile of genes by share of mean natural-donor expression in each compartment."""
    nat = M.group == "natural"
    mu = pd.DataFrame({c: P[nat & (M.compartment == c)].mean() for c in ("stroma", "epithelium", "endothelium", "immune")})
    share = mu.div(mu.sum(1).replace(0, np.nan), axis=0).fillna(0)
    return {c: set(share[c].nlargest(int(len(share) * frac)).index) for c in COMPS}


def clocks(P, M, n_hvg=2000, min_n=50, genes=None):
    rows = []
    for comp in COMPS:
        m = (M.compartment == comp) & (M.n >= min_n); Mc = M[m]
        X = (P.loc[m, sorted(genes[comp])] if genes else P[m]).to_numpy()
        tr = Mc.y.notna().to_numpy(); y = Mc.y.to_numpy()
        pred = np.full(len(Mc), np.nan); pred[tr] = lodo_h(X[tr], y[tr], n_hvg)
        pred[~tr] = fit(X[tr], y[tr], n_hvg)(X[~tr])
        rows.append(pd.DataFrame({"donor": Mc.donor, "compartment": comp, "clock": pred}))
    D = pd.concat(rows).pivot_table(index="donor", columns="compartment", values="clock")
    info = M.drop_duplicates("donor").set_index("donor")[["group", "Hormonal treatment", "Stage", "Endometrial_pathology"]]
    D = D.join(info).dropna(subset=list(COMPS)); D["dyssynchrony"] = D.stroma - D.epithelium
    return D


def lodo_h(X, y, n_hvg):
    out = np.empty(len(y))
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        out[i] = fit(X[m], y[m], n_hvg)(X[i:i + 1])[0]
    return out


def tests(D, tg, perm=False):
    nat, tr = D[D.group == "natural"], D[D.group == tg]
    if len(tr) < 3:
        return dict(n_treated=len(tr)), pd.Series(dtype=float)
    r_nat, r_tr = residuals(nat.stroma, nat.epithelium, (tr.stroma, tr.epithelium))
    sec = nat[nat.Stage.astype(str).str.startswith("Secretory")]
    out = dict(n_treated=len(tr), n_natural=len(nat), n_natural_secretory=len(sec),
               dys_median_treated=round(float(tr.dyssynchrony.median()), 3),
               p_h1=float(f"{mannwhitneyu(tr.dyssynchrony, nat.dyssynchrony, alternative='greater').pvalue:.3g}"),
               p_vs_secretory=float(f"{mannwhitneyu(tr.dyssynchrony, sec.dyssynchrony, alternative='greater').pvalue:.3g}"),
               residual_median_treated=round(float(np.median(r_tr)), 3), residual_median_natural=round(float(np.median(r_nat)), 3),
               p_s1_below_0=float(f"{wilcoxon(r_tr, alternative='less').pvalue:.3g}"),
               p_s1_vs_natural=float(f"{mannwhitneyu(r_tr, r_nat, alternative='less').pvalue:.3g}"))
    out["s1_passed"] = bool(out["p_s1_below_0"] < 0.05 and out["p_s1_vs_natural"] < 0.05)
    if perm:
        out["perm_h1"] = perm_p(tr.dyssynchrony, nat.dyssynchrony); out["perm_s1"] = perm_p(r_tr, r_nat, less=True)
    return out, pd.Series(np.r_[r_nat, r_tr], index=list(nat.index) + list(tr.index))


def woi_test(P, M, tg, keep):
    ep = M[(M.compartment == "epithelium") & (M.n >= 50) & M.donor.isin(keep)]
    mid, p0 = ep.index[(ep.group == "natural") & (ep.y == 2)], ep.index[(ep.group == "natural") & (ep.y == 0)]
    g = (P.loc[mid].mean() - P.loc[p0].mean()).nlargest(N_WOI).index; sc = P.loc[ep.index, g].mean(1); t = ep.index[ep.group == tg]
    return dict(n_treated=len(t), n_mid=len(mid), p=float(f"{mannwhitneyu(sc[t], sc[mid], alternative='less').pvalue:.3g}"))


if __name__ == "__main__":
    s = sys.argv[1] if len(sys.argv) > 1 else "nuclei"
    tg = sys.argv[2] if len(sys.argv) > 2 else "progestin_only"; tag = s if tg == "progestin_only" else f"{s}_{tg}"
    P = pd.read_parquet(os.path.join(W, f"profiles_{s}.parquet")); M = pd.read_csv(os.path.join(W, f"profiles_{s}_meta.csv"), index_col=0)
    M = M.loc[P.index]; M["group"] = M["Hormonal treatment"].map(treatment_group); M["y"] = M.Stage.map(stage_value)
    M.loc[M.group != "natural", "y"] = np.nan
    P = P.loc[:, (P > 0).mean(0) > 0.5]
    summ = {"set": s, "treated_group": tg}
    D = clocks(P, M)
    summ["main"], res = tests(D, tg, perm=True)
    D["residual"] = res
    # S2 endometriosis
    endo = D.Endometrial_pathology.astype(str).str.contains("E")
    summ["s2_no_endometriosis"] = tests(D[~endo], tg)[0]
    summ["s2_endometriosis_only"] = tests(D[endo], tg)[0]
    summ["s2_woi_no_endometriosis"] = woi_test(P, M, tg, set(M.donor[~M.Endometrial_pathology.astype(str).str.contains("E")]))
    # S3 per drug
    tr = D[D.group == tg].assign(drug=lambda x: x["Hormonal treatment"].map(drug))
    summ["s3_per_drug"] = tr.groupby("drug").agg(n=("stroma", "size"), stroma=("stroma", "median"), epithelium=("epithelium", "median"),
                                                  dyssynchrony=("dyssynchrony", "median"), residual=("residual", "median")).round(3).to_dict("index")
    summ["s3_leave_one_drug_out"] = {d: tests(D.drop(tr.index[tr.drug == d]), tg)[0] for d in tr.drug.unique()}
    # S5 clock variants
    spec = specific_genes(P, M)
    variants = {"hvg1000": dict(n_hvg=1000), "hvg5000": dict(n_hvg=5000), "min100": dict(min_n=100),
                "specific_genes": dict(genes=spec, n_hvg=1000)}
    summ["s5"] = {k: tests(clocks(P, M, **v), tg)[0] for k, v in variants.items()}
    D.to_csv(os.path.join(ROOT, "outputs", f"s4_{tag}_donors.csv"), float_format="%.4g")
    json.dump(summ, open(os.path.join(ROOT, "outputs", f"s4_{tag}_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))
