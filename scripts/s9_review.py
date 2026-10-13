"""S9: review response analyses R1 to R8 (PLAN.md, review response, all post hoc).
Writes outputs/s9_review.json and outputs/s9_*.csv. Usage: s9_review.py"""
import json, os
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy.stats import mannwhitneyu, wilcoxon, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from s2_clock import stage_value, treatment_group, ROOT, W, MIN_N, N_HVG, ALPHAS, fit
from s4_sensitivity import residuals, lodo_h
from s6_external_huang import zscore, huang_profiles, heca, COMPS
from s7_external_bulk import series_matrix, gpl6244_symbols

OUT = os.path.join(ROOT, "outputs")
SETS = {"nuclei": "progestin_only", "cells_combined_pill": "combined_pill"}
PROFILE = {"nuclei": "nuclei", "cells_combined_pill": "cells"}


def pv(p):
    return None if p is None or not np.isfinite(p) else float(f"{p:.3g}")


def mw(a, b, alt):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    return pv(mannwhitneyu(a, b, alternative=alt).pvalue) if len(a) > 1 and len(b) > 1 else None


def holm(ps):
    """Holm step-down adjusted P values, same order as input (None kept)."""
    idx = [i for i, p in enumerate(ps) if p is not None]
    order = sorted(idx, key=lambda i: ps[i]); m = len(order); adj = [None] * len(ps); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i])); adj[i] = run
    return adj


def load(tag):
    P = pd.read_parquet(os.path.join(W, f"profiles_{PROFILE[tag]}.parquet")); M = pd.read_csv(os.path.join(W, f"profiles_{PROFILE[tag]}_meta.csv"), index_col=0)
    M = M.loc[P.index]; M["group"] = M["Hormonal treatment"].map(treatment_group); M["y"] = M.Stage.map(stage_value)
    M.loc[M.group != "natural", "y"] = np.nan
    return P.loc[:, (P > 0).mean(0) > 0.5], M


def r1_all_compartments(tag, tg):
    d = pd.read_csv(os.path.join(OUT, f"s2_{tag}_donors.csv"))
    Wd = d.pivot_table(index=["donor", "group"], columns="compartment", values="clock").reset_index()
    out = {}
    core = ["stroma", "epithelium", "immune"]
    for x in core + ["endothelium"]:
        others = [c for c in core if c != x]
        D = Wd.dropna(subset=[x] + others).copy(); D["ref"] = D[others].mean(1)
        nat, tr = D[D.group == "natural"], D[D.group == tg]
        if len(tr) < 3:
            out[x] = dict(n_treated=len(tr)); continue
        rn, rt = residuals(nat.ref, nat[x], (tr.ref, tr[x]))
        out[x] = dict(n_treated=len(tr), n_natural=len(nat), residual_treated=round(float(np.median(rt)), 3),
                      residual_natural=round(float(np.median(rn)), 3), p_two_sided=mw(rt, rn, "two-sided"))
    return out


def r2_staged(tag, tg):
    d = pd.read_csv(os.path.join(OUT, f"s4_{tag}_donors.csv"))
    st = d.Stage.map(stage_value)
    nat, tr = d[(d.group == "natural") & st.notna()], d[d.group == tg]
    rn, rt = residuals(nat.stroma, nat.epithelium, (tr.stroma, tr.epithelium))
    return dict(n_natural_staged=len(nat), residual_treated=round(float(np.median(rt)), 3), residual_natural=round(float(np.median(rn)), 3),
                p=mw(rt, rn, "less"))


def r2_huang_lodo():
    Ph, Mh = huang_profiles(); Pe, Me = heca()
    genes = sorted(set(Pe.columns) & set(Ph.columns))
    cyc = Mh.drop_duplicates("donor").set_index("donor").cycle
    nat_lodo, D = {}, pd.DataFrame(index=cyc.index)
    for comp in COMPS:
        m, mh = (Me.compartment == comp) & (Me.n >= MIN_N) & Me.y.notna(), (Mh.compartment == comp) & (Mh.n >= MIN_N)
        Xe, Xh, y = zscore(Pe.loc[m, genes].to_numpy()), zscore(Ph.loc[mh, genes].to_numpy()), Me.y[m].to_numpy()
        pred = np.empty(len(y))
        for i in range(len(y)):
            k = np.arange(len(y)) != i
            pred[i] = fit(Xe[k], y[k])(Xe[i:i + 1])[0]
        nat_lodo[comp] = pd.Series(pred, index=Me.donor[m].values)
        D[comp] = pd.Series(fit(Xe, y)(Xh), index=Mh.donor[mh].values)
    both = pd.concat(nat_lodo, axis=1).dropna(); b = np.polyfit(both.stroma, both.epithelium, 1)
    D["residual"] = D.epithelium - np.polyval(b, D.stroma); D["cycle"] = cyc
    return dict(p_e3_lodo_line=mw(D.residual[D.cycle == "Controlled"], D.residual[D.cycle == "Natural"], "less"),
                residual_programmed=round(float(D.residual[D.cycle == "Controlled"].median()), 3),
                residual_natural=round(float(D.residual[D.cycle == "Natural"].median()), 3))


def r3_ood(tag, tg):
    P, M = load(tag); out = {}
    for comp in COMPS:
        m = (M.compartment == comp) & (M.n >= MIN_N); Mc = M[m]; X = P[m].to_numpy()
        nat = (Mc.group == "natural").to_numpy(); tr = (Mc.group == tg).to_numpy()
        hv = np.argsort(X[nat].var(0))[::-1][:N_HVG]; mu, sd = X[nat][:, hv].mean(0), X[nat][:, hv].std(0); sd[sd == 0] = 1
        Z = (X[:, hv] - mu) / sd; pca = PCA(10).fit(Z[nat]); S = pca.transform(Z) / np.sqrt(pca.explained_variance_)
        dist = np.sqrt((S ** 2).sum(1))
        q95 = np.percentile(dist[nat], 95)
        out[comp] = dict(natural_median=round(float(np.median(dist[nat])), 2), treated_median=round(float(np.median(dist[tr])), 2),
                         natural_q95=round(float(q95), 2), treated_above_q95=f"{int((dist[tr] > q95).sum())}/{int(tr.sum())}")
    return out


def r4_joint(tag, tg):
    d = pd.read_csv(os.path.join(OUT, f"s4_{tag}_donors.csv")).set_index("donor")
    P, M = load(tag)
    info = M.drop_duplicates("donor").set_index("donor")
    n = M[M.compartment.isin(COMPS)].groupby("donor").n.min()
    d = d[d.group.isin(["natural", tg])].copy()
    d["treated"] = (d.group == tg).astype(float)
    d["endometriosis"] = d.Endometrial_pathology.astype(str).str.contains("E").astype(float)
    d["age"] = pd.to_numeric(info.Age.reindex(d.index), errors="coerce")
    d["log_nuclei"] = np.log(n.reindex(d.index).astype(float))
    d["study_tan"] = (info.dataset.reindex(d.index).astype(str) == "Tan").astype(float)
    out = {}
    base = d.dropna(subset=["residual", "treated", "endometriosis"]).index
    for name, cov in (("treatment_only", ["treated"]), ("treatment_endometriosis", ["treated", "endometriosis"]),
                      ("plus_age_nuclei", ["treated", "endometriosis", "age", "log_nuclei"]),
                      ("plus_study", ["treated", "endometriosis", "age", "log_nuclei", "study_tan"])):
        if name == "plus_study" and d.study_tan.nunique() < 2:
            continue
        x = d.loc[base] if name == "treatment_only" else d.dropna(subset=["residual"] + cov)
        f = sm.OLS(x.residual, sm.add_constant(x[cov])).fit(cov_type="HC3")
        ci = f.conf_int().loc["treated"]
        out[name] = dict(n=int(len(x)), treated_coef=round(float(f.params["treated"]), 3), ci=[round(float(ci[0]), 3), round(float(ci[1]), 3)],
                         p=pv(f.pvalues["treated"]))
        if "endometriosis" in cov:
            out[name].update(endometriosis_coef=round(float(f.params["endometriosis"]), 3), endometriosis_p=pv(f.pvalues["endometriosis"]))
    return out


def r4_factorial(tag, tg):
    """Stroma minus epithelium index and distance from the natural line by endometriosis and exposure (natural or tg)."""
    d = pd.read_csv(os.path.join(OUT, f"s4_{tag}_donors.csv"))
    d = d[d.group.isin(["natural", tg])].dropna(subset=["dyssynchrony"])
    d["endometriosis"] = d.Endometrial_pathology.astype(str).str.contains("E")
    cell = lambda g, e: d[(d.group == g) & (d.endometriosis == e)]
    out = {f"{g}_{'endometriosis' if e else 'no_endometriosis'}": dict(n=len(cell(g, e)), index_median=round(float(cell(g, e).dyssynchrony.median()), 3),
                                                                        distance_median=round(float(cell(g, e).residual.median()), 3))
           for g in ("natural", tg) for e in (True, False)}
    ne, nn = cell("natural", True), cell("natural", False)
    out["p_natural_endometriosis_vs_none_index"] = mw(ne.dyssynchrony, nn.dyssynchrony, "two-sided")
    out["p_natural_endometriosis_vs_none_distance"] = mw(ne.residual, nn.residual, "two-sided")
    out["p_exposed_no_endometriosis_index_above_natural_no_endometriosis"] = mw(cell(tg, False).dyssynchrony, nn.dyssynchrony, "greater")
    return out


def min_nuclei_clocks(P, M, tg, min_n, train_min_n=MIN_N):
    """Stroma and epithelium clocks as in s4: trained on staged natural donors with at least train_min_n nuclei
    (leave-one-donor-out for them), other natural donors with at least train_min_n and exposed donors (tg) with at
    least min_n nuclei scored by the full fit. Returns donor x [group, stroma, epithelium, dyssynchrony] for donors
    with both compartments."""
    rows = []
    for comp in ("stroma", "epithelium"):
        c = M[(M.compartment == comp) & (((M.group == "natural") & (M.n >= train_min_n)) | ((M.group == tg) & (M.n >= min_n)))]
        tr = c.y.notna().to_numpy(); X, y = P.loc[c.index].to_numpy(), c.y.to_numpy()
        pred = np.full(len(c), np.nan); pred[tr] = lodo_h(X[tr], y[tr], N_HVG); pred[~tr] = fit(X[tr], y[tr])(X[~tr])
        rows.append(pd.DataFrame({"donor": c.donor.values, "group": c.group.values, "compartment": comp, "clock": pred}))
    D = pd.concat(rows).pivot_table(index=["donor", "group"], columns="compartment", values="clock").reset_index().dropna()
    D["dyssynchrony"] = D.stroma - D.epithelium
    return D.set_index("donor")


def r10_min_nuclei(tag, tg, thresholds=(10, 20, 30, 50)):
    """H1 (stroma minus epithelium index, exposed above natural) and position-matched residual (exposed below natural)
    with the minimum nuclei per compartment lowered for exposed donors only. Training and natural donors stay at MIN_N."""
    P, M = load(tag); out = {"training_min_n": MIN_N}
    for t in thresholds:
        D = min_nuclei_clocks(P, M, tg, t)
        nat, tr = D[D.group == "natural"], D[D.group == tg]
        rn, rt = residuals(nat.stroma, nat.epithelium, (tr.stroma, tr.epithelium))
        out[f"min{t}"] = dict(n_exposed=len(tr), n_natural=len(nat), exposed_donors=sorted(tr.index),
                              index_median_exposed=round(float(tr.dyssynchrony.median()), 3), index_median_natural=round(float(nat.dyssynchrony.median()), 3),
                              p_h1=mw(tr.dyssynchrony, nat.dyssynchrony, "greater"),
                              residual_median_exposed=round(float(np.median(rt)), 3), residual_median_natural=round(float(np.median(rn)), 3),
                              p_residual=mw(rt, rn, "less"))
    return out


def axis_genes(P, M, comp, n, up=True):
    nat = M[(M.compartment == comp) & (M.n >= MIN_N) & (M.group == "natural")]
    p0, p2 = P.loc[nat.index[nat.y == 0]].mean(), P.loc[nat.index[nat.y == 2]].mean()
    d = p2 - p0
    return d[d > 0.1].nlargest(n).index if up else d[d < -0.1].nsmallest(n).index


def r5_cross(tag, tg):
    other = "cells_combined_pill" if tag == "nuclei" else "nuclei"
    P, M = load(tag); Po, Mo = load(other)
    shared = P.columns.intersection(Po.columns)
    out = {}
    woi = axis_genes(Po[shared], Mo, "epithelium", 50)
    ep = M[(M.compartment == "epithelium") & (M.n >= MIN_N)]
    sc = P.loc[ep.index, woi].mean(1)
    t, mid = sc[(ep.group == tg).values], sc[((ep.group == "natural") & (ep.y == 2)).values]
    out["woi_genes_from_other_set"] = dict(genes=len(woi), treated=round(float(t.median()), 3), natural_mid=round(float(mid.median()), 3), p=mw(t, mid, "less"))
    fr = {}
    for comp in COMPS:
        g = axis_genes(Po[shared], Mo, comp, 200)
        c = M[(M.compartment == comp) & (M.n >= MIN_N)]
        mu0, mu2 = P.loc[c.index[(c.group == "natural") & (c.y == 0)], g].mean(), P.loc[c.index[(c.group == "natural") & (c.y == 2)], g].mean()
        T = P.loc[c.index[c.group == tg], g]
        fr[comp] = pd.Series(((T - mu0) / (mu2 - mu0)).median(1).values, index=c.donor[c.group == tg].values)
    F = pd.concat(fr, axis=1).dropna()
    out["donor_level_fraction"] = dict(n=len(F), stroma_median=round(float(F.stroma.median()), 3), epithelium_median=round(float(F.epithelium.median()), 3),
                                       p_paired_stroma_greater=pv(wilcoxon(F.stroma, F.epithelium, alternative="greater").pvalue),
                                       donors_epithelium_lower=f"{int((F.epithelium < F.stroma).sum())}/{len(F)}")
    return out


def r6_ambient(tag, tg):
    P, M = load(tag); out = {}
    for comp, genes in (("stroma", ["EPCAM", "KRT8", "KRT18"]), ("epithelium", ["PDGFRB", "DCN", "COL1A2"])):
        c = M[(M.compartment == comp) & (M.n >= MIN_N)]
        for g in genes:
            if g in P.columns:
                t, n = P.loc[c.index[c.group == tg], g], P.loc[c.index[c.group == "natural"], g]
                out[f"{g}_in_{comp}"] = dict(treated=round(float(t.median()), 3), natural=round(float(n.median()), 3), p_two_sided=mw(t, n, "two-sided"))
    return out


EPI = ["EPCAM", "CDH1", "KRT8", "KRT18", "KRT19", "CLDN3"]
STR = ["PDGFRB", "DCN", "COL1A2", "COL3A1", "LUM"]


def r7_bulk():
    S = pd.read_csv(os.path.join(OUT, "s7_bulk_samples.csv")); out = {}; Xs = {}
    for gse, tr in (("GSE60129", ["LNG-IUS", "DMPA"]), ("GSE137765", ["LNG-IUS", "COC"])):
        X, meta = series_matrix(gse); X = gpl6244_symbols(X).T
        d = S[S.set == gse].set_index("sample"); X = X.loc[d.index]; Xs[gse] = X
        Z = pd.DataFrame(zscore(X.to_numpy()), index=X.index, columns=X.columns)
        d["epi_content"] = Z[[g for g in EPI if g in Z]].mean(1); d["str_content"] = Z[[g for g in STR if g in Z]].mean(1)
        d["treated"] = d.group.isin(tr).astype(float)
        r = dict(epi_content_p_two_sided=mw(d.epi_content[d.treated == 1], d.epi_content[d.treated == 0], "two-sided"),
                 epi_content_treated=round(float(d.epi_content[d.treated == 1].median()), 3), epi_content_control=round(float(d.epi_content[d.treated == 0].median()), 3))
        for y in ("dyssynchrony", "epithelium"):
            f = sm.OLS(d[y], sm.add_constant(d[["treated", "epi_content", "str_content"]])).fit(cov_type="HC3")
            ci = f.conf_int().loc["treated"]
            r[f"{y}_adjusted"] = dict(treated_coef=round(float(f.params["treated"]), 3), ci=[round(float(ci[0]), 3), round(float(ci[1]), 3)], p=pv(f.pvalues["treated"]))
        out[gse] = r
    a, b = Xs["GSE60129"], Xs["GSE137765"]; g = a.columns.intersection(b.columns)
    A = a[g].sub(a[g].mean()); B = b[g].sub(b[g].mean())
    C = np.corrcoef(np.vstack([A.to_numpy(), B.to_numpy()]))
    na = len(A); cross = C[:na, na:]; within = C[na:, na:].copy(); np.fill_diagonal(within, np.nan)
    out["shared_controls_check"] = dict(max_cross_correlation=round(float(np.nanmax(cross)), 3), median_within_max=round(float(np.nanmedian(np.nanmax(within, 1))), 3),
                                        pairs_above_0_9=int((cross > 0.9).sum()))
    return out


if __name__ == "__main__":
    R = {}
    for tag, tg in SETS.items():
        R[tag] = dict(r1_all_compartments=r1_all_compartments(tag, tg), r2_staged_only=r2_staged(tag, tg), r3_ood=r3_ood(tag, tg),
                      r4_joint=r4_joint(tag, tg), r4_factorial=r4_factorial(tag, tg), r5_cross_dataset=r5_cross(tag, tg), r6_ambient=r6_ambient(tag, tg),
                      r10_min_nuclei=r10_min_nuclei(tag, tg))
    R["huang_r2_lodo_line"] = r2_huang_lodo()
    R["bulk_r7"] = r7_bulk()
    n2, c2 = json.load(open(os.path.join(OUT, "s2_nuclei_summary.json"))), json.load(open(os.path.join(OUT, "s2_cells_combined_pill_summary.json")))
    n4, c4 = json.load(open(os.path.join(OUT, "s4_nuclei_summary.json"))), json.load(open(os.path.join(OUT, "s4_cells_combined_pill_summary.json")))
    h, b = json.load(open(os.path.join(OUT, "s6_huang_summary.json"))), json.load(open(os.path.join(OUT, "s7_bulk_summary.json")))
    fam = {"index": [n2["h1"]["p_dyssynchrony_higher"], c2["h1"]["p_dyssynchrony_higher"], h["unscaled"]["p_e1_dyssynchrony_higher"],
                     b["GSE60129"]["p_dyssynchrony_higher"], b["GSE137765"]["p_dyssynchrony_higher"]],
           "delay": [n4["main"]["p_s1_vs_natural"], c4["main"]["p_s1_vs_natural"], h["unscaled"]["p_e1_dyssynchrony_higher"],
                     b["GSE60129"]["p_dyssynchrony_higher"], b["GSE137765"]["p_dyssynchrony_higher"]],
           "woi": [n2["h2"]["p_treated_lower_than_mid"], c2["h2"]["p_treated_lower_than_mid"], h["unscaled"]["p_e2_woi_lower"],
                   b["GSE60129"]["p_woi_lower"], b["GSE137765"]["p_woi_lower"]]}
    R["r8_holm"] = {k: [pv(x) for x in holm(v)] for k, v in fam.items()}
    R["h1_subtests"] = {"nuclei": {k: n2["h1"][k] for k in ("p_stroma_above_1", "p_epithelium_below_1")},
                        "cells_combined_pill": {k: c2["h1"][k] for k in ("p_stroma_above_1", "p_epithelium_below_1")}}
    json.dump(R, open(os.path.join(OUT, "s9_review.json"), "w"), indent=1)
    print(json.dumps(R, indent=1))
