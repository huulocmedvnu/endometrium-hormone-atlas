"""S13 (R12 and R13, post hoc).
R12: Huang 2026 programmed against natural cycles on within-profile rank quantiles, clocks trained on HECA snRNA-seq
natural profiles in rank space, EC0 transfer check, E1 index, E2 WOI score, E3 residual against the HECA natural line.
R13: NNLS deconvolution with HECA natural-donor signatures (from s12), checked on HECA donors with known nuclei
fractions, applied to GSE137765 and the natural arrays, R9 residual adjusted for the epithelial fraction.
Writes outputs/s13_rank_huang_deconv.json and outputs/s13_deconvolution.csv. Usage: s13_rank_huang_deconv.py"""
import json, os
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy.optimize import nnls
from scipy.stats import mannwhitneyu, spearmanr
from s2_clock import fit, lodo, ROOT, W, MIN_N
from s6_external_huang import huang_profiles, heca, COMPS
from s7_external_bulk import series_matrix, gpl6244_symbols
from s10_rank_bulk import rank_quantiles, load_array, phase_of, PHASE, ENDO, NAT

OUT = os.path.join(ROOT, "outputs")
N_MARK = 50
GROUPS = ["stroma", "epithelium", "endothelium", "immune", "pv_smc"]


def mw(a, b, alt):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    return float(f"{mannwhitneyu(a, b, alternative=alt).pvalue:.3g}") if len(a) > 1 and len(b) > 1 else None


def markers(S, n=N_MARK, min_level=1.0):
    """Top n genes by share of signal per group, among genes reaching min_level in that group."""
    share = S.div(S.sum(0), axis=1)
    return sorted({g for grp in S.index for g in share.loc[grp][S.loc[grp] >= min_level].nlargest(n).index})


def deconvolve(S, mix):
    """NNLS fractions per sample. S groups x genes (linear), mix samples x genes (linear), rows scaled by S gene mean."""
    g = [x for x in S.columns if x in mix.columns]; sc = S[g].mean(0).replace(0, np.nan)
    A = (S[g] / sc).T.fillna(0).to_numpy(); out = []
    for s, row in mix[g].iterrows():
        f, _ = nnls(A, (row / sc).fillna(0).to_numpy()); out.append(f / f.sum() if f.sum() > 0 else f)
    return pd.DataFrame(out, index=mix.index, columns=S.index)


def r12():
    Ph, Mh = huang_profiles(); Pe, Me = heca()
    U = sorted(set(Pe.columns) & set(Ph.columns))
    Re, Rh = rank_quantiles(Pe[U]), rank_quantiles(Ph[U])
    cyc = Mh.drop_duplicates("donor").set_index("donor").cycle
    D, nat_lodo = pd.DataFrame(index=cyc.index), {}
    for comp in COMPS:
        m, mh = (Me.compartment == comp) & (Me.n >= MIN_N) & Me.y.notna(), (Mh.compartment == comp) & (Mh.n >= MIN_N)
        X, y = Re.loc[m].to_numpy(), Me.y[m].to_numpy()
        nat_lodo[comp] = pd.Series(lodo(X, y), index=Me.donor[m].values)
        D[comp] = pd.Series(fit(X, y)(Rh.loc[mh].to_numpy()), index=Mh.donor[mh].values)
    s2 = json.load(open(os.path.join(OUT, "s2_nuclei_summary.json")))
    woi = [g for g in s2["h2"]["genes"] if g in U]
    ep = Mh[(Mh.compartment == "epithelium") & (Mh.n >= MIN_N)]
    D["woi"] = pd.Series(Rh.loc[ep.index, woi].mean(1).values, index=ep.donor.values)
    both = pd.concat(nat_lodo, axis=1).dropna(); b = np.polyfit(both.stroma, both.epithelium, 1)
    D["index"] = D.stroma - D.epithelium; D["residual"] = D.epithelium - np.polyval(b, D.stroma); D["cycle"] = cyc
    nat, pc = D[D.cycle == "Natural"], D[D.cycle == "Controlled"]
    return dict(genes=len(U), heca_lodo_rho={c: round(float(spearmanr(nat_lodo[c], Me.y[(Me.compartment == c) & (Me.n >= MIN_N) & Me.y.notna()])[0]), 3) for c in COMPS},
                ec0={c: dict(natural_median=round(float(nat[c].median()), 3), passed=bool(nat[c].median() > 1)) for c in COMPS},
                medians={k: {c: round(float(g[c].median()), 3) for c in ("stroma", "epithelium", "index", "residual", "woi")} for k, g in (("natural", nat), ("programmed", pc))},
                n=dict(natural=int(nat["index"].notna().sum()), programmed=int(pc["index"].notna().sum())),
                p_e1_index=mw(pc["index"], nat["index"], "greater"), p_e2_woi=mw(pc.woi, nat.woi, "less"), p_e3_residual=mw(pc.residual, nat.residual, "less"),
                p_stroma_two_sided=mw(pc.stroma, nat.stroma, "two-sided"), p_epithelium_two_sided=mw(pc.epithelium, nat.epithelium, "two-sided"))


def r13():
    S = pd.read_parquet(os.path.join(W, "signature_nuclei.parquet")).loc[GROUPS]
    mk = markers(S); Sm = S[mk]; out = {"markers": len(mk)}
    # check on HECA donors: approximate linear whole-tissue profiles from compartment means and nuclei counts
    P = pd.read_parquet(os.path.join(W, "profiles_nuclei.parquet")); M = pd.read_csv(os.path.join(W, "profiles_nuclei_meta.csv"), index_col=0).loc[P.index]
    lin = np.expm1(P[[g for g in mk if g in P.columns]])
    wt = lin.mul(M.n, axis=0).groupby(M.donor).sum().div(M.groupby("donor").n.sum(), axis=0)
    true = M.pivot_table(index="donor", columns="compartment", values="n").fillna(0); true = true.div(true.sum(1), axis=0)
    est = deconvolve(Sm, wt)
    out["heca_check_rho"] = {c: round(float(spearmanr(est[c], true[c].reindex(est.index))[0]), 3) for c in ("stroma", "epithelium", "immune")}
    rows = []
    X, meta = series_matrix("GSE137765"); X = gpl6244_symbols(X).T
    src = pd.Series(meta.source_name_ch1.values, index=meta.geo_accession.values)
    keep = src[src.str.contains("endometrial")].index
    f = deconvolve(Sm, 2 ** X.loc[keep]); f["set"] = "gse137765"; f["group"] = src[keep].str.replace("endometrial biopsy using ", "").values; rows.append(f)
    g570 = os.path.join(NAT, "GPL570.annot.gz")
    for gse in ("GSE4888", "GSE6364"):
        A, m = load_array(gse, g570)
        A = 2 ** A if A.max().max() < 30 else A
        ph = pd.concat([phase_of(m, PHASE)]); en = phase_of(m, ENDO)
        f = deconvolve(Sm, A); f["set"] = gse.lower(); f["group"] = np.where(ph.reindex(f.index).notna(), "natural", np.where(en.reindex(f.index).notna(), "endometriosis", "other"))
        f["phase"] = ph.reindex(f.index).fillna(en.reindex(f.index)); rows.append(f)
    F = pd.concat(rows)
    R9 = pd.read_csv(os.path.join(OUT, "s10_rank_bulk_samples.csv"), index_col=0)
    F = F.join(R9[["residual", "epithelium", "stroma"]].rename(columns={"epithelium": "epi_clock", "stroma": "str_clock"}))
    d = F[F.set == "gse137765"].copy(); tr = d.group.isin(["LNG-IUD", "COC"]); ct = d.group.isin(["control", "cu-IUD"])
    out["gse137765"] = dict(epithelium_fraction=d.groupby("group").epithelium.median().round(3).to_dict(),
                            p_epithelium_fraction_lng_vs_control=mw(d.epithelium[d.group == "LNG-IUD"], d.epithelium[ct], "two-sided"))
    for name, sel in (("hormonal", tr), ("LNG-IUS", d.group == "LNG-IUD")):
        x = d[sel | ct].assign(treated=lambda v: (~v.group.isin(["control", "cu-IUD"])).astype(float))
        fm = sm.OLS(x.residual, sm.add_constant(x[["treated", "epithelium"]])).fit(cov_type="HC3"); ci = fm.conf_int().loc["treated"]
        out["gse137765"][f"residual_adjusted_{name}"] = dict(coef=round(float(fm.params["treated"]), 3), ci=[round(float(ci[0]), 3), round(float(ci[1]), 3)],
                                                             p=float(f"{fm.pvalues['treated']:.3g}"))
    n = F[F.group == "natural"]
    out["natural_arrays_epithelium_fraction_by_phase"] = {str(int(p)): round(float(n.epithelium[n.phase == p].median()), 3) for p in (0.0, 1.0, 2.0)}
    F.rename_axis("sample").to_csv(os.path.join(OUT, "s13_deconvolution.csv"), float_format="%.4g")
    return out


if __name__ == "__main__":
    summ = {"r12_huang_rank": r12(), "r13_deconvolution": r13()}
    json.dump(summ, open(os.path.join(OUT, "s13_rank_huang_deconv.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))
