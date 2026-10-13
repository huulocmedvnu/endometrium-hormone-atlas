"""S10 (R9, post hoc): rank-based bulk clocks with an absolute per-sample scale, calibrated on natural dated arrays.
Features: within-sample rank quantiles over genes shared by HECA, GPL570 and GPL6244, restricted to genes specific to
each compartment. Ridge trained on HECA whole-tissue natural staged donors. Natural arrays GSE4888 and GSE6364
(controls) give the natural epithelium-on-stroma line, which is applied to the contraceptive cohorts.
Writes outputs/s10_rank_bulk.json and outputs/s10_rank_bulk_samples.csv. Usage: s10_rank_bulk.py"""
import gzip, io, json, os
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon
from sklearn.linear_model import RidgeCV
from s2_clock import ROOT, W, ALPHAS
from s4_sensitivity import specific_genes
from s7_external_bulk import series_matrix, heca, whole_tissue, B as BULK

NAT = os.path.join(W, "ext", "natural")
OUT = os.path.join(ROOT, "outputs")
N_FEAT = 1000
PHASE = {"Proliferative phase PE": 0.0, "Early secretory phase ESE": 1.0, "Mid secretory phase MSE": 2.0,
         "Proliferative Phase Normal": 0.0, "Early Secretory Phase Normal": 1.0, "Mid Secretory Phase Normal": 2.0}
ENDO = {"Proliferative Phase Endometriosis": 0.0, "Early Secretory Phase Endometriosis": 1.0, "Mid Secretory Phase Endometriosis": 2.0}
EPI = ["EPCAM", "CDH1", "KRT8", "KRT18", "KRT19", "CLDN3"]
STR = ["PDGFRB", "DCN", "COL1A2", "COL3A1", "LUM"]


def annot_symbols(path, X):
    """Probe x sample table to gene symbol x sample, probes with one symbol only, mean over probes."""
    lines = gzip.open(path, "rt").read().splitlines(); i = lines.index("!platform_table_begin")
    a = pd.read_csv(io.StringIO("\n".join(lines[i + 1:])), sep="\t", usecols=["ID", "Gene symbol"], dtype=str, comment="!").dropna()
    a = a[~a["Gene symbol"].str.contains("///")].set_index("ID")["Gene symbol"]
    X.index = X.index.astype(str)
    return X.loc[X.index.intersection(a.index)].groupby(a).mean()


def rank_quantiles(X):
    """Samples x genes to within-sample rank quantiles in (0, 1]."""
    return X.rank(axis=1) / X.shape[1]


def load_array(gse, path, base=NAT):
    X, meta = series_matrix_from(os.path.join(base, f"{gse}_series_matrix.txt.gz"))
    return annot_symbols(path, X).T, meta


def series_matrix_from(path):
    lines = gzip.open(path, "rt").read().splitlines(); meta = {}
    for l in lines:
        if l.startswith("!Sample_"):
            k = l.split("\t")[0][8:]; v = [x.strip('"') for x in l.split("\t")[1:]]
            meta[k if k not in meta else f"{k}_{sum(x.startswith(k) for x in meta)}"] = v
    i, j = lines.index("!series_matrix_table_begin"), lines.index("!series_matrix_table_end")
    X = pd.read_csv(io.StringIO("\n".join(lines[i + 1:j])), sep="\t", index_col=0)
    n = len(meta["geo_accession"])
    return X, pd.DataFrame({k: v for k, v in meta.items() if len(v) == n})


def phase_of(meta, table):
    ph = pd.Series(np.nan, index=meta.geo_accession.values)
    for c in [c for c in meta.columns if c.startswith("characteristics_ch1")]:
        ph = ph.fillna(pd.Series(meta[c].map(table).values, index=meta.geo_accession.values))
    return ph


def mw(a, b, alt):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    return float(f"{mannwhitneyu(a, b, alternative=alt).pvalue:.3g}") if len(a) > 1 and len(b) > 1 else None


if __name__ == "__main__":
    P, M = heca(); spec = specific_genes(P, M)
    Pw = whole_tissue(P, M); yw = M.drop_duplicates("donor").set_index("donor").y.reindex(Pw.index)
    g570 = os.path.join(NAT, "GPL570.annot.gz")
    A1, m1 = load_array("GSE4888", g570); A2, m2 = load_array("GSE6364", g570)
    C1, c1 = series_matrix("GSE60129"); C2, c2 = series_matrix("GSE137765")
    from s7_external_bulk import gpl6244_symbols
    C1, C2 = gpl6244_symbols(C1).T, gpl6244_symbols(C2).T
    U = sorted(set(Pw.columns) & set(A1.columns) & set(A2.columns) & set(C1.columns) & set(C2.columns))
    R = {k: rank_quantiles(v[U]) for k, v in dict(heca=Pw, gse4888=A1, gse6364=A2, gse60129=C1, gse137765=C2).items()}
    ytr = yw[yw.notna()]; summ = {"genes_shared": len(U)}
    nat_ph = pd.concat([phase_of(m1, PHASE), phase_of(m2, PHASE)]).dropna()
    endo_ph = phase_of(m2, ENDO).dropna()
    natX = pd.concat([R["gse4888"], R["gse6364"]])
    clocks = {}
    for comp in ("stroma", "epithelium"):
        g = sorted(spec[comp] & set(U)); Xtr = R["heca"].loc[ytr.index, g]
        hv = Xtr.var().nlargest(N_FEAT).index
        lodo = np.empty(len(ytr))
        for i in range(len(ytr)):
            k = np.arange(len(ytr)) != i
            lodo[i] = RidgeCV(alphas=ALPHAS).fit(Xtr.iloc[k][hv], ytr.iloc[k]).predict(Xtr.iloc[[i]][hv])[0]
        m = RidgeCV(alphas=ALPHAS).fit(Xtr[hv], ytr)
        clocks[comp] = {k: pd.Series(m.predict(v[hv]), index=v.index) for k, v in R.items() if k != "heca"}
        nat = clocks[comp]["gse4888"].combine_first(clocks[comp]["gse6364"]).reindex(nat_ph.index)
        summ[f"rc0_{comp}"] = dict(heca_lodo_rho=round(float(spearmanr(lodo, ytr)[0]), 3), genes=len(g),
                                   natural_array_rho=round(float(spearmanr(nat, nat_ph)[0]), 3),
                                   natural_array_median_by_phase={str(int(p)): round(float(nat[nat_ph == p].median()), 3) for p in (0.0, 1.0, 2.0)})
    summ["rc0_passed"] = bool(all(summ[f"rc0_{c}"]["natural_array_rho"] >= 0.5 for c in ("stroma", "epithelium")))
    D = []
    for key, meta, src in (("gse4888", m1, None), ("gse6364", m2, None), ("gse60129", c1, "source_name_ch1"), ("gse137765", c2, "source_name_ch1")):
        d = pd.DataFrame({"set": key, "stroma": clocks["stroma"][key], "epithelium": clocks["epithelium"][key]})
        if src:
            grp = pd.Series(meta[src].values, index=meta.geo_accession.values)
            d["group"] = grp.reindex(d.index).str.replace("Endometrium, ", "").str.replace("endometrial biopsy using ", "")
            d = d[grp.reindex(d.index).str.contains("ndometri").fillna(False)]
            Z = (C1 if key == "gse60129" else C2).loc[d.index]; Z = (Z - Z.mean()) / Z.std()
            d["epi_content"] = Z[[g for g in EPI if g in Z]].mean(1); d["str_content"] = Z[[g for g in STR if g in Z]].mean(1)
        else:
            d["phase"] = nat_ph.reindex(d.index); d["endo_phase"] = endo_ph.reindex(d.index)
            d["group"] = np.where(d.phase.notna(), "natural", np.where(d.endo_phase.notna(), "endometriosis", "other"))
        D.append(d)
    D = pd.concat(D)
    nat = D[D.group == "natural"]
    b = np.polyfit(nat.stroma, nat.epithelium, 1)
    D["residual"] = D.epithelium - np.polyval(b, D.stroma)
    loo = []
    for i in range(len(nat)):
        k = np.arange(len(nat)) != i; bb = np.polyfit(nat.stroma.values[k], nat.epithelium.values[k], 1)
        loo.append(nat.epithelium.values[i] - np.polyval(bb, nat.stroma.values[i]))
    D.loc[nat.index, "residual"] = loo
    summ["natural_line"] = dict(n=len(nat), slope=round(float(b[0]), 3), intercept=round(float(b[1]), 3))
    r_nat = D.residual[D.group == "natural"]
    for key, tr, ct in (("gse60129", ["LNG-IUS", "DMPA"], ["control"]), ("gse137765", ["LNG-IUD", "COC"], ["control", "cu-IUD"])):
        d = D[D.set == key]; t, c = d[d.group.isin(tr)], d[d.group.isin(ct)]
        x = d[d.group.isin(tr + ct)].assign(treated=lambda v: v.group.isin(tr).astype(float))
        f = sm.OLS(x.residual, sm.add_constant(x[["treated", "epi_content", "str_content"]])).fit(cov_type="HC3")
        ci = f.conf_int().loc["treated"]
        summ[key] = dict(n={k: int(v) for k, v in d.group.value_counts().items()},
                         medians=d.groupby("group")[["stroma", "epithelium", "residual"]].median().round(3).to_dict("index"),
                         rc1_controls_residual_p_two_sided=float(f"{wilcoxon(c.residual).pvalue:.3g}") if key == "gse60129" else None,
                         p_treated_below_natural_arrays=mw(t.residual, r_nat, "less"), p_treated_below_own_controls=mw(t.residual, c.residual, "less"),
                         adjusted_for_content=dict(coef=round(float(f.params["treated"]), 3), ci=[round(float(ci[0]), 3), round(float(ci[1]), 3)],
                                                   p=float(f"{f.pvalues['treated']:.3g}")),
                         per_drug={g: mw(d.residual[d.group == g], c.residual, "less") for g in tr})
    e = D[(D.set == "gse6364")]
    summ["exploratory_endometriosis"] = dict(n_endo=int((e.group == "endometriosis").sum()), n_normal=int((e.group == "natural").sum()),
                                            residual_endo=round(float(e.residual[e.group == "endometriosis"].median()), 3),
                                            residual_normal=round(float(e.residual[e.group == "natural"].median()), 3),
                                            p_two_sided=mw(e.residual[e.group == "endometriosis"], e.residual[e.group == "natural"], "two-sided"))
    D.rename_axis("sample").to_csv(os.path.join(OUT, "s10_rank_bulk_samples.csv"), float_format="%.4g")
    json.dump(summ, open(os.path.join(OUT, "s10_rank_bulk.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))
