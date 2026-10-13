"""S7: external validation in bulk data (PLAN.md, external validation and amendments).
Transfer check GSE97929 (sorted epithelium or stroma, LH+2 against LH+8): compartment clocks from HECA snRNA-seq
compartment profiles. Progestin sets GSE60129 (LNG-IUS, DMPA, mid-secretory controls) and GSE137765 (LNG-IUS, COC,
copper IUD, controls): bulk clocks trained on HECA whole-tissue profiles over compartment-specific genes.
All sets z-scored within set. Writes outputs/s7_bulk_samples.csv and outputs/s7_bulk_summary.json.
Usage: s7_external_bulk.py"""
import gzip, io, json, os, re
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.linear_model import RidgeCV
from s2_clock import stage_value, treatment_group, ROOT, W, MIN_N, ALPHAS
from s4_sensitivity import specific_genes
from s6_external_huang import zscore

B = os.path.join(W, "ext", "bulk")
N_FEAT = 1000


def series_matrix(gse):
    """Expression table (probes x GSM) and per-sample metadata from a GEO series matrix."""
    lines = gzip.open(os.path.join(B, f"{gse}_series_matrix.txt.gz"), "rt").read().splitlines()
    meta = {}
    for l in lines:
        if l.startswith("!Sample_"):
            k = l.split("\t")[0][8:]; v = [x.strip('"') for x in l.split("\t")[1:]]
            meta[k if k not in meta else f"{k}_{sum(x.startswith(k) for x in meta)}"] = v
    i, j = lines.index("!series_matrix_table_begin"), lines.index("!series_matrix_table_end")
    X = pd.read_csv(io.StringIO("\n".join(lines[i + 1:j])), sep="\t", index_col=0)
    n = len(meta["geo_accession"])
    return X, pd.DataFrame({k: v for k, v in meta.items() if len(v) == n})


def gpl6244_symbols(X):
    a = pd.read_csv(os.path.join(B, "GPL6244.annot.gz"), sep="\t", skiprows=27, usecols=["ID", "Gene symbol"], dtype=str).dropna()
    a = a[~a["Gene symbol"].str.contains("///")].set_index("ID")["Gene symbol"]
    X.index = X.index.astype(str)
    return X.loc[X.index.intersection(a.index)].groupby(a).mean()


def heca():
    P = pd.read_parquet(os.path.join(W, "profiles_nuclei.parquet")); M = pd.read_csv(os.path.join(W, "profiles_nuclei_meta.csv"), index_col=0)
    M = M.loc[P.index]; M["group"] = M["Hormonal treatment"].map(treatment_group); M["y"] = M.Stage.map(stage_value)
    M.loc[M.group != "natural", "y"] = np.nan
    return P.loc[:, (P > 0).mean(0) > 0.5], M


def whole_tissue(P, M):
    """Donor whole-tissue profile: compartment means weighted by number of nuclei."""
    w = P.mul(M.n, axis=0).groupby(M.donor).sum()
    return w.div(M.groupby("donor").n.sum(), axis=0)


def train(Xtr, y, Xnew):
    """Ridge clock on within-set z-scores, top N_FEAT variable features of the training set."""
    Ztr, Znew = zscore(Xtr), zscore(Xnew)
    hv = np.argsort(np.asarray(Xtr).var(0))[::-1][:N_FEAT]
    r = RidgeCV(alphas=ALPHAS).fit(Ztr[:, hv], y)
    return r.predict(Znew[:, hv])


def mw(a, b, alt):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    return float(f"{mannwhitneyu(a, b, alternative=alt).pvalue:.3g}") if len(a) > 1 and len(b) > 1 else None


def gse97929(P, M):
    R = pd.read_csv(os.path.join(B, "GSE97929_reads_all.txt.gz"), sep="\t", index_col=0)
    _, meta = series_matrix("GSE97929")
    t = meta.title.str.extract(r"^(?P<donor>[^_]+)_(?P<lib>\d)_(?P<comp>epithelium|stroma)_LH(?P<lh>\d+)$")
    info = t.drop_duplicates("donor").set_index("donor")
    lab = pd.Series([c.split("|")[1] for c in R.columns], index=R.columns).str.replace("_1$", "", regex=True)
    R = R.loc[:, lab.isin(info.index)]; R = R.T.groupby(lab[R.columns]).sum().T
    X = np.log1p(R / R.sum() * 1e4).T
    out, rows = {}, []
    for comp in ("stroma", "epithelium"):
        d = info.index[info.comp == comp]; m = (M.compartment == comp) & (M.n >= MIN_N) & M.y.notna()
        g = sorted(set(P.columns) & set(X.columns)); g = [x for x in g if X.loc[d, x].gt(0).mean() > 0.5]
        pred = pd.Series(train(P.loc[m, g].to_numpy(), M.y[m].to_numpy(), X.loc[d, g].to_numpy()), index=d)
        lh = info.lh[d].astype(int)
        out[comp] = dict(n_lh2=int((lh <= 2).sum()), n_lh8=int((lh >= 8).sum()), median_lh2=round(float(pred[lh <= 2].median()), 3),
                         median_lh8=round(float(pred[lh >= 8].median()), 3), p_lh8_higher=mw(pred[lh >= 8], pred[lh <= 2], "greater"))
        rows.append(pd.DataFrame({"set": "GSE97929", "sample": d, "group": "LH" + info.lh[d].values, comp: pred.values}))
    return out, rows


def progestin_set(gse, group_of, treated, control, Pw, yw, spec, woi):
    X, meta = series_matrix(gse)
    X = gpl6244_symbols(X).T
    src = meta.source_name_ch1
    grp = src.map(group_of); keep = grp.notna().to_numpy()
    X, grp = X.loc[meta.geo_accession[keep].values], pd.Series(grp[keep].values, index=meta.geo_accession[keep].values)
    D = pd.DataFrame({"set": gse, "sample": X.index, "group": grp.values}, index=X.index)
    for comp in ("stroma", "epithelium"):
        g = sorted(spec[comp] & set(X.columns) & set(Pw.columns))
        D[comp] = train(Pw[g].to_numpy(), yw, X[g].to_numpy())
    D["dyssynchrony"] = D.stroma - D.epithelium
    w = [x for x in woi if x in X.columns]; D["woi"] = zscore(X[w].to_numpy()).mean(1)
    t, c = D[D.group.isin(treated)], D[D.group.isin(control)]
    res = dict(n={k: int(v) for k, v in D.group.value_counts().items()}, woi_genes=len(w),
               medians=D.groupby("group")[["stroma", "epithelium", "dyssynchrony", "woi"]].median().round(3).to_dict("index"),
               p_dyssynchrony_higher=mw(t.dyssynchrony, c.dyssynchrony, "greater"), p_woi_lower=mw(t.woi, c.woi, "less"),
               p_stroma_two_sided=mw(t.stroma, c.stroma, "two-sided"), p_epithelium_two_sided=mw(t.epithelium, c.epithelium, "two-sided"))
    res["per_drug"] = {d: dict(p_dyssynchrony_higher=mw(D.dyssynchrony[D.group == d], c.dyssynchrony, "greater"),
                               p_woi_lower=mw(D.woi[D.group == d], c.woi, "less")) for d in treated}
    return res, D


if __name__ == "__main__":
    P, M = heca(); summ = {}
    summ["GSE97929_transfer"], rows = gse97929(P, M)
    spec = specific_genes(P, M)
    Pw = whole_tissue(P, M); yw = M.drop_duplicates("donor").set_index("donor").y.reindex(Pw.index)
    Pw, yw = Pw[yw.notna()], yw[yw.notna()].to_numpy()
    woi = json.load(open(os.path.join(ROOT, "outputs", "s2_nuclei_summary.json")))["h2"]["genes"]
    summ["bulk_clock_lodo_rho"] = {}
    for comp in ("stroma", "epithelium"):
        g = sorted(spec[comp] & set(Pw.columns)); X = Pw[g].to_numpy(); pred = np.empty(len(yw))
        for i in range(len(yw)):
            m = np.arange(len(yw)) != i
            pred[i] = train(X[m], yw[m], X)[i]
        summ["bulk_clock_lodo_rho"][comp] = round(float(spearmanr(pred, yw)[0]), 3)
    g60 = {"Endometrium, control": "control", "Endometrium, LNG-IUS": "LNG-IUS", "Endometrium, DMPA": "DMPA"}
    summ["GSE60129"], D1 = progestin_set("GSE60129", g60, ["LNG-IUS", "DMPA"], ["control"], Pw, yw, spec, woi)
    g13 = {"endometrial biopsy using control": "control", "endometrial biopsy using cu-IUD": "Cu-IUD",
           "endometrial biopsy using LNG-IUD": "LNG-IUS", "endometrial biopsy using COC": "COC"}
    summ["GSE137765"], D2 = progestin_set("GSE137765", g13, ["LNG-IUS", "COC"], ["control", "Cu-IUD"], Pw, yw, spec, woi)
    pd.concat(rows + [D1, D2]).to_csv(os.path.join(ROOT, "outputs", "s7_bulk_samples.csv"), index=False, float_format="%.4g")
    json.dump(summ, open(os.path.join(ROOT, "outputs", "s7_bulk_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))
