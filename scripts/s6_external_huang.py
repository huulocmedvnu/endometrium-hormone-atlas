"""S6: external validation in Huang et al. 2026 (JCI Insight), programmed (HRT) against natural cycles, snRNA-seq
(PLAN.md, external validation and amendment). Donor x compartment log1p(CP10k) profiles from raw counts, clocks
trained on HECA snRNA-seq natural donors over shared genes, Huang donors scored unscaled and after within-set z-scoring.
EC0 transfer check, E1 dyssynchrony, E2 epithelial WOI score, E3 position-matched residual.
Writes outputs/s6_huang_donors.csv and outputs/s6_huang_summary.json. Usage: s6_external_huang.py"""
import json, os
import numpy as np, pandas as pd, h5py, scipy.sparse as sp
from anndata.io import read_elem
from scipy.stats import mannwhitneyu
from sklearn.linear_model import RidgeCV
from s2_clock import stage_value, treatment_group, ROOT, W, MIN_N, N_HVG, ALPHAS
from s1_profiles import lognorm

H5 = os.path.join(W, "ext", "huang", "huang2026.h5ad")
COMP = {"stromal cell of endometrium": "stroma", "glandular epithelial cell of endometrium": "epithelium",
        "luminal epithelial cell of endometrium": "epithelium", "ciliated epithelial cell": "epithelium",
        "endothelial cell of uterus": "endothelium", "uterine natural killer cell": "immune", "macrophage": "immune", "B cell": "immune"}
COMPS = ("stroma", "epithelium")


def zscore(X):
    """Column z-score, constant columns set to 0."""
    X = np.asarray(X, float); sd = X.std(0); sd[sd == 0] = 1
    return (X - X.mean(0)) / sd


def huang_profiles():
    with h5py.File(H5, "r") as f:
        obs = read_elem(f["obs"]); X = read_elem(f["raw/X"]); var = read_elem(f["raw/var"]) if "var" in f["raw"] else read_elem(f["var"])
    sym = (var["feature_name"] if "feature_name" in var else read_elem(h5py.File(H5, "r")["var"])["feature_name"]).astype(str).values
    X = lognorm(sp.csr_matrix(X))
    comp = obs.cell_type.astype(str).map(COMP)
    lab = obs.donor_id.astype(str) + "|" + comp.astype(str); ok = comp.notna().to_numpy()
    cats = pd.Index(sorted(lab[ok].unique())); codes = cats.get_indexer(lab[ok])
    G = sp.csr_matrix((np.ones(len(codes)), (codes, np.flatnonzero(ok))), shape=(len(cats), len(obs)))
    n = np.bincount(codes, minlength=len(cats))
    P = pd.DataFrame(np.asarray((G @ X).todense()) / n[:, None], index=cats, columns=sym)
    P = P.T.groupby(level=0).mean().T  # duplicated symbols averaged
    cyc = obs.drop_duplicates("donor_id").set_index("donor_id").CycleType.astype(str)
    M = pd.DataFrame({"donor": cats.str.split("|").str[0], "compartment": cats.str.split("|").str[1], "n": n}, index=cats)
    M["cycle"] = M.donor.map(cyc)
    return P, M


def heca():
    P = pd.read_parquet(os.path.join(W, "profiles_nuclei.parquet")); M = pd.read_csv(os.path.join(W, "profiles_nuclei_meta.csv"), index_col=0)
    M = M.loc[P.index]; M["group"] = M["Hormonal treatment"].map(treatment_group); M["y"] = M.Stage.map(stage_value)
    M.loc[M.group != "natural", "y"] = np.nan
    return P.loc[:, (P > 0).mean(0) > 0.5], M


def score(Ph, Mh, Pe, Me, genes, comp, z):
    m, mh = (Me.compartment == comp) & (Me.n >= MIN_N) & Me.y.notna(), (Mh.compartment == comp) & (Mh.n >= MIN_N)
    Xe, Xh = Pe.loc[m, genes].to_numpy(), Ph.loc[mh, genes].to_numpy()
    if z:
        Xe, Xh = zscore(Xe), zscore(Xh)
    hv = np.argsort(Xe.var(0))[::-1][:N_HVG]
    mu, sd = Xe[:, hv].mean(0), Xe[:, hv].std(0); sd[sd == 0] = 1
    r = RidgeCV(alphas=ALPHAS).fit((Xe[:, hv] - mu) / sd, Me.y[m].to_numpy())
    pe = pd.Series(r.predict((Xe[:, hv] - mu) / sd), index=Me.donor[m].values)
    return pd.Series(r.predict((Xh[:, hv] - mu) / sd), index=Mh.donor[mh].values), pe


def mw(a, b, alt):
    return float(f"{mannwhitneyu(a.dropna(), b.dropna(), alternative=alt).pvalue:.3g}")


if __name__ == "__main__":
    Ph, Mh = huang_profiles(); Pe, Me = heca()
    genes = sorted(set(Pe.columns) & set(Ph.columns))
    s2 = json.load(open(os.path.join(ROOT, "outputs", "s2_nuclei_summary.json")))
    woi = [g for g in s2["h2"]["genes"] if g in Ph.columns]
    cyc = Mh.drop_duplicates("donor").set_index("donor").cycle
    summ = {"shared_genes": len(genes), "woi_genes_present": len(woi), "donors": cyc.value_counts().to_dict(),
            "nuclei_per_compartment": Mh.groupby("compartment").n.sum().to_dict()}
    rows = []
    for version, z in (("unscaled", False), ("zscored", True)):
        D = pd.DataFrame(index=cyc.index); nat_e = {}
        for comp in COMPS:
            D[comp], nat_e[comp] = score(Ph, Mh, Pe, Me, genes, comp, z)
        D["cycle"] = cyc; D["dyssynchrony"] = D.stroma - D.epithelium
        ep = Mh[(Mh.compartment == "epithelium") & (Mh.n >= MIN_N)]
        D["woi"] = pd.Series(Ph.loc[ep.index, woi].mean(1).values, index=ep.donor.values)
        both = pd.concat(nat_e, axis=1).dropna(); b = np.polyfit(both.stroma, both.epithelium, 1)
        D["residual"] = D.epithelium - np.polyval(b, D.stroma)
        nat, pc = D[D.cycle == "Natural"], D[D.cycle == "Controlled"]
        summ[version] = dict(
            ec0={c: dict(natural_median=round(float(nat[c].median()), 3), passed=bool(nat[c].median() > 1)) for c in COMPS},
            medians={k: {c: round(float(g[c].median()), 3) for c in ("stroma", "epithelium", "dyssynchrony", "woi", "residual")}
                     for k, g in (("natural", nat), ("programmed", pc))},
            n=dict(natural=int(nat.dyssynchrony.notna().sum()), programmed=int(pc.dyssynchrony.notna().sum())),
            p_e1_dyssynchrony_higher=mw(pc.dyssynchrony, nat.dyssynchrony, "greater"),
            p_e2_woi_lower=mw(pc.woi, nat.woi, "less"),
            p_e3_residual_lower=mw(pc.residual, nat.residual, "less"),
            p_stroma_two_sided=mw(pc.stroma, nat.stroma, "two-sided"), p_epithelium_two_sided=mw(pc.epithelium, nat.epithelium, "two-sided"))
        rows.append(D.assign(version=version))
    summ["primary"] = "unscaled" if all(v["passed"] for v in summ["unscaled"]["ec0"].values()) else "zscored"
    pd.concat(rows).rename_axis("donor").to_csv(os.path.join(ROOT, "outputs", "s6_huang_donors.csv"), float_format="%.4g")
    json.dump(summ, open(os.path.join(ROOT, "outputs", "s6_huang_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))
