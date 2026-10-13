"""S5: paper extension S6, which programmes follow and which stay behind under hormones (exploratory, PLAN.md).
(a) Cycle-axis genes per compartment: the 200 genes most increased and the 200 most decreased from natural
proliferative to natural secretory mid donors. Response fraction per gene = (treated median - proliferative mean) /
(secretory mid mean - proliferative mean), 0 = stays proliferative, 1 = reaches mid-secretory.
(b) Steroid receptor expression per compartment. (c) Cell-state composition of epithelium and stroma per donor.
Writes outputs/s5_<set>_genes.csv, outputs/s5_<set>_composition.csv and outputs/s5_<set>_summary.json.
Usage: s5_mechanism.py nuclei|cells [treated group, default progestin_only]"""
import json, os, sys
import numpy as np, pandas as pd, h5py
from anndata.io import read_elem
from scipy.stats import mannwhitneyu
from s2_clock import stage_value, treatment_group, ROOT, W, MIN_N
from s1_profiles import HECA, FILES

N_AXIS, MIN_DIFF = 200, 0.1
RECEPTORS = ["ESR1", "ESR2", "PGR", "AR", "NR3C1", "PGRMC1", "OXTR"]
LINEAGES = {"epithelium": "Epithelial", "stroma": "Mesenchymal"}
NOT_STROMA = ("ePV", "mPV", "uSMC")


def response_fraction(mu0, mu2, treated):
    """(treated - mu0) / (mu2 - mu0), elementwise."""
    mu0, mu2, treated = (np.asarray(v, float) for v in (mu0, mu2, treated))
    return (treated - mu0) / (mu2 - mu0)


def p2(a, b):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    return float(f"{mannwhitneyu(a, b).pvalue:.3g}") if len(a) >= 2 and len(b) >= 2 else None


def label(M, tg):
    st = M.Stage.astype(str)
    lab = np.where(M.group == tg, "treated", np.where(M.group != "natural", "other",
                   np.where(st.str.startswith("Proliferative"), "proliferative",
                   np.where(st == "Secretory Early", "secretory_early", np.where(st == "Secretory Mid", "secretory_mid", "other")))))
    return pd.Series(lab, index=M.index)


if __name__ == "__main__":
    s = sys.argv[1] if len(sys.argv) > 1 else "nuclei"
    tg = sys.argv[2] if len(sys.argv) > 2 else "progestin_only"; tag = s if tg == "progestin_only" else f"{s}_{tg}"
    P = pd.read_parquet(os.path.join(W, f"profiles_{s}.parquet")); M = pd.read_csv(os.path.join(W, f"profiles_{s}_meta.csv"), index_col=0)
    M = M.loc[P.index]; M["group"] = M["Hormonal treatment"].map(treatment_group); M["lab"] = label(M, tg)
    summ, gene_rows = {"set": s, "treated_group": tg, "axis": {}, "receptors": {}}, []
    for comp in ("stroma", "epithelium"):
        Mc = M[(M.compartment == comp) & (M.n >= MIN_N)]; Pc = P.loc[Mc.index]
        Pc = Pc.loc[:, (Pc > 0).mean(0) > 0.5]
        g0, g2, gt = (Pc[Mc.lab == k] for k in ("proliferative", "secretory_mid", "treated"))
        d = g2.mean() - g0.mean()
        for direction, genes in (("up", d[d > MIN_DIFF].nlargest(N_AXIS).index), ("down", d[d < -MIN_DIFF].nsmallest(N_AXIS).index)):
            f = response_fraction(g0[genes].mean(), g2[genes].mean(), gt[genes].median())
            gene_rows.append(pd.DataFrame({"compartment": comp, "direction": direction, "gene": genes, "axis_diff": d[genes].values, "fraction": f}))
            summ["axis"][f"{comp}_{direction}"] = dict(genes=len(genes), median_fraction=round(float(np.median(f)), 3),
                                                       share_below_025=round(float(np.mean(f < 0.25)), 3), share_above_075=round(float(np.mean(f > 0.75)), 3))
        summ["receptors"][comp] = {r: dict(proliferative=round(float(g0[r].median()), 3), secretory_mid=round(float(g2[r].median()), 3),
                                           treated=round(float(gt[r].median()), 3), p_vs_proliferative=p2(gt[r], g0[r]), p_vs_mid=p2(gt[r], g2[r]))
                                   for r in RECEPTORS if r in Pc.columns}
        summ["n_" + comp] = Mc.lab.value_counts().to_dict()
    G = pd.concat(gene_rows)
    up = G[G.direction == "up"]
    summ["axis"]["p_up_epithelium_lags_stroma"] = float(f"{mannwhitneyu(up[up.compartment == 'epithelium'].fraction, up[up.compartment == 'stroma'].fraction, alternative='less').pvalue:.3g}")
    e = up[up.compartment == "epithelium"].sort_values("axis_diff", ascending=False)
    summ["epithelium_up_stay_proliferative"] = e[e.fraction < 0.25].gene.head(30).tolist()
    summ["epithelium_up_follow"] = e[e.fraction > 0.75].gene.head(30).tolist()
    # (c) composition from cell-level labels
    fn, _, _ = FILES[s]
    with h5py.File(os.path.join(HECA, fn), "r") as f:
        obs = read_elem(f["obs"])[["genotype", "lineage", "celltype"]].astype(str)
    lab = M.drop_duplicates("donor").set_index("donor").lab
    comp_rows = []
    for comp, lin in LINEAGES.items():
        o = obs[obs.lineage == lin]
        if comp == "stroma":
            o = o[~o.celltype.str.startswith(NOT_STROMA)]
        t = pd.crosstab(o.genotype, o.celltype); t = t[t.sum(1) >= MIN_N]; t = t.loc[:, t.sum() > 0]
        fr = t.div(t.sum(1), axis=0).assign(lab=lab.reindex(t.index).values)
        long = fr.melt(id_vars="lab", ignore_index=False, var_name="celltype", value_name="fraction").reset_index().rename(columns={"genotype": "donor"})
        comp_rows.append(long.assign(compartment=comp))
        summ["composition_" + comp] = {c: dict(proliferative=round(float(fr.loc[fr.lab == "proliferative", c].median()), 3),
                                               secretory_mid=round(float(fr.loc[fr.lab == "secretory_mid", c].median()), 3),
                                               treated=round(float(fr.loc[fr.lab == "treated", c].median()), 3),
                                               p_vs_proliferative=p2(fr.loc[fr.lab == "treated", c], fr.loc[fr.lab == "proliferative", c]),
                                               p_vs_mid=p2(fr.loc[fr.lab == "treated", c], fr.loc[fr.lab == "secretory_mid", c]))
                                       for c in t.columns}
    G.to_csv(os.path.join(ROOT, "outputs", f"s5_{tag}_genes.csv"), index=False, float_format="%.4g")
    pd.concat(comp_rows).to_csv(os.path.join(ROOT, "outputs", f"s5_{tag}_composition.csv"), index=False, float_format="%.4g")
    json.dump(summ, open(os.path.join(ROOT, "outputs", f"s5_{tag}_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))
