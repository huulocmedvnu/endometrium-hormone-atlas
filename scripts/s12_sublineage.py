"""S12 (R11 and R13 signatures, post hoc). One pass over HECA snRNA-seq.
R11: donor x epithelial sub-lineage profiles (luminal, glandular functionalis, eHormones, basalis) and stroma,
clocks per sub-lineage trained on natural donors, tests against the stromal clock.
R13: natural-donor signatures in linear CP10k for stroma, epithelium, endothelium, immune, perivascular and smooth
muscle, written to $WORK/signature_nuclei.parquet for s13. Writes outputs/s12_sublineage.json.
Usage: s12_sublineage.py [--workers 8]"""
import argparse, json, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd, scipy.sparse as sp, h5py
from anndata.io import read_elem
from scipy.stats import mannwhitneyu
from s1_profiles import HECA
from s2_clock import fit, lodo, stage_value, treatment_group, ROOT, W
from s4_sensitivity import residuals

PATH = os.path.join(HECA, "endometriumAtlasV2_nuclei.h5ad")
MIN_SUB, TG = 30, "progestin_only"
SUB = {"SOX9_luminal": "luminal", "preLuminal": "luminal", "Luminal": "luminal", "SOX9_functionalis_I": "glandular",
       "SOX9_functionalis_II": "glandular", "preGlandular": "glandular", "Glandular": "glandular", "Glandular_secretory": "glandular",
       "eHormones": "eHormones", "SOX9_basalis": "basalis"}


def sig_group(lineage, celltype):
    if lineage == "Mesenchymal":
        return "pv_smc" if celltype.startswith(("ePV", "mPV", "uSMC")) else "stroma"
    return {"Epithelial": "epithelium", "Endothelial": "endothelium", "Immune": "immune"}.get(lineage)


def _block(args):
    lo, hi, c1, n1, c2, n2 = args
    with h5py.File(PATH, "r") as f:
        g = f["X"]; ptr = g["indptr"][lo:hi + 1]; a, b = int(ptr[0]), int(ptr[-1])
        X = sp.csr_matrix((g["data"][a:b], g["indices"][a:b], ptr - a), shape=(hi - lo, int(g.attrs["shape"][1])))
    out = []
    for codes, n, lin in ((c1[lo:hi], n1, False), (c2[lo:hi], n2, True)):
        k = codes >= 0; Y = X.copy()
        if lin:
            Y.data = np.expm1(Y.data)
        G = sp.csr_matrix((np.ones(k.sum()), (codes[k], np.flatnonzero(k))), shape=(n, hi - lo))
        out.append(np.asarray((G @ Y).todense()))
    return out


def encode(lab):
    ok = lab.notna().to_numpy(); cats = pd.Index(sorted(lab[ok].unique()))
    return np.where(ok, cats.get_indexer(lab.fillna("")), -1), cats


def tests(clk_sub, clk_st, group):
    D = pd.concat({"stroma": clk_st, "sub": clk_sub}, axis=1).dropna().join(group)
    nat, t = D[D.group == "natural"], D[D.group == TG]
    if len(t) < 3:
        return dict(n_treated=len(t))
    rn, rt = residuals(nat.stroma, nat["sub"], (t.stroma, t["sub"]))
    return dict(n_treated=len(t), n_natural=len(nat), sub_median_treated=round(float(t["sub"].median()), 3), stroma_median_treated=round(float(t.stroma.median()), 3),
                residual_treated=round(float(np.median(rt)), 3), p_residual=float(f"{mannwhitneyu(rt, rn, alternative='less').pvalue:.3g}"),
                p_index=float(f"{mannwhitneyu(t.stroma - t['sub'], nat.stroma - nat['sub'], alternative='greater').pvalue:.3g}"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=8); a = ap.parse_args()
    with h5py.File(PATH, "r") as f:
        obs = read_elem(f["obs"])[["genotype", "lineage", "celltype", "Hormonal treatment", "Stage"]].astype(str); var = pd.Index(read_elem(f["var"]).index)
    grp = obs["Hormonal treatment"].map(treatment_group)
    sg = pd.Series([sig_group(l, c) for l, c in zip(obs.lineage, obs.celltype)], index=obs.index)
    sub = obs.celltype.map(SUB).where(obs.lineage == "Epithelial")
    sub = sub.fillna(sg.where(sg == "stroma"))
    c1, cats1 = encode((obs.genotype + "|" + sub).where(sub.notna()))
    c2, cats2 = encode(sg.where(grp == "natural"))
    jobs = [(lo, min(lo + 10000, len(obs)), c1, len(cats1), c2, len(cats2)) for lo in range(0, len(obs), 10000)]
    S1, S2 = np.zeros((len(cats1), len(var))), np.zeros((len(cats2), len(var)))
    with ProcessPoolExecutor(a.workers) as ex:
        for r1, r2 in ex.map(_block, jobs):
            S1 += r1; S2 += r2
    n1 = np.bincount(c1[c1 >= 0], minlength=len(cats1)); n2 = np.bincount(c2[c2 >= 0], minlength=len(cats2))
    pd.DataFrame(S2 / n2[:, None], index=cats2, columns=var).astype(np.float32).to_parquet(os.path.join(W, "signature_nuclei.parquet"))
    sums = pd.DataFrame(S1, index=cats1, columns=var); n = pd.Series(n1, index=cats1)
    # glandular plus eHormones variant
    donors = sorted({c.split("|")[0] for c in cats1})
    for dnr in donors:
        ks = [f"{dnr}|glandular", f"{dnr}|eHormones"]; ks = [k for k in ks if k in sums.index]
        if ks:
            sums.loc[f"{dnr}|glandular_eh"] = sums.loc[ks].sum(); n[f"{dnr}|glandular_eh"] = n[ks].sum()
    Pm = sums.div(n, axis=0)
    info = obs.assign(group=grp).drop_duplicates("genotype").set_index("genotype")
    meta = pd.DataFrame({"donor": Pm.index.str.split("|").str[0], "sub": Pm.index.str.split("|").str[1], "n": n.reindex(Pm.index).values}, index=Pm.index)
    meta = meta.join(info[["group", "Stage"]], on="donor"); meta["y"] = np.where(meta.group == "natural", meta.Stage.map(stage_value), np.nan)
    expressed = Pm.columns[(Pm > 0).mean(0) > 0.5]
    group = meta.drop_duplicates("donor").set_index("donor").group
    clocks = {}
    for s in ("stroma", "luminal", "glandular", "glandular_eh"):
        m = (meta["sub"] == s) & (meta.n >= (50 if s == "stroma" else MIN_SUB)); Mc, X = meta[m], Pm.loc[m, expressed].to_numpy()
        tr = Mc.y.notna().to_numpy(); pred = np.full(len(Mc), np.nan)
        pred[tr] = lodo(X[tr], Mc.y.to_numpy()[tr]); pred[~tr] = fit(X[tr], Mc.y.to_numpy()[tr])(X[~tr])
        clocks[s] = pd.Series(pred, index=Mc.donor.values)
        from scipy.stats import spearmanr
        clocks[s + "_rho"] = round(float(spearmanr(pred[tr], Mc.y.to_numpy()[tr])[0]), 3)
    share = meta[meta["sub"].isin(["luminal", "glandular", "eHormones", "basalis"])].pivot_table(index="donor", columns="sub", values="n").fillna(0)
    share = share.div(share.sum(1), axis=0).join(group)
    summ = {"min_nuclei_sublineage": MIN_SUB,
            "lodo_rho": {s: clocks[s + "_rho"] for s in ("stroma", "luminal", "glandular", "glandular_eh")},
            "basalis_share_median": {g: round(float(share.basalis[share.group == g].median()), 4) for g in ("natural", TG)},
            "luminal": tests(clocks["luminal"], clocks["stroma"], group), "glandular": tests(clocks["glandular"], clocks["stroma"], group),
            "glandular_with_ehormones": tests(clocks["glandular_eh"], clocks["stroma"], group)}
    json.dump(summ, open(os.path.join(ROOT, "outputs", "s12_sublineage.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))
