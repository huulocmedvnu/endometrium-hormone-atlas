"""S3: H4 (endometriosis timing) and H5 (age), PLAN.md extension. Natural-cycle donors only. Clocks refitted on
control donors (controls scored leave-one-donor-out, endometriosis cases by the model fitted on all controls).
Residual = predicted minus histological stage. Writes outputs/s3_<set>_donors.csv and outputs/s3_summary.json.
Usage: s3_endometriosis_age.py"""
import json, os
import numpy as np, pandas as pd, h5py
from anndata.io import read_elem
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon, norm
from s2_clock import fit, lodo, treatment_group, W, ROOT, MIN_N
from s1_profiles import HECA, FILES

STAGE = {"Secretory Early": 1.0, "Secretory Early-Mid": 1.5, "Secretory Mid": 2.0}


def stage_value(s):
    s = str(s)
    return 0.0 if s.startswith("Proliferative") else STAGE.get(s, np.nan)


def partial_spearman(x, y, z):
    """Spearman correlation of x and y after removing the rank-linear effect of z from both."""
    r = lambda v: pd.Series(v).rank().to_numpy()
    rx, ry, rz = r(x), r(y), r(z)
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)
    return spearmanr(res(rx), res(ry))


def stouffer(ps):
    ps = [p for p in ps if np.isfinite(p)]
    return float(norm.sf(sum(norm.isf(p) for p in ps) / np.sqrt(len(ps)))) if ps else float("nan")


def run(s):
    P = pd.read_parquet(os.path.join(W, f"profiles_{s}.parquet")); M = pd.read_csv(os.path.join(W, f"profiles_{s}_meta.csv"), index_col=0).loc[P.index]
    with h5py.File(os.path.join(HECA, FILES[s][0]), "r") as f:
        o = read_elem(f["obs"])
    rasrm = o.groupby(o.genotype.astype(str))["Endometriosis_stage"].first().astype(str)
    M["group"] = M["Hormonal treatment"].map(treatment_group); M["y"] = M.Stage.map(stage_value)
    M["case"] = M.Endometrial_pathology.astype(str).str.contains("E"); M["rasrm"] = pd.to_numeric(M.donor.map(rasrm), errors="coerce")
    M["age"] = pd.to_numeric(M.Age, errors="coerce")
    P = P.loc[:, (P > 0).mean(0) > 0.5]
    rows = []
    for comp in ("stroma", "epithelium"):
        m = (M.compartment == comp) & (M.n >= MIN_N) & (M.group == "natural") & M.y.notna(); Mc, X = M[m], P[m].to_numpy()
        ctl = (~Mc.case).to_numpy(); pred = np.empty(len(Mc))
        pred[ctl] = lodo(X[ctl], Mc.y.to_numpy()[ctl]); pred[~ctl] = fit(X[ctl], Mc.y.to_numpy()[ctl])(X[~ctl])
        rows.append(Mc[["donor", "case", "rasrm", "age", "y", "Stage"]].assign(set=s, compartment=comp, clock=pred, residual=pred - Mc.y.to_numpy()))
    return pd.concat(rows)


if __name__ == "__main__":
    D = pd.concat([run(s) for s in ("nuclei", "cells")])
    D.to_csv(os.path.join(ROOT, "outputs", "s3_donors.csv"), index=False, float_format="%.4g")
    summ = {}
    for comp in ("stroma", "epithelium"):
        r = {}
        for s in ("nuclei", "cells"):
            d = D[(D.set == s) & (D.compartment == comp)]; sec = d[d.y > 0]
            ca, co = sec[sec.case], sec[~sec.case]
            p_case = mannwhitneyu(ca.residual, co.residual, alternative="less").pvalue if len(ca) >= 3 and len(co) >= 3 else np.nan
            rho, p2 = spearmanr(sec.rasrm, sec.residual) if sec.rasrm.notna().sum() >= 5 else (np.nan, np.nan)
            p_stage = p2 / 2 if rho < 0 else 1 - p2 / 2 if np.isfinite(rho) else np.nan
            ctl_all = d[~d.case]
            pa = partial_spearman(d.age, d.residual, d.y) if d.age.notna().all() else spearmanr(d.age, d.residual, nan_policy="omit")
            r[s] = dict(secretory_cases=len(ca), secretory_controls=len(co), residual_median_cases=round(float(ca.residual.median()), 3),
                        residual_median_controls=round(float(co.residual.median()), 3), p_cases_lower=float(f"{p_case:.3g}"),
                        rho_rasrm=round(float(rho), 3), p_rasrm_negative=float(f"{p_stage:.3g}"),
                        controls_residual_p_two_sided=float(f"{wilcoxon(ctl_all.residual).pvalue:.3g}") if len(ctl_all) >= 6 else None,
                        age_partial_rho=round(float(pa[0]), 3), age_p=float(f"{pa[1]:.3g}"), donors_age=int(d.age.notna().sum()))
        r["combined_p_cases_lower"] = float(f"{stouffer([r[s]['p_cases_lower'] for s in ('nuclei', 'cells')]):.3g}")
        r["combined_p_rasrm_negative"] = float(f"{stouffer([r[s]['p_rasrm_negative'] for s in ('nuclei', 'cells')]):.3g}")
        summ[comp] = r
    json.dump(summ, open(os.path.join(ROOT, "outputs", "s3_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))
