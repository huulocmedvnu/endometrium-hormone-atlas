"""S8: effect sizes for the main comparisons, Hodges-Lehmann shift (median of all treated minus control differences)
with a percentile bootstrap 95% CI (groups resampled separately, 5,000 draws). Reads the donor and sample tables in
outputs/. Writes outputs/s8_effects.csv and outputs/s8_effects.json. Usage: s8_effects.py"""
import json, os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
N_BOOT = 5000


def hodges_lehmann(a, b):
    """Median of all pairwise differences a_i - b_j."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.median(a[:, None] - b[None, :]))


def hl_ci(a, b, n=N_BOOT, seed=0):
    a, b = np.asarray(pd.Series(a).dropna(), float), np.asarray(pd.Series(b).dropna(), float)
    rng = np.random.default_rng(seed)
    boot = [hodges_lehmann(rng.choice(a, len(a)), rng.choice(b, len(b))) for _ in range(n)]
    return hodges_lehmann(a, b), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), len(a), len(b)


def woi_by_group(tag, tg):
    d = pd.read_csv(os.path.join(OUT, f"s2_{tag}_donors.csv")); d = d[d.compartment == "epithelium"]
    return d.woi_score[d.group == tg], d.woi_score[(d.group == "natural") & (d.stage == "Secretory Mid")]


if __name__ == "__main__":
    rows = []
    add = lambda cohort, exposure, measure, comparator, a, b: rows.append(dict(zip(
        ["cohort", "exposure", "measure", "comparator", "shift", "ci_low", "ci_high", "n_exposed", "n_comparator"],
        [cohort, exposure, measure, comparator, *hl_ci(a, b)])))
    for tag, cohort, tg, exp in (("nuclei", "HECA snRNA-seq", "progestin_only", "progestin only"),
                                 ("cells_combined_pill", "HECA scRNA-seq", "combined_pill", "combined pill")):
        d = pd.read_csv(os.path.join(OUT, f"s4_{tag}_donors.csv"))
        add(cohort, exp, "position-matched epithelial residual", "natural", d.residual[d.group == tg], d.residual[d.group == "natural"])
        add(cohort, exp, "dyssynchrony", "natural", d.dyssynchrony[d.group == tg], d.dyssynchrony[d.group == "natural"])
        add(cohort, exp, "epithelial WOI score", "natural mid-secretory", *woi_by_group("nuclei" if tag == "nuclei" else "cells_combined_pill", tg))
    H = pd.read_csv(os.path.join(OUT, "s6_huang_donors.csv"))
    for v in ("unscaled", "zscored"):
        h = H[H.version == v]
        add("Huang 2026", "programmed FET cycle", f"dyssynchrony ({v})", "natural cycle", h.dyssynchrony[h.cycle == "Controlled"], h.dyssynchrony[h.cycle == "Natural"])
    h = H[H.version == "unscaled"]
    add("Huang 2026", "programmed FET cycle", "epithelial WOI score", "natural cycle", h.woi[h.cycle == "Controlled"], h.woi[h.cycle == "Natural"])
    S = pd.read_csv(os.path.join(OUT, "s7_bulk_samples.csv"))
    for gse, tr, ct, exp, cmp in (("GSE60129", ["LNG-IUS", "DMPA"], ["control"], "LNG-IUS, DMPA", "mid-secretory control"),
                                  ("GSE137765", ["LNG-IUS", "COC"], ["control", "Cu-IUD"], "LNG-IUS, combined pill", "non-hormonal")):
        d = S[S.set == gse]
        for m in ("dyssynchrony", "epithelium", "stroma", "woi"):
            add(gse, exp, {"epithelium": "epithelium clock", "stroma": "stroma clock", "woi": "epithelial WOI score"}.get(m, m), cmp,
                d[m][d.group.isin(tr)], d[m][d.group.isin(ct)])
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "s8_effects.csv"), index=False, float_format="%.4g")
    json.dump({f"{r.cohort}|{r.measure}": dict(shift=round(r.shift, 3), ci=[round(r.ci_low, 3), round(r.ci_high, 3)], n=[int(r.n_exposed), int(r.n_comparator)])
               for r in R.itertuples()}, open(os.path.join(OUT, "s8_effects.json"), "w"), indent=1)
    print(R.round(3).to_string(index=False))
