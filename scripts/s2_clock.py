"""S2: cycle clocks per compartment, positive controls PC1 and PC2, hypotheses H1 and H2 (PLAN.md).
Ridge regression of stage (proliferative 0, secretory early 1, secretory mid 2) on the 2,000 most variable genes of
the natural-cycle training donors, leave-one-donor-out evaluation (gene selection inside each fold), treated donors
scored by the model fitted on all natural donors. Writes outputs/s2_<set>_donors.csv and outputs/s2_<set>_summary.json.
Usage: s2_clock.py nuclei|cells [treated group, default progestin_only]"""
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr, mannwhitneyu, wilcoxon
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = os.environ.get("WORK", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work"))
MIN_N, N_HVG, N_WOI = 50, 2000, 50
ALPHAS = np.logspace(-1, 5, 13)


def stage_value(stage):
    s = str(stage)
    if s.startswith("Proliferative"):
        return 0.0
    return {"Secretory Early": 1.0, "Secretory Mid": 2.0}.get(s, np.nan)


def treatment_group(h):
    h = str(h)
    if h in ("nan", "-", "None", ""):
        return "natural"
    if h.startswith(("POP", "Implant", "Mirena", "Desogestrel")):
        return "progestin_only"
    if h.startswith(("COCP", "norethindrone")):
        return "combined_pill"
    return "other_" + h.split(" ")[0]


def fit(X, y, n_hvg=N_HVG):
    hv = np.argsort(X.var(0))[::-1][:n_hvg]
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)).fit(X[:, hv], y)
    return lambda Z: m.predict(Z[:, hv])


def lodo(X, y):
    out = np.empty(len(y))
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        out[i] = fit(X[m], y[m])(X[i:i + 1])[0]
    return out


if __name__ == "__main__":
    s = sys.argv[1] if len(sys.argv) > 1 else "nuclei"
    tg = sys.argv[2] if len(sys.argv) > 2 else "progestin_only"; tag = s if tg == "progestin_only" else f"{s}_{tg}"
    P = pd.read_parquet(os.path.join(W, f"profiles_{s}.parquet")); M = pd.read_csv(os.path.join(W, f"profiles_{s}_meta.csv"), index_col=0)
    M = M.loc[P.index]; M["group"] = M["Hormonal treatment"].map(treatment_group); M["y"] = M.Stage.map(stage_value)
    M.loc[M.group != "natural", "y"] = np.nan
    expressed = (P > 0).mean(0) > 0.5; P = P.loc[:, expressed]
    rows, summ = [], {"set": s, "treated_group": tg, "pc1": {}, "groups": M.drop_duplicates("donor").group.value_counts().to_dict()}
    for comp in ["stroma", "epithelium", "endothelium", "immune"]:
        m = (M.compartment == comp) & (M.n >= MIN_N); Mc, X = M[m], P[m].to_numpy()
        tr = Mc.y.notna().to_numpy()
        pred = np.full(len(Mc), np.nan); pred[tr] = lodo(X[tr], Mc.y.to_numpy()[tr])
        full = fit(X[tr], Mc.y.to_numpy()[tr]); pred[~tr] = full(X[~tr])
        rho, p = spearmanr(pred[tr], Mc.y.to_numpy()[tr])
        summ["pc1"][comp] = dict(donors_train=int(tr.sum()), rho=round(float(rho), 3), p=float(f"{p:.3g}"), passed=bool(rho >= 0.5 and p < 0.05))
        rows.append(pd.DataFrame({"donor": Mc.donor, "compartment": comp, "group": Mc.group, "treatment": Mc["Hormonal treatment"],
                                  "stage": Mc.Stage, "pathology": Mc.Endometrial_pathology, "y": Mc.y, "clock": pred, "n": Mc.n}))
    D = pd.concat(rows)
    # PC2: pseudo-decidualisation markers in stroma
    st = M[(M.compartment == "stroma") & (M.n >= MIN_N)]
    po, prol = st.group == tg, (st.group == "natural") & (st.y == 0)
    summ["pc2"] = {g: dict(treated_median=round(float(P.loc[st.index[po], g].median()), 3), natural_prolif_median=round(float(P.loc[st.index[prol], g].median()), 3),
                           p=float(f"{mannwhitneyu(P.loc[st.index[po], g], P.loc[st.index[prol], g], alternative='greater').pvalue:.3g}"))
                   for g in ("PRL", "IGFBP1") if g in P.columns}
    summ["pc2"]["passed"] = bool(any(v["p"] < 0.05 for v in summ["pc2"].values()))
    # H1: dyssynchrony index
    W2 = D.pivot_table(index=["donor", "group"], columns="compartment", values="clock").reset_index()
    W2["dyssynchrony"] = W2.stroma - W2.epithelium
    nat, pro = W2[W2.group == "natural"].dropna(subset=["dyssynchrony"]), W2[W2.group == tg].dropna(subset=["dyssynchrony"])
    summ["h1"] = dict(n_treated=len(pro), n_natural=len(nat),
                      stroma_median_treated=round(float(pro.stroma.median()), 3), epithelium_median_treated=round(float(pro.epithelium.median()), 3),
                      dyssynchrony_median_treated=round(float(pro.dyssynchrony.median()), 3), dyssynchrony_median_natural=round(float(nat.dyssynchrony.median()), 3),
                      p_dyssynchrony_higher=float(f"{mannwhitneyu(pro.dyssynchrony, nat.dyssynchrony, alternative='greater').pvalue:.3g}"),
                      p_stroma_above_1=float(f"{wilcoxon(pro.stroma - 1, alternative='greater').pvalue:.3g}"),
                      p_epithelium_below_1=float(f"{wilcoxon(pro.epithelium - 1, alternative='less').pvalue:.3g}"))
    # H2: window-of-implantation programme in epithelium
    ep = M[(M.compartment == "epithelium") & (M.n >= MIN_N)]
    mid, pro0 = ep.index[(ep.group == "natural") & (ep.y == 2)], ep.index[(ep.group == "natural") & (ep.y == 0)]
    woi = (P.loc[mid].mean() - P.loc[pro0].mean()).nlargest(N_WOI).index
    sc = P.loc[ep.index, woi].mean(1)
    pe = ep.index[ep.group == tg]
    summ["h2"] = dict(genes=list(woi), treated_median=round(float(sc[pe].median()), 3), natural_mid_median=round(float(sc[mid].median()), 3),
                      natural_prolif_median=round(float(sc[pro0].median()), 3),
                      p_treated_lower_than_mid=float(f"{mannwhitneyu(sc[pe], sc[mid], alternative='less').pvalue:.3g}"))
    D = D.merge(pd.DataFrame({"donor": ep.donor, "woi_score": sc.values}), on="donor", how="left")
    D.to_csv(os.path.join(ROOT, "outputs", f"s2_{tag}_donors.csv"), index=False, float_format="%.4g")
    json.dump(summ, open(os.path.join(ROOT, "outputs", f"s2_{tag}_summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "h2"} | {"h2": {k: v for k, v in summ["h2"].items() if k != "genes"}}, indent=1))
