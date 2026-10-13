"""S11 (R10, post hoc): cell-level checks in HECA snRNA-seq.
(a) 50 nuclei per donor and compartment, 20 random draws, clocks retrained per draw (leave-one-donor-out for natural
donors), H1 index and position-matched residual tests per draw. (b) Epithelial profiles without the eHormones state.
Genes: union of the 3,000 most variable genes among natural profiles of stroma and of epithelium.
Writes outputs/s11_cell_level.json. Usage: s11_cell_level.py [--workers 8] [--draws 20]"""
import argparse, json, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd, scipy.sparse as sp, h5py
from anndata.io import read_elem
from scipy.stats import mannwhitneyu
from s1_profiles import HECA, compartment
from s2_clock import fit, lodo, stage_value, treatment_group, ROOT, W
from s4_sensitivity import residuals

PATH = os.path.join(HECA, "endometriumAtlasV2_nuclei.h5ad")
K, N_SUPER, TG = 50, 3000, "progestin_only"


def _rows(args):
    lo, hi, keep, cols = args
    with h5py.File(PATH, "r") as f:
        g = f["X"]; ptr = g["indptr"][lo:hi + 1]; a, b = int(ptr[0]), int(ptr[-1])
        X = sp.csr_matrix((g["data"][a:b], g["indices"][a:b], ptr - a), shape=(hi - lo, int(g.attrs["shape"][1])))
    return X[keep[lo:hi]][:, cols]


def profiles(X, labels):
    """Mean rows per label, labels as a pandas Series aligned to X rows."""
    cats = pd.Index(sorted(labels.unique())); codes = cats.get_indexer(labels)
    G = sp.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(len(cats), len(codes)))
    n = np.bincount(codes, minlength=len(cats))
    return pd.DataFrame(np.asarray((G @ X).todense()) / n[:, None], index=cats), pd.Series(n, index=cats)


def tests(Pr, meta):
    """Clocks per compartment, H1 index test and position-matched residual test."""
    clk = {}
    for comp in ("stroma", "epithelium"):
        m = meta.compartment == comp; Mc, X = meta[m], Pr.loc[meta.index[m]].to_numpy()
        tr = Mc.y.notna().to_numpy(); pred = np.full(len(Mc), np.nan)
        pred[tr] = lodo(X[tr], Mc.y.to_numpy()[tr]); pred[~tr] = fit(X[tr], Mc.y.to_numpy()[tr])(X[~tr])
        clk[comp] = pd.Series(pred, index=Mc.donor.values)
    D = pd.concat(clk, axis=1).dropna().join(meta.drop_duplicates("donor").set_index("donor").group)
    D["index"] = D.stroma - D.epithelium
    nat, t = D[D.group == "natural"], D[D.group == TG]
    rn, rt = residuals(nat.stroma, nat.epithelium, (t.stroma, t.epithelium))
    return dict(n_treated=len(t), p_h1=float(mannwhitneyu(t["index"], nat["index"], alternative="greater").pvalue),
                p_residual=float(mannwhitneyu(rt, rn, alternative="less").pvalue), residual_treated=float(np.median(rt)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--draws", type=int, default=20); a = ap.parse_args()
    Pp = pd.read_parquet(os.path.join(W, "profiles_nuclei.parquet")); Mp = pd.read_csv(os.path.join(W, "profiles_nuclei_meta.csv"), index_col=0).loc[Pp.index]
    Mp["group"] = Mp["Hormonal treatment"].map(treatment_group); Mp["y"] = Mp.Stage.map(stage_value); Mp.loc[Mp.group != "natural", "y"] = np.nan
    Pp = Pp.loc[:, (Pp > 0).mean(0) > 0.5]
    genes = set()
    for comp in ("stroma", "epithelium"):
        nat = Pp[(Mp.compartment == comp) & (Mp.group == "natural") & (Mp.n >= K)]
        genes |= set(nat.var().nlargest(N_SUPER).index)
    with h5py.File(PATH, "r") as f:
        obs = read_elem(f["obs"])[["genotype", "lineage", "celltype"]].astype(str); var = pd.Index(read_elem(f["var"]).index)
    comp = pd.Series([compartment(l, c) for l, c in zip(obs.lineage, obs.celltype)], index=obs.index)
    keep = comp.isin(["stroma", "epithelium"]).to_numpy()
    gl = sorted(genes); cols = var.get_indexer(gl)
    jobs = [(lo, min(lo + 20000, len(obs)), keep, cols) for lo in range(0, len(obs), 20000)]
    with ProcessPoolExecutor(a.workers) as ex:
        X = sp.vstack(list(ex.map(_rows, jobs))).tocsr()
    cells = obs[keep].assign(comp=comp[keep].values).reset_index(drop=True)
    cells["label"] = cells.genotype + "|" + cells["comp"]
    info = Mp.drop_duplicates("donor").set_index("donor")[["group", "Stage"]]

    def meta_for(idx):
        m = pd.DataFrame({"donor": idx.str.split("|").str[0], "compartment": idx.str.split("|").str[1]}, index=idx)
        m = m.join(info, on="donor"); m["y"] = np.where(m.group == "natural", m.Stage.map(stage_value), np.nan)
        return m

    rng = np.random.default_rng(0); draws = []
    big = cells.groupby("label").size(); big = big[big >= K].index
    for b in range(a.draws):
        sel = np.concatenate([rng.choice(np.flatnonzero(cells.label.values == l), K, replace=False) for l in big])
        Pr, n = profiles(X[sel], cells.label.iloc[sel].reset_index(drop=True))
        r = tests(Pr, meta_for(Pr.index)); r["draw"] = b; draws.append(r)
        print(b, {k: round(v, 4) for k, v in r.items()}, flush=True)
    Dd = pd.DataFrame(draws)
    noeh = (cells.celltype != "eHormones").to_numpy()
    Pr, n = profiles(X[noeh], cells.label[noeh].reset_index(drop=True)); Pr = Pr[n.reindex(Pr.index).values >= K]
    eh = tests(Pr, meta_for(Pr.index))
    Pr, n = profiles(X, cells.label); Pr = Pr[n.values >= K]; full = tests(Pr, meta_for(Pr.index))
    summ = dict(genes=len(gl), nuclei_per_profile=K, draws=a.draws,
                downsampled=dict(n_treated=int(Dd.n_treated.median()), median_p_h1=float(f"{Dd.p_h1.median():.3g}"), share_p_h1_below_005=float(np.mean(Dd.p_h1 < 0.05)),
                                 median_p_residual=float(f"{Dd.p_residual.median():.3g}"), share_p_residual_below_005=float(np.mean(Dd.p_residual < 0.05)),
                                 median_residual_treated=round(float(Dd.residual_treated.median()), 3)),
                without_ehormones={k: (float(f"{v:.3g}") if isinstance(v, float) else v) for k, v in eh.items()},
                all_cells_gene_superset={k: (float(f"{v:.3g}") if isinstance(v, float) else v) for k, v in full.items()})
    json.dump(summ, open(os.path.join(ROOT, "outputs", "s11_cell_level.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))
