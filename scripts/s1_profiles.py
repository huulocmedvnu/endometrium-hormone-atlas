"""S1: donor x compartment mean expression profiles from the HECA h5ad files, read by h5py row blocks in a process
pool. snRNA-seq X is already log-normalised and is averaged as is. scRNA-seq raw counts are turned into
log1p(counts per 10,000) per cell first. Compartments: stroma (Mesenchymal without ePV, mPV, uSMCs), epithelium,
endothelium, immune. Writes $WORK/profiles_<set>.parquet (rows donor|compartment) and $WORK/profiles_<set>_meta.csv.
Usage: s1_profiles.py nuclei|cells [--workers 8]"""
import argparse, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd, scipy.sparse as sp, h5py
from anndata.io import read_elem

HECA = os.environ.get("HECA_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "heca"))
FILES = {"nuclei": ("endometriumAtlasV2_nuclei.h5ad", "X", False), "cells": ("endometriumAtlasV2_cells_with_counts.h5ad", "layers/counts", True)}
W = os.environ.get("WORK", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work"))
NOT_STROMA = ("ePV", "mPV", "uSMC")


def compartment(lineage, celltype):
    """Stroma, epithelium, endothelium or immune from HECA lineage and cell type (perivascular and smooth muscle cells
    are left out of stroma and get None)."""
    lin, ct = str(lineage), str(celltype)
    if lin == "Mesenchymal":
        return None if ct.startswith(NOT_STROMA) else "stroma"
    return {"Epithelial": "epithelium", "Endothelial": "endothelium", "Immune": "immune"}.get(lin)


def lognorm(X):
    lib = np.asarray(X.sum(1)).ravel(); lib[lib == 0] = 1
    Y = sp.csr_matrix(X.multiply(1e4 / lib[:, None])); Y.data = np.log1p(Y.data)
    return Y


def _block(args):
    path, key, lo, hi, codes, n, norm = args
    with h5py.File(path, "r") as f:
        g = f[key]; ptr = g["indptr"][lo:hi + 1]; a, b = int(ptr[0]), int(ptr[-1])
        X = sp.csr_matrix((g["data"][a:b], g["indices"][a:b], ptr - a), shape=(hi - lo, int(g.attrs["shape"][1])))
    if norm:
        X = lognorm(X)
    c = codes[lo:hi]; keep = c >= 0
    G = sp.csr_matrix((np.ones(keep.sum()), (c[keep], np.flatnonzero(keep))), shape=(n, hi - lo))
    return np.asarray((G @ X).todense())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("set", choices=list(FILES)); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--block", type=int, default=10000); a = ap.parse_args()
    fn, key, norm = FILES[a.set]; path = os.path.join(HECA, fn)
    with h5py.File(path, "r") as f:
        obs = read_elem(f["obs"]); genes = pd.Index(read_elem(f["var"]).index)
    comp = np.array([compartment(l, c) for l, c in zip(obs.lineage, obs.celltype)], dtype=object)
    lab = pd.Series(obs.genotype.astype(str).values + "|" + pd.Series(comp).astype(str).values)
    ok = pd.notna(comp)
    cats = pd.Index(sorted(lab[ok].unique())); codes = np.where(ok, cats.get_indexer(lab), -1)
    jobs = [(path, key, lo, min(lo + a.block, len(obs)), codes, len(cats), norm) for lo in range(0, len(obs), a.block)]
    with ProcessPoolExecutor(a.workers) as ex:
        S = sum(ex.map(_block, jobs))
    n = np.bincount(codes[codes >= 0], minlength=len(cats))
    P = pd.DataFrame(S / n[:, None], index=cats, columns=genes).astype(np.float32)
    d = obs.assign(_lab=lab.values)[ok].groupby("_lab", observed=True)
    meta = pd.DataFrame({"donor": cats.str.split("|").str[0], "compartment": cats.str.split("|").str[1], "n": n}, index=cats)
    for c in ["Hormonal treatment", "Stage", "Binary Stage", "Endometrial_pathology", "dataset", "Age"]:
        meta[c] = d[c].first().astype(str).reindex(cats).values
    os.makedirs(W, exist_ok=True)
    P.to_parquet(os.path.join(W, f"profiles_{a.set}.parquet")); meta.to_csv(os.path.join(W, f"profiles_{a.set}_meta.csv"))
    print(P.shape, meta.groupby("compartment").donor.nunique().to_dict())
