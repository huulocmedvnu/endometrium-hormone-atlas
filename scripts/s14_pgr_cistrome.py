"""S14 (R14, post hoc): PGR binding near the genes that follow or stay behind under progestins.
PGR ChIP-seq of whole endometrium (GSE132712, hg19 bigWig, normalised coverage). Score per gene: maximum over its
RefSeq TSSs of the mean signal in TSS +/- 10 kb, ChIP minus input, averaged over two replicates, for mid-secretory and
proliferative samples. Gene sets from S5 response fractions. Writes outputs/s14_pgr_cistrome.json and
outputs/s14_pgr_gene_scores.csv. Usage: s14_pgr_cistrome.py"""
import json, os
import numpy as np, pandas as pd, pyBigWig
from scipy.stats import mannwhitneyu
from s2_clock import ROOT, W

D = os.path.join(W, "ext", "pgr")
FLANK = 10_000
PAIRS = {"mid": [("GSM3890633_4_N275i10", "GSM3890634_5_Inputi79"), ("GSM3890636_2_N274i94", "GSM3890638_4_N274i96")],
         "prolif": [("GSM3890632_3_N252i8", "GSM3890634_5_Inputi79"), ("GSM3890635_1_N293i93", "GSM3890637_3_N293i95")]}
OUT = os.path.join(ROOT, "outputs")


def tss_table():
    r = pd.read_csv(os.path.join(D, "refGene_hg19.txt.gz"), sep="\t", header=None, usecols=[2, 3, 4, 5, 12])
    r.columns = ["chrom", "strand", "start", "end", "symbol"]
    r = r[r.chrom.str.match(r"^chr[0-9XY]+$")]
    r["tss"] = np.where(r.strand == "+", r.start, r.end)
    return r[["symbol", "chrom", "tss"]].drop_duplicates()


def window_means(bw_path, T):
    bw = pyBigWig.open(bw_path); size = bw.chroms(); out = np.full(len(T), np.nan)
    for i, (c, t) in enumerate(zip(T.chrom.values, T.tss.values)):
        if c in size:
            v = bw.stats(c, max(0, int(t) - FLANK), min(size[c], int(t) + FLANK), type="mean")[0]
            out[i] = 0.0 if v is None else v
    bw.close()
    return out


def follow_stay(G, comp, hi=0.75, lo=0.25):
    g = G[(G.compartment == comp) & (G.direction == "up")]
    return set(g.gene[g.fraction > hi]), set(g.gene[g.fraction < lo])


if __name__ == "__main__":
    T = tss_table()
    sig = {}
    for phase, pairs in PAIRS.items():
        reps = [window_means(os.path.join(D, f"{c}.bw"), T) - window_means(os.path.join(D, f"{i}.bw"), T) for c, i in pairs]
        sig[phase] = np.nanmean(reps, axis=0)
    T = T.assign(mid=sig["mid"], prolif=sig["prolif"])
    S = T.groupby("symbol")[["mid", "prolif"]].max(); S["change"] = S.mid - S.prolif
    P = pd.read_parquet(os.path.join(W, "profiles_nuclei.parquet")); expressed = P.columns[(P > 0).mean(0) > 0.5]
    S = S[S.index.isin(expressed)]
    summ = {"genes_scored": int(len(S)), "flank_bp": FLANK}
    for tag in ("nuclei", "cells_combined_pill"):
        G = pd.read_csv(os.path.join(OUT, f"s5_{tag}_genes.csv"))
        for comp in ("epithelium", "stroma"):
            f, s = follow_stay(G, comp); fs, ss = S.reindex(sorted(f)).dropna(), S.reindex(sorted(s)).dropna()
            up = S.reindex(G.gene[(G.compartment == comp) & (G.direction == "up")]).dropna()
            summ[f"{tag}_{comp}"] = dict(
                n_follow=len(fs), n_stay=len(ss), mid_follow=round(float(fs.mid.median()), 4), mid_stay=round(float(ss.mid.median()), 4),
                p_follow_more_bound=float(f"{mannwhitneyu(fs.mid, ss.mid, alternative='greater').pvalue:.3g}") if len(fs) > 1 and len(ss) > 1 else None,
                change_follow=round(float(fs.change.median()), 4), change_stay=round(float(ss.change.median()), 4),
                p_change=float(f"{mannwhitneyu(fs.change, ss.change, alternative='greater').pvalue:.3g}") if len(fs) > 1 and len(ss) > 1 else None,
                mid_all_up=round(float(up.mid.median()), 4), mid_background=round(float(S.mid.median()), 4),
                p_up_vs_background=float(f"{mannwhitneyu(up.mid, S.mid.drop(up.index, errors='ignore'), alternative='greater').pvalue:.3g}"))
    S.round(4).rename_axis("symbol").to_csv(os.path.join(OUT, "s14_pgr_gene_scores.csv"))
    json.dump(summ, open(os.path.join(OUT, "s14_pgr_cistrome.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))
