# Compartment-specific transcriptomic clocks in progestin-exposed endometrium

Code to reproduce the study *Compartment-specific transcriptomic clocks reveal stroma-epithelium discordance in
progestin-exposed endometrium* (T.H.L. Tran).

The study trains one transcriptomic cycle clock for the endometrial stroma and one for the epithelium on natural-cycle
donors of the Human Endometrial Cell Atlas, and applies both clocks to women using continuous progestins, combined oral
contraceptives, and programmed or natural cycles. All data are public and de-identified.

## Quick start

```bash
git clone https://github.com/huulocmedvnu/endometrium-hormone-atlas.git
cd endometrium-hormone-atlas
pip install -r requirements.txt
jupyter lab notebooks/reproduce.ipynb      # then Kernel > Restart Kernel and Run All Cells
```

The notebook

1. downloads every raw input (about 28 GB) and checks each file against `inputs/data_manifest.json`,
2. runs the analysis steps S1 to S15 in order, with a log for each step in `work/logs/`,
3. redraws Figures 1 to 4 and S1 to S4 in `figures/`,
4. runs the unit tests,
5. compares every regenerated number with the committed `outputs/` and prints `REPRODUCED` when all agree.

## Requirements

- Python 3.12 or 3.13 with the packages in `requirements.txt`, and `curl`.
- CPU only. About 30 GB of RAM at the peak, about 60 GB of free disk, about one hour on 16 cores after the download.

By default, downloads and derived files go to `work/` and the two atlas h5ad files to `data/heca/` inside the
repository. To use other locations, set them before starting Jupyter:

```bash
export WORK=/path/to/work
export HECA_DIR=/path/to/heca
```

The settings cell at the top of the notebook also sets the number of worker processes (`WORKERS`) and BLAS threads
(`THREADS`). Lower them on a smaller machine.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| S1 | `s1_profiles.py` | Mean expression per woman and compartment from the atlas h5ad files |
| S2 | `s2_clock.py` | Ridge regression cycle clocks per compartment, positive controls, primary hypothesis |
| S3 | `s3_endometriosis_age.py` | Endometriosis and age in natural-cycle donors |
| S4 | `s4_sensitivity.py` | Position-matched residual, per drug, without endometriosis |
| S5 | `s5_mechanism.py` | Genes that follow the cycle under hormones, window-of-implantation score, steroid receptors |
| S6 | `s6_external_huang.py` | Programmed against natural cycles (single-nucleus data) |
| S7 | `s7_external_bulk.py` | LH-timed sorted cells and contraceptive microarray cohorts |
| S8 | `s8_effects.py` | Hodges-Lehmann effect sizes with bootstrap confidence intervals |
| S9 to S15 | `s9_review.py` to `s15_clock_rigor.py` | Robustness analyses R1 to R18 (post hoc) |
| | `figures.py` | Figures from `outputs/` |

The pre-specified hypotheses, positive controls and dated amendments are in `PLAN.md`. Analyses added after the main
results are marked post hoc there.

## Repository layout

| Path | Content |
|---|---|
| `notebooks/reproduce.ipynb` | Runs the whole study from the public data |
| `scripts/` | Analysis steps S1 to S15, data download, figures |
| `outputs/` | Every reported number (JSON and CSV), used by the notebook as the reference |
| `inputs/data_manifest.json` | URL, size and SHA-256 of every raw input |
| `tests/` | Unit tests for the estimators |
| `PLAN.md` | Pre-specified analysis plan with dated amendments |

## Data

| Dataset | Source |
|---|---|
| Human Endometrial Cell Atlas, single-nucleus and single-cell RNA-seq (Marečková et al. 2024) | Reproductive Cell Atlas, ArrayExpress E-MTAB-14039 |
| Programmed and natural cycles, single-nucleus RNA-seq (Huang et al. 2026) | CZ CELLxGENE Discover, collection e80f5450-0a0f-4312-a728-431d50aeb8fe |
| LNG-IUS and DMPA users, microarray | GEO GSE60129 |
| Healthy volunteers using LNG-IUS, combined pill or copper IUD, microarray | GEO GSE137765 |
| Sorted stroma and epithelium at LH+2 and LH+8 | GEO GSE97929 |
| Natural dated endometrium, microarray | GEO GSE4888, GSE6364 |
| Progesterone receptor ChIP-seq of mid-secretory endometrium | GEO GSE132712 |

`scripts/fetch_data.py` downloads all of them.

## Contact

Tran Thai Huu Loc, MD, Department of Obstetrics and Gynecology, Hong Hung Hospital, Tay Ninh City, Viet Nam.
ORCID 0009-0009-1810-010X. tranthaihuuloc2@gmail.com
