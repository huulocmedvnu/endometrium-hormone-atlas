# Plan: the human endometrium under progestins and combined hormonal therapy, a cell-resolved map with cycle clocks

Written 2026-10-11, before any comparison of treated and untreated donors. Deadline: ASPIRE 2027 abstract,
6 November 2026.

## Question

Under progestin-only therapy (desogestrel pill, etonogestrel implant, levonorgestrel coil) or combined oral
contraception, which natural-cycle position does each endometrial compartment take, are the compartments out of step
with each other, and how far is the window-of-implantation programme suppressed? Clinical context: progestin-primed
ovarian stimulation (freeze-all), levonorgestrel coil or GnRH agonist before frozen embryo transfer in endometriosis
and adenomyosis, and oral contraceptive scheduling before stimulation.

## What is known (checked 2026-10-11)

- HECA (Mareckova et al. 2024, Nat Genet) profiled donors on exogenous hormones but excluded them from all main
  analyses, showing only composition and UMAPs (Extended Data Fig. 5), and noted that hormones strongly change the
  epithelial transcriptome.
- No single-cell or single-nucleus study of progestin-only contraception in endometrium was found. Prior work is bulk
  or in vitro (progestin transcriptomes in stromal fibroblasts 2020, LAPC and endothelium 2014, T cells by flow).

## Data (HECA, Reproductive Cell Atlas, already on the server)

| set | treated donors | natural-cycle donors | values |
|---|---|---|---|
| snRNA-seq (primary), one lab and protocol | progestin-only 11 (desogestrel or Cerazette 4, implant 3, Mirena 4), combined pill 2, Zoladex 1 | 49 | log-normalised (no raw counts in the file) |
| scRNA-seq (secondary) | combined pill 10 (all Tan et al.), others 1 each | about 47 | raw counts |

Compartments: stroma = Mesenchymal lineage without perivascular cells (ePV, mPV) and uSMCs, epithelium = Epithelial
lineage (includes the eHormones state), endothelium, immune. The donor is the unit.

## Methods (fixed now)

- Donor x compartment profiles: mean log-normalised expression over nuclei (snRNA-seq), compartments with at least
  50 nuclei.
- Cycle clock per compartment: ridge regression on the natural-cycle donors with a clear stage, target
  proliferative (any proliferative label) = 0, secretory early = 1, secretory mid = 2, features = the 2,000 most
  variable genes among those donors, penalty chosen by inner cross-validation. Evaluation by leave-one-donor-out.
- Treated donors: predicted cycle position per compartment from the model trained on all natural donors.
- Dyssynchrony index per donor: stroma position minus epithelium position.
- Window-of-implantation programme: mean expression of the 50 genes most increased from proliferative to
  secretory mid in natural epithelium (defined on natural donors only), scored per donor.

## Positive controls and rules

- PC1 clock validity: leave-one-donor-out Spearman between predicted and true stage at least 0.5 (p < 0.05) for
  stroma and for epithelium. If either fails, that compartment's clock is not interpreted.
- PC2 known biology: progestins induce pseudo-decidualisation, so PRL and IGFBP1 in stroma are higher in
  progestin-only donors than in natural proliferative donors (one-sided Wilcoxon p < 0.05 for at least one of them).
  If PC2 fails, the treated-donor results are reported as descriptive only.

## Hypotheses

- H1: progestin-only stroma sits at a secretory-like position (above 1) while epithelium sits near proliferative
  (below 1), giving a dyssynchrony index above the natural range (one-sided Wilcoxon against natural donors).
- H2: the epithelial window-of-implantation programme is lower in progestin-only donors than in natural secretory mid
  donors (one-sided Wilcoxon).
- H3 (secondary, scRNA-seq): the combined pill gives the same direction, using a clock trained on the scRNA-seq
  natural donors.
- Exploratory: the three progestins separately, endometriosis status within treated donors, immune composition.

## Limitations to state

Three progestins pooled, all four Mirena users have endometriosis, n = 11, single-nucleus values are log-normalised,
the combined-pill donors come from one study.

## Compute and timeline

CPU, at most 16 workers, h5py row blocks. by 18 Oct profiles, clocks and controls. by 25 Oct H1 to H3, figure.
by 1 Nov abstract. 6 Nov submit.

## Extension (added 2026-10-11, before any endometriosis or age comparison was run)

HECA tested endometriosis by composition and differential expression, never by cycle timing, and did not study age.

- H4 (endometriosis, primary of the extension): natural-cycle donors only. Clocks are refitted on control donors
  only (controls scored leave-one-donor-out, cases by the model fitted on all controls). Clock residual = predicted
  minus histological stage (proliferative 0, secretory early 1, secretory early-mid 1.5, secretory mid 2). In
  secretory-phase donors, stromal residual is lower in endometriosis than in controls (one-sided Wilcoxon, p < 0.05),
  and falls with rASRM stage (Spearman rho < 0, one-sided p < 0.05). snRNA-seq and scRNA-seq are tested separately
  and combined by Stouffer's method. Epithelium is reported the same way.
- H5 (age, exploratory): Spearman correlation of age with clock residual and with the window-of-implantation score in
  natural-cycle donors, with phase as a covariate (partial correlation).
- Control: the residual in controls is centred on 0 (two-sided Wilcoxon, reported).

## Paper extension (added 2026-10-12, before any of these analyses was run)

Goal moves from abstract to full paper. Known before writing this section: H1 to H3 results, and that natural
secretory donors also have a small positive dyssynchrony (median about 0.1), so H1 against all natural donors may be
partly driven by clock position.

- S1 position-matched dyssynchrony (primary robustness check): in natural donors, regress epithelium clock on stroma
  clock (linear, plus a LOWESS check); treated residual = observed epithelium minus expected epithelium at the same
  stroma position. Pass if treated residuals are below 0 (one-sided Wilcoxon signed-rank p < 0.05) and below natural
  residuals (one-sided Mann-Whitney p < 0.05, natural residuals leave-one-donor-out). Also against natural secretory
  donors only (Mann-Whitney on the raw index).
- S2 endometriosis: H1, H2 and S1 repeated without donors whose pathology includes endometriosis (E), and with
  endometriosis-only natural donors as comparators (pathology-matched).
- S3 per progestin: medians per drug (desogestrel, implant, LNG-IUS), leave-one-drug-out H1 and S1.
- S4 permutation: 10,000 label permutations for H1, H2 and S1 (treated vs natural), exact p reported.
- S5 clock robustness: 1,000 and 5,000 variable genes, minimum 100 nuclei per compartment, clock trained on
  genes specific to each compartment only (top decile specificity, so that epithelium and stroma clocks share no
  genes); H1 and S1 repeated.
- S6 which epithelial programmes stay proliferative: genes of the natural proliferative-to-secretory-mid epithelial
  axis split by response in treated epithelium (follows stroma, stays proliferative), receptor expression (ESR1,
  PGR, AR) per compartment, epithelial substate composition (descriptive, exploratory).
- External validation: public bulk or single-cell data of progestin, contraceptive or programmed (HRT) cycles, chosen
  from a dataset search done before loading any of them. Compartment-specific clocks are moved to bulk by restricting
  each to genes specific to its compartment. Rule: report whatever comes out, direction expected as in H1.

Primary claim of the paper stands if S1 passes. If S1 fails, the paper reports stroma pseudo-decidualisation with WOI
suppression (H2), and dyssynchrony as not separable from clock position.

## External validation (added 2026-10-12, after the dataset search and before any external file was loaded)

Search result (agent, Europe PMC, GEO FTP, BioStudies, CELLxGENE): no paper reports per-compartment cycle clocks or
stroma-epithelium dyssynchrony under hormones. Chosen datasets:

- E-primary: Huang et al. 2026 (JCI Insight, doi 10.1172/jci.insight.202443), CELLxGENE collection
  e80f5450-0a0f-4312-a728-431d50aeb8fe, snRNA-seq (10x Flex), programmed (HRT) frozen transfer cycle 7 donors on
  progesterone day 6 against natural cycle 9 donors at LH+7. This is the clinical setting of the question.
  - Profiles: donor x compartment mean of log1p(counts per 10,000) from raw counts, compartments mapped from the
    authors' cell type labels as in S1 (perivascular and smooth muscle cells left out of stroma), at least 50 nuclei.
  - Clocks: trained on all HECA snRNA-seq natural donors (as S2), genes restricted to those measured in both sets,
    each set's genes z-scored within set before scoring (the two sets differ in chemistry). Huang donors scored only.
  - EC0 (transfer check): natural-cycle LH+7 donors score above 1 on both the stroma and the epithelium clock
    (median). If EC0 fails for a compartment, its clock is not interpreted in this set.
  - E1 (primary): dyssynchrony (stroma minus epithelium) higher in programmed than natural cycles (one-sided
    Mann-Whitney, p < 0.05).
  - E2: epithelial window-of-implantation score (the 50 HECA genes of H2) lower in programmed than natural cycles
    (one-sided Mann-Whitney).
  - E3 (reported): position-matched epithelial residual as in S1, natural fit taken from the HECA natural donors.
- E-secondary (bulk, reported with no gate): GSE137765 (LNG-IUS 11, COC 12, copper IUD 13, controls 11) and GSE60129
  (LNG-IUS 13, DMPA 8, mid-secretory controls 12). Clocks restricted to compartment-specific genes (S5 decile) and
  retrained on HECA pseudobulk of all compartments combined. Dyssynchrony expected higher in progestin users.
- Compartment transfer check (reported): GSE97929, sorted epithelium and stroma at LH+2 and LH+8 in the same women,
  each compartment clock should rise from LH+2 to LH+8 (paired one-sided Wilcoxon).
- Amendment (2026-10-12, after reading only the file structure of the Huang set, before any expression value was
  computed): z-scoring genes within each set removes absolute cycle position, so EC0 cannot be tested on it. Both
  sets are log1p(counts per 10,000), so the unscaled version is run as well. EC0 is tested on the unscaled version.
  E1 and E2 are run on both. The unscaled version is primary if EC0 passes there, otherwise the z-scored version is
  primary for E1 and E2 (a between-group contrast inside one set does not need absolute position).
  Compartments in Huang: stroma = stromal cell of endometrium, epithelium = glandular, luminal and ciliated
  epithelial cells (cervical epithelium left out), endothelium, immune = uNK, macrophage, B cell.
- Amendment (2026-10-12, after reading sample sheets only, before any expression value): in GSE97929 each woman gave
  either epithelium or stroma, not both, so the transfer check is unpaired (one-sided Mann-Whitney LH+8 above LH+2
  per compartment). Libraries of the same sample (_1, _2) are summed. Bulk clocks are trained on HECA snRNA-seq
  whole-tissue profiles (compartment means weighted by nuclei) over compartment-specific genes; both sets z-scored
  within set. Groups: GSE60129 progestin = LNG-IUS + DMPA against mid-secretory controls; GSE137765 hormonal =
  LNG-IUS + COC against non-hormonal = controls + copper IUD; each drug also reported alone.

## Review response (added 2026-10-12 after an internal review by two readers, all analyses below are post hoc)

The position-matched residual S1 was itself added after H1 results were known and is reported as a post hoc
robustness test. H1 (dyssynchrony index, with its sub-tests) remains the pre-specified primary test.

- R1 all compartments: residual of each compartment clock against the mean of the other compartments (natural fit),
  to separate stromal advance from epithelial delay.
- R2 fit restricted to staged natural donors, Huang E3 line fitted on leave-one-donor-out HECA values.
- R3 out-of-distribution: Mahalanobis distance of treated donors in the PCA space of natural donors per compartment.
- R4 joint model: residual ~ treatment + endometriosis + age + log nuclei, pooled natural and treated donors.
- R5 circularity: WOI and axis genes selected in one HECA dataset and tested in the other, donor-level paired test of
  stromal against epithelial response fraction.
- R6 ambient RNA: compartment-exclusive markers (EPCAM, KRT8 in stroma, PDGFRB, DCN in epithelium) by group.
- R7 bulk composition: epithelial and stromal content scores from structural markers, group effects adjusted for
  content, and a check for shared controls between GSE60129 and GSE137765.
- R8 Holm adjustment within the families of Table 2.
- R9 rank-based bulk clocks (added 2026-10-12 before any natural array was scored, post hoc): features are
  within-sample rank quantiles over genes shared by HECA, GPL570 and GPL6244, restricted to compartment-specific genes,
  ridge trained on HECA whole-tissue natural staged donors. Natural dated arrays: GSE4888 (Talbi 2006) and GSE6364
  controls (Burney 2007), proliferative 0, early secretory 1, mid secretory 2 (late secretory and ambiguous left out).
  RC0 transfer check: Spearman rho at least 0.5 with phase for both clocks in natural arrays. Natural line:
  epithelium on stroma across the natural arrays. RC1 platform check: residuals of GSE60129 mid-secretory controls
  centred on 0 (two-sided Wilcoxon, reported). Test: residual of progestin users below the natural-array residuals
  and below their own controls (one-sided Mann-Whitney), also adjusted for epithelial and stromal content.
  Exploratory: GSE6364 endometriosis against normal residual by phase.
- R10 cell-level checks (snRNA-seq, post hoc): (a) 50 nuclei per donor and compartment, 20 random draws, clocks
  retrained per draw (leave-one-donor-out for natural donors), residual and H1 tests per draw, reported as the median
  P and the share of draws with P < 0.05. (b) Epithelial profiles without the eHormones state, H1 and residual tests.

## Second review round (added 2026-10-12, before running, all post hoc)

- R11 epithelial sub-lineages (snRNA-seq): luminal (SOX9_luminal, preLuminal, Luminal) and glandular functionalis
  (SOX9_functionalis_I and II, preGlandular, Glandular, Glandular_secretory) profiles, at least 30 nuclei, eHormones
  left out (main) or added to the glandular set (variant), SOX9_basalis reported as a share only. Clock per
  sub-lineage trained on natural donors, residual against the stromal clock and stroma minus sub-lineage index,
  exposed against natural (one-sided Mann-Whitney).
- R12 Huang 2026 on rank quantiles: within-profile rank quantiles over genes shared with HECA, compartment clocks
  trained on HECA snRNA-seq natural profiles in rank space, transfer check as EC0, then E1 to E3. This does not
  replace the pre-specified unscaled test.
- R13 deconvolution: non-negative least squares with HECA snRNA-seq signatures (natural donors, linear CP10k, top
  50 specific genes per cell group: stroma, epithelium, endothelium, immune, perivascular and smooth muscle), checked
  on HECA whole-tissue profiles with known fractions, applied to GSE137765 and to the natural arrays (GSE60129 is
  gene-centred and has no absolute scale, so it is left out). Epithelial fraction by group and the R9 residual
  adjusted for it.

## R14 PGR cistrome (added 2026-10-12 before any ChIP-seq file was loaded, post hoc)

Data: GSE132712 (Chi et al. 2020, PGR ChIP-seq of whole endometrium, proliferative and mid-secretory, 2 replicates
each, inputs). Score per gene: mean mid-secretory PGR signal minus input in TSS +/- 10 kb, averaged over the two
replicates (genome build taken from the bigWig chromosome sizes). Gene sets from S5 (snRNA-seq, progestin-only):
epithelial cycle-up genes that follow (response fraction above 0.75) against those that stay (below 0.25), and the
same split for stroma. Hypothesis: genes that follow are bound more by PGR than genes that stay (one-sided
Mann-Whitney), consistent with direct PGR control of the genes that respond and indirect control of those that stay.
Reported also: the mid-secretory minus proliferative change in binding, and all expressed genes as background.

## Clock rigour (added 2026-10-12 before running, post hoc)

Known before writing: the snRNA-seq training set has 24 proliferative, 3 early and 9 mid-secretory donors, leave-one-
donor-out MAE 0.16 (stroma) and 0.24 (epithelium), calibration slopes 0.90 and 0.81, and the 3 early secretory donors
read 0.59 and 0.41.

- R15 mid-range validation: Wang 2020 10x donors (in the HECA scRNA-seq file, not used to train the snRNA-seq clocks),
  phase coded proliferative 0, early secretory 1, early-mid 1.5, mid 2, late secretory reported separately. snRNA-seq
  clocks in rank space (as R12) applied to Wang profiles. Reported: prediction by phase, Spearman with phase, MAE.
- R16 calibration: for each compartment, linear map from leave-one-donor-out prediction to stage fitted on natural
  donors, applied to all donors. H1 index and residual tests repeated on calibrated values.
- R17 uncertainty: 200 bootstrap resamples of natural training donors (stratified by stage), clocks refitted, natural
  donors scored out of bag and exposed donors by each refit. Per-donor 95% intervals, H1 index shift and P per
  replicate, reported as median and 2.5 to 97.5 percentiles and the share of replicates with P < 0.05.
- R18 model comparison by leave-one-donor-out: ridge (main), elastic net, multinomial logistic expected stage,
  k-nearest neighbours, and a marker score (mean of the 50 genes most increased minus the 50 most decreased from
  proliferative to mid-secretory, selected inside each fold, rescaled on training donors). MAE, RMSE, Spearman,
  calibration slope, and the H1 index test under each model.
