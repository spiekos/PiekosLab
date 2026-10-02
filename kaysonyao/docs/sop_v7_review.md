# SOP v7 review: code conformance and Jenn's suggested edits

30 Sep 2026. SOP file: `DP3_Metabolomics_Lipidomics_Preprocessing_SOP_v7 JK Edits 08122026.docx`.
"Base text" = v7 with Jenn Ko's tracked changes rejected (the authoritative SOP, per the PI).
"Suggestions" = her 40 tracked changes (75 insertions, 26 deletions). Code checked:
`01_data_cleaning/sop_omics_pipeline.py` (Jenn's version plus our fixes). Nothing has been
changed in the code or the SOP yet.

## Part A. Does the code follow the base text?

### A1. Critical, in both the SOP and the code: Steps 4 → 9 → 10 work on the wrong scale

Step 4 divides every feature by the ISTD geometric mean (or its class ISTD), which turns peak
areas of 10^4–10^9 into ratios of roughly 0.001–10. Step 9 then applies `log2(x + 1)`. For
x much smaller than 1, `log2(1 + x) ≈ 1.44·x`, so the "log transform" leaves most values
almost linear. Measured on the current `data/processed` outputs:

| Dataset | Median value | Values < 1 | Values < 0 |
|---|---|---|---|
| MTBL plasma | 0.008 | 92% | 3% |
| LIPD plasma | 0.000 | 99.9% | 35% |
| MTBL urine | 0.37 | 72% | 2% |
| MTBL placenta | 0.08 | 82% | 1% |

Consequences:
- **Step 10 imputation breaks.** "Minimum − 1" is a halving only on a true log2 scale. Here it
  puts imputed values at about −1 while observed values sit around 0.0001 (LIPD plasma POS
  median), thousands of times outside the observed spread. Only 0.4% of cells are imputed.
- **ComBat (Step 13) runs on skewed, non-log data**, against the SOP's own warning in Step 9, and
  those few extreme imputed cells dominate the per-feature means and variances it estimates.
  Tracing LIPD plasma through the pipeline: before ComBat nothing is negative except the 0.4%
  imputed cells; after ComBat 32% of POS values and 7% of NEG values are negative. The negative
  values in the table above come from this step. It is also a likely contributor to the ComBat
  over-compression we saw earlier.
- **Step 18's absolute thresholds** (IQR floor 0.1, ceiling 5.0 "on log2 scale") are meaningless.
  In all ten dataset × mode runs the pipeline's own log shows the 5th-percentile IQR below the 0.1
  floor and the 95th-percentile IQR (0.009–1.9) below the 5.0 ceiling, so the hypervariable rule
  never fires and the "constant" rule simply removes the bottom 5% by rank.
- Both our old pipeline and Jenn's have this, so every metabolomics/lipidomics result so far
  (including Table 4) carries it. Proteomics is unaffected (NPX is already log2).

Fix (needs one added sentence in SOP Step 4): after dividing by the ISTD reference, multiply by
that reference's median across all samples (per ionization mode; per ISTD for class-matched
normalization). Values return to the peak-area scale, relative differences are unchanged,
`log2(x+1)` becomes a real log transform, and Steps 10, 13 and 18 behave as written. Step 17 must
then compute RSD on the linear scale (see A5).

### A2. Other places the code differs from the base text

| # | Step | Base SOP says | Code does | Change |
|---|---|---|---|---|
| A2 | 18 IQR filter | IQR within the gestational-age bins of Appendix D | Uses the visit letter A–E at the end of the sample ID. For placenta that letter is the late-enrolment "E" in the ID, so IQR comes from those 21 of 124 samples only | Use Appendix D bins from `SampleGestAge`; placenta = one bin of all samples |
| A3 | 16 ISTD sample QC | Lipid POS: a sample failing only the neutral-lipid ISTDs (TG-d7, DG-d7, CE-d7) is noted and reviewed, not auto-excluded | Excludes every failing sample (an earlier run: 16 of 39 plasma POS flags and 5 of 7 placenta flags were neutral-lipid-only) | Flag those for review instead of excluding |
| A4 | 4 / App. C | "SM, Cer, HexCer, other sphingolipids" → SM-d9; log per-class routing and fallback counts | Hex2Cer, Hex3Cer and CerPE fall back to the pooled mean (plasma POS 19 / 11 / 13 features; placenta 29 / 27 / 10); log gives totals only | Route them to SM-d9; write the per-class table to the log |
| A5 | 17 RSD filter | One RSD across the QC pools, on batch-corrected data | Fails a feature if RSD > 30% in any single batch. Stricter: MTBL plasma POS 3,378 vs 2,694 features fail; NEG 2,642 vs 2,009; LIPD plasma POS 2,217 vs 1,919; NEG 371 vs 267 | Decide which rule (PI). After the A1 fix, compute RSD on back-transformed (linear) values |
| A6 | 7 feature missingness | Across all biological samples; drop if > 20% missing in either group | Jenn's suggested rule (per timepoint; drop only if both groups > 20%). Also counts samples with no Group as cases | Depends on suggestion B3; fix the missing-Group bug either way |
| A7 | 2(c) injection order | From the file index | Urine falls back to row order (its batch file has no raw-file names) | Minor: only the drift plots use it |

### A3. SOP text that is out of date with the code (update the SOP, not the code)

- Overview and Step 13 say ComBat uses pycombat, or sva via rpy2. We use `combat_ref.ComBatRef`, a
  Python port of sva::ComBat with reference-batch support, validated against sva to ~5e-14
  (pycombat has no reference-batch mode, which Step 13 requires).
- Step 13 says ref.batch "anchors batch parameter estimation to the bridge samples". It anchors to
  the reference batch; bridge samples are not used specially.
- Step 13 for prediction runs: we drop the outcome covariates from the ComBat design
  (`--blind-combat`) and fit on development participants only. Worth a sentence in the SOP.
- Steps 3, 4b and 5 contradict each other: Step 3 sends ISTD-assessed data to Step 4 and Step 4b
  says "proceed to Step 6" (skipping Step 5), while Step 5's note says it is fine with QC pools run
  at the end. The code runs both 4 and 5. Clarify which is intended.
- Appendix D: "Term <37.0" should be "≥ 37.0". Appendix D is also used by Steps 7, 8 and the
  binning, not only Step 18.
- Step 34 still says "timepoints A through E"; should be the Appendix D bins (T1–T5).

## Part B. Jenn's suggestions

| # | Where | Suggestion | Verdict |
|---|---|---|---|
| B1 | Config | Add `RT_ARTIFACT_MAX = 0.5 min`, `RT_ISOMER_MAX = 3.0 min` | **Accept.** Names the Step 23 cut-offs; the code already uses them |
| B2 | Step 6 | Choose 3–5 check features (metabolomics: highest mean intensity, non-ISTD; lipidomics: ≥ 2 classes), POS and NEG separately; reorder paragraphs | **Accept.** Clearer; the code does this |
| B3 | Step 7 | Compute missingness within each timepoint; drop only if > 20% missing in both cases and controls | **Accept with edits** (below) |
| B4 | Step 8 | Rename to "Sample-Level Filter and Flag"; label (a) "Missingness" | **Accept** |
| B5 | Step 8(b) | Flag samples ≤ 0.1 week before delivery; exclude post-delivery samples | **Accept, and go further for modelling** (below) |
| B6 | Step 8(c) | One sample per timepoint, closest to the "median of the gestational age range" | **Accept, reword** to "midpoint of the Appendix D bin (ties → earlier sample)". `bin_echo_windows.py` already does exactly this |
| B7 | Step 18 | "Remove features only if BOTH constant and hypervariable conditions are met" | **Reject.** A feature cannot be both; the base meaning (both thresholds within one category) is correct and is what the code does. Keep base text, fix the missing line break. The "(e)" label is fine |
| B8 | Part 3A intro | New outline of how dedup units are built (Steps 20–23, scoring, grouping keys) | **Accept with a fix.** The edit deleted "has already grouped under a named compound", which leaves a broken sentence and drops the scope of Part 3A. Restore it |
| B9 | Step 34, lipids | LPC(16:0) (falls) and Cer(d18:1_16:0) (rises); plot all adducts | **Accept.** Both are in our data and match: Spearman ρ with gestational age −0.77 and +0.53. The base panel's PC(38:6), PC(40:6), PE(40:6) are not in the final matrix under those names |
| B10 | Step 34, metabolites | Palmitoylcarnitine and L-kynurenine rise, L-threonine falls | **Revise before accepting** (below) |
| B11 | Several | Paragraph splits, no wording change | **Accept** |

**B3, Step 7.** Keeping a feature if at least one group is well detected is the standard "80%
rule" applied per group, and computing it per timepoint protects features that change over
pregnancy. Three edits:
1. It uses outcome labels, so for prediction it must be fitted inside the CV folds (already on
   the in-fold list). The base rule uses the groups too, so this is not new.
2. Say how timepoints combine. The code drops a feature if it fails at any timepoint, which
   removes features absent early but present later, against the rule's own rationale. "Drop only
   if it fails at every timepoint" fits the intent better. PI decision.
3. Define "time point" as the Appendix D bins; placenta is a single bin.

**B5, Step 8(b).** Excluding post-delivery samples is clearly right. For prediction models,
samples taken at delivery should also be excluded, not only flagged: for preterm cases the sample
date is the delivery date, which gives away the outcome. Earlier we found 7 of 30 W4 samples at or
after delivery, all complications. Suggested wording: "delivery sample: 0 ≤ GA_delivery −
GA_sample ≤ 0.1 weeks; post-delivery: GA_sample > GA_delivery". The code currently only flags both.

**B10, Step 34 metabolites.** Replacing the steroids is right: progesterone and cortisol are not
in the plasma export at all, and "estradiol" appears only as a spurious "17-Estradiol cyclooctyl
acetate". But the proposed set does not behave as stated in our data:
- palmitoylcarnitine is in the raw export but is filtered out before the final matrix;
- L-threonine rises with gestational age (ρ = +0.76), opposite to the stated decrease;
- L-kynurenine falls slightly (ρ = −0.14) instead of rising.
Pick metabolites that survive filtering and have literature-backed trends, or correct the
expected directions with citations. Update Appendix F ("steroids") to match. These correlations
are rank-based, so the A1 scale problem barely affects them, but re-check after the fix.

## Decisions for the PI

1. Approve the A1 rescaling fix (and rerun everything downstream).
2. Step 7 (B3): accept the per-group rule; drop at any timepoint or only at every timepoint?
3. Step 8(b) (B5): exclude delivery samples from modelling, or only flag them?
4. Step 17 (A5): one pooled RSD (SOP) or per-batch (code)?
5. Steps 3/4b/5: is Step 5 meant to run after Step 4?
6. Step 34 (B10): which metabolites?

## Code changes, once decided

- A1 rescale in Step 4; A5 linear-scale RSD (and pooled/per-batch rule as decided).
- A2 Step 18 on Appendix D bins; placenta as one bin of all samples.
- A3 Step 16 neutral-lipid-only → review flag.
- A4 route Hex2Cer / Hex3Cer / CerPE to SM-d9; per-class routing table in the log.
- A6 missing-Group bug; Step 7 timepoint rule as decided.
- B5 exclude post-delivery (and, if agreed, delivery) samples.
- B10 trajectory analyte list as decided.
