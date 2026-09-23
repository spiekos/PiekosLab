# Legacy archive — 2026-08-13: everything predating SOP-compliant ComBat

All artefacts here were produced with `pycombat_standard` batch correction.
**SOP Step 13 mandates reference-batch (`ref.batch`) anchoring, which none of
these runs used.** They are superseded by `data/cleaned/sop_pipeline_R_20260813`
and the results derived from it.

## Why they were superseded

`sva::ComBat(ref.batch=...)` was unavailable for the entire history of this
project. The pipeline attempted it, failed silently, and fell back to
`pycombat`, which cannot do reference-batch anchoring at all. Every log said
`pycombat_standard` and nobody noticed, because the run still succeeded. This
affected the collaborator's 2026-08-12 delivery as well.

R + sva were installed on 2026-08-13. `combat_fit_transform.R` now splits sva's
estimator into `combat_fit()` / `combat_apply()`, validated against
`sva::ComBat` to max |diff| = 3.6e-15, giving reference-batch anchoring *and*
fit-on-train / apply-to-test simultaneously — which no published package
provides.

## How much it mattered

Comparing the last pre-fix output (`sop_pipeline_ft2_20260813`, pycombat) with
the SOP-compliant one (`sop_pipeline_R_20260813`, sva ref.batch); both are
blind-ComBat and dev-fit, so batch correction is the only difference:

| dataset        | features (pycombat) | features (sva) | mean abs diff (shared) | max abs diff |
|----------------|--------------------:|---------------:|-----------------------:|-------------:|
| MTBL_plasma    |                 664 |           1399 |                  0.106 |        4.473 |
| LIPD_plasma    |                1386 |           1327 |                  0.022 |        8.350 |
| MTBL_placenta  |                1295 |           1588 |                  0.139 |        5.092 |
| LIPD_placenta  |                1822 |           2006 |                  0.003 |        3.100 |

MTBL_plasma retains more than twice as many features. Different batch
correction changes the variance structure, which changes what the RSD and IQR
filters remove. This is not a cosmetic difference.

## superseded_data/

| Directory | What it was |
|---|---|
| `sop_pipeline_20260812/` | Collaborator delivery. Label-aware ComBat (`protected=['Subgroup','SampleGestAge','GestAgeDelivery']`), fitted on all 530 samples. Outcome encoded in the predictors. |
| `sop_pipeline_devfit_20260812/` | First blind + dev-restricted run. Test participants *excluded* rather than transformed. |
| `sop_pipeline_fittransform_20260812/` | First `--fit-split dev` run, before the imputation fit-row ordering fix. |
| `sop_pipeline_ft2_20260813/` | Same with the ordering fixed. Last pycombat run. |
| `windows_*`, `visits_AE_20260813/` | Binnings derived from the above. |

## superseded_results/

| Directory | What it was |
|---|---|
| `models_5T/`, `models_ECHO2/` | Base models on the label-aware collaborator data. Tree models reached PR-AUC ~1.00 — leakage, not signal. |
| `models_5T_corrected/`, `models_ECHO2_corrected/` | Blind + dev-fit versions. These are the two zips sent to the PI on 2026-08-12. |
| `holdout_5T/`, `holdout_ECHO2/` | First held-out evaluations. |
| `holdout_*_halfmin/` | Reruns after moving half-minimum imputation into the CV folds. |
| `holdout_visits_AE/` | A–E visit timepoints, the PI's intended definition. |

**The two zips sent to the PI (`DP3_metrics_A_UNCORRECTED_*`,
`DP3_metrics_B_CORRECTED_*`) are built from `models_*_corrected/` and therefore
predate the ComBat fix.** The biased-vs-corrected contrast they show is still
qualitatively valid — that comparison was about label leakage, which is
independent of ref.batch — but the absolute numbers are superseded.

## Test-set usage

The held-out set was opened five times across the archived runs. Each was a
specified methodological change rather than a response to results, but F.5
requires a single look. The current plan is development-set only until
preprocessing is frozen, then one final evaluation.
