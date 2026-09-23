# Legacy archive — 2026-08-11

Superseded artefacts retired during the August 2026 audit. Nothing here is
referenced by any current script; all four scripts that pointed at
`sop_omics_pipeline_v2` were repointed to `sop_omics_pipeline` before the move.

Total: ~586 MB.

---

## data_cleaned/

| Folder | Retired because |
|---|---|
| `sop_omics_pipeline_v2/` | **Misleading name — it is older, not newer.** Last run 2026-05-09, and only 2 of 4 datasets completed (no `MTBL_placenta` or `LIPD_placenta` logs). The live output is `data/cleaned/sop_omics_pipeline/`, which is the pipeline's default `--output-root` and now holds all 4 datasets regenerated 2026-08-11. |
| `omics_pipeline/` | Pre-SOP-v4 run (2026-04-22). Predates the v4 step reordering (Step 8 before log2; Steps 15–18 post-ComBat). |
| `metabolomics/` | Pre-SOP collaborator-integrated cleaning (2026-03-25). Superseded by the SOP-native pipeline. |
| `metabolomics_combat/` | Standalone ComBat experiment (2026-04-03). ComBat is now Step 13 inside the pipeline. |
| `metabolomics_dedup/` | Standalone deduplication experiment (2026-05-05). Dedup is now Steps 20–33 inside the pipeline. |
| `lipids/` | Pre-SOP lipid cleaning (2026-04-02). Superseded by `LIPD_plasma` / `LIPD_placenta`. |

## data_raw_intermediate/

| Folder | Retired because |
|---|---|
| `metabolomics_prenormalized/` | Byte-identical copy of `kaylaxu/data/MTBL_plasma/cleaned/prenormalized/`. Duplicate of an upstream intermediate; read by nothing. |

## model_results/

| Folder | Retired because |
|---|---|
| `models/` (1761 files, 219 MB) | **Results are void, not merely stale.** Every binary run used `lasso_feature_selection_binary()`, which called `LogisticRegressionCV(l1_ratios=..., solver="saga")` **without `penalty="elasticnet"`**. sklearn then defaults to `penalty="l2"`, silently ignores `l1_ratios`, and — because L2 never drives a coefficient to exactly zero — the downstream `coef != 0` test selected **every** feature. Fingerprint: identical selection counts at every timepoint (`metabolomics/plasma` = 1887 at A–E; `MTBL_sop_nodiff/plasma` = 502 at A–E). No dimensionality reduction ever occurred, so the `lasso_selected_features.csv` files and any biology read off them are unreliable. Fixed 2026-08-11. |

Also affected by the same runs, independent of the selector bug:
- Each per-timepoint model drew its **own independent random split**; among subjects present in all five timepoint models, 20 of 27 received different split assignments. A single locked participant-level split now replaces this.
- Runs consumed the old `dp3 master table v2.xlsx`, in which DP3-0343 was still misclassified as `sptb` rather than `FGR`.

## backups/

| Folder | Contents |
|---|---|
| `_backup_pre_20260811_142448/` | Pre-replacement copies of `dp3 master table v2.xlsx` (DP3-0343 = `sptb`) and `dp3 n=133 clinical and metadata.xlsx`. Kept for provenance of the reclassification. |

## docs/

| File | Retired because |
|---|---|
| `DP3_..._SOP_v3.docx` | Superseded by SOP v4 (2026-04-30), which the pipeline implements. |

---

## Related backup kept outside this archive

`kaylaxu/_backup_data_pre_20260811_144128/` — kayla's pre-re-extraction
`{MTBL,LIPD}_{plasma,placenta}` CSVs. Left in her folder because it is hers.
The placenta extractions there were built from an **older** workbook than the
May-2026 exports now in `data/`; the plasma ones are byte-identical to current.
