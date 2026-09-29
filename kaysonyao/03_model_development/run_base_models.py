"""Aim 3A base models across gestational windows and placenta.

Inputs
------
Full analyte matrix - no differential pre-filter. Selecting features from
differential results computed on the whole cohort lets the test set help decide
which features exist. Elastic net does the dimensionality reduction instead,
during fitting.

Datasets
--------
Plasma  : windowed by gestational age (scheme set by the binning step, e.g.
          T1-T5). Metabolomics and lipidomics by default; pass --datasets to
          include proteomics_plasma.
Placenta: a single unwindowed dataset. Placenta is collected once at delivery
          and carries no SampleGestAge, so it cannot be binned; this matches
          the previous pipeline, which ran placenta with timepoint="all".

Development set only. The locked 30% test set is never read here - it is opened
once, at final evaluation (F.5).

Artifacts written per dataset/window, restoring the previous pipeline's outputs
so `feature_interpretation.py` and `run_permutation_test.py` can consume them:

    sample_splits.csv          SampleID, split, Group, label
    selected_features.csv      union of features surviving selection
    tuned_hyperparams.json     per-fold params + params of the dev-refit model
    cv_results.csv             per-outer-fold metrics per model
    oof_predictions.csv        out-of-fold probability per sample per model
    summary.json               headline metrics, best model by OOF PR-AUC
    <model>.joblib             refit on the full development set
    scaler.joblib              fitted preprocessing of the best model
    X_dev_scaled.csv           dev matrix after impute+scale
    X_trainval_scaled.csv      alias of X_dev_scaled (downstream contract)
    y_test.csv                 dev labels, indexed by SampleID
    <model>_pr_curve.png / _roc_curve.png / _feature_importance.png
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

sys.path.insert(0, os.path.dirname(__file__))
from nested_cv import MODEL_NAMES, build_model, run_nested_cv  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

COMPLICATIONS = ["HDP", "FGR", "sPTB"]
METADATA_COLS = {
    "SampleID", "SubjectID", "Batch", "Group", "Subgroup",
    "GestAgeDelivery", "SampleGestAge", "MetadataCanonicalID",
    "Timepoint", "Tissue",
}


# Explicit mapping. The previous rule ("MTBL" -> Metabolomics, anything else ->
# Lipidomics) would have silently filed proteomics results as lipidomics.
ASSAYS = {"MTBL": "Metabolomics", "LIPD": "Lipidomics", "proteomics": "Proteomics"}


def assay_name(ds: str) -> str:
    for prefix, name in ASSAYS.items():
        if ds.startswith(prefix):
            return name
    raise ValueError(f"Unknown assay for dataset {ds!r}; add it to ASSAYS.")


def analyte_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in METADATA_COLS]


def _load(path: str, locked: pd.DataFrame):
    if not os.path.exists(path):
        logger.warning("  not found: %s", path)
        return None
    df = pd.read_csv(path, low_memory=False)
    df = df[df["Group"].isin(["Control"] + COMPLICATIONS)].copy()
    dev = set(locked.loc[locked["split"] == "dev", "SubjectID"])
    df = df[df["SubjectID"].isin(dev)]
    if df.empty:
        return None

    X = df[analyte_columns(df)].apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=1, how="all")

    # Safety net: the pipeline now drops post-merge samples missing more than
    # SAMPLE_MISSING_THRESHOLD of analytes (one ionization mode absent after the
    # POS/NEG join). This catches any matrix generated before that change, so an
    # older file cannot silently reintroduce whole-panel imputation.
    _frac = X.isna().mean(axis=1)
    _bad = _frac > 0.50
    if _bad.any():
        logger.warning(
            "%d sample(s) missing >50%% of analytes (likely one polarity absent) - "
            "dropping. Regenerate the matrix with the current pipeline to remove "
            "this warning.", int(_bad.sum()),
        )
        keep = ~_bad.values
        df, X = df[keep].reset_index(drop=True), X[keep].reset_index(drop=True)
    # NaNs are left in place: imputation happens inside the CV pipeline so it
    # refits per fold rather than borrowing medians across folds.
    y = df["Group"].isin(COMPLICATIONS).astype(int)
    return (df.reset_index(drop=True), X.reset_index(drop=True),
            y.reset_index(drop=True), df["SubjectID"].reset_index(drop=True))


def _write_artifacts(out_dir, meta, X, y, res, locked, random_state=42):
    os.makedirs(out_dir, exist_ok=True)
    sid = meta["SampleID"].astype(str)

    # sample_splits.csv - dev rows here; locked test rows appended for provenance
    dev_rows = pd.DataFrame({"SampleID": sid, "split": "train",
                             "Group": meta["Group"].values, "label": y.values})
    test_ids = locked.loc[locked["split"] == "test", ["SubjectID", "Group"]]
    test_rows = pd.DataFrame({
        "SampleID": test_ids["SubjectID"].astype(str), "split": "test",
        "Group": test_ids["Group"].values,
        "label": test_ids["Group"].isin(COMPLICATIONS).astype(int).values,
    })
    pd.concat([dev_rows, test_rows], ignore_index=True).to_csv(
        os.path.join(out_dir, "sample_splits.csv"), index=False)

    # per-fold metrics + OOF predictions
    cv_rows, oof = [], {"SampleID": sid}
    for model, m in res.items():
        for r in m["per_fold"]:
            cv_rows.append({"model": model, **r})
        oof[model] = m["oof_prob"]
    pd.DataFrame(cv_rows).to_csv(os.path.join(out_dir, "cv_results.csv"), index=False)
    pd.DataFrame(oof).to_csv(os.path.join(out_dir, "oof_predictions.csv"), index=False)

    best = max(res, key=lambda k: res[k]["pr_auc_oof"])

    # refit each model on the full development set
    hp = {"model_params": {}, "oof_pr_auc": {}, "params_per_fold": {}}
    selected = set()
    for model, m in res.items():
        # most common fold params as the refit configuration
        params = m["best_params_per_fold"][
            int(np.argmax([f["pr_auc"] for f in m["per_fold"]]))
        ]
        hp["model_params"][model] = params
        hp["oof_pr_auc"][model] = m["pr_auc_oof"]
        hp["params_per_fold"][model] = m["best_params_per_fold"]

        pipe = build_model(model, params, y, random_state)
        pipe.fit(X, y)
        joblib.dump(pipe, os.path.join(out_dir, f"{model.replace(' ', '_').replace('(', '').replace(')', '')}.joblib"))

        if "select" in pipe.named_steps:
            mask = pipe.named_steps["select"].get_support()
            selected |= set(X.columns[mask])
        elif hasattr(pipe.named_steps["clf"], "coef_"):
            coef = pipe.named_steps["clf"].coef_.ravel()
            selected |= set(X.columns[coef != 0])

        # curves + importance for the best model
        if model == best:
            prob = np.asarray(m["oof_prob"])
            for disp, fname, title in (
                (PrecisionRecallDisplay, "pr_curve", "PR curve"),
                (RocCurveDisplay, "roc_curve", "ROC curve"),
            ):
                fig, ax = plt.subplots(figsize=(5, 4))
                disp.from_predictions(y, prob, ax=ax, name=model)
                ax.set_title(f"{title} (out-of-fold) - {model}")
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, f"{fname}.png"), dpi=150)
                plt.close(fig)

            clf = pipe.named_steps["clf"]
            imp = getattr(clf, "feature_importances_", None)
            if imp is None and hasattr(clf, "coef_"):
                imp = np.abs(clf.coef_.ravel())
            if imp is not None:
                names = (X.columns[pipe.named_steps["select"].get_support()]
                         if "select" in pipe.named_steps else X.columns)
                s = pd.Series(imp, index=names).sort_values(ascending=False).head(25)
                fig, ax = plt.subplots(figsize=(7, 6))
                s[::-1].plot.barh(ax=ax)
                ax.set_title(f"Top 25 features - {model}")
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, "feature_importance.png"), dpi=150)
                plt.close(fig)
                s.to_csv(os.path.join(out_dir, "feature_importance.csv"), header=["importance"])

            prep = Pipeline_prefix(pipe)
            joblib.dump(prep, os.path.join(out_dir, "scaler.joblib"))
            Xs = pd.DataFrame(prep.transform(X), columns=X.columns, index=sid)
            Xs.to_csv(os.path.join(out_dir, "X_dev_scaled.csv"))
            Xs.to_csv(os.path.join(out_dir, "X_trainval_scaled.csv"))
            pd.Series(y.values, index=sid, name="label").to_csv(
                os.path.join(out_dir, "y_test.csv"), header=True)

    pd.Series(sorted(selected), name="feature").to_csv(
        os.path.join(out_dir, "selected_features.csv"), index=False)
    with open(os.path.join(out_dir, "tuned_hyperparams.json"), "w") as f:
        json.dump(hp, f, indent=2, default=float)

    summary = {
        "n_dev": int(len(y)),
        "n_participants": int(meta["SubjectID"].nunique()),
        "class_dist": {"control": int((y == 0).sum()), "complication": int((y == 1).sum())},
        "n_features_in": int(X.shape[1]),
        "n_features_selected_union": len(selected),
        "best_model_val": best,
        "best_oof_pr_auc": res[best]["pr_auc_oof"],
        "note": "Development set only; the locked 30% test set has not been read.",
        "metrics": {m: {k: v for k, v in d.items()
                        if k not in ("per_fold", "oof_prob", "oof_fold", "best_params_per_fold")}
                    for m, d in res.items()},
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    return best


def Pipeline_prefix(pipe):
    """Return the fitted impute+scale prefix of a pipeline, for reuse downstream."""
    from sklearn.pipeline import Pipeline
    return Pipeline([(n, s) for n, s in pipe.steps if n in ("impute", "scale")])


def main() -> None:
    ap = argparse.ArgumentParser(description="Aim 3A base models.")
    ap.add_argument("--repo-root", default=os.getcwd())
    ap.add_argument("--windows-root", default="data/cleaned/windows3_devfit")
    ap.add_argument("--placenta-root", default="data/cleaned/sop_omics_pipeline_devfit")
    ap.add_argument("--windows", nargs="+", default=["W1_early", "W2_mid", "W3_third"])
    ap.add_argument("--out", default="04_results_and_figures/models_v3")
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--outer-splits", type=int, default=5)
    ap.add_argument("--inner-splits", type=int, default=3)
    ap.add_argument("--models", nargs="+", default=MODEL_NAMES)
    ap.add_argument("--skip-placenta", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=["MTBL_plasma", "LIPD_plasma"],
                    help="Plasma datasets to model, each a folder under --windows-root "
                         "(e.g. MTBL_plasma LIPD_plasma proteomics_plasma).")
    args = ap.parse_args()

    root = os.path.abspath(args.repo_root)
    locked = pd.read_csv(os.path.join(root, "data", "cleaned", "locked_split.csv"))
    out_root = os.path.join(root, args.out)

    jobs = []
    for ds in args.datasets:
        for w in args.windows:
            jobs.append((ds, w, os.path.join(root, args.windows_root, ds, f"{ds}_echo_{w}.csv")))
    if not args.skip_placenta:
        for ds in ("MTBL_placenta", "LIPD_placenta"):
            jobs.append((ds, "all", os.path.join(
                root, args.placenta_root, ds, f"{ds}_cleaned_with_metadata.csv")))

    rows = []
    for ds, window, path in jobs:
        logger.info("=== %s / %s ===", ds, window)
        loaded = _load(path, locked)
        if loaded is None:
            continue
        meta, X, y, groups = loaded
        if y.nunique() < 2 or min(int(y.sum()), int((y == 0).sum())) < 5:
            logger.warning("  too few in one class - skipping")
            continue
        logger.info("  X=%s | complication=%d control=%d | participants=%d",
                    X.shape, int(y.sum()), int((y == 0).sum()), groups.nunique())

        res = run_nested_cv(X, y, groups, outer_splits=args.outer_splits,
                            inner_splits=args.inner_splits, n_trials=args.n_trials,
                            models=args.models)
        if not res:
            continue
        out_dir = os.path.join(out_root, ds, window)
        best = _write_artifacts(out_dir, meta, X, y, res, locked)
        logger.info("  best by OOF PR-AUC: %s", best)

        assay = assay_name(ds)
        tissue = "placenta" if "placenta" in ds else "plasma"
        for model, m in res.items():
            rows.append({
                "assay": assay, "tissue": tissue, "window": window, "model": model,
                "n": int(len(y)), "prevalence": round(m["prevalence"], 3),
                "feats_in": int(X.shape[1]),
                "feats_sel": round(m["n_features_mean"], 1) if m["n_features_mean"] else None,
                "pr_auc": round(m["pr_auc_mean"], 3),
                "pr_auc_oof": round(m["pr_auc_oof"], 3),
                "pr_lo": round(m["pr_auc_ci95"][0], 3), "pr_hi": round(m["pr_auc_ci95"][1], 3),
                # Pooled out-of-fold ROC-AUC, the same quantity the bootstrap CI
                # (roc_lo/roc_hi) and pr_auc_oof are computed on. This column used
                # to hold the mean over the 5 outer folds, which did not match its
                # own CI; the fold mean is kept separately for reference.
                "roc_auc": round(m["roc_auc_oof"], 3),
                "roc_auc_foldmean": round(m["roc_auc_mean"], 3),
                "roc_lo": round(m["roc_auc_ci95"][0], 3), "roc_hi": round(m["roc_auc_ci95"][1], 3),
                "accuracy": round(m["accuracy_mean"], 3), "f1": round(m["f1_mean"], 3),
                "precision": round(m["precision_mean"], 3),
                "recall_sens": round(m["recall_mean"], 3),
                "specificity": round(m["specificity_mean"], 3),
                "brier": round(m["brier_mean"], 3),
            })
        pd.DataFrame(rows).to_csv(os.path.join(out_root, "model_metrics_all.csv"), index=False)

    if rows:
        d = pd.DataFrame(rows)
        for assay, tag in (("Metabolomics", "metabolomics"), ("Lipidomics", "lipidomics"),
                           ("Proteomics", "proteomics")):
            sub = d[d.assay == assay]
            if not sub.empty:  # no empty per-assay files for assays not run
                sub.to_csv(os.path.join(out_root, f"model_metrics_{tag}.csv"), index=False)
        logger.info("Summary -> %s", os.path.join(out_root, "model_metrics_all.csv"))
    logger.info("Base models complete.")


if __name__ == "__main__":
    main()
