"""Final held-out evaluation (Aim 3C / F.5).

Procedure per dataset-window:
  1. Nested CV on the development participants -> honest development estimate
     and selection of the best model by out-of-fold PR-AUC.
  2. Refit that model on the FULL development set.
  3. Predict once on the held-out participants and report test metrics.

This requires preprocessing produced with `--fit-split dev`: ComBat and the
imputation floors estimated on development participants only, then applied to
everyone. Held-out participants are therefore present in the matrix but never
influenced any parameter.

THE TEST SET SURVIVES ONE LOOK. Re-running this after seeing the numbers, with
anything changed in response, converts it back into a development set. The
script refuses to overwrite an existing result directory for that reason.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

sys.path.insert(0, os.path.dirname(__file__))
from nested_cv import MODEL_NAMES, build_model, run_nested_cv, _bootstrap_ci  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

COMPLICATIONS = ["HDP", "FGR", "sPTB"]
METADATA_COLS = {
    "SampleID", "SubjectID", "Batch", "Group", "Subgroup",
    "GestAgeDelivery", "SampleGestAge", "MetadataCanonicalID",
    "Timepoint", "Tissue",
    # Added by sop_omics_pipeline.py (1-5 or "Delivery"). Must never be a feature:
    # it is ~constant within a window except for delivery samples, which it flags.
    "SampleTimepoint",
}


def analyte_columns(df):
    return [c for c in df.columns if c not in METADATA_COLS]


def _metrics(y, prob, seed=42):
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    pr_lo, pr_hi = _bootstrap_ci(y, prob, average_precision_score, random_state=seed)
    roc_lo, roc_hi = _bootstrap_ci(y, prob, roc_auc_score, random_state=seed)
    return {
        "pr_auc": average_precision_score(y, prob), "pr_lo": pr_lo, "pr_hi": pr_hi,
        "roc_auc": roc_auc_score(y, prob) if len(np.unique(y)) > 1 else np.nan,
        "roc_lo": roc_lo, "roc_hi": roc_hi,
        "accuracy": accuracy_score(y, pred), "f1": f1_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "recall_sens": recall_score(y, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "brier": brier_score_loss(y, prob),
        "prevalence": float(np.mean(y)),
    }


def _load(path, locked):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, low_memory=False)
    df = df[df["Group"].isin(["Control"] + COMPLICATIONS)].copy()
    dev = set(locked.loc[locked["split"] == "dev", "SubjectID"])
    tst = set(locked.loc[locked["split"] == "test", "SubjectID"])
    df["_split"] = np.where(df["SubjectID"].isin(dev), "dev",
                            np.where(df["SubjectID"].isin(tst), "test", "other"))
    df = df[df["_split"].isin(["dev", "test"])]
    if df.empty:
        return None
    cols = analyte_columns(df.drop(columns=["_split"]))
    X = df[cols].apply(pd.to_numeric, errors="coerce")
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
    y = df["Group"].isin(COMPLICATIONS).astype(int)
    return df.reset_index(drop=True), X.reset_index(drop=True), y.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="Held-out evaluation (one look).")
    ap.add_argument("--repo-root", default=os.getcwd())
    ap.add_argument("--windows-root", required=True)
    ap.add_argument("--placenta-root", default=None)
    ap.add_argument("--windows", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing result dir. The test set is meant to be "
                         "looked at once; only pass this if you know why.")
    args = ap.parse_args()

    root = os.path.abspath(args.repo_root)
    out_root = os.path.join(root, args.out)
    if os.path.exists(out_root) and not args.force:
        raise SystemExit(f"{out_root} exists. The held-out set survives one look; pass --force "
                         "only if you intend to overwrite a previous evaluation.")
    os.makedirs(out_root, exist_ok=True)
    locked = pd.read_csv(os.path.join(root, "data", "processed", "locked_split.csv"))

    jobs = []
    for ds in ("MTBL_plasma", "LIPD_plasma"):
        for w in args.windows:
            jobs.append((ds, w, os.path.join(root, args.windows_root, ds, f"{ds}_echo_{w}.csv")))
    if args.placenta_root:
        for ds in ("MTBL_placenta", "LIPD_placenta"):
            jobs.append((ds, "all", os.path.join(root, args.placenta_root, *ds.split("_", 1),
                                                 f"{ds}_cleaned_with_metadata.csv")))

    rows = []
    for ds, window, path in jobs:
        loaded = _load(path, locked)
        if loaded is None:
            logger.warning("%s / %s: not found", ds, window)
            continue
        meta, X, y = loaded
        dev_m = (meta["_split"] == "dev").values
        tst_m = (meta["_split"] == "test").values
        Xd, yd, gd = X[dev_m].reset_index(drop=True), y[dev_m].reset_index(drop=True), meta.loc[dev_m, "SubjectID"].reset_index(drop=True)
        Xt, yt = X[tst_m].reset_index(drop=True), y[tst_m].reset_index(drop=True)
        if yd.nunique() < 2 or yt.nunique() < 2 or min(int(yt.sum()), int((yt == 0).sum())) < 3:
            logger.warning("%s / %s: held-out set too small or single-class - skipping", ds, window)
            continue
        logger.info("=== %s / %s | dev n=%d (%d pos) | TEST n=%d (%d pos) ===",
                    ds, window, len(yd), int(yd.sum()), len(yt), int(yt.sum()))

        res = run_nested_cv(Xd, yd, gd, outer_splits=5, inner_splits=3,
                            n_trials=args.n_trials, models=MODEL_NAMES)
        if not res:
            continue
        best = max(res, key=lambda k: res[k]["pr_auc_oof"])
        logger.info("  best on dev (OOF PR-AUC): %s = %.3f", best, res[best]["pr_auc_oof"])

        assay = "Metabolomics" if ds.startswith("MTBL") else "Lipidomics"
        tissue = ds.split("_", 1)[1]  # plasma / placenta / urine
        for model, m in res.items():
            params = m["best_params_per_fold"][
                int(np.argmax([f["pr_auc"] for f in m["per_fold"]]))]
            pipe = build_model(model, params, yd)
            pipe.fit(Xd, yd)
            tm = _metrics(yt.values, pipe.predict_proba(Xt)[:, 1])
            rows.append({
                "assay": assay, "tissue": tissue, "window": window, "model": model,
                "selected_on_dev": model == best,
                "n_dev": int(len(yd)), "n_test": int(len(yt)),
                "dev_pr_auc_oof": round(m["pr_auc_oof"], 3),
                "dev_pr_lo": round(m["pr_auc_ci95"][0], 3), "dev_pr_hi": round(m["pr_auc_ci95"][1], 3),
                "dev_roc_auc": round(m["roc_auc_oof"], 3),
                "TEST_pr_auc": round(tm["pr_auc"], 3),
                "TEST_pr_lo": round(tm["pr_lo"], 3), "TEST_pr_hi": round(tm["pr_hi"], 3),
                "TEST_roc_auc": round(tm["roc_auc"], 3),
                "TEST_accuracy": round(tm["accuracy"], 3), "TEST_f1": round(tm["f1"], 3),
                "TEST_precision": round(tm["precision"], 3),
                "TEST_recall_sens": round(tm["recall_sens"], 3),
                "TEST_specificity": round(tm["specificity"], 3),
                "TEST_brier": round(tm["brier"], 3),
                "TEST_prevalence": round(tm["prevalence"], 3),
            })
            logger.info("    %-32s dev OOF %.3f -> TEST %.3f (baseline %.3f)",
                        model, m["pr_auc_oof"], tm["pr_auc"], tm["prevalence"])
        pd.DataFrame(rows).to_csv(os.path.join(out_root, "holdout_metrics_all.csv"), index=False)

    if rows:
        d = pd.DataFrame(rows)
        for assay, tag in (("Metabolomics", "metabolomics"), ("Lipidomics", "lipidomics")):
            d[d.assay == assay].to_csv(os.path.join(out_root, f"holdout_metrics_{tag}.csv"), index=False)
        with open(os.path.join(out_root, "PROVENANCE.json"), "w") as f:
            json.dump({"windows_root": args.windows_root, "n_trials": args.n_trials,
                       "note": "Test set evaluated once. Preprocessing fitted on dev only "
                               "(--blind-combat --fit-split dev)."}, f, indent=2)
    logger.info("Held-out evaluation complete.")


if __name__ == "__main__":
    main()
