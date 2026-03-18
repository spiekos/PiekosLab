"""
Binary classification pipeline for DP3 proteomics data.

For each tissue (plasma / placenta) × timepoint, predict any pregnancy complication
(Control=0 vs pooled Complication=1, where Complication = HDP + FGR + sPTB).

Steps per tissue × timepoint:
  1. Load cleaned data; label Control=0, any of HDP/FGR/sPTB=1 (Complication)
  2. 70 / 15 / 15 stratified train / val / test split
  3. Pearson correlation matrix of training features (saved as PNG)
  4. LASSO (L1 LogisticRegressionCV) feature selection on training set
  5. Optuna TPE hyperparameter tuning for each model: train on X_train, score PR-AUC on X_val
  6. 10-fold stratified CV on {LogisticRegression, RandomForest, XGBoost, SVM} with tuned params
  7. Best model (by val PR-AUC from tuning) retrained on train+val, evaluated on test set
  8. All results written to 04_results_and_figures/models/binary/<tissue>/<timepoint>/

Compare with multilabel_classifier.py which predicts each specific outcome separately.

Usage
-----
Run from the project root:

    python 03_model_development/binary_classifier.py [OPTIONS]

Options
-------
--plasma-dir    DIR   Directory with per-timepoint plasma CSVs
                      [default: data/cleaned/proteomics/normalized_sliced_by_suffix/]
--placenta-csv  FILE  Path to placenta CSV
                      [default: data/cleaned/proteomics/normalized_full_results/
                                proteomics_placenta_cleaned_with_metadata.csv]
--output-dir    DIR   Root output directory
                      [default: 04_results_and_figures/models/binary/]
--timepoints    A B   Which plasma timepoints to run (default: A B C D E)
--complications ...   Which outcome labels to pool as "Complication" (default: HDP FGR sPTB)
--n-trials      INT   Optuna trials per model (default: 50)
--skip-plasma         Skip plasma analysis
--skip-placenta       Skip placenta analysis
"""

# Standard library
import argparse
import json
import logging
import os
import sys

# Third-party
import joblib
import numpy as np
import pandas as pd

# Local
sys.path.insert(0, os.path.dirname(__file__))
from utilities import (
    OUTCOMES,
    TIMEPOINTS,
    RANDOM_STATE,
    load_data,
    load_significant_analytes,
    get_analyte_columns,
    normalise_group_labels,
    split_70_15_15,
    plot_correlation_matrix,
    lasso_feature_selection_binary,
    get_base_models_binary,
    tune_hyperparams_binary,
    build_tuned_model_binary,
    run_cv_binary,
    evaluate_binary,
    plot_pr_curve,
    plot_roc_curve,
    save_feature_importance,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_complication_labels(
    df: pd.DataFrame,
    complications: list,
) -> pd.Series | None:
    """
    Return a binary label Series: Control=0, Complication=1.
    Any sample whose Group is in *complications* is labelled 1.
    Samples with other Group values are dropped.
    Returns None if either class has fewer than 5 samples.
    """
    keep = set(["Control"] + complications)
    mask = df["Group"].isin(keep)
    sub  = df.loc[mask, "Group"].copy()
    y    = sub.isin(complications).astype(int)
    counts = y.value_counts()
    n_ctrl  = counts.get(0, 0)
    n_compl = counts.get(1, 0)
    if min(n_ctrl, n_compl) < 5:
        logger.warning(
            "  Control=%d  Complication=%d — too few samples, skipping.", n_ctrl, n_compl
        )
        return None
    logger.info("  Labels: Control=%d  Complication=%d", n_ctrl, n_compl)
    return y


def _get_feature_importance(model_name: str, model) -> np.ndarray | None:
    """Extract feature importances or coefficients from a fitted model."""
    if hasattr(model, "coef_"):
        return model.coef_.ravel()
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    return None


# ---------------------------------------------------------------------------
# Core per-dataset runner
# ---------------------------------------------------------------------------

def run_binary_pipeline(
    df: pd.DataFrame,
    tissue: str,
    timepoint: str,
    complications: list,
    output_dir: str,
    n_trials: int = 50,
    sig_analytes: list | None = None,
) -> dict | None:
    """
    Full binary classification pipeline for one (tissue, timepoint).

    Labels: Control=0, pooled Complication (HDP/FGR/sPTB)=1.
    Hyperparameters are tuned via Optuna TPE using the val set (PR-AUC objective).

    Parameters
    ----------
    sig_analytes : optional pre-filter list of analyte names from differential analysis.
                   When provided, only those columns are used as features.
                   If None or empty, all analyte columns are used (fallback).
    """
    os.makedirs(output_dir, exist_ok=True)
    tag = f"[{tissue} | tp={timepoint} | Control vs Complication]"
    logger.info("%s Starting binary pipeline.", tag)

    # ── 1. Build binary labels ─────────────────────────────────────────────
    df = normalise_group_labels(df)
    y_all = _make_complication_labels(df, complications)
    if y_all is None:
        return None

    all_analyte_cols = get_analyte_columns(df)

    if sig_analytes:
        # Intersect with columns actually present in this dataset
        analyte_cols = [c for c in sig_analytes if c in df.columns]
        if not analyte_cols:
            logger.warning(
                "%s None of the %d significant analytes found in dataset — "
                "falling back to all %d analytes.",
                tag, len(sig_analytes), len(all_analyte_cols),
            )
            analyte_cols = all_analyte_cols
        else:
            logger.info(
                "%s Pre-filtered to %d / %d significant analytes.",
                tag, len(analyte_cols), len(all_analyte_cols),
            )
    else:
        logger.info(
            "%s No significant analytes list — using all %d analytes.",
            tag, len(all_analyte_cols),
        )
        analyte_cols = all_analyte_cols

    X_all = df.loc[y_all.index, analyte_cols]

    # ── 2. 70 / 15 / 15 stratified split ──────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = split_70_15_15(X_all, y_all)
    logger.info(
        "%s Split — train=%d  val=%d  test=%d",
        tag, len(y_train), len(y_val), len(y_test),
    )

    # Save split indices with original Group labels for traceability
    split_df = pd.concat([
        pd.DataFrame({"SampleID": X_train.index, "split": "train"}),
        pd.DataFrame({"SampleID": X_val.index,   "split": "val"}),
        pd.DataFrame({"SampleID": X_test.index,  "split": "test"}),
    ])
    split_df["Group"] = df.loc[split_df["SampleID"], "Group"].values
    split_df["label"] = y_all.loc[split_df["SampleID"]].values
    split_df.to_csv(os.path.join(output_dir, "sample_splits.csv"), index=False)

    # ── 3. Correlation matrix (training features, pre-LASSO) ──────────────
    plot_correlation_matrix(
        X_train,
        output_path=os.path.join(output_dir, "correlation_matrix_pretrain.png"),
        title=f"Pearson correlation — {tissue} {timepoint} (Control vs Complication)",
    )

    # ── 4. LASSO feature selection on training set ─────────────────────────
    selected_features = lasso_feature_selection_binary(X_train, y_train)
    if len(selected_features) == 0:
        logger.warning("%s LASSO selected 0 features — skipping.", tag)
        return None

    pd.Series(selected_features, name="feature").to_csv(
        os.path.join(output_dir, "lasso_selected_features.csv"), index=False
    )

    X_train_sel = X_train[selected_features]
    X_val_sel   = X_val[selected_features]
    X_test_sel  = X_test[selected_features]

    plot_correlation_matrix(
        X_train_sel,
        output_path=os.path.join(output_dir, "correlation_matrix_postlasso.png"),
        title=f"Pearson correlation (LASSO) — {tissue} {timepoint} (Control vs Complication)",
    )

    # ── 5. Optuna TPE tuning on val set ───────────────────────────────────
    model_names = list(get_base_models_binary().keys())
    logger.info("%s Optuna tuning (%d trials/model) …", tag, n_trials)

    tuned_params     = {}
    tuned_val_scores = {}
    for model_name in model_names:
        logger.info("%s  Tuning %s …", tag, model_name)
        best_params, best_val_pr = tune_hyperparams_binary(
            model_name, X_train_sel, y_train, X_val_sel, y_val,
            n_trials=n_trials, random_state=RANDOM_STATE,
        )
        tuned_params[model_name]     = best_params
        tuned_val_scores[model_name] = best_val_pr

    # Persist tuned hyperparameters (JSON-serialisable; max_depth=None → null)
    with open(os.path.join(output_dir, "tuned_hyperparams.json"), "w") as fh:
        json.dump({"model_params": tuned_params, "val_pr_auc": tuned_val_scores}, fh, indent=2)

    # ── 6. 10-fold CV on train set with tuned params ───────────────────────
    cv_results = {}
    for model_name in model_names:
        model  = build_tuned_model_binary(model_name, tuned_params[model_name], y_train)
        cv_res = run_cv_binary(model, X_train_sel, y_train)
        cv_results[model_name] = cv_res
        logger.info(
            "%s  [CV] %-20s  PR-AUC=%.3f±%.3f  ROC-AUC=%.3f±%.3f",
            tag, model_name,
            cv_res["pr_auc_mean"],  cv_res["pr_auc_std"],
            cv_res["roc_auc_mean"], cv_res["roc_auc_std"],
        )

    cv_df = pd.DataFrame(cv_results).T
    cv_df.index.name = "model"
    cv_df.to_csv(os.path.join(output_dir, "cv_results.csv"))

    # ── 7. Best model by val PR-AUC, retrained on train+val → test ────────
    best_model_name = max(tuned_val_scores, key=lambda m: tuned_val_scores[m])
    logger.info(
        "%s Best model (val PR-AUC=%.4f): %s",
        tag, tuned_val_scores[best_model_name], best_model_name,
    )

    X_trainval = pd.concat([X_train_sel, X_val_sel])
    y_trainval = pd.concat([y_train, y_val])

    test_results = {}
    _saved_scaler = None
    for model_name in model_names:
        model = build_tuned_model_binary(model_name, tuned_params[model_name], y_trainval)
        metrics, scaler = evaluate_binary(model, X_trainval, y_trainval, X_test_sel, y_test)
        test_results[model_name] = metrics

        # Persist fitted model
        joblib.dump(model, os.path.join(output_dir, f"{model_name}.joblib"))

        if model_name == best_model_name:
            _saved_scaler = scaler
            X_te_s = scaler.transform(X_test_sel)
            y_prob = model.predict_proba(X_te_s)[:, 1]
            plot_pr_curve(
                y_test.values, y_prob,
                title=f"PR curve — {model_name} | {tissue} {timepoint} (Control vs Complication)",
                output_path=os.path.join(output_dir, f"{model_name}_pr_curve.png"),
            )
            plot_roc_curve(
                y_test.values, y_prob,
                title=f"ROC curve — {model_name} | {tissue} {timepoint} (Control vs Complication)",
                output_path=os.path.join(output_dir, f"{model_name}_roc_curve.png"),
            )

        fi = _get_feature_importance(model_name, model)
        if fi is not None:
            save_feature_importance(
                selected_features, fi,
                output_path=os.path.join(output_dir, f"{model_name}_feature_importance.png"),
                title=f"Top features — {model_name} | {tissue} {timepoint}",
            )

        logger.info(
            "%s  [Test] %-20s  PR-AUC=%.3f  ROC-AUC=%.3f  F1=%.3f  Acc=%.3f",
            tag, model_name,
            metrics["pr_auc"], metrics["roc_auc"], metrics["f1"], metrics["accuracy"],
        )

    # Persist scaler + scaled datasets for feature interpretation
    if _saved_scaler is not None:
        joblib.dump(_saved_scaler, os.path.join(output_dir, "scaler.joblib"))
        X_tv_sc = _saved_scaler.transform(X_trainval)
        X_te_sc = _saved_scaler.transform(X_test_sel)
        pd.DataFrame(X_tv_sc, columns=selected_features, index=X_trainval.index).to_csv(
            os.path.join(output_dir, "X_trainval_scaled.csv")
        )
        pd.DataFrame(X_te_sc, columns=selected_features, index=X_test_sel.index).to_csv(
            os.path.join(output_dir, "X_test_scaled.csv")
        )
        y_test.to_csv(os.path.join(output_dir, "y_test.csv"), header=True)

    test_df = pd.DataFrame(test_results).T
    test_df.index.name = "model"
    test_df.to_csv(os.path.join(output_dir, "test_results.csv"))

    def _class_dist(y: pd.Series) -> dict:
        vc = y.value_counts()
        return {"control": int(vc.get(0, 0)), "complication": int(vc.get(1, 0))}

    summary = {
        "tissue":                tissue,
        "timepoint":             timepoint,
        "outcome":               "Complication",
        "complications_pooled":  complications,
        "n_train":               int(len(y_train)),
        "n_val":                 int(len(y_val)),
        "n_test":                int(len(y_test)),
        "class_dist_train":      _class_dist(y_train),
        "class_dist_val":        _class_dist(y_val),
        "class_dist_test":       _class_dist(y_test),
        "n_features_pretlasso":  len(analyte_cols),
        "n_features_postlasso":  len(selected_features),
        "best_model_val":        best_model_name,
        "best_val_pr_auc":       tuned_val_scores[best_model_name],
        "test_metrics":          test_results,
    }

    with open(os.path.join(output_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    logger.info("%s Done.", tag)
    return summary


# ---------------------------------------------------------------------------
# Plasma / placenta loops
# ---------------------------------------------------------------------------

def run_plasma(
    plasma_dir: str,
    output_root: str,
    timepoints: list,
    complications: list,
    n_trials: int,
    sig_analytes_dir: str | None = None,
) -> list:
    summaries = []
    for tp in timepoints:
        csv_path = os.path.join(
            plasma_dir, f"proteomics_plasma_formatted_suffix_{tp}.csv"
        )
        if not os.path.exists(csv_path):
            logger.warning("Plasma timepoint %s CSV not found: %s — skipping.", tp, csv_path)
            continue

        df = load_data(csv_path)
        logger.info("Plasma timepoint %s loaded: %d samples × %d cols", tp, *df.shape)

        sig_analytes = None
        if sig_analytes_dir:
            sig_csv = os.path.join(
                sig_analytes_dir, "plasma", "cross_sectional", tp,
                "Control_vs_Complication_differential_results.csv",
            )
            sig_analytes = load_significant_analytes(sig_csv)
            if sig_analytes:
                logger.info("Plasma tp=%s: %d significant analytes loaded (q<0.05).", tp, len(sig_analytes))
            else:
                logger.warning(
                    "Plasma tp=%s: no significant analytes found (q<0.05) — using all features.", tp
                )

        out_dir = os.path.join(output_root, "plasma", tp)
        result  = run_binary_pipeline(
            df, "plasma", tp, complications, out_dir, n_trials, sig_analytes
        )
        if result:
            summaries.append(result)
    return summaries


def run_placenta(
    placenta_csv: str,
    output_root: str,
    complications: list,
    n_trials: int,
    sig_analytes_dir: str | None = None,
) -> list:
    if not os.path.exists(placenta_csv):
        logger.warning("Placenta CSV not found: %s — skipping.", placenta_csv)
        return []

    df = load_data(placenta_csv)
    logger.info("Placenta loaded: %d samples × %d cols", *df.shape)

    sig_analytes = None
    if sig_analytes_dir:
        sig_csv = os.path.join(
            sig_analytes_dir, "placenta", "cross_sectional",
            "Control_vs_Complication_differential_results.csv",
        )
        sig_analytes = load_significant_analytes(sig_csv)
        if sig_analytes:
            logger.info("Placenta: %d significant analytes loaded (q<0.05).", len(sig_analytes))
        else:
            logger.warning("Placenta: no significant analytes found (q<0.05) — using all features.")

    out_dir = os.path.join(output_root, "placenta", "all")
    result  = run_binary_pipeline(
        df, "placenta", "all", complications, out_dir, n_trials, sig_analytes
    )
    return [result] if result else []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    wkdir = os.getcwd()
    p = argparse.ArgumentParser(
        description=(
            "Binary classification: Control vs pooled Complication (HDP+FGR+sPTB). "
            "Hyperparameters tuned via Optuna TPE on the validation set."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--plasma-dir",
        default=os.path.join(
            wkdir, "data", "cleaned", "proteomics", "normalized_sliced_by_suffix"
        ),
    )
    p.add_argument(
        "--placenta-csv",
        default=os.path.join(
            wkdir, "data", "cleaned", "proteomics", "normalized_full_results",
            "proteomics_placenta_cleaned_with_metadata.csv",
        ),
    )
    p.add_argument(
        "--output-dir",
        default=os.path.join(wkdir, "04_results_and_figures", "models", "binary"),
    )
    p.add_argument("--timepoints",    nargs="+", default=TIMEPOINTS)
    p.add_argument(
        "--complications", nargs="+", default=OUTCOMES,
        help="Group labels to pool as 'Complication'.",
    )
    p.add_argument(
        "--n-trials", type=int, default=50,
        help="Optuna trials per model per condition.",
    )
    p.add_argument(
        "--sig-analytes-dir",
        default=os.path.join(wkdir, "04_results_and_figures", "differential_analysis"),
        help=(
            "Root directory of differential analysis outputs containing "
            "Control_vs_Complication_significant_analytes.csv files. "
            "Features are restricted to significant analytes where available; "
            "timepoints with no significant analytes fall back to all features. "
            "Pass '' to disable and always use all features."
        ),
    )
    p.add_argument("--skip-plasma",   action="store_true")
    p.add_argument("--skip-placenta", action="store_true")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    logger.info("Pooling as Complication: %s", args.complications)
    logger.info("Optuna trials per model: %d", args.n_trials)
    sig_dir = args.sig_analytes_dir if args.sig_analytes_dir else None
    if sig_dir:
        logger.info("Significant analytes dir: %s", sig_dir)
    else:
        logger.info("No significant analytes filter — using all features.")
    all_summaries = []

    if not args.skip_plasma:
        logger.info("=== PLASMA ===")
        all_summaries += run_plasma(
            args.plasma_dir, args.output_dir,
            args.timepoints, args.complications, args.n_trials,
            sig_analytes_dir=sig_dir,
        )

    if not args.skip_placenta:
        logger.info("=== PLACENTA ===")
        all_summaries += run_placenta(
            args.placenta_csv, args.output_dir,
            args.complications, args.n_trials,
            sig_analytes_dir=sig_dir,
        )

    if all_summaries:
        rows = []
        for s in all_summaries:
            for model_name, metrics in s["test_metrics"].items():
                row = {
                    "tissue":    s["tissue"],
                    "timepoint": s["timepoint"],
                    "model":     model_name,
                }
                row.update({k: round(v, 4) for k, v in metrics.items()})
                rows.append(row)

        os.makedirs(args.output_dir, exist_ok=True)

        summary_df = pd.DataFrame(rows)
        csv_path   = os.path.join(args.output_dir, "all_results_summary.csv")
        summary_df.to_csv(csv_path, index=False)
        logger.info("Aggregate summary saved → %s", csv_path)

    logger.info("Binary classifier pipeline complete.")


if __name__ == "__main__":
    main()
