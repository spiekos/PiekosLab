"""
Multi-label classification pipeline for DP3 proteomics data.

For each tissue (plasma / placenta) x timepoint, jointly predict HDP / FGR / sPTB:
  1. Load cleaned data; keep Control + all complication rows; encode as binary columns
  2. 70 / 15 / 15 train / val / test split (no stratification - multi-label)
  3. Pearson correlation matrix of training features (saved as PNG)
  4. Multi-task LASSO (MultiTaskLassoCV) for joint feature selection
  5. Optuna TPE hyperparameter tuning for each model: train on X_train, score
     macro-average PR-AUC on X_val across all outcomes
  6. 10-fold KFold CV on {LogisticRegression, RandomForest, XGBoost, SVM}
     (wrapped in MultiOutputClassifier) with tuned hyperparameters
  7. Best model (by val macro PR-AUC from tuning) retrained on train+val, evaluated on test
  8. All results written to 04_results_and_figures/models/multilabel/<tissue>/<timepoint>/

Usage
-----
Run from the project root:

    python 03_model_development/multilabel_classifier.py [OPTIONS]

Options
-------
--plasma-dir    DIR   Directory with per-timepoint plasma CSVs
                      [default: data/cleaned/proteomics/normalized_sliced_by_suffix/]
--placenta-csv  FILE  Path to placenta CSV
                      [default: data/cleaned/proteomics/normalized_full_results/
                                proteomics_placenta_cleaned_with_metadata.csv]
--output-dir    DIR   Root output directory
                      [default: 04_results_and_figures/models/multilabel/]
--timepoints    A B   Which plasma timepoints to run (default: A B C D E)
--outcomes      ...   Which outcomes to model jointly (default: HDP FGR sPTB)
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
    lasso_feature_selection_multilabel,
    tune_hyperparams_multilabel,
    build_tuned_model_multilabel,
    run_cv_multilabel,
    evaluate_multilabel,
    plot_pr_curve,
    plot_roc_curve,
    save_feature_importance,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_multilabel_targets(
    df: pd.DataFrame,
    outcomes: list,
) -> pd.DataFrame | None:
    """
    Construct a binary indicator DataFrame (0/1) for each outcome column.
    Rows whose Group is not in {Control} union outcomes are dropped.

    Returns None if any outcome has fewer than 5 positive samples.
    """
    keep_groups = set(["Control"] + outcomes)
    mask = df["Group"].isin(keep_groups)
    sub  = df.loc[mask]

    Y = pd.DataFrame(index=sub.index)
    for outcome in outcomes:
        Y[outcome] = (sub["Group"] == outcome).astype(int)
        if Y[outcome].sum() < 5:
            logger.warning(
                "Outcome '%s' has only %d positive samples - skipping tissue/timepoint.",
                outcome, Y[outcome].sum()
            )
            return None

    return Y

# ---------------------------------------------------------------------------
# Core per-dataset runner
# ---------------------------------------------------------------------------

def run_multilabel_pipeline(
    df: pd.DataFrame,
    tissue: str,
    timepoint: str,
    outcomes: list,
    output_dir: str,
    n_trials: int = 50,
    sig_analytes: list | None = None,
) -> dict | None:
    """
    Run the multi-label classification pipeline for one (tissue, timepoint).

    Hyperparameters are tuned via Optuna TPE using the val set
    (macro-average PR-AUC across all outcomes as the objective).

    Parameters
   ----------
    df           : cleaned wide-format DataFrame
    tissue       : 'plasma' or 'placenta'
    timepoint    : timepoint label, e.g. 'A'; use 'all' for placenta
    outcomes     : list of outcome strings, e.g. ['HDP', 'FGR', 'sPTB']
    output_dir   : where to save results
    n_trials     : number of Optuna trials per model
    sig_analytes : optional pre-filter list of analyte names from differential analysis.
                   When provided, only those columns are used as features.
                   If None or empty, all analyte columns are used (fallback).

    Returns
   -------
    summary dict or None if skipped
    """
    os.makedirs(output_dir, exist_ok=True)
    tag = f"[{tissue} | tp={timepoint} | multilabel]"
    logger.info("%s Starting multi-label pipeline. Outcomes: %s", tag, outcomes)

    # 1. Build multi-label targets
    df = normalise_group_labels(df)
    Y_all = _build_multilabel_targets(df, outcomes)
    if Y_all is None:
        return None

    all_analyte_cols = get_analyte_columns(df)

    if sig_analytes:
        analyte_cols = [c for c in sig_analytes if c in df.columns]
        if not analyte_cols:
            logger.warning(
                "%s None of the %d significant analytes found in dataset - "
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
            "%s No significant analytes list - using all %d analytes.",
            tag, len(all_analyte_cols),
        )
        analyte_cols = all_analyte_cols

    X_all = df.loc[Y_all.index, analyte_cols]

    # 2. 70 / 15 / 15 split (stratify on first outcome as proxy)
    strat_col = Y_all.iloc[:, 0]
    X_train, X_val, X_test, y_train_s, y_val_s, y_test_s = split_70_15_15(
        X_all, strat_col, stratify=True
    )
    Y_train = Y_all.loc[X_train.index]
    Y_val   = Y_all.loc[X_val.index]
    Y_test  = Y_all.loc[X_test.index]

    logger.info(
        "%s Split - train=%d  val=%d  test=%d",
        tag, len(Y_train), len(Y_val), len(Y_test)
    )

    split_rows = (
        [{"SampleID": i, "split": "train"} for i in X_train.index]
        + [{"SampleID": i, "split": "val"}   for i in X_val.index]
        + [{"SampleID": i, "split": "test"}  for i in X_test.index]
    )
    pd.DataFrame(split_rows).to_csv(
        os.path.join(output_dir, "sample_splits.csv"), index=False
    )

    # 3. Correlation matrix (pre-LASSO)
    plot_correlation_matrix(
        X_train,
        output_path=os.path.join(output_dir, "correlation_matrix_pretrain.png"),
        title=f"Pearson correlation - {tissue} {timepoint} (multi-label)",
    )

    # 4. Multi-task LASSO feature selection
    selected_features = lasso_feature_selection_multilabel(X_train, Y_train)
    if len(selected_features) == 0:
        logger.warning("%s Multi-task LASSO selected 0 features - skipping.", tag)
        return None

    pd.Series(selected_features, name="feature").to_csv(
        os.path.join(output_dir, "lasso_selected_features.csv"), index=False
    )

    X_train_sel = X_train[selected_features]
    X_val_sel   = X_val[selected_features]
    X_test_sel  = X_test[selected_features]

    # Correlation matrix post-LASSO
    plot_correlation_matrix(
        X_train_sel,
        output_path=os.path.join(output_dir, "correlation_matrix_postlasso.png"),
        title=f"Pearson correlation (LASSO features) - {tissue} {timepoint} (multi-label)",
    )

    # 5. Optuna TPE tuning on val set
    model_names = ["LogisticRegression", "RandomForest", "XGBoost", "SVM"]
    logger.info("%s Optuna tuning (%d trials/model) ...", tag, n_trials)

    tuned_params     = {}
    tuned_val_scores = {}
    for model_name in model_names:
        logger.info("%s  Tuning %s ...", tag, model_name)
        best_params, best_val_macro_pr = tune_hyperparams_multilabel(
            model_name, X_train_sel, Y_train, X_val_sel, Y_val,
            n_trials=n_trials, random_state=RANDOM_STATE,
        )
        tuned_params[model_name]     = best_params
        tuned_val_scores[model_name] = best_val_macro_pr

    with open(os.path.join(output_dir, "tuned_hyperparams.json"), "w") as fh:
        json.dump(
            {"model_params": tuned_params, "val_macro_pr_auc": tuned_val_scores},
            fh, indent=2,
        )

    # 6. 10-fold CV on train set with tuned params
    cv_results = {}
    for model_name in model_names:
        model  = build_tuned_model_multilabel(model_name, tuned_params[model_name])
        cv_res = run_cv_multilabel(model, X_train_sel, Y_train)
        cv_results[model_name] = cv_res
        macro  = cv_res["macro_avg"]
        logger.info(
            "%s  [CV] %-20s  Macro PR-AUC=%.3f  Macro ROC-AUC=%.3f",
            tag, model_name, macro["pr_auc_mean"], macro["roc_auc_mean"],
        )

    cv_rows = []
    for model_name, res in cv_results.items():
        for key, vals in res.items():
            if key == "macro_avg":
                row = {"model": model_name, "outcome": "macro_avg"}
                row.update(vals)
            else:
                row = {"model": model_name, "outcome": key}
                row.update(vals)
            cv_rows.append(row)
    pd.DataFrame(cv_rows).to_csv(
        os.path.join(output_dir, "cv_results.csv"), index=False
    )

    # 7. Best model by val macro PR-AUC; retrained on train+val -> test
    best_model_name = max(tuned_val_scores, key=lambda m: tuned_val_scores[m])
    logger.info(
        "%s Best model (val macro PR-AUC=%.4f): %s",
        tag, tuned_val_scores[best_model_name], best_model_name,
    )

    X_trainval = pd.concat([X_train_sel, X_val_sel])
    Y_trainval = pd.concat([Y_train, Y_val])

    test_results = {}
    for model_name in model_names:
        model = build_tuned_model_multilabel(model_name, tuned_params[model_name])
        metrics, scaler = evaluate_multilabel(
            model, X_trainval, Y_trainval, X_test_sel, Y_test
        )
        test_results[model_name] = metrics

        # Per-outcome PR / ROC curves for best model only
        if model_name == best_model_name:
            X_te_s = scaler.transform(X_test_sel)
            Y_prob = model.predict_proba(X_te_s)
            for i, outcome in enumerate(outcomes):
                prob = Y_prob[i][:, 1] if hasattr(Y_prob[i], "shape") else Y_prob[:, i]
                plot_pr_curve(
                    Y_test[outcome].values, prob,
                    title=f"PR - {model_name} | {tissue} {timepoint} | {outcome}",
                    output_path=os.path.join(
                        output_dir, f"{model_name}_{outcome}_pr_curve.png"
                    ),
                )
                plot_roc_curve(
                    Y_test[outcome].values, prob,
                    title=f"ROC - {model_name} | {tissue} {timepoint} | {outcome}",
                    output_path=os.path.join(
                        output_dir, f"{model_name}_{outcome}_roc_curve.png"
                    ),
                )

        # Feature importance averaged across outcomes (RF / XGBoost)
        estimators = model.estimators_
        if hasattr(estimators[0], "feature_importances_"):
            avg_imp = np.mean(
                [e.feature_importances_ for e in estimators], axis=0
            )
            save_feature_importance(
                selected_features, avg_imp,
                output_path=os.path.join(
                    output_dir, f"{model_name}_feature_importance_avg.png"
                ),
                title=f"Avg feature importance - {model_name} | {tissue} {timepoint}",
            )
        elif hasattr(estimators[0], "coef_"):
            avg_coef = np.mean(
                [e.coef_.ravel() for e in estimators], axis=0
            )
            save_feature_importance(
                selected_features, avg_coef,
                output_path=os.path.join(
                    output_dir, f"{model_name}_feature_importance_avg.png"
                ),
                title=f"Avg coefficient - {model_name} | {tissue} {timepoint}",
            )

        # Log per-outcome test metrics
        for outcome, m in metrics.items():
            logger.info(
                "%s  [Test] %-20s  %-8s  PR-AUC=%.3f  ROC-AUC=%.3f  F1=%.3f",
                tag, model_name, outcome,
                m["pr_auc"], m["roc_auc"], m["f1"],
            )

    # Save test results
    test_rows = []
    for model_name, per_outcome in test_results.items():
        for outcome, metrics in per_outcome.items():
            row = {"model": model_name, "outcome": outcome}
            row.update(metrics)
            test_rows.append(row)
    pd.DataFrame(test_rows).to_csv(
        os.path.join(output_dir, "test_results.csv"), index=False
    )

    summary = {
        "tissue":                   tissue,
        "timepoint":                timepoint,
        "outcomes":                 outcomes,
        "n_train":                  int(len(Y_train)),
        "n_val":                    int(len(Y_val)),
        "n_test":                   int(len(Y_test)),
        "n_features_pretlasso":     len(analyte_cols),
        "n_features_postlasso":     len(selected_features),
        "best_model_val":           best_model_name,
        "best_val_macro_pr_auc":    tuned_val_scores[best_model_name],
        "test_metrics":             test_results,
    }

    with open(os.path.join(output_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    logger.info("%s Done.", tag)
    return summary

# ---------------------------------------------------------------------------
# Plasma loop
# ---------------------------------------------------------------------------

def run_plasma(
    plasma_dir: str,
    output_root: str,
    timepoints: list,
    outcomes: list,
    n_trials: int,
    sig_analytes_dir: str | None = None,
) -> list:
    summaries = []
    for tp in timepoints:
        csv_path = os.path.join(
            plasma_dir,
            f"proteomics_plasma_formatted_suffix_{tp}.csv",
        )
        if not os.path.exists(csv_path):
            logger.warning("Plasma timepoint %s CSV not found: %s - skipping.", tp, csv_path)
            continue

        df = load_data(csv_path)
        logger.info("Plasma timepoint %s loaded: %d samples x %d cols", tp, *df.shape)

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
                    "Plasma tp=%s: no significant analytes found (q<0.05) - using all features.", tp
                )

        out_dir = os.path.join(output_root, "plasma", tp)
        result  = run_multilabel_pipeline(
            df, "plasma", tp, outcomes, out_dir, n_trials, sig_analytes
        )
        if result:
            summaries.append(result)

    return summaries

# ---------------------------------------------------------------------------
# Placenta loop
# ---------------------------------------------------------------------------

def run_placenta(
    placenta_csv: str,
    output_root: str,
    outcomes: list,
    n_trials: int,
    sig_analytes_dir: str | None = None,
) -> list:
    if not os.path.exists(placenta_csv):
        logger.warning("Placenta CSV not found: %s - skipping.", placenta_csv)
        return []

    df = load_data(placenta_csv)
    logger.info("Placenta loaded: %d samples x %d cols", *df.shape)

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
            logger.warning("Placenta: no significant analytes found (q<0.05) - using all features.")

    out_dir = os.path.join(output_root, "placenta", "all")
    result  = run_multilabel_pipeline(
        df, "placenta", "all", outcomes, out_dir, n_trials, sig_analytes
    )
    return [result] if result else []

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    wkdir = os.getcwd()
    p = argparse.ArgumentParser(
        description=(
            "Multi-label classification: joint HDP + FGR + sPTB prediction. "
            "Hyperparameters tuned via Optuna TPE on the validation set "
            "(macro-average PR-AUC across all outcomes)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--plasma-dir",
        default=os.path.join(
            wkdir, "data", "cleaned", "proteomics", "normalized_sliced_by_suffix"
        ),
        help="Directory with per-timepoint plasma CSVs.",
    )
    p.add_argument(
        "--placenta-csv",
        default=os.path.join(
            wkdir,
            "data", "cleaned", "proteomics", "normalized_full_results",
            "proteomics_placenta_cleaned_with_metadata.csv",
        ),
        help="Path to placenta cleaned CSV.",
    )
    p.add_argument(
        "--output-dir",
        default=os.path.join(wkdir, "04_results_and_figures", "models", "multilabel"),
        help="Root output directory.",
    )
    p.add_argument(
        "--timepoints",
        nargs="+",
        default=TIMEPOINTS,
        help="Plasma timepoints to process.",
    )
    p.add_argument(
        "--outcomes",
        nargs="+",
        default=OUTCOMES,
        help="Outcomes to model jointly.",
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
    p.add_argument("--skip-plasma",   action="store_true", help="Skip plasma analysis.")
    p.add_argument("--skip-placenta", action="store_true", help="Skip placenta analysis.")
    return p

def main() -> None:
    args = _build_parser().parse_args()
    logger.info("Outcomes modelled jointly: %s", args.outcomes)
    logger.info("Optuna trials per model:   %d", args.n_trials)
    sig_dir = args.sig_analytes_dir if args.sig_analytes_dir else None
    if sig_dir:
        logger.info("Significant analytes dir: %s", sig_dir)
    else:
        logger.info("No significant analytes filter - using all features.")
    all_summaries = []

    if not args.skip_plasma:
        logger.info("=== PLASMA ===")
        all_summaries += run_plasma(
            args.plasma_dir,
            args.output_dir,
            args.timepoints,
            args.outcomes,
            args.n_trials,
            sig_analytes_dir=sig_dir,
        )

    if not args.skip_placenta:
        logger.info("=== PLACENTA ===")
        all_summaries += run_placenta(
            args.placenta_csv,
            args.output_dir,
            args.outcomes,
            args.n_trials,
            sig_analytes_dir=sig_dir,
        )

    # Aggregate summary table
    if all_summaries:
        rows = []
        for s in all_summaries:
            for model_name, per_outcome in s["test_metrics"].items():
                for outcome, metrics in per_outcome.items():
                    row = {
                        "tissue":    s["tissue"],
                        "timepoint": s["timepoint"],
                        "outcome":   outcome,
                        "model":     model_name,
                    }
                    row.update({k: round(v, 4) for k, v in metrics.items()})
                    rows.append(row)

        os.makedirs(args.output_dir, exist_ok=True)
        summary_df = pd.DataFrame(rows)
        out_path   = os.path.join(args.output_dir, "all_results_summary.csv")
        summary_df.to_csv(out_path, index=False)
        logger.info("Aggregate summary saved -> %s", out_path)

    logger.info("Multi-label classifier pipeline complete.")

if __name__ == "__main__":
    main()
