"""
Run the existing binary and multi-label model pipelines on survey data.

This wrapper reshapes cleaned survey files into model-ready wide matrices and
then reuses the same modeling functions used for omics data.

Surveys included:
  - EPDS
  - PSS
  - PUQE24
  - Water

Diet is intentionally excluded because it remains raw categorical strings and
has not yet been numerically encoded.

Outputs
-------
Model-ready inputs:
  data/cleaned/survey/model_ready/<survey>/
    normalized_sliced_by_suffix/   (EPDS / PSS / PUQE24 per visit)
    normalized_full_results/       (water)

Model results:
  04_results_and_figures/models/binary/survey/<survey>/<timepoint or all>/
  04_results_and_figures/models/multilabel/survey/<survey>/<timepoint or all>/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Iterable

import pandas as pd

from binary_classifier import run_binary_pipeline
from multilabel_classifier import run_multilabel_pipeline
from utilities import OUTCOMES


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SURVEY_DIR = os.path.join(ROOT, "data", "cleaned", "survey")
MODEL_READY_ROOT = os.path.join(ROOT, "data", "cleaned", "survey", "model_ready")
OUT_BINARY_ROOT = os.path.join(ROOT, "04_results_and_figures", "models", "binary", "survey")
OUT_MULTI_ROOT = os.path.join(ROOT, "04_results_and_figures", "models", "multilabel", "survey")

VISITS = ["A", "C", "D", "PP"]


def _safe_group(df: pd.DataFrame, group_keys: list[str], numeric_cols: list[str]) -> pd.DataFrame:
    """Average numeric duplicates while keeping first metadata values."""
    meta_cols = [c for c in df.columns if c not in numeric_cols and c not in group_keys]
    grouped_num = df.groupby(group_keys)[numeric_cols].mean().reset_index()
    if not meta_cols:
        return grouped_num
    grouped_meta = df[group_keys + meta_cols].groupby(group_keys, as_index=False).first()
    return grouped_num.merge(grouped_meta, on=group_keys, how="left")


def _write_visit_matrix(
    df: pd.DataFrame,
    visit: str,
    out_dir: str,
    prefix: str,
    feature_cols: list[str],
) -> str | None:
    sub = df[df["Visit"] == visit].copy()
    if sub.empty:
        logger.warning("%s visit %s: no rows after filtering.", prefix, visit)
        return None

    numeric_cols = [c for c in feature_cols if c in sub.columns]
    keep_cols = ["SubjectID", "Group", "Subgroup"] + numeric_cols
    sub = sub[keep_cols]

    # One row per subject per visit.
    sub = _safe_group(sub, ["SubjectID"], numeric_cols)
    sub["SampleID"] = sub["SubjectID"].astype(str) + visit
    sub = sub.set_index("SampleID")

    out = sub[["SubjectID", "Group", "Subgroup"] + numeric_cols]
    out.index.name = "SampleID"

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{prefix}_plasma_formatted_suffix_{visit}.csv")
    out.to_csv(path)
    logger.info("%s visit %s: saved model-ready matrix %s (%d samples x %d features)",
                prefix.upper(), visit, path, len(out), len(numeric_cols))
    return path


def _prepare_scored_survey(
    survey_key: str,
    input_csv: str,
    out_root: str,
    feature_cols: list[str],
) -> str:
    df = pd.read_csv(input_csv)
    df = df[df["Visit"].isin(VISITS)].copy()
    out_dir = os.path.join(out_root, survey_key, "normalized_sliced_by_suffix")
    for visit in VISITS:
        _write_visit_matrix(df, visit, out_dir, survey_key, feature_cols)
    return out_dir


def _prepare_epds() -> str:
    feature_cols = [f"epds_q{i}.1" for i in range(1, 11)] + ["score"]
    return _prepare_scored_survey(
        "epds",
        os.path.join(SURVEY_DIR, "epds_cleaned.csv"),
        MODEL_READY_ROOT,
        feature_cols,
    )


def _prepare_pss() -> str:
    feature_cols = [f"pss_q{i}.1" for i in range(1, 11)] + ["score"]
    return _prepare_scored_survey(
        "pss",
        os.path.join(SURVEY_DIR, "pss_cleaned.csv"),
        MODEL_READY_ROOT,
        feature_cols,
    )


def _prepare_puqe24() -> str:
    feature_cols = ["puqe_q1", "puqe_q2", "puqe_q3", "score"]
    return _prepare_scored_survey(
        "puqe24",
        os.path.join(SURVEY_DIR, "puqe24_cleaned.csv"),
        MODEL_READY_ROOT,
        feature_cols,
    )


def _prepare_water() -> str:
    df = pd.read_csv(os.path.join(SURVEY_DIR, "water_cleaned.csv"))
    feature_cols = [
        "TTHM_avg", "Br.THM_avg", "CHCl3_avg", "CHBr3_avg", "BDCM_avg", "CDBM_avg",
        "TTHM_max", "Br.THM_max", "CHCl3_max", "CHBr3_max", "BDCM_max", "CDBM_max",
        "TTHM_exceed_rate", "CHCl3_exceed_rate", "CHBr3_exceed_rate",
        "BDCM_exceed_rate", "CDBM_exceed_rate",
    ]
    keep_cols = ["SubjectID", "Group", "Subgroup"] + feature_cols
    df = df[keep_cols].copy()
    df = _safe_group(df, ["SubjectID"], feature_cols)
    df = df.set_index("SubjectID")
    df.index.name = "SampleID"
    df.insert(0, "SubjectID", df.index)

    out_dir = os.path.join(MODEL_READY_ROOT, "water", "normalized_full_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "water_cleaned_with_metadata.csv")
    df.to_csv(out_path)
    logger.info("WATER: saved model-ready matrix %s (%d samples x %d features)",
                out_path, len(df), len(feature_cols))
    return out_path


def _summary_rows_binary(survey_key: str, summaries: Iterable[dict]) -> list[dict]:
    rows = []
    for s in summaries:
        for model_name, metrics in s["test_metrics"].items():
            row = {"survey": survey_key, "condition": s["timepoint"], "model": model_name}
            row.update({k: round(v, 4) for k, v in metrics.items()})
            rows.append(row)
    return rows


def _summary_rows_multi(survey_key: str, summaries: Iterable[dict]) -> list[dict]:
    rows = []
    for s in summaries:
        for model_name, per_outcome in s["test_metrics"].items():
            for outcome, metrics in per_outcome.items():
                row = {
                    "survey": survey_key,
                    "condition": s["timepoint"],
                    "outcome": outcome,
                    "model": model_name,
                }
                row.update({k: round(v, 4) for k, v in metrics.items()})
                rows.append(row)
    return rows


def _run_visit_models(
    survey_key: str,
    plasma_dir: str,
    visits: list[str],
    n_trials: int,
) -> tuple[list[dict], list[dict]]:
    binary_summaries = []
    multi_summaries = []

    for visit in visits:
        csv_path = os.path.join(plasma_dir, f"{survey_key}_plasma_formatted_suffix_{visit}.csv")
        if not os.path.exists(csv_path):
            logger.warning("%s visit %s: missing model-ready CSV, skipping.", survey_key.upper(), visit)
            continue
        df = pd.read_csv(csv_path, index_col=0)
        logger.info("%s visit %s loaded: %d samples x %d cols", survey_key.upper(), visit, *df.shape)

        out_bin = os.path.join(OUT_BINARY_ROOT, survey_key, visit)
        out_mul = os.path.join(OUT_MULTI_ROOT, survey_key, visit)

        b = run_binary_pipeline(
            df=df,
            tissue=survey_key,
            timepoint=visit,
            complications=OUTCOMES,
            output_dir=out_bin,
            n_trials=n_trials,
            sig_analytes=None,
            data_path=os.path.abspath(csv_path),
        )
        if b:
            binary_summaries.append(b)

        m = run_multilabel_pipeline(
            df=df,
            tissue=survey_key,
            timepoint=visit,
            outcomes=OUTCOMES,
            output_dir=out_mul,
            n_trials=n_trials,
            sig_analytes=None,
        )
        if m:
            multi_summaries.append(m)

    return binary_summaries, multi_summaries


def _run_water_models(water_csv: str, n_trials: int) -> tuple[list[dict], list[dict]]:
    df = pd.read_csv(water_csv, index_col=0)
    logger.info("WATER loaded: %d samples x %d cols", *df.shape)

    out_bin = os.path.join(OUT_BINARY_ROOT, "water", "all")
    out_mul = os.path.join(OUT_MULTI_ROOT, "water", "all")

    binary_summaries = []
    multi_summaries = []

    b = run_binary_pipeline(
        df=df,
        tissue="water",
        timepoint="all",
        complications=OUTCOMES,
        output_dir=out_bin,
        n_trials=n_trials,
        sig_analytes=None,
        data_path=os.path.abspath(water_csv),
    )
    if b:
        binary_summaries.append(b)

    m = run_multilabel_pipeline(
        df=df,
        tissue="water",
        timepoint="all",
        outcomes=OUTCOMES,
        output_dir=out_mul,
        n_trials=n_trials,
        sig_analytes=None,
    )
    if m:
        multi_summaries.append(m)

    return binary_summaries, multi_summaries


def _save_aggregate(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info("Aggregate summary saved -> %s", path)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run survey data through the binary and multilabel model pipelines."
    )
    p.add_argument(
        "--surveys",
        nargs="+",
        choices=["epds", "pss", "puqe24", "water"],
        default=["epds", "pss", "puqe24", "water"],
        help="Survey datasets to run.",
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Optuna trials per model per condition. Lower than omics default to keep survey runs practical.",
    )
    args = p.parse_args()

    logger.info("Survey modeling start | surveys=%s | n_trials=%d", args.surveys, args.n_trials)

    binary_rows = []
    multi_rows = []

    if "epds" in args.surveys:
        epds_dir = _prepare_epds()
        b, m = _run_visit_models("epds", epds_dir, VISITS, args.n_trials)
        binary_rows.extend(_summary_rows_binary("epds", b))
        multi_rows.extend(_summary_rows_multi("epds", m))

    if "pss" in args.surveys:
        pss_dir = _prepare_pss()
        b, m = _run_visit_models("pss", pss_dir, VISITS, args.n_trials)
        binary_rows.extend(_summary_rows_binary("pss", b))
        multi_rows.extend(_summary_rows_multi("pss", m))

    if "puqe24" in args.surveys:
        puqe_dir = _prepare_puqe24()
        b, m = _run_visit_models("puqe24", puqe_dir, VISITS, args.n_trials)
        binary_rows.extend(_summary_rows_binary("puqe24", b))
        multi_rows.extend(_summary_rows_multi("puqe24", m))

    if "water" in args.surveys:
        water_csv = _prepare_water()
        b, m = _run_water_models(water_csv, args.n_trials)
        binary_rows.extend(_summary_rows_binary("water", b))
        multi_rows.extend(_summary_rows_multi("water", m))

    if binary_rows:
        _save_aggregate(os.path.join(OUT_BINARY_ROOT, "all_results_summary.csv"), binary_rows)
    if multi_rows:
        _save_aggregate(os.path.join(OUT_MULTI_ROOT, "all_results_summary.csv"), multi_rows)

    with open(os.path.join(OUT_BINARY_ROOT, "run_config.json"), "w") as fh:
        json.dump({"surveys": args.surveys, "n_trials": args.n_trials}, fh, indent=2)

    logger.info("Survey modeling complete.")


if __name__ == "__main__":
    main()
