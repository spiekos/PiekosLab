"""
SOP-native untargeted metabolomics and lipidomics preprocessing pipeline.

This module implements the April 2026 DP3 SOP starting from the raw
Compound Discoverer style exports stored under ../kaylaxu/data/.

It intentionally does NOT overwrite the older collaborator-integrated
scripts. Instead, it provides a clean replacement entrypoint that follows
the SOP ordering:

1. Missing-value standardization
2. Sample type / batch / injection-order parsing
3. Pre-normalization drift diagnostics
4. ISTD normalization
5. Median fold-change batch normalization
6. Post-normalization drift diagnostics
7. Feature missingness filter
8. QC-pool RSD filter
9. Sample missingness filter
10. Sample-level ISTD MAD QC
11. Log2 transformation
12. Half-minimum imputation
13. Pre-correction PCA
14. Batch-confounding checks
15. Batch correction
16. Post-correction PCA
17. Bridge-sample averaging
18-31. Deduplication, annotation, metadata integration, and file outputs
32. Biological trajectory plots
33. Human-readable pipeline log

Notes
-----
- When the original raw workbook is available, injection order is parsed
  from the raw-file headers (`F#` for metabolomics, file sequence for the
  current lipidomics export). Datasets without a raw workbook still fall
  back to within-batch row order, and that fallback is recorded in the log.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
import subprocess
import tempfile
import textwrap
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

os.environ.setdefault(
    "MPLCONFIGDIR",
    tempfile.mkdtemp(prefix="dp3-matplotlib-"),
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGGER = logging.getLogger("sop_omics_pipeline")

PROTON_MASS = 1.007276
VALID_TIMEPOINTS = set("ABCDE")


@dataclass(frozen=True)
class Thresholds:
    mass_tolerance_parent: float = 0.02
    mass_tolerance_non_parent: float = 0.10
    mass_tolerance_dedup_no_formula: float = 0.02
    rt_tolerance: float = 0.20
    ppm_tolerance_annotation: float = 5.0
    sample_mad_threshold: float = 5.0
    sample_missing_threshold: float = 0.50
    feature_missing_threshold: float = 0.20
    rsd_threshold: float = 30.0
    mzcloud_l2_threshold: float = 80.0
    annot_source_l2_threshold: int = 3


@dataclass(frozen=True)
class DatasetConfig:
    dataset_id: str
    database: str
    tissue: str
    meta_sheet: str
    meta_sample_col: str
    input_dir: Path
    output_dir: Path
    positive_istd_names: tuple[str, ...]
    negative_istd_names: tuple[str, ...]
    bridge_expected: bool = True
    raw_workbook: Path | None = None
    raw_sheet_pos: str | None = None
    raw_sheet_neg: str | None = None
    raw_sample_row: int | None = None
    raw_file_row: int | None = None


@dataclass
class PolarityRun:
    polarity: str
    expression: pd.DataFrame
    sample_info: pd.DataFrame
    feature_meta: pd.DataFrame
    raw_istd: pd.DataFrame
    injection_order_source: str = "row_order_proxy"
    retained_feature_meta: pd.DataFrame | None = None
    retained_expression: pd.DataFrame | None = None
    dropped_features: list[dict] = field(default_factory=list)


@dataclass
class PipelineArtifacts:
    sample_filter_log: list[dict] = field(default_factory=list)
    feature_filter_log: list[dict] = field(default_factory=list)
    dedup_log: list[dict] = field(default_factory=list)
    qc_warnings: list[str] = field(default_factory=list)
    batch_scaling_factors: dict[str, float] = field(default_factory=dict)
    drift_flags: list[str] = field(default_factory=list)
    method_log: list[str] = field(default_factory=list)
    bridge_counts: dict[str, int] = field(default_factory=dict)
    # Step 33: structured per-step tracking
    # Each entry: {polarity, step, label, n_features, n_samples}
    step_counts: list[dict] = field(default_factory=list)
    # polarity -> list of ISTD feature names that were removed during normalization
    istd_names: dict[str, list[str]] = field(default_factory=dict)
    # polarity -> batch -> count of biological samples at pipeline entry
    initial_sample_counts: dict[str, dict[str, int]] = field(default_factory=dict)


def _bh_fdr(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg q-values for a Series indexed like the input."""
    p = p_values.astype(float).copy()
    valid = p.notna()
    if not valid.any():
        return pd.Series(np.nan, index=p.index, dtype=float)

    pv = p[valid].values
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        candidate = ranked[i] * n / rank
        prev = min(prev, candidate)
        q[i] = prev
    out = pd.Series(np.nan, index=p.index, dtype=float)
    out.loc[valid] = q[np.argsort(order)]
    return out


def _normalise_name(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").strip()).lower()


def _canonical_sample_name(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").strip())


def _canonical_batch_label(value: str) -> str:
    digits = re.sub(r"\D", "", str(value or "").strip())
    if digits:
        try:
            return str(int(digits))
        except ValueError:
            return digits.lstrip("0") or "0"
    return str(value or "").strip()


def _metadata_alias_candidates(sample_id: str) -> list[str]:
    sample_id = _canonical_sample_name(sample_id)
    candidates: list[str] = []
    with_e = re.match(r"^(DP3-\d{4})E([A-E])$", sample_id)
    if with_e:
        candidates.append(f"{with_e.group(1)}{with_e.group(2)}")
    without_e = re.match(r"^(DP3-\d{4})([A-E])$", sample_id)
    if without_e:
        candidates.append(f"{without_e.group(1)}E{without_e.group(2)}")
    return [c for c in candidates if c != sample_id]


def _extract_injection_order_value(*values: str) -> float:
    patterns = [
        r"\(F\s*([0-9]+)\)",
        r"\bF\s*([0-9]+)\b",
        r"file[_\s-]*index[_\s-]*([0-9]+)",
    ]
    for value in values:
        text = str(value or "")
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return float(match.group(1))
        if re.search(r"(raw|pool|set[0-9]+|_pos_|_neg_|\bpos\b|\bneg\b)", text, flags=re.IGNORECASE):
            match = re.search(r"([0-9]+)(?:\.raw)?$", text, flags=re.IGNORECASE)
            if match:
                return float(match.group(1))
    return float("nan")


def _extract_batch_label_from_text(*values: str) -> str:
    for value in values:
        text = str(value or "")
        match = re.search(r"(?<!\d)(\d{5,6})(?!\d)", text)
        if match:
            return _canonical_batch_label(match.group(1))
    return ""


def _safe_float(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if out == 0:
        return float("nan")
    return out


def _geometric_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(arr))))


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _make_unique(names: list[str]) -> list[str]:
    counts: Counter[str] = Counter()
    unique: list[str] = []
    for name in names:
        counts[name] += 1
        if counts[name] == 1:
            unique.append(name)
        else:
            unique.append(f"{name}__{counts[name]}")
    return unique


def _parse_batch_date(value: str) -> datetime:
    digits = re.sub(r"\D", "", str(value or "").strip())
    if len(digits) == 5:
        month = int(digits[0])
        day = int(digits[1:3])
        year = 2000 + int(digits[3:5])
    elif len(digits) == 6:
        month = int(digits[0:2])
        day = int(digits[2:4])
        year = 2000 + int(digits[4:6])
    else:
        raise ValueError(f"Unrecognized batch-date format: {value!r}")
    return datetime(year, month, day)


def _pick_reference_batch(candidate_batches: Iterable[str]) -> str:
    batches = [str(batch) for batch in candidate_batches]
    if not batches:
        raise ValueError("No candidate batches were provided.")
    try:
        return min(batches, key=_parse_batch_date)
    except Exception:
        return sorted(batches)[0]


def _default_modifications() -> pd.DataFrame:
    """
    Canonical adduct/isotope/fragment mass shifts used for classification.

    This list intentionally covers the high-frequency metabolomics and
    lipidomics shifts needed by the SOP. The file-based canonical list
    mentioned in the SOP is not present in the repo, so the pipeline vendors
    a pragmatic default here.
    """
    rows = [
        ("Isotope", "13C", 1.003355, "Single 13C isotope"),
        ("Isotope", "2x13C", 2.006710, "Double 13C isotope"),
        ("Isotope", "15N", 0.997035, "Single 15N isotope"),
        ("Isotope", "18O", 2.004245, "Single 18O isotope"),
        ("Isotope", "34S", 1.995796, "Single 34S isotope"),
        ("Adduct", "M+Na", 21.981943, "Sodium adduct"),
        ("Adduct", "M+NH4", 17.026549, "Ammonium adduct"),
        ("Adduct", "M+K", 37.955882, "Potassium adduct"),
        ("Adduct", "M+H-H2O", -18.010565, "Water loss"),
        ("Adduct", "M+Cl", 34.969402, "Chloride adduct"),
        ("Adduct", "M+FA-H", 44.998201, "Formate adduct"),
        ("Adduct", "M+Ac-H", 59.013851, "Acetate adduct"),
        ("Adduct", "M+HCOO", 44.998201, "Formate adduct alias"),
        ("Fragment", "H2O loss", -18.010565, "Neutral loss H2O"),
        ("Fragment", "CO2 loss", -43.989830, "Neutral loss CO2"),
        ("Fragment", "NH3 loss", -17.026549, "Neutral loss NH3"),
        ("Fragment", "CH3 loss", -15.023475, "Neutral loss CH3"),
        ("Fragment", "phosphate loss", -79.966331, "Neutral loss phosphate"),
        ("Fragment", "hexose loss", -162.052824, "Neutral loss hexose"),
    ]
    return pd.DataFrame(
        rows, columns=["Type", "Name", "Delta_m/z", "Description"]
    )


def _excel_col_to_idx(cell_ref: str) -> int:
    letters = re.match(r"([A-Z]+)", cell_ref)
    if not letters:
        return 0
    idx = 0
    for char in letters.group(1):
        idx = idx * 26 + (ord(char) - ord("A") + 1)
    return idx - 1


def _read_xlsx_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    """
    Lightweight XLSX reader for environments without openpyxl.

    Supports the subset needed by the DP3 metadata workbook: shared strings,
    numeric cells, and inline strings.
    """
    ns = {
        "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "pkgrel": "http://schemas.openxmlformats.org/package/2006/relationships",
    }
    with zipfile.ZipFile(path) as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for item in root.findall("main:si", ns):
                text = "".join(node.text or "" for node in item.findall(".//main:t", ns))
                shared_strings.append(text)

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels.findall("pkgrel:Relationship", ns)
        }
        sheet_target = None
        for sheet in workbook.findall("main:sheets/main:sheet", ns):
            if sheet.attrib.get("name") == sheet_name:
                rel_id = sheet.attrib.get(f"{{{ns['rel']}}}id")
                sheet_target = rel_map.get(rel_id)
                break
        if sheet_target is None:
            raise ValueError(f"Sheet '{sheet_name}' not found in {path}.")

        if not sheet_target.startswith("xl/"):
            sheet_target = f"xl/{sheet_target}"
        sheet_xml = ET.fromstring(zf.read(sheet_target))
        rows: list[list[str]] = []
        max_cols = 0
        for row in sheet_xml.findall(".//main:sheetData/main:row", ns):
            values: dict[int, str] = {}
            for cell in row.findall("main:c", ns):
                ref = cell.attrib.get("r", "A1")
                col_idx = _excel_col_to_idx(ref)
                cell_type = cell.attrib.get("t")
                value = ""
                if cell_type == "s":
                    raw = cell.findtext("main:v", default="", namespaces=ns)
                    if raw:
                        value = shared_strings[int(raw)]
                elif cell_type == "inlineStr":
                    value = "".join(
                        node.text or "" for node in cell.findall(".//main:t", ns)
                    )
                else:
                    value = cell.findtext("main:v", default="", namespaces=ns)
                values[col_idx] = value
                max_cols = max(max_cols, col_idx + 1)
            row_values = [""] * max_cols
            for idx, value in values.items():
                row_values[idx] = value
            rows.append(row_values)
    if not rows:
        return pd.DataFrame()
    header = rows[0]
    data = [row + [""] * (len(header) - len(row)) for row in rows[1:]]
    return pd.DataFrame(data, columns=header)


def _read_xlsx_selected_rows(
    path: Path,
    sheet_name: str,
    target_rows: set[int],
) -> dict[int, dict[int, str]]:
    """
    Return only the requested XLSX rows as {row_number: {col_idx: value}}.

    This keeps the raw-workbook parsing lightweight even for very large
    untargeted exports.
    """
    ns = {
        "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "pkgrel": "http://schemas.openxmlformats.org/package/2006/relationships",
    }
    if not target_rows:
        return {}

    with zipfile.ZipFile(path) as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for item in root.findall("main:si", ns):
                text = "".join(node.text or "" for node in item.findall(".//main:t", ns))
                shared_strings.append(text)

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels.findall("pkgrel:Relationship", ns)
        }
        sheet_target = None
        for sheet in workbook.findall("main:sheets/main:sheet", ns):
            if sheet.attrib.get("name") == sheet_name:
                rel_id = sheet.attrib.get(f"{{{ns['rel']}}}id")
                sheet_target = rel_map.get(rel_id)
                break
        if sheet_target is None:
            raise ValueError(f"Sheet '{sheet_name}' not found in {path}.")

        if not sheet_target.startswith("xl/"):
            sheet_target = f"xl/{sheet_target}"

        rows: dict[int, dict[int, str]] = {}
        max_target = max(target_rows)
        for _, elem in ET.iterparse(zf.open(sheet_target), events=("end",)):
            if not elem.tag.endswith("row"):
                continue
            row_number = int(elem.attrib.get("r", "0") or "0")
            if row_number in target_rows:
                row_values: dict[int, str] = {}
                for cell in elem:
                    if not cell.tag.endswith("c"):
                        continue
                    ref = cell.attrib.get("r", "A1")
                    col_idx = _excel_col_to_idx(ref)
                    cell_type = cell.attrib.get("t")
                    value = ""
                    if cell_type == "s":
                        raw = next(
                            (child.text for child in cell if child.tag.endswith("v")),
                            "",
                        )
                        if raw:
                            value = shared_strings[int(raw)]
                    elif cell_type == "inlineStr":
                        value = "".join(
                            child.text or ""
                            for child in cell.iter()
                            if child.tag.endswith("t")
                        )
                    else:
                        value = next(
                            (child.text or "" for child in cell if child.tag.endswith("v")),
                            "",
                        )
                    row_values[col_idx] = value
                rows[row_number] = row_values
            elem.clear()
            if row_number >= max_target and target_rows.issubset(rows.keys()):
                break
    return rows


def _sample_match_key(sample_name: str, sample_type: str) -> str:
    return "" if sample_type == "qc_pool" else _canonical_sample_name(sample_name)


def _load_raw_acquisition_map(
    config: DatasetConfig,
    polarity: str,
) -> tuple[pd.DataFrame | None, str | None]:
    workbook = config.raw_workbook
    sample_row = config.raw_sample_row
    file_row = config.raw_file_row
    sheet_name = config.raw_sheet_pos if polarity.lower() == "pos" else config.raw_sheet_neg
    if (
        workbook is None
        or not workbook.exists()
        or sheet_name is None
        or sample_row is None
        or file_row is None
    ):
        return None, None

    header_rows = _read_xlsx_selected_rows(workbook, sheet_name, {sample_row, file_row})
    sample_cells = header_rows.get(sample_row, {})
    file_cells = header_rows.get(file_row, {})
    if not sample_cells or not file_cells:
        return None, None

    records: list[dict] = []
    saw_f_number = False
    saw_file_sequence = False
    for rank, col_idx in enumerate(sorted(set(sample_cells) & set(file_cells)), start=1):
        sample_label = str(sample_cells.get(col_idx, "") or "").strip()
        raw_file = str(file_cells.get(col_idx, "") or "").strip()
        if not raw_file:
            continue
        batch = _extract_batch_label_from_text(raw_file, sample_label)
        if not batch:
            continue
        sample_type = "qc_pool" if sample_label.lower().startswith("pooled") else "biological"
        order = _extract_injection_order_value(raw_file, sample_label)
        if not np.isfinite(order):
            order = float(rank)
        if re.search(r"\(F\s*[0-9]+\)|\bF\s*[0-9]+\b", raw_file, flags=re.IGNORECASE):
            saw_f_number = True
        elif re.search(r"([0-9]+)(?:\.raw)?$", raw_file, flags=re.IGNORECASE):
            saw_file_sequence = True
        records.append(
            {
                "sample_name": _canonical_sample_name(sample_label),
                "sample_type": sample_type,
                "batch": batch,
                "raw_file": raw_file,
                "raw_injection_order": order,
            }
        )
    if not records:
        return None, None

    source = "raw_workbook"
    if saw_f_number:
        source = "raw_workbook_f_number"
    elif saw_file_sequence:
        source = "raw_workbook_file_sequence"
    return pd.DataFrame.from_records(records), source


def _load_metadata(meta_path: Path, config: DatasetConfig) -> pd.DataFrame:
    cols = [
        config.meta_sample_col,
        "group",
        "subgroup",
        "gest age del",
        "omics set#",
    ]
    if config.tissue == "plasma":
        cols.append("sample gest Age")

    meta = _read_xlsx_sheet(meta_path, config.meta_sheet)
    available = [c for c in cols if c in meta.columns]
    meta = meta[available].dropna(subset=[config.meta_sample_col]).copy()
    meta[config.meta_sample_col] = meta[config.meta_sample_col].map(_canonical_sample_name)
    meta = meta.rename(
        columns={
            config.meta_sample_col: "SampleID",
            "group": "Group",
            "subgroup": "Subgroup",
            "gest age del": "GestAgeDelivery",
            "omics set#": "Batch",
            "sample gest Age": "SampleGestAge",
        }
    )
    meta["Group"] = meta["Group"].replace({"sptb": "sPTB", "SPTB": "sPTB"})
    meta = meta.set_index("SampleID")
    meta = meta[~meta.index.duplicated(keep="first")].copy()
    meta["MetadataCanonicalID"] = meta.index

    alias_rows: list[pd.Series] = []
    for sample_id, row in meta.iterrows():
        for alias in _metadata_alias_candidates(sample_id):
            if alias in meta.index:
                continue
            alias_row = row.copy()
            alias_row.name = alias
            alias_rows.append(alias_row)
    if alias_rows:
        meta = pd.concat([meta, pd.DataFrame(alias_rows)], axis=0)
        meta = meta[~meta.index.duplicated(keep="first")].copy()
    return meta


def _standardize_expression(exp: pd.DataFrame) -> pd.DataFrame:
    return exp.apply(lambda col: col.map(_safe_float))


def _standardize_feature_metadata(comp: pd.DataFrame, polarity: str) -> pd.DataFrame:
    comp = comp.copy()
    if "Export Order" in comp.columns:
        comp = comp.set_index("Export Order")

    is_lipid = "LipidID" in comp.columns and "Name" not in comp.columns

    if is_lipid:
        comp["annotation_name"] = comp["LipidID"].fillna("")
        adduct = comp.get("AdductIon", pd.Series("", index=comp.index)).fillna("").astype(str)
        base_name = comp["annotation_name"].astype(str)
        has_adduct = adduct != ""
        stripped = base_name.where(
            ~has_adduct,
            base_name.str.replace(r"([+-].+)$", "", regex=True),
        )
        comp["compound_group_name"] = stripped.fillna("")
        comp["formula"] = comp.get("IonFormula", pd.Series("", index=comp.index)).fillna("")
        comp["calc_mw"] = np.nan
        comp["mz"] = pd.to_numeric(comp.get("CalcMz"), errors="coerce")
        comp["rt"] = pd.to_numeric(comp.get("BaseRt"), errors="coerce")
        comp["area_max"] = np.nan
        comp["peak_rating"] = np.nan
        comp["rsd_qc"] = np.nan
        comp["ms2"] = ""
        comp["mzcloud_confidence"] = np.nan
        comp["mass_list_matches"] = ""
        comp["annotation_source_matches"] = 0
    else:
        annot_cols = [
            c
            for c in [
                "Annot. Source: Predicted Compositions",
                "Annot. Source: mzCloud Search",
                "Annot. Source: mzVault Search",
                "Annot. Source: Metabolika Search",
                "Annot. Source: ChemSpider Search",
                "Annot. Source: MassList Search",
            ]
            if c in comp.columns
        ]
        if annot_cols:
            full_matches = (comp[annot_cols] == "Full match").sum(axis=1)
        else:
            full_matches = pd.Series(0, index=comp.index)

        mass_list_col = next(
            (c for c in comp.columns if c.startswith("Mass List Match: HMDB")),
            "",
        )
        comp["annotation_name"] = comp.get("Name", pd.Series("", index=comp.index)).fillna("")
        comp["compound_group_name"] = comp["annotation_name"]
        comp["formula"] = comp.get("Formula", pd.Series("", index=comp.index)).fillna("")
        comp["calc_mw"] = pd.to_numeric(comp.get("Calc. MW"), errors="coerce")
        comp["mz"] = pd.to_numeric(comp.get("m/z"), errors="coerce")
        comp["rt"] = pd.to_numeric(comp.get("RT [min]"), errors="coerce")
        comp["area_max"] = pd.to_numeric(comp.get("Area (Max.)"), errors="coerce")
        comp["peak_rating"] = pd.to_numeric(comp.get("Peak Rating (Max.)"), errors="coerce")
        comp["rsd_qc"] = pd.to_numeric(comp.get("RSD QC Areas [%]"), errors="coerce")
        comp["ms2"] = comp.get("MS2", pd.Series("", index=comp.index)).fillna("")
        comp["mzcloud_confidence"] = pd.to_numeric(
            comp.get("mzCloud Best Match Confidence"), errors="coerce"
        )
        comp["mass_list_matches"] = (
            comp.get(mass_list_col, pd.Series("", index=comp.index)).fillna("")
            if mass_list_col
            else ""
        )
        comp["annotation_source_matches"] = full_matches

    comp["annotation_name"] = comp["annotation_name"].fillna("").astype(str)
    comp["compound_group_name"] = comp["compound_group_name"].fillna("").astype(str)
    comp["formula"] = comp["formula"].fillna("").astype(str)
    comp["mass_list_matches"] = comp["mass_list_matches"].fillna("").astype(str)
    comp["is_named"] = comp["annotation_name"].map(
        lambda x: bool(x and x.lower() not in {"not named", "nan"})
    )
    comp["polarity"] = polarity
    comp.index = comp.index.map(str)
    return comp


def _load_polarity_run(
    input_dir: Path,
    polarity: str,
    istd_names: tuple[str, ...],
    config: DatasetConfig,
) -> PolarityRun:
    exp = pd.read_csv(input_dir / f"{polarity}_expression.csv", index_col=0, low_memory=False)
    batch = pd.read_csv(input_dir / f"{polarity}_batch.csv", index_col=0)
    comp = pd.read_csv(input_dir / f"{polarity}_compounds.csv")

    exp = _standardize_expression(exp)
    comp = _standardize_feature_metadata(comp, polarity.upper())

    raw_sample_labels = exp.index.to_series().astype(str)
    sample_names = raw_sample_labels.map(_canonical_sample_name).tolist()
    batch_labels = [_canonical_batch_label(x) for x in batch.iloc[:, 0].astype(str).tolist()]
    if len(sample_names) != len(batch_labels):
        raise ValueError(
            f"{polarity} sample rows ({len(sample_names)}) do not match batch labels "
            f"({len(batch_labels)})."
        )

    row_ids = [
        f"{name}__{batch_labels[i]}__{i:04d}" for i, name in enumerate(sample_names)
    ]
    exp.index = row_ids

    sample_info = pd.DataFrame(
        {
            "row_id": row_ids,
            "sample_name": sample_names,
            "batch": batch_labels,
        }
    ).set_index("row_id")
    sample_info["sample_type"] = np.where(
        sample_info["sample_name"].str.lower().str.startswith("pooled"),
        "qc_pool",
        "biological",
    )
    batch_index_labels = batch.index.to_series().astype(str).tolist()
    sample_info["raw_file"] = ""
    order_cols = [
        col
        for col in batch.columns
        if _normalise_name(col) in {"injectionorder", "acquisitionorder", "fnumber", "fileindex"}
    ]
    if order_cols:
        parsed_order = pd.to_numeric(batch[order_cols[0]], errors="coerce").tolist()
        order_source = f"batch_csv_column:{order_cols[0]}"
    else:
        raw_file_cols = [
            col
            for col in batch.columns
            if _normalise_name(col) in {"rawfile", "filename", "filepath", "file"}
        ]
        raw_file_values = batch[raw_file_cols[0]].astype(str).tolist() if raw_file_cols else [""] * len(batch)
        parsed_order = [
            _extract_injection_order_value(raw_sample_labels.iloc[i], batch_index_labels[i], raw_file_values[i])
            for i in range(len(sample_names))
        ]
        if raw_file_cols and any(np.isfinite(v) for v in parsed_order):
            order_source = f"parsed_f_number:{raw_file_cols[0]}"
        elif any(np.isfinite(v) for v in parsed_order):
            order_source = "parsed_f_number:sample_labels"
        else:
            order_source = "row_order_proxy"
    sample_info["injection_order"] = parsed_order
    raw_map, raw_order_source = _load_raw_acquisition_map(config, polarity)
    if raw_map is not None and not raw_map.empty:
        current = sample_info.reset_index().rename(columns={"index": "row_id"})
        current["sample_key"] = [
            _sample_match_key(name, sample_type)
            for name, sample_type in zip(current["sample_name"], current["sample_type"])
        ]
        current["match_order"] = current.groupby(["sample_key", "batch"]).cumcount()
        raw_map = raw_map.copy()
        raw_map["sample_key"] = [
            _sample_match_key(name, sample_type)
            for name, sample_type in zip(raw_map["sample_name"], raw_map["sample_type"])
        ]
        raw_map["match_order"] = raw_map.groupby(["sample_key", "batch"]).cumcount()
        raw_map = raw_map.rename(columns={"raw_file": "raw_file_from_workbook"})
        merged = current.merge(
            raw_map[
                [
                    "sample_key",
                    "batch",
                    "match_order",
                    "raw_file_from_workbook",
                    "raw_injection_order",
                ]
            ],
            on=["sample_key", "batch", "match_order"],
            how="left",
        )
        raw_orders = pd.to_numeric(merged["raw_injection_order"], errors="coerce")
        matched = raw_orders.apply(np.isfinite)
        if matched.any():
            current.loc[matched, "injection_order"] = raw_orders.loc[matched].astype(float)
            current.loc[matched, "raw_file"] = (
                merged.loc[matched, "raw_file_from_workbook"].fillna("")
            )
            order_source = (
                raw_order_source if matched.all() else f"{raw_order_source}+row_order_proxy"
            )
            sample_info = current.set_index("row_id")[sample_info.columns.tolist()]
    missing_order = ~pd.to_numeric(sample_info["injection_order"], errors="coerce").apply(
        np.isfinite
    )
    if missing_order.any():
        fallback_order = sample_info.groupby("batch").cumcount() + 1
        sample_info.loc[missing_order, "injection_order"] = fallback_order.loc[missing_order]
        if missing_order.all():
            order_source = "row_order_proxy"
        elif "row_order_proxy" not in order_source:
            order_source = f"{order_source}+row_order_proxy"
    sample_info["injection_order"] = sample_info["injection_order"].astype(float)
    biological = sample_info["sample_type"] == "biological"
    sample_info["is_bridge"] = False
    if biological.any():
        batch_counts = (
            sample_info.loc[biological, ["sample_name", "batch"]]
            .drop_duplicates()
            .groupby("sample_name")["batch"]
            .nunique()
        )
        bridge_names = set(batch_counts[batch_counts > 1].index)
        sample_info.loc[biological, "is_bridge"] = sample_info.loc[
            biological, "sample_name"
        ].isin(bridge_names)

    feature_ids = [c for c in exp.columns if c in comp.index]
    exp = exp[feature_ids].copy()
    comp = comp.loc[feature_ids].copy()

    raw_istd_ids = [
        fid
        for fid in comp.index
        if fid.lower().startswith("c")
        or _normalise_name(comp.at[fid, "annotation_name"])
        in {_normalise_name(x) for x in istd_names}
    ]
    raw_istd = exp[raw_istd_ids].copy()
    return PolarityRun(
        polarity=polarity.upper(),
        expression=exp,
        sample_info=sample_info,
        feature_meta=comp,
        raw_istd=raw_istd,
        injection_order_source=order_source,
    )


def _tic(df: pd.DataFrame) -> pd.Series:
    return df.sum(axis=1, skipna=True)


def _plot_istd_diagnostics(
    run: PolarityRun,
    output_dir: Path,
    prefix: str,
) -> list[str]:
    flags: list[str] = []
    _ensure_dir(output_dir)
    biological_or_qc = run.sample_info.copy()
    biological_or_qc["tic"] = _tic(run.expression.drop(columns=run.raw_istd.columns, errors="ignore"))
    x_label = (
        "Injection order (F#)"
        if "f_number" in run.injection_order_source
        else "Injection order"
        if "raw_workbook" in run.injection_order_source or "file_sequence" in run.injection_order_source
        else "Injection order proxy"
    )

    for batch, batch_rows in biological_or_qc.groupby("batch"):
        batch_rows = batch_rows.sort_values("injection_order")
        row_ids = batch_rows.index.tolist()
        if not row_ids:
            continue
        fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
        for column in run.raw_istd.columns:
            axes[0].plot(
                batch_rows["injection_order"],
                run.raw_istd.loc[row_ids, column].values,
                marker="o",
                label=column,
            )
        axes[0].set_title(f"{prefix} batch {batch}: raw ISTD signal")
        axes[0].set_ylabel("raw intensity")
        axes[0].legend(fontsize=8)

        for column in run.raw_istd.columns:
            values = run.raw_istd.loc[row_ids, column].astype(float)
            mean_val = values.mean(skipna=True)
            scaled = values / mean_val if mean_val and np.isfinite(mean_val) else values
            axes[1].plot(
                batch_rows["injection_order"],
                scaled.values,
                marker="o",
                label=column,
            )
        axes[1].set_title("ISTD concordance (scaled to per-ISTD mean)")
        axes[1].set_ylabel("scaled signal")
        axes[1].legend(fontsize=8)

        axes[2].plot(
            batch_rows["injection_order"],
            batch_rows["tic"].values,
            marker="o",
            color="black",
        )
        axes[2].set_title("Total ion current")
        axes[2].set_xlabel(x_label)
        axes[2].set_ylabel("TIC")
        out_path = output_dir / f"{prefix.lower()}_batch_{batch}_drift.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)

        if run.raw_istd.shape[1] >= 2:
            corr = run.raw_istd.loc[row_ids].corr(min_periods=3).values
            if corr.size and np.isfinite(corr).any():
                finite = corr[np.isfinite(corr)]
                if finite.size and np.nanmin(finite) < 0.50:
                    flags.append(
                        f"{prefix} batch {batch}: ISTD concordance is weak "
                        f"(minimum pairwise r={np.nanmin(finite):.2f})."
                    )
    return flags


def _select_feature_traces(df: pd.DataFrame, n_features: int = 5) -> list[str]:
    means = df.mean(axis=0, skipna=True).sort_values(ascending=False)
    return means.head(n_features).index.tolist()


def _plot_post_normalization_diagnostics(
    run: PolarityRun,
    normalized: pd.DataFrame,
    output_dir: Path,
    prefix: str,
) -> list[str]:
    flags: list[str] = []
    _ensure_dir(output_dir)
    traces = _select_feature_traces(normalized.drop(columns=run.raw_istd.columns, errors="ignore"))
    x_label = (
        "Injection order (F#)"
        if "f_number" in run.injection_order_source
        else "Injection order"
        if "raw_workbook" in run.injection_order_source or "file_sequence" in run.injection_order_source
        else "Injection order proxy"
    )

    for batch, batch_rows in run.sample_info.groupby("batch"):
        batch_rows = batch_rows.sort_values("injection_order")
        row_ids = batch_rows.index.tolist()
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        tic = _tic(normalized.loc[row_ids])
        axes[0].plot(batch_rows["injection_order"], tic.values, marker="o", color="black")
        axes[0].set_title(f"{prefix} batch {batch}: post-normalization TIC")
        axes[0].set_ylabel("TIC")

        for feature in traces:
            axes[1].plot(
                batch_rows["injection_order"],
                normalized.loc[row_ids, feature].values,
                marker="o",
                label=feature,
            )
        axes[1].set_title("Post-normalization high-abundance traces")
        axes[1].set_xlabel(x_label)
        axes[1].set_ylabel("normalized intensity")
        axes[1].legend(fontsize=7, ncol=2)
        out_path = output_dir / f"{prefix.lower()}_batch_{batch}_postnorm.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)

        if len(tic) >= 3:
            slope = np.polyfit(batch_rows["injection_order"], tic.values, deg=1)[0]
            if np.isfinite(slope):
                scale = np.nanmean(np.abs(tic.values)) or 1.0
                if abs(slope) / scale > 0.05:
                    flags.append(
                        f"{prefix} batch {batch}: post-normalization TIC still trends with "
                        f"injection order (relative slope={abs(slope)/scale:.3f})."
                    )
    if traces:
        batch_frames = [
            (batch, batch_rows.sort_values("injection_order"))
            for batch, batch_rows in run.sample_info.groupby("batch")
        ]
        fig, axes = plt.subplots(len(traces) + 1, 1, figsize=(11, 3.1 * (len(traces) + 1)))
        if len(traces) == 0:
            axes = [axes]
        for batch, batch_rows in batch_frames:
            row_ids = batch_rows.index.tolist()
            axes[0].plot(
                batch_rows["injection_order"],
                _tic(normalized.loc[row_ids]).values,
                marker="o",
                label=batch,
            )
        axes[0].set_title(f"{prefix} cross-batch post-normalization TIC overlay")
        axes[0].set_ylabel("TIC")
        axes[0].legend(fontsize=8, ncol=2)
        for ax, feature in zip(axes[1:], traces):
            for batch, batch_rows in batch_frames:
                row_ids = batch_rows.index.tolist()
                ax.plot(
                    batch_rows["injection_order"],
                    normalized.loc[row_ids, feature].values,
                    marker="o",
                    label=batch,
                )
            ax.set_title(feature)
            ax.set_ylabel("normalized intensity")
        axes[-1].set_xlabel(x_label)
        axes[1].legend(fontsize=8, ncol=2)
        overlay_path = output_dir / f"{prefix.lower()}_cross_batch_postnorm_overlay.png"
        fig.tight_layout()
        fig.savefig(overlay_path, dpi=180)
        plt.close(fig)
    return flags


def _write_metadata_audit(
    pos_run: PolarityRun,
    neg_run: PolarityRun,
    sample_metadata: pd.DataFrame,
    output_dir: Path,
    config: DatasetConfig,
    artifacts: PipelineArtifacts,
) -> None:
    _ensure_dir(output_dir)
    pos_bio = pos_run.sample_info[pos_run.sample_info["sample_type"] == "biological"].copy()
    neg_bio = neg_run.sample_info[neg_run.sample_info["sample_type"] == "biological"].copy()
    pos_names = set(pos_bio["sample_name"])
    neg_names = set(neg_bio["sample_name"])
    all_names = sorted(pos_names | neg_names)

    rows: list[dict] = []
    alias_resolutions: list[str] = []
    missing_metadata: list[str] = []
    pos_only: list[str] = sorted(pos_names - neg_names)
    neg_only: list[str] = sorted(neg_names - pos_names)

    for sample_name in all_names:
        meta_row = sample_metadata.loc[sample_name] if sample_name in sample_metadata.index else None
        if isinstance(meta_row, pd.DataFrame):
            meta_row = meta_row.iloc[0]
        if meta_row is None:
            status = "missing"
            canonical_id = ""
            missing_metadata.append(sample_name)
            group = subgroup = gest_age = sample_gest_age = meta_batch = ""
        else:
            canonical_id = str(meta_row.get("MetadataCanonicalID", sample_name))
            status = "exact" if canonical_id == sample_name else "alias"
            if status == "alias":
                alias_resolutions.append(f"{sample_name}->{canonical_id}")
            group = meta_row.get("Group", "")
            subgroup = meta_row.get("Subgroup", "")
            gest_age = meta_row.get("GestAgeDelivery", "")
            sample_gest_age = meta_row.get("SampleGestAge", "")
            meta_batch = meta_row.get("Batch", "")

        pos_batches = sorted(pos_bio.loc[pos_bio["sample_name"] == sample_name, "batch"].astype(str).unique())
        neg_batches = sorted(neg_bio.loc[neg_bio["sample_name"] == sample_name, "batch"].astype(str).unique())
        rows.append(
            {
                "sample_name": sample_name,
                "in_pos": sample_name in pos_names,
                "in_neg": sample_name in neg_names,
                "pos_batches": ";".join(pos_batches),
                "neg_batches": ";".join(neg_batches),
                "metadata_status": status,
                "metadata_sample_id": canonical_id,
                "Group": group,
                "Subgroup": subgroup,
                "GestAgeDelivery": gest_age,
                "SampleGestAge": sample_gest_age,
                "MetadataBatch": meta_batch,
            }
        )

    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(output_dir / f"{config.dataset_id}_metadata_audit.csv", index=False)

    if alias_resolutions:
        artifacts.method_log.append(
            f"{config.dataset_id}: resolved {len(alias_resolutions)} sample IDs via metadata aliases before normalization."
        )
    if missing_metadata:
        artifacts.qc_warnings.append(
            f"{config.dataset_id}: samples missing metadata before normalization: {missing_metadata}."
        )
    if pos_only:
        artifacts.qc_warnings.append(
            f"{config.dataset_id}: biological samples present only in POS mode: {pos_only}."
        )
    if neg_only:
        artifacts.qc_warnings.append(
            f"{config.dataset_id}: biological samples present only in NEG mode: {neg_only}."
        )


def _istd_normalize(
    run: PolarityRun,
    artifacts: PipelineArtifacts,
) -> tuple[pd.DataFrame, list[str]]:
    feature_ids = run.feature_meta.index.tolist()
    non_istd_features = [c for c in feature_ids if c not in run.raw_istd.columns]
    normalized = run.expression[feature_ids].copy()
    if run.raw_istd.empty:
        artifacts.qc_warnings.append(
            f"{run.polarity}: no internal standards were detected; ISTD normalization was skipped."
        )
        return normalized[non_istd_features], []
    geo = run.raw_istd.apply(_geometric_mean, axis=1)
    finite_geo = geo[np.isfinite(geo) & (geo > 0)]
    if finite_geo.empty:
        artifacts.qc_warnings.append(
            f"{run.polarity}: all internal-standard geometric means were missing; "
            "ISTD normalization was skipped."
        )
        return normalized[non_istd_features], list(run.raw_istd.columns)

    geo_filled = geo.copy()
    batch_geo = geo.groupby(run.sample_info["batch"]).transform(
        lambda s: s[np.isfinite(s) & (s > 0)].median()
    )
    geo_filled = geo_filled.where(np.isfinite(geo_filled) & (geo_filled > 0), batch_geo)
    global_geo = float(finite_geo.median())
    geo_filled = geo_filled.fillna(global_geo)
    imputed_rows = int(((~np.isfinite(geo)) | (geo <= 0)).sum())
    if imputed_rows:
        artifacts.method_log.append(
            f"{run.polarity}: filled missing sample-level ISTD geometric means for "
            f"{imputed_rows} rows using batch/global medians."
        )

    normalized = normalized.divide(geo_filled.replace(0, np.nan), axis=0)
    normalized = normalized.replace([np.inf, -np.inf], np.nan)
    artifacts.method_log.append(
        f"{run.polarity}: ISTD normalization used geometric mean of {list(run.raw_istd.columns)}."
    )
    return normalized[non_istd_features], list(run.raw_istd.columns)


def _median_fold_change_batch_normalize(
    df: pd.DataFrame,
    sample_info: pd.DataFrame,
    ref_batch: str,
    artifacts: PipelineArtifacts,
) -> pd.DataFrame:
    corrected = df.copy()
    qc_mask = sample_info["sample_type"] == "qc_pool"
    ref_mask = qc_mask & (sample_info["batch"] == ref_batch)
    ref_qc = corrected.loc[ref_mask]
    if ref_qc.empty:
        artifacts.qc_warnings.append(
            f"Reference batch {ref_batch} has no QC pools; batch scaling factors default to 1.0."
        )
        return corrected

    for batch in sorted(sample_info["batch"].unique()):
        if batch == ref_batch:
            artifacts.batch_scaling_factors[batch] = 1.0
            continue
        batch_mask = qc_mask & (sample_info["batch"] == batch)
        batch_qc = corrected.loc[batch_mask]
        if batch_qc.empty:
            artifacts.batch_scaling_factors[batch] = 1.0
            artifacts.qc_warnings.append(
                f"Batch {batch} has no QC pools; scaling factor left at 1.0."
            )
            continue
        per_feature_ratio = batch_qc.median(axis=0, skipna=True) / ref_qc.median(axis=0, skipna=True)
        finite_ratio = per_feature_ratio.replace([np.inf, -np.inf], np.nan).dropna()
        scaling_factor = float(finite_ratio.median()) if not finite_ratio.empty else float("nan")
        if not np.isfinite(scaling_factor) or scaling_factor == 0:
            scaling_factor = 1.0
        corrected.loc[sample_info["batch"] == batch] = (
            corrected.loc[sample_info["batch"] == batch] / scaling_factor
        )
        artifacts.batch_scaling_factors[batch] = scaling_factor
    return corrected


def _feature_missingness_filter(
    df: pd.DataFrame,
    sample_info: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    thresholds: Thresholds,
    artifacts: PipelineArtifacts,
    polarity: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    biological = sample_info["sample_type"] == "biological"
    bio_df = df.loc[biological].copy()
    meta = sample_metadata.reindex(sample_info.loc[biological, "sample_name"]).copy()
    groups = meta["Group"].fillna("Unknown").values
    kept_cols: list[str] = []
    rows: list[dict] = []
    for col in bio_df.columns:
        series = bio_df[col]
        miss_any = float(series.isna().mean())
        control_mask = groups == "Control"
        case_mask = ~control_mask
        control_missing = (
            float(series[control_mask].isna().mean()) if control_mask.any() else float("nan")
        )
        case_missing = (
            float(series[case_mask].isna().mean()) if case_mask.any() else float("nan")
        )
        drop = (
            (np.isfinite(control_missing) and control_missing > thresholds.feature_missing_threshold)
            or (np.isfinite(case_missing) and case_missing > thresholds.feature_missing_threshold)
        )
        rows.append(
            {
                "feature_id": col,
                "polarity": polarity,
                "overall_missing": miss_any,
                "control_missing": control_missing,
                "case_missing": case_missing,
                "dropped": drop,
            }
        )
        if drop:
            artifacts.feature_filter_log.append(
                {
                    "step": 7,
                    "feature_id": col,
                    "polarity": polarity,
                    "reason": (
                        f"missingness control={control_missing:.3f}, "
                        f"case={case_missing:.3f}"
                    ),
                }
            )
            if (
                np.isfinite(control_missing)
                and np.isfinite(case_missing)
                and abs(control_missing - case_missing) > 0.10
            ):
                artifacts.qc_warnings.append(
                    f"{polarity} feature {col} shows differential missingness "
                    f"(control={control_missing:.2f}, case={case_missing:.2f})."
                )
        else:
            kept_cols.append(col)
    return df[kept_cols].copy(), pd.DataFrame(rows)


def _rsd_filter(
    df: pd.DataFrame,
    sample_info: pd.DataFrame,
    thresholds: Thresholds,
    artifacts: PipelineArtifacts,
    polarity: str,
) -> pd.DataFrame:
    """Drop features whose QC-pool RSD exceeds the threshold in ANY batch.

    Mirrors Kayla's MTBL_datacleaning.py ``rsd()`` function: RSD is computed
    separately for each batch's QC pools, and a feature is removed if it fails
    in at least one batch.  Computing a single pooled RSD across all batches
    could mask high within-batch variability that averages away.
    """
    qc_mask = sample_info["sample_type"] == "qc_pool"
    unique_batches = sample_info.loc[qc_mask, "batch"].unique()
    if len(unique_batches) == 0:
        artifacts.qc_warnings.append(f"{polarity}: no QC pools available for RSD filtering.")
        return df

    # per-batch RSD table: columns = batches, rows = features
    batch_rsd: dict[str, pd.Series] = {}
    for batch in unique_batches:
        qc_idx = sample_info.index[(qc_mask) & (sample_info["batch"] == batch)]
        if len(qc_idx) < 2:
            # Need at least 2 QC pools to compute meaningful RSD
            artifacts.qc_warnings.append(
                f"{polarity} batch {batch}: only {len(qc_idx)} QC pool(s); "
                "batch skipped in RSD filter."
            )
            continue
        qc_batch = df.loc[qc_idx]
        batch_rsd[batch] = (
            qc_batch.std(axis=0, skipna=True) / qc_batch.mean(axis=0, skipna=True)
        ) * 100.0

    if not batch_rsd:
        artifacts.qc_warnings.append(
            f"{polarity}: no batch had ≥2 QC pools; RSD filter skipped."
        )
        return df

    # Feature fails if it exceeds the threshold in ANY batch (Kayla's logic)
    fail_mask = pd.Series(False, index=df.columns)
    for rsd_series in batch_rsd.values():
        fail_mask |= rsd_series.reindex(df.columns).fillna(np.inf) > thresholds.rsd_threshold

    for col in fail_mask.index[fail_mask]:
        per_batch_str = ", ".join(
            f"batch{b}={batch_rsd[b].get(col, float('nan')):.1f}%"
            for b in sorted(batch_rsd)
            if col in batch_rsd[b].index
        )
        artifacts.feature_filter_log.append(
            {
                "step": 8,
                "feature_id": col,
                "polarity": polarity,
                "reason": f"QC RSD >30% in ≥1 batch ({per_batch_str})",
            }
        )

    keep = ~fail_mask
    return df.loc[:, keep.index[keep]].copy()


def _sample_missingness_filter(
    df: pd.DataFrame,
    sample_info: pd.DataFrame,
    thresholds: Thresholds,
    artifacts: PipelineArtifacts,
    polarity: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keep_mask = pd.Series(True, index=df.index)
    biological = sample_info["sample_type"] == "biological"
    miss = df.loc[biological].isna().mean(axis=1)
    for row_id, frac in miss.items():
        if frac > thresholds.sample_missing_threshold:
            keep_mask.loc[row_id] = False
            artifacts.sample_filter_log.append(
                {
                    "step": 9,
                    "row_id": row_id,
                    "sample_name": sample_info.at[row_id, "sample_name"],
                    "polarity": polarity,
                    "reason": f"missingness={frac:.3f}",
                }
            )
    return df.loc[keep_mask].copy(), sample_info.loc[keep_mask].copy()


def _sample_istd_mad_filter(
    run: PolarityRun,
    sample_info: pd.DataFrame,
    thresholds: Thresholds,
    artifacts: PipelineArtifacts,
) -> set[str]:
    bad_row_ids: set[str] = set()
    biological = sample_info["sample_type"] == "biological"
    for batch, rows in sample_info.loc[biological].groupby("batch"):
        row_ids = rows.index.tolist()
        for column in run.raw_istd.columns:
            values = run.raw_istd.loc[row_ids, column].astype(float)
            med = values.median(skipna=True)
            mad = stats.median_abs_deviation(values.dropna(), nan_policy="omit")
            if not np.isfinite(mad) or mad == 0:
                continue
            lo = med - thresholds.sample_mad_threshold * mad
            hi = med + thresholds.sample_mad_threshold * mad
            failures = values[(values < lo) | (values > hi)]
            for row_id, value in failures.items():
                bad_row_ids.add(row_id)
                artifacts.sample_filter_log.append(
                    {
                        "step": 10,
                        "row_id": row_id,
                        "sample_name": sample_info.at[row_id, "sample_name"],
                        "polarity": run.polarity,
                        "batch": batch,
                        "reason": (
                            f"{column} raw ISTD={value:.3e} outside [{lo:.3e}, {hi:.3e}] "
                            f"(median={med:.3e}, MAD={mad:.3e})"
                        ),
                    }
                )
    return bad_row_ids


def _half_minimum_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values with half the per-feature minimum.

    The input is expected to be in log2 space (log2(x + 1)).  In log2 space
    dividing a value by 2 in linear scale is equivalent to subtracting 1:
        log2(x / 2) = log2(x) - 1
    Using ``min / 2`` would instead give the *square root* of the minimum in
    linear space, which is incorrect.
    """
    out = df.copy()
    for col in out.columns:
        valid = out[col].dropna()
        if valid.empty:
            continue
        fill = float(valid.min()) - 1.0  # half-minimum in log2 space
        out[col] = out[col].fillna(fill)
    return out


def _pca_plot(
    df: pd.DataFrame,
    labels: pd.Series,
    title: str,
    out_path: Path,
) -> None:
    X = df.copy()
    X = X.apply(lambda col: col.fillna(col.median(skipna=True)), axis=0)
    X = X.dropna(axis=1, how="all")
    if X.shape[0] < 3 or X.shape[1] < 2:
        return
    pca = PCA(n_components=2)
    scaled = StandardScaler().fit_transform(X.values)
    coords = pca.fit_transform(scaled)
    pca_df = pd.DataFrame(coords, index=X.index, columns=["PC1", "PC2"])
    if isinstance(labels, pd.Series):
        if labels.index.equals(X.index):
            label_values = labels.reindex(X.index)
        elif len(labels) == len(X):
            label_values = pd.Series(labels.to_numpy(), index=X.index)
        else:
            label_values = pd.Series("Unknown", index=X.index)
    else:
        label_values = pd.Series(labels, index=X.index)
    pca_df["label"] = label_values.fillna("Unknown").astype(str).values
    fig, ax = plt.subplots(figsize=(8, 6))
    for label in sorted(pca_df["label"].unique()):
        subset = pca_df[pca_df["label"] == label]
        ax.scatter(subset["PC1"], subset["PC2"], label=label, alpha=0.75, s=45)
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _batch_confounding_checks(
    sample_info: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    artifacts: PipelineArtifacts,
) -> list[dict]:
    rows: list[dict] = []
    sample_names = sample_info["sample_name"]
    meta = sample_metadata.reindex(sample_names.values).copy()
    meta.index = sample_info.index
    batch = sample_info["batch"]
    for covariate in ["Group", "Subgroup"]:
        if covariate not in meta.columns:
            continue
        table = pd.crosstab(batch.values, meta[covariate].values)
        if table.shape[0] > 1 and table.shape[1] > 1:
            stat, p_value, _, _ = stats.chi2_contingency(table)
            rows.append({"covariate": covariate, "test": "chi_square", "p_value": p_value})
            if p_value < 0.05:
                artifacts.qc_warnings.append(
                    f"Batch confounding warning: {covariate} is associated with batch "
                    f"(chi-square p={p_value:.4g})."
                )
    for covariate in ["SampleGestAge", "GestAgeDelivery"]:
        if covariate not in meta.columns:
            continue
        grouped = []
        for batch_id, series in meta.groupby(batch)[covariate]:
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if len(numeric) >= 2:
                grouped.append(numeric.values)
        if len(grouped) >= 2:
            stat, p_value = stats.f_oneway(*grouped)
            rows.append({"covariate": covariate, "test": "anova", "p_value": p_value})
            if p_value < 0.05:
                artifacts.qc_warnings.append(
                    f"Batch confounding warning: {covariate} differs by batch "
                    f"(ANOVA p={p_value:.4g})."
                )
    return rows


def _combat_design_matrix(
    sample_info: pd.DataFrame,
    sample_metadata: pd.DataFrame,
) -> tuple[pd.DataFrame | None, list[str]]:
    meta = sample_metadata.reindex(sample_info["sample_name"].values).copy()
    meta.index = sample_info.index

    frames: list[pd.DataFrame] = []
    used: list[str] = []

    if "Subgroup" in meta.columns and meta["Subgroup"].dropna().nunique() > 1:
        dummies = pd.get_dummies(
            meta["Subgroup"].fillna("Unknown").astype(str),
            prefix="Subgroup",
            drop_first=True,
            dtype=float,
        )
        if not dummies.empty:
            frames.append(dummies)
            used.append("Subgroup")
    elif "Group" in meta.columns and meta["Group"].dropna().nunique() > 1:
        dummies = pd.get_dummies(
            meta["Group"].fillna("Unknown").astype(str),
            prefix="Group",
            drop_first=True,
            dtype=float,
        )
        if not dummies.empty:
            frames.append(dummies)
            used.append("Group")

    for covariate in ["SampleGestAge", "GestAgeDelivery"]:
        if covariate not in meta.columns:
            continue
        numeric = pd.to_numeric(meta[covariate], errors="coerce")
        if numeric.notna().sum() < 2 or numeric.nunique(dropna=True) <= 1:
            continue
        frames.append(
            numeric.fillna(numeric.median(skipna=True)).astype(float).rename(covariate).to_frame()
        )
        used.append(covariate)

    if not frames:
        return None, []

    design = pd.concat(frames, axis=1)
    design = design.loc[:, design.nunique(dropna=False) > 1]
    if design.empty:
        return None, []

    keep_cols: list[str] = []
    for column in design.columns:
        trial = design[keep_cols + [column]].to_numpy(dtype=float)
        if np.linalg.matrix_rank(trial) > len(keep_cols):
            keep_cols.append(column)
    design = design[keep_cols]
    if design.empty:
        return None, []
    return design, used


def _combat_r_ref_batch(
    df_log2_imputed: pd.DataFrame,
    batch_labels: pd.Series,
    ref_batch: str,
    covariates: pd.DataFrame | None,
) -> pd.DataFrame:
    with tempfile.TemporaryDirectory(prefix="dp3-combat-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        matrix_path = tmpdir_path / "matrix.csv"
        batch_path = tmpdir_path / "batch.csv"
        covariate_path = tmpdir_path / "covariates.csv"
        output_path = tmpdir_path / "corrected.csv"
        script_path = tmpdir_path / "combat_ref_batch.R"

        df_log2_imputed.to_csv(matrix_path)
        pd.DataFrame({"batch": batch_labels.astype(str)}).to_csv(batch_path)
        if covariates is not None and not covariates.empty:
            covariates.reindex(df_log2_imputed.index).to_csv(covariate_path)
        else:
            covariate_path.write_text("")

        script_path.write_text(
            textwrap.dedent(
                """
                args <- commandArgs(trailingOnly = TRUE)
                matrix_path <- args[1]
                batch_path <- args[2]
                ref_batch <- args[3]
                covariate_path <- args[4]
                output_path <- args[5]

                suppressPackageStartupMessages(library(sva))

                mat <- read.csv(matrix_path, row.names = 1, check.names = FALSE)
                batch_df <- read.csv(batch_path, row.names = 1, check.names = FALSE)
                batch <- factor(batch_df$batch)
                mod <- NULL

                if (file.info(covariate_path)$size > 0) {
                  mod_df <- read.csv(covariate_path, row.names = 1, check.names = FALSE)
                  if (ncol(mod_df) > 0) {
                    mod <- as.matrix(mod_df)
                  }
                }

                dat <- t(as.matrix(mat))
                corrected <- ComBat(
                  dat = dat,
                  batch = batch,
                  mod = mod,
                  par.prior = TRUE,
                  prior.plots = FALSE,
                  ref.batch = ref_batch
                )
                write.csv(t(corrected), output_path, quote = FALSE)
                """
            )
        )
        result = subprocess.run(
            [
                "Rscript",
                str(script_path),
                str(matrix_path),
                str(batch_path),
                str(ref_batch),
                str(covariate_path),
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        corrected = pd.read_csv(output_path, index_col=0)
        corrected = corrected.reindex(index=df_log2_imputed.index, columns=df_log2_imputed.columns)
        if corrected.isna().all().all():
            raise RuntimeError(f"R ComBat produced an all-NaN matrix. stderr={result.stderr.strip()}")
        return corrected


def _combat_pycombat(
    df_log2_imputed: pd.DataFrame,
    batch_labels: pd.Series,
    covariates: pd.DataFrame | None,
) -> pd.DataFrame:
    from pycombat import Combat

    model = Combat()
    X = covariates.to_numpy(dtype=float) if covariates is not None and not covariates.empty else None
    corrected = model.fit_transform(
        df_log2_imputed.to_numpy(dtype=float),
        batch_labels.astype(str).to_numpy(),
        X=X,
    )
    return pd.DataFrame(corrected, index=df_log2_imputed.index, columns=df_log2_imputed.columns)


def _reference_location_scale_correction(
    df_log2_imputed: pd.DataFrame,
    batch_labels: pd.Series,
    ref_batch: str,
) -> pd.DataFrame:
    """
    Fallback when a true ComBat backend is unavailable.

    Per-feature, each non-reference batch is transformed to match the
    reference batch mean and standard deviation on the imputed log2 scale.
    """
    corrected = df_log2_imputed.copy()
    ref_mask = batch_labels == ref_batch
    if ref_mask.sum() < 2:
        return corrected
    for col in corrected.columns:
        ref = corrected.loc[ref_mask, col]
        ref_mean = ref.mean()
        ref_std = ref.std(ddof=1)
        if not np.isfinite(ref_std) or ref_std == 0:
            ref_std = 1.0
        for batch in sorted(batch_labels.unique()):
            if batch == ref_batch:
                continue
            mask = batch_labels == batch
            values = corrected.loc[mask, col]
            mean = values.mean()
            std = values.std(ddof=1)
            if not np.isfinite(std) or std == 0:
                std = ref_std
            corrected.loc[mask, col] = ((values - mean) / std) * ref_std + ref_mean
    return corrected


def _apply_batch_correction(
    df_log2_imputed: pd.DataFrame,
    batch_labels: pd.Series,
    ref_batch: str,
    covariates: pd.DataFrame | None,
    protected_covariates: list[str],
    bridge_count: int,
    artifacts: PipelineArtifacts,
) -> tuple[pd.DataFrame, str]:
    aligned_batches = batch_labels.reindex(df_log2_imputed.index)
    if aligned_batches.nunique(dropna=True) < 2:
        artifacts.method_log.append("Single batch detected; batch correction skipped.")
        return df_log2_imputed.copy(), "no_batch_correction_single_batch"

    if covariates is not None and not covariates.empty:
        covariates = covariates.reindex(df_log2_imputed.index)

    try:
        corrected = _combat_r_ref_batch(df_log2_imputed, aligned_batches, ref_batch, covariates)
        detail = f"R sva::ComBat(ref.batch={ref_batch})"
        if protected_covariates:
            detail += f", protected={protected_covariates}"
        detail += f", bridge_samples={bridge_count}"
        artifacts.method_log.append(detail)
        return corrected, "sva_combat_ref_batch"
    except Exception as exc:
        artifacts.qc_warnings.append(
            f"R ref.batch ComBat failed ({exc}); trying Python pycombat without ref.batch."
        )

    try:
        corrected = _combat_pycombat(df_log2_imputed, aligned_batches, covariates)
        detail = "pycombat without ref.batch"
        if protected_covariates:
            detail += f", protected={protected_covariates}"
        detail += f", bridge_samples={bridge_count}"
        artifacts.method_log.append(detail)
        artifacts.qc_warnings.append(
            "Python pycombat does not support ref.batch; used standard empirical-Bayes ComBat."
        )
        return corrected, "pycombat_standard"
    except Exception as exc:
        artifacts.qc_warnings.append(
            f"Python pycombat failed ({exc}); used reference-batch location/scale fallback."
        )

    method_name = "reference_location_scale_fallback"
    corrected = _reference_location_scale_correction(df_log2_imputed, aligned_batches, ref_batch)
    artifacts.qc_warnings.append(
        "True ComBat backend unavailable; used reference-batch location/scale "
        "harmonization instead."
    )
    return corrected, method_name


def _annotation_score(meta: pd.DataFrame, expression: pd.DataFrame) -> pd.Series:
    intensity = meta["area_max"].copy()
    if intensity.isna().all():
        intensity = expression.mean(axis=0, skipna=True).reindex(meta.index)
    intensity = intensity.fillna(0.0)

    def scaled_rank(values: pd.Series) -> pd.Series:
        if values.max() <= values.min():
            return pd.Series(10.0, index=values.index)
        scaled = (values - values.min()) / (values.max() - values.min())
        return scaled * 10.0

    peak = meta["peak_rating"].apply(
        lambda x: 10
        if pd.notna(x) and x >= 7
        else 7
        if pd.notna(x) and x >= 5
        else 4
        if pd.notna(x) and x >= 3
        else 1
    )
    rsd = meta["rsd_qc"].apply(
        lambda x: 10
        if pd.notna(x) and x < 10
        else 8
        if pd.notna(x) and x < 15
        else 6
        if pd.notna(x) and x < 20
        else 4
        if pd.notna(x) and x < 25
        else 2
        if pd.notna(x) and x < 30
        else 0
    )
    ms2 = meta["ms2"].fillna("").map(
        lambda x: 10
        if x == "DDA for preferred ion"
        else 6
        if x == "DDA for other ion"
        else 4
        if x == "DDA available"
        else 0
    )
    annotation = meta["mzcloud_confidence"].apply(
        lambda x: 10
        if pd.notna(x) and x >= 90
        else 9
        if pd.notna(x) and x >= 80
        else 8
        if pd.notna(x) and x >= 70
        else 0
    )
    no_mzcloud = meta["mzcloud_confidence"].isna()
    annotation.loc[no_mzcloud] = meta.loc[no_mzcloud, "annotation_source_matches"].apply(
        lambda x: 10
        if x >= 6
        else 9
        if x == 5
        else 8
        if x == 4
        else 7
        if x == 3
        else 6
        if x == 2
        else 5
        if x == 1
        else 0
    )
    return peak + rsd + ms2 + scaled_rank(intensity) + annotation


def _candidate_classifications(
    group_meta: pd.DataFrame,
    polarity: str,
    modifications: pd.DataFrame,
    thresholds: Thresholds,
) -> pd.DataFrame:
    group_meta = group_meta.copy()
    if group_meta["calc_mw"].notna().all():
        expected = (
            group_meta["calc_mw"] + PROTON_MASS
            if polarity == "POS"
            else group_meta["calc_mw"] - PROTON_MASS
        )
        delta = group_meta["mz"] - expected
        group_meta["expected_parent_mz"] = expected
        group_meta["classification"] = "Unknown/Measurement Error"
        parent_mask = delta.abs() <= thresholds.mass_tolerance_parent
        group_meta.loc[parent_mask, "classification"] = "Parent"
        for idx, value in delta.loc[~parent_mask].items():
            matched = modifications.loc[
                (modifications["Delta_m/z"] - value).abs() <= thresholds.mass_tolerance_non_parent
            ]
            if not matched.empty:
                group_meta.at[idx, "classification"] = matched.iloc[0]["Type"]
                group_meta.at[idx, "modification_name"] = matched.iloc[0]["Name"]
            else:
                group_meta.at[idx, "modification_name"] = ""
    else:
        group_meta["classification"] = "Unclassified"
        group_meta["modification_name"] = ""
    return group_meta


def _resolve_named_groups(
    run: PolarityRun,
    expression: pd.DataFrame,
    thresholds: Thresholds,
    modifications: pd.DataFrame,
    artifacts: PipelineArtifacts,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = run.feature_meta.loc[expression.columns].copy()
    meta["quality_score"] = _annotation_score(meta, expression)
    keep_ids: set[str] = set()

    named = meta[meta["is_named"]].copy()
    unnamed = meta[~meta["is_named"]].copy()
    for name, group in named.groupby("compound_group_name"):
        classified = _candidate_classifications(group, run.polarity, modifications, thresholds)
        parent_ids = classified.index[classified["classification"] == "Parent"].tolist()
        if parent_ids:
            candidate_ids = parent_ids
            dropped = [idx for idx in classified.index if idx not in candidate_ids]
        else:
            candidate_ids = classified.index[
                classified["classification"] != "Unknown/Measurement Error"
            ].tolist()
            if not candidate_ids:
                candidate_ids = classified.index.tolist()
            dropped = [idx for idx in classified.index if idx not in candidate_ids]

        candidate_meta = classified.loc[candidate_ids].copy()
        rt_spread = candidate_meta["rt"].max(skipna=True) - candidate_meta["rt"].min(skipna=True)
        if len(candidate_meta) > 1 and np.isfinite(rt_spread):
            if rt_spread < 0.5:
                rt_flag = "chromatographic_artifact"
            elif rt_spread <= 3.0:
                rt_flag = "potential_structural_isomers"
            else:
                rt_flag = "likely_annotation_error"
        else:
            rt_flag = "single_candidate"

        ranked = candidate_meta.sort_values(
            ["quality_score", "area_max", "rsd_qc", "rt"],
            ascending=[False, False, True, True],
        )
        winner = ranked.index[0]
        keep_ids.add(winner)
        for loser in ranked.index[1:].tolist() + dropped:
            artifacts.dedup_log.append(
                {
                    "phase": "named_within_compound",
                    "polarity": run.polarity,
                    "group_name": name,
                    "representative": winner,
                    "dropped_feature": loser,
                    "rt_flag": rt_flag,
                }
            )

    named_kept = meta.loc[sorted(keep_ids)].copy() if keep_ids else meta.iloc[0:0].copy()
    return named_kept, unnamed.copy()


def _cluster_indices(
    meta: pd.DataFrame,
    mz_tol: float,
    rt_tol: float,
    require_formula: bool | None = None,
) -> list[list[str]]:
    if meta.empty:
        return []
    work = meta.copy().sort_values(["rt", "mz"])
    clusters: list[list[str]] = []
    used: set[str] = set()
    records = work.to_dict("index")
    ids = work.index.tolist()
    for i, fid in enumerate(ids):
        if fid in used:
            continue
        seed = [fid]
        used.add(fid)
        base = records[fid]
        for other in ids[i + 1 :]:
            if other in used:
                continue
            rec = records[other]
            if abs(rec["rt"] - base["rt"]) > rt_tol:
                if rec["rt"] > base["rt"]:
                    break
                continue
            if require_formula is True and rec["formula"] != base["formula"]:
                continue
            if require_formula is None and not math.isfinite(base["mz"]):
                continue
            tol = mz_tol
            if abs(rec["mz"] - base["mz"]) <= tol:
                seed.append(other)
                used.add(other)
        clusters.append(seed)
    return clusters


def _collapse_group_by_quality(
    group_ids: list[str],
    meta: pd.DataFrame,
    artifacts: PipelineArtifacts,
    phase: str,
    representative_reason: str,
) -> str:
    ranked = meta.loc[group_ids].sort_values(
        ["quality_score", "area_max", "rsd_qc", "rt"],
        ascending=[False, False, True, True],
    )
    winner = ranked.index[0]
    for loser in ranked.index[1:]:
        artifacts.dedup_log.append(
            {
                "phase": phase,
                "representative": winner,
                "dropped_feature": loser,
                "reason": representative_reason,
            }
        )
    return winner


def _resolve_unnamed_groups(
    run: PolarityRun,
    unnamed_meta: pd.DataFrame,
    expression: pd.DataFrame,
    thresholds: Thresholds,
    modifications: pd.DataFrame,
    artifacts: PipelineArtifacts,
) -> pd.DataFrame:
    if unnamed_meta.empty:
        return unnamed_meta
    meta = unnamed_meta.copy()
    meta["quality_score"] = _annotation_score(meta, expression)

    keep: set[str] = set()

    formula_meta = meta[meta["formula"].astype(str).str.strip() != ""].copy()
    for cluster in _cluster_indices(
        formula_meta,
        mz_tol=thresholds.mass_tolerance_non_parent,
        rt_tol=thresholds.rt_tolerance,
        require_formula=True,
    ):
        keep.add(
            _collapse_group_by_quality(
                cluster, meta, artifacts, "unnamed_formula", "formula+RT collapse"
            )
        )

    no_formula = meta[meta["formula"].astype(str).str.strip() == ""].copy()
    for cluster in _cluster_indices(
        no_formula,
        mz_tol=thresholds.mass_tolerance_dedup_no_formula,
        rt_tol=thresholds.rt_tolerance,
        require_formula=False,
    ):
        keep.add(
            _collapse_group_by_quality(
                cluster, meta, artifacts, "unnamed_mass_rt", "mz+RT collapse"
            )
        )

    reduced = meta.loc[sorted(keep)] if keep else meta.iloc[0:0].copy()
    if reduced.empty:
        return reduced

    # Adduct relationship collapsing among remaining unnamed features.
    ids = reduced.sort_values(["rt", "mz"]).index.tolist()
    parent: dict[str, str] = {fid: fid for fid in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, left in enumerate(ids):
        left_rt = reduced.at[left, "rt"]
        left_mz = reduced.at[left, "mz"]
        if not np.isfinite(left_rt) or not np.isfinite(left_mz):
            continue
        for right in ids[i + 1 :]:
            right_rt = reduced.at[right, "rt"]
            right_mz = reduced.at[right, "mz"]
            if right_rt - left_rt > thresholds.rt_tolerance:
                break
            if abs(right_rt - left_rt) > thresholds.rt_tolerance:
                continue
            delta = abs(right_mz - left_mz)
            matched = modifications.loc[
                (modifications["Delta_m/z"].abs() - delta).abs()
                <= thresholds.mass_tolerance_non_parent
            ]
            if not matched.empty:
                union(left, right)

    components: defaultdict[str, list[str]] = defaultdict(list)
    for fid in ids:
        components[find(fid)].append(fid)

    winners: list[str] = []
    for component in components.values():
        winners.append(
            _collapse_group_by_quality(
                component,
                reduced,
                artifacts,
                "unnamed_adduct_graph",
                "adduct/isotope/fragment graph collapse",
            )
        )
    ordered_winners = sorted(set(winners), key=winners.index)
    return reduced.loc[ordered_winners].copy()


def _finalize_feature_ids(
    meta: pd.DataFrame,
    database: str,
    thresholds: Thresholds,
    istd_names: set[str],
) -> pd.DataFrame:
    meta = meta.copy()
    final_ids: list[str] = []
    schymanski: list[str] = []
    for fid, row in meta.iterrows():
        polarity = row["polarity"]
        name = str(row.get("compound_group_name", row["annotation_name"])).strip()
        formula = str(row["formula"]).strip()
        mz = row["mz"]
        rt = row["rt"]

        if name and name.lower() not in {"not named", "nan"}:
            final_id = f"{name}_{polarity}"
        else:
            mz_text = f"{mz:.4f}" if np.isfinite(mz) else "nan"
            rt_text = f"{rt:.2f}" if np.isfinite(rt) else "nan"
            final_id = f"unk_{fid}_{polarity}_{mz_text}_RT{rt_text}"
        final_ids.append(final_id)

        if _normalise_name(name) in istd_names or "istd" in _normalise_name(name):
            schymanski.append("Level 1")
        elif name and name.lower() not in {"not named", "nan"}:
            if (
                pd.notna(row["mzcloud_confidence"])
                and row["mzcloud_confidence"] >= thresholds.mzcloud_l2_threshold
            ) or row["annotation_source_matches"] >= thresholds.annot_source_l2_threshold:
                schymanski.append("Level 2")
            else:
                schymanski.append("Level 3")
        elif formula:
            candidates = [
                token.strip()
                for token in re.split(r"[;|]", row["mass_list_matches"])
                if token.strip()
            ]
            if len(candidates) > 1:
                schymanski.append("Level 3")
            else:
                schymanski.append("Level 4")
        else:
            schymanski.append("Level 5")

    meta["final_feature_id"] = _make_unique(final_ids)
    meta["schymanski_level"] = schymanski
    meta["database"] = database
    return meta


def _merge_polarities(
    runs: list[tuple[PolarityRun, pd.DataFrame, pd.DataFrame]],
    sample_metadata: pd.DataFrame,
    config: DatasetConfig,
    thresholds: Thresholds,
    artifacts: PipelineArtifacts,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_tables: list[pd.DataFrame] = []
    matrices: list[pd.DataFrame] = []

    istd_names = {
        _normalise_name(name)
        for name in config.positive_istd_names + config.negative_istd_names
    }

    for run, corrected_before_average, sample_info in runs:
        # Average bridge samples after correction, on the log scale.
        biological_mask = sample_info["sample_type"] == "biological"
        biological_names = sample_info.loc[biological_mask, "sample_name"]
        averaged = corrected_before_average.loc[biological_mask].groupby(biological_names).mean()
        meta = run.retained_feature_meta.loc[corrected_before_average.columns].copy()
        meta = _finalize_feature_ids(meta, config.database, thresholds, istd_names)
        averaged.columns = meta["final_feature_id"].values
        meta.index = averaged.columns
        feature_tables.append(meta)
        matrices.append(averaged)

    merged = pd.concat(matrices, axis=1)
    feature_meta = pd.concat(feature_tables, axis=0)
    feature_meta = feature_meta[~feature_meta.index.duplicated(keep="first")]

    merged.index.name = "SampleID"
    merged = merged.join(sample_metadata, how="left")
    missing_samples = merged.index[merged["Group"].isna()].tolist()
    if missing_samples:
        artifacts.qc_warnings.append(
            f"{config.dataset_id}: averaged samples lacking metadata were dropped: {missing_samples}."
        )
    merged = merged[merged["Group"].notna()].copy()
    subject_ids = merged.index.to_series().str.replace(r"[A-E]$", "", regex=True)
    ordered_meta_cols = [
        "Group",
        "Subgroup",
        "Batch",
        "GestAgeDelivery",
        "SampleGestAge",
    ]
    present_meta_cols = [c for c in ordered_meta_cols if c in merged.columns]
    analyte_cols = [c for c in merged.columns if c not in present_meta_cols]
    final_matrix = pd.concat(
        [subject_ids.rename("SubjectID"), merged[present_meta_cols], merged[analyte_cols]],
        axis=1,
    )
    return final_matrix, feature_meta


def _trajectory_plots(
    final_matrix: pd.DataFrame,
    feature_meta: pd.DataFrame,
    output_dir: Path,
    config: DatasetConfig,
    artifacts: PipelineArtifacts,
) -> None:
    if config.tissue != "plasma" or "SampleGestAge" not in final_matrix.columns:
        return
    targets = ("progesterone", "cortisol", "estradiol")
    analyte_names = {
        fid: str(feature_meta.at[fid, "annotation_name"]).lower()
        for fid in feature_meta.index
    }
    selected = [
        fid for fid, name in analyte_names.items() if any(keyword in name for keyword in targets)
    ]
    if not selected:
        artifacts.qc_warnings.append(
            f"{config.dataset_id}: no curated pregnancy-trajectory metabolites were retained."
        )
        return
    traj_dir = output_dir / "trajectory_plots"
    _ensure_dir(traj_dir)
    x = pd.to_numeric(final_matrix["SampleGestAge"], errors="coerce")
    for fid in selected[:6]:
        y = pd.to_numeric(final_matrix[fid], errors="coerce")
        valid = x.notna() & y.notna()
        if valid.sum() < 3:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(x[valid], y[valid], alpha=0.7)
        ax.set_title(f"{feature_meta.at[fid, 'annotation_name']} trajectory")
        ax.set_xlabel("Gestational age at collection")
        ax.set_ylabel("Processed abundance")
        fig.tight_layout()
        fig.savefig(traj_dir / f"{fid}.png", dpi=180)
        plt.close(fig)


def _write_plain_text_log(
    output_dir: Path,
    config: DatasetConfig,
    artifacts: PipelineArtifacts,
    sample_matrix: pd.DataFrame,
    feature_meta: pd.DataFrame,
    missingness_reports: dict[str, pd.DataFrame],
    confounding_rows: list[dict],
) -> None:
    """Write the SOP Step 33 pipeline log.

    Includes: ISTD names, drift detection flags, per-step feature and sample
    counts (Steps 7-10), per-batch initial sample counts, flagged samples per
    step, Schymanski annotation tier breakdown, batch normalization details,
    ComBat method, confounding check results, bridge counts, and all QC
    warnings.
    """
    level_counts = feature_meta["schymanski_level"].value_counts(dropna=False)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sep = "=" * 72

    lines: list[str] = [
        sep,
        f"DP3 SOP Preprocessing Log  —  {config.dataset_id}",
        f"Generated : {now}",
        f"Database  : {config.database}",
        sep,
    ]

    # ── Internal standards ────────────────────────────────────────────────────
    lines.append("")
    lines.append("INTERNAL STANDARDS USED")
    lines.append("-" * 40)
    if artifacts.istd_names:
        for pol, names in sorted(artifacts.istd_names.items()):
            if names:
                lines.append(f"  {pol}: {', '.join(names)}")
            else:
                lines.append(f"  {pol}: (none detected — ISTD normalization skipped)")
    else:
        lines.append("  (no ISTD information recorded)")

    # ── Normalization / drift ─────────────────────────────────────────────────
    lines.append("")
    lines.append("NORMALIZATION SUMMARY (Steps 1–6)")
    lines.append("-" * 40)
    for method in artifacts.method_log:
        lines.append(f"  {method}")
    if artifacts.batch_scaling_factors:
        lines.append("")
        lines.append("  Median fold-change batch scaling factors:")
        for batch, factor in sorted(artifacts.batch_scaling_factors.items()):
            lines.append(f"    Batch {batch}: {factor:.6f}")
    lines.append("")
    if artifacts.drift_flags:
        lines.append("  Post-normalization drift flags:")
        for flag in artifacts.drift_flags:
            lines.append(f"    ! {flag}")
    else:
        lines.append("  Post-normalization drift: none flagged.")

    # ── Initial sample counts per batch ──────────────────────────────────────
    lines.append("")
    lines.append("INITIAL BIOLOGICAL SAMPLE COUNTS BY BATCH (before filtering)")
    lines.append("-" * 40)
    if artifacts.initial_sample_counts:
        for pol, batch_dict in sorted(artifacts.initial_sample_counts.items()):
            lines.append(f"  {pol}:")
            total = 0
            for batch in sorted(batch_dict, key=lambda b: (b.isdigit(), int(b) if b.isdigit() else b)):
                n = batch_dict[batch]
                total += n
                lines.append(f"    Batch {batch}: {n} samples")
            lines.append(f"    Total: {total} samples")
    else:
        lines.append("  (not recorded)")

    # ── Per-step feature / sample count table ─────────────────────────────────
    lines.append("")
    lines.append("PER-STEP FEATURE AND SAMPLE COUNTS (Steps 7–10)")
    lines.append("-" * 40)
    if artifacts.step_counts:
        for pol in sorted({sc["polarity"] for sc in artifacts.step_counts}):
            pol_steps = [sc for sc in artifacts.step_counts if sc["polarity"] == pol]
            lines.append(f"  {pol}:")
            lines.append(f"    {'Step':<42} {'Features':>10} {'Bio Samples':>12}")
            lines.append(f"    {'-'*42} {'-'*10} {'-'*12}")
            for sc in pol_steps:
                lines.append(
                    f"    {sc['label']:<42} {sc['n_features']:>10,} {sc['n_samples']:>12,}"
                )
        # Final counts after deduplication
        lines.append("")
        lines.append(f"  Final (post-dedup) feature count: {feature_meta.shape[0]:,}")
        lines.append(f"  Final biological sample count   : {len(sample_matrix):,}")
    else:
        lines.append("  (step-count tracking not available)")

    # ── Samples flagged per filtering step ───────────────────────────────────
    lines.append("")
    lines.append("SAMPLES FLAGGED BY FILTER STEP")
    lines.append("-" * 40)
    sample_steps: dict[tuple[str, int], list[str]] = {}
    for entry in artifacts.sample_filter_log:
        key = (entry.get("polarity", "?"), int(entry.get("step", 0)))
        sample_steps.setdefault(key, []).append(
            f"{entry.get('sample_name', entry.get('row_id', '?'))} "
            f"({entry.get('reason', '')})"
        )
    if sample_steps:
        step_labels = {9: "Step 9 — sample missingness", 10: "Step 10 — ISTD MAD"}
        for (pol, step), names in sorted(sample_steps.items()):
            label = step_labels.get(step, f"Step {step}")
            lines.append(f"  {pol} {label}: {len(names)} sample(s) removed")
            for nm in names:
                lines.append(f"    - {nm}")
    else:
        lines.append("  No samples were removed during filtering.")

    # ── Features flagged per filtering step ──────────────────────────────────
    lines.append("")
    lines.append("FEATURE FILTER SUMMARY BY STEP")
    lines.append("-" * 40)
    feat_by_step: dict[tuple[str, int], int] = {}
    for entry in artifacts.feature_filter_log:
        key = (entry.get("polarity", "?"), int(entry.get("step", 0)))
        feat_by_step[key] = feat_by_step.get(key, 0) + 1
    if feat_by_step:
        step_labels_feat = {
            7: "Step 7 — feature missingness",
            8: "Step 8 — QC RSD (per-batch)",
        }
        for (pol, step), count in sorted(feat_by_step.items()):
            label = step_labels_feat.get(step, f"Step {step}")
            lines.append(f"  {pol} {label}: {count:,} feature(s) removed")
    else:
        lines.append("  No features were removed during filtering.")

    # ── Deduplication ─────────────────────────────────────────────────────────
    lines.append("")
    lines.append("DEDUPLICATION (Steps 18–26)")
    lines.append("-" * 40)
    lines.append(f"  Total deduplication drop events: {len(artifacts.dedup_log):,}")

    # ── Batch correction ──────────────────────────────────────────────────────
    lines.append("")
    lines.append("BATCH CORRECTION (Step 15)")
    lines.append("-" * 40)
    combat_lines = [m for m in artifacts.method_log if "batch correction" in m.lower()]
    for m in combat_lines:
        lines.append(f"  {m}")
    if artifacts.bridge_counts:
        lines.append("")
        lines.append("  Bridge samples used for ComBat anchor:")
        for pol, count in sorted(artifacts.bridge_counts.items()):
            lines.append(f"    {pol}: {count} unique bridge sample(s)")

    # ── Confounding checks ────────────────────────────────────────────────────
    if confounding_rows:
        lines.append("")
        lines.append("BATCH-CONFOUNDING CHECKS (Step 14)")
        lines.append("-" * 40)
        for row in confounding_rows:
            sig = " *** SIGNIFICANT" if row["p_value"] < 0.05 else ""
            lines.append(
                f"  {row['covariate']} via {row['test']}: p={row['p_value']:.4g}{sig}"
            )

    # ── Annotation tiers ─────────────────────────────────────────────────────
    lines.append("")
    lines.append("SCHYMANSKI ANNOTATION TIER BREAKDOWN")
    lines.append("-" * 40)
    total_feat = max(len(feature_meta), 1)
    for level in ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]:
        count = int(level_counts.get(level, 0))
        pct = count / total_feat * 100
        lines.append(f"  {level}: {count:>6,}  ({pct:5.1f}%)")
    lines.append(f"  {'Total':<8}: {total_feat:>6,}")

    # ── Feature missingness detail ────────────────────────────────────────────
    if missingness_reports:
        lines.append("")
        lines.append("FEATURE MISSINGNESS DETAIL (Step 7)")
        lines.append("-" * 40)
        for pol, report in missingness_reports.items():
            if report.empty:
                lines.append(f"  {pol}: no features reported.")
                continue
            dropped = int(report["dropped"].sum())
            lines.append(f"  {pol}: {dropped} feature(s) removed for missingness.")
            if dropped:
                # Show differential-missingness warnings if any
                diff = report[
                    report.get("case_missing", pd.Series(dtype=float)).notna()
                    & (
                        (report.get("case_missing", pd.Series(dtype=float)) -
                         report.get("control_missing", pd.Series(dtype=float))).abs() > 0.10
                    )
                ] if "case_missing" in report.columns else pd.DataFrame()
                if not diff.empty:
                    lines.append(
                        f"    ({len(diff)} feature(s) also show differential missingness "
                        f">10% between groups)"
                    )

    # ── QC warnings ───────────────────────────────────────────────────────────
    lines.append("")
    if artifacts.qc_warnings:
        lines.append("QC WARNINGS")
        lines.append("-" * 40)
        for warning in artifacts.qc_warnings:
            lines.append(f"  ! {warning}")
    else:
        lines.append("QC WARNINGS: none.")

    lines.append("")
    lines.append(sep)
    lines.append("END OF LOG")
    lines.append(sep)

    (output_dir / "pipeline_log.txt").write_text("\n".join(lines) + "\n")


def _process_polarity(
    run: PolarityRun,
    sample_metadata: pd.DataFrame,
    ref_batch: str,
    thresholds: Thresholds,
    modifications: pd.DataFrame,
    output_dir: Path,
    artifacts: PipelineArtifacts,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    diag_dir = output_dir / "diagnostics" / run.polarity.lower()
    _ensure_dir(diag_dir)

    artifacts.drift_flags.extend(
        _plot_istd_diagnostics(run, diag_dir / "pre_norm", f"{run.polarity}_pre")
    )

    # ── Step 33 tracking: initial counts before any filtering ────────────────
    pol = run.polarity
    bio_mask = run.sample_info["sample_type"] == "biological"
    artifacts.initial_sample_counts[pol] = (
        run.sample_info.loc[bio_mask]
        .groupby("batch")
        .size()
        .to_dict()
    )

    def _snap(label: str, feat_df: pd.DataFrame, samp_df: pd.DataFrame) -> None:
        """Append one step-count snapshot to artifacts."""
        artifacts.step_counts.append(
            {
                "polarity": pol,
                "label": label,
                "n_features": feat_df.shape[1],
                "n_samples": int((samp_df["sample_type"] == "biological").sum()),
            }
        )

    _snap("entry (post-ISTD removal)", run.expression, run.sample_info)
    # ─────────────────────────────────────────────────────────────────────────

    normalized, removed_istds = _istd_normalize(run, artifacts)
    artifacts.istd_names[pol] = removed_istds  # Step 33: record ISTD names used

    normalized = _median_fold_change_batch_normalize(
        normalized,
        run.sample_info,
        ref_batch,
        artifacts,
    )
    artifacts.drift_flags.extend(
        _plot_post_normalization_diagnostics(
            run, normalized, diag_dir / "post_norm", f"{run.polarity}_post"
        )
    )

    filtered, missingness_report = _feature_missingness_filter(
        normalized,
        run.sample_info,
        sample_metadata,
        thresholds,
        artifacts,
        run.polarity,
    )
    _snap("after Step 7 (feature missingness)", filtered, run.sample_info)

    filtered = _rsd_filter(filtered, run.sample_info, thresholds, artifacts, run.polarity)
    _snap("after Step 8 (QC RSD filter)", filtered, run.sample_info)

    filtered, sample_info = _sample_missingness_filter(
        filtered, run.sample_info, thresholds, artifacts, run.polarity
    )
    _snap("after Step 9 (sample missingness)", filtered, sample_info)

    bad_samples = _sample_istd_mad_filter(run, sample_info, thresholds, artifacts)
    if bad_samples:
        keep_mask = ~sample_info.index.isin(bad_samples)
        filtered = filtered.loc[keep_mask].copy()
        sample_info = sample_info.loc[keep_mask].copy()
    _snap("after Step 10 (sample ISTD MAD)", filtered, sample_info)

    log2_df = np.log2(filtered + 1.0)
    imputed = _half_minimum_impute(log2_df)

    batch_labels = sample_info["batch"]
    _pca_plot(
        imputed,
        batch_labels,
        f"{run.polarity} pre-correction PCA (batch)",
        diag_dir / "pca_pre_correction_batch.png",
    )
    groups = sample_metadata.reindex(sample_info["sample_name"])["Group"].fillna("Unknown")
    groups.index = sample_info.index
    confounding_rows = _batch_confounding_checks(sample_info, sample_metadata, artifacts)
    combat_covariates, protected_covariates = _combat_design_matrix(sample_info, sample_metadata)
    bridge_count = int(sample_info.loc[sample_info["is_bridge"], "sample_name"].nunique())
    artifacts.bridge_counts[run.polarity] = bridge_count

    corrected, method_name = _apply_batch_correction(
        imputed,
        batch_labels,
        ref_batch,
        combat_covariates,
        protected_covariates,
        bridge_count,
        artifacts,
    )
    artifacts.method_log.append(
        f"{run.polarity}: batch correction method={method_name}, reference batch={ref_batch}."
    )

    _pca_plot(
        corrected,
        batch_labels,
        f"{run.polarity} post-correction PCA (batch)",
        diag_dir / "pca_post_correction_batch.png",
    )
    _pca_plot(
        corrected,
        groups.reset_index(drop=True) if isinstance(groups, pd.Series) else groups,
        f"{run.polarity} post-correction PCA (group)",
        diag_dir / "pca_post_correction_group.png",
    )

    named_kept, unnamed = _resolve_named_groups(
        run, corrected, thresholds, modifications, artifacts
    )
    unnamed_kept = _resolve_unnamed_groups(
        run, unnamed, corrected, thresholds, modifications, artifacts
    )
    retained_meta = pd.concat([named_kept, unnamed_kept], axis=0)
    retained_meta = retained_meta[~retained_meta.index.duplicated(keep="first")]
    retained_expression = corrected[retained_meta.index.tolist()].copy()
    run.retained_feature_meta = retained_meta
    run.retained_expression = retained_expression
    return retained_expression, retained_meta, sample_info, missingness_report, confounding_rows


def run_dataset(
    config: DatasetConfig,
    meta_path: Path,
    thresholds: Thresholds,
) -> None:
    artifacts = PipelineArtifacts()
    _ensure_dir(config.output_dir)
    sample_metadata = _load_metadata(meta_path, config)
    modifications = _default_modifications()
    ref_batch = ""

    pos_run = _load_polarity_run(config.input_dir, "pos", config.positive_istd_names, config)
    neg_run = _load_polarity_run(config.input_dir, "neg", config.negative_istd_names, config)
    artifacts.method_log.append(
        f"POS injection-order source: {pos_run.injection_order_source}."
    )
    artifacts.method_log.append(
        f"NEG injection-order source: {neg_run.injection_order_source}."
    )
    _write_metadata_audit(
        pos_run,
        neg_run,
        sample_metadata,
        config.output_dir,
        config,
        artifacts,
    )

    candidate_batches = sorted(
        set(pos_run.sample_info["batch"].unique()) | set(neg_run.sample_info["batch"].unique())
    )
    if not candidate_batches:
        raise RuntimeError(f"{config.dataset_id}: no batches detected.")
    ref_batch = _pick_reference_batch(candidate_batches)

    pos_expression, pos_meta, pos_samples, pos_missing, pos_conf = _process_polarity(
        pos_run,
        sample_metadata,
        ref_batch,
        thresholds,
        modifications,
        config.output_dir,
        artifacts,
    )
    neg_expression, neg_meta, neg_samples, neg_missing, neg_conf = _process_polarity(
        neg_run,
        sample_metadata,
        ref_batch,
        thresholds,
        modifications,
        config.output_dir,
        artifacts,
    )

    final_matrix, feature_meta = _merge_polarities(
        [
            (pos_run, pos_expression, pos_samples),
            (neg_run, neg_expression, neg_samples),
        ],
        sample_metadata,
        config,
        thresholds,
        artifacts,
    )

    full_path = config.output_dir / f"{config.dataset_id}_cleaned_with_metadata.csv"
    final_matrix.to_csv(full_path)

    feature_path = config.output_dir / f"{config.dataset_id}_feature_metadata.csv"
    feature_meta.to_csv(feature_path)

    if artifacts.sample_filter_log:
        pd.DataFrame(artifacts.sample_filter_log).to_csv(
            config.output_dir / f"{config.dataset_id}_sample_filters.csv", index=False
        )
    if artifacts.feature_filter_log:
        pd.DataFrame(artifacts.feature_filter_log).to_csv(
            config.output_dir / f"{config.dataset_id}_feature_filters.csv", index=False
        )
    if artifacts.dedup_log:
        pd.DataFrame(artifacts.dedup_log).to_csv(
            config.output_dir / f"{config.dataset_id}_dedup_log.csv", index=False
        )

    if config.tissue == "plasma":
        for tp in sorted(VALID_TIMEPOINTS):
            tp_mask = final_matrix.index.str.endswith(tp)
            if not tp_mask.any():
                continue
            tp_df = final_matrix.loc[tp_mask].copy()
            tp_df.index = tp_df.index.str[:-1]
            tp_df.index.name = "SampleID"
            tp_df.to_csv(config.output_dir / f"{config.dataset_id}_suffix_{tp}.csv")

    _trajectory_plots(final_matrix, feature_meta, config.output_dir, config, artifacts)
    _write_plain_text_log(
        config.output_dir,
        config,
        artifacts,
        final_matrix,
        feature_meta,
        {
            "POS": pos_missing,
            "NEG": neg_missing,
        },
        pos_conf + neg_conf,
    )

    LOGGER.info(
        "[%s] wrote %s and %s",
        config.dataset_id,
        full_path,
        feature_path,
    )


def _build_configs(repo_root: Path, kayla_root: Path, output_root: Path) -> dict[str, DatasetConfig]:
    return {
        "MTBL_plasma": DatasetConfig(
            dataset_id="MTBL_plasma",
            database="HMDB",
            tissue="plasma",
            meta_sheet="n=133 metabolomics",
            meta_sample_col="Sample ID",
            input_dir=kayla_root / "data" / "MTBL_plasma",
            output_dir=output_root / "MTBL_plasma",
            positive_istd_names=("D3-Alanine-ISTD", "D3-Creatinine-ISTD"),
            negative_istd_names=("D4-Taurine-ISTD", "D3-Lactate-ISTD"),
            raw_workbook=repo_root / "data" / "metabolomics_raw" / "050725_Sadovsky DP3 Plasma Polar Untargeted_ALL copy.xlsx",
            raw_sheet_pos="POS Compounds",
            raw_sheet_neg="NEG Compounds",
            raw_sample_row=2,
            raw_file_row=3,
        ),
        "MTBL_placenta": DatasetConfig(
            dataset_id="MTBL_placenta",
            database="HMDB",
            tissue="placenta",
            meta_sheet="n=133 placenta",
            meta_sample_col="ID",
            input_dir=kayla_root / "data" / "MTBL_placenta",
            output_dir=output_root / "MTBL_placenta",
            positive_istd_names=("D3-Alanine-ISTD", "D3-Creatinine-ISTD"),
            negative_istd_names=("D4-Taurine-ISTD", "D3-Lactate-ISTD"),
        ),
        "LIPD_plasma": DatasetConfig(
            dataset_id="LIPD_plasma",
            database="LIPID_MAPS",
            tissue="plasma",
            meta_sheet="n=133 metabolomics",
            meta_sample_col="Sample ID",
            input_dir=kayla_root / "data" / "LIPD_plasma",
            output_dir=output_root / "LIPD_plasma",
            positive_istd_names=("18:1_LPC-d7", "18:1_SM-d9"),
            negative_istd_names=("15:0-18:1(d7)-PC", "18:1-18:1(d9)-PE"),
            raw_workbook=repo_root / "data" / "lipids" / "072925 Sadovsky Plasma Lipids Untargeted ALL.xlsx",
            raw_sheet_pos="Plasma POS Lipids",
            raw_sheet_neg="Plasma NEG Lipids",
            raw_sample_row=3,
            raw_file_row=4,
        ),
        "LIPD_placenta": DatasetConfig(
            dataset_id="LIPD_placenta",
            database="LIPID_MAPS",
            tissue="placenta",
            meta_sheet="n=133 placenta",
            meta_sample_col="ID",
            input_dir=kayla_root / "data" / "LIPD_placenta",
            output_dir=output_root / "LIPD_placenta",
            positive_istd_names=("18:1_LPC-d7", "18:1_SM-d9"),
            negative_istd_names=("15:0-18:1(d7)-PC", "18:1-18:1(d9)-PE"),
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="DP3 SOP-native omics preprocessing pipeline.")
    parser.add_argument(
        "--kayla-root",
        default=str(root.parent / "kaylaxu"),
        help="Path to the Kayla Xu raw-export repository root.",
    )
    parser.add_argument(
        "--metadata",
        default=str(root / "data" / "dp3 master table v2.xlsx"),
        help="Path to the master metadata workbook.",
    )
    parser.add_argument(
        "--output-root",
        default=str(root / "data" / "cleaned" / "sop_omics_pipeline"),
        help="Root directory for SOP-native outputs.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["MTBL_plasma", "MTBL_placenta", "LIPD_plasma", "LIPD_placenta"],
        help="Datasets to process.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    repo_root = Path(__file__).resolve().parents[1]
    kayla_root = Path(args.kayla_root).resolve()
    output_root = Path(args.output_root).resolve()
    metadata = Path(args.metadata).resolve()
    configs = _build_configs(repo_root, kayla_root, output_root)
    thresholds = Thresholds()

    for dataset_id in args.datasets:
        if dataset_id not in configs:
            raise SystemExit(f"Unknown dataset '{dataset_id}'.")
        config = configs[dataset_id]
        LOGGER.info("Starting SOP pipeline for %s", dataset_id)
        run_dataset(config, metadata, thresholds)


if __name__ == "__main__":
    main()
