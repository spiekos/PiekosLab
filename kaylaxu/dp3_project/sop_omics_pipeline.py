"""
SOP-native untargeted metabolomics and lipidomics preprocessing pipeline.

This module implements the August 2026 DP3 SOP starting from the raw
Compound Discoverer style exports

Does NOT overwrite the older collaborator-integrated scripts. Provides a
clean replacement entrypoint that follows the SOP ordering:
Part One: Normalization and Drift Correction
1. Standardize Missing Values
2. Identify Sample Types, Batches, and Injection Order
3. Drift Assessment
4. ISTD normalization
5. Median fold-change batch normalization
6. Post-normalization drift check

Part Two: Data Cleaning and Batch Correction
7. Feature missingness filter
8. Sample missingness filter and flag
9. Log2 transformation
10. Half-minimum imputation
11. Pre-correction PCA
12. Verify Batch Confounding with Biological Covariates
13. ComBat Batch Correction
14. Post-correction PCA
15. Post-ComBat intensity check
16. Sample-level QC via Internal Standards
17. RSD Filter on QC Pools
18. IQR filter (within-timepoint)
19. Average Bridge Samples

Part Three: Feature Deduplication
Part 3A: Within-Compound Feature Deduplication
20. Expected Parent Ion Calculation
21. Feature Classification
22. Feature Filtering
23. Multiple Parent Handling
24. Quality Scoring
25. Feature Selection
Part 3B: Cross-Feature Deduplication for Unnanotated Features (Metabolomics)
26. Formula-Based Grouping for Unannotated Features with Formulas
27. Mass-Based Grouping for Unannotated Features without Formulas
28. Adduct-Relationship Collapsing for Unannotated Features

Part Four: Metadata Integration and Annotation
29. Replace Placeholder Feature IDs with Names
30. Merge Ionization Modes
31. Assign Confidence Levels
32. Drop Mass-Only Features
33. Add Metadata Columns

Part Five: Pipeline Validation and Sanity Checks
34. Biological trajectory plots for longitudinal tissues
35. Write Final Pipeline Log 

Notes
-----
- When the original raw workbook is available, injection order is parsed
  from the raw-file headers (`F#` for metabolomics, file sequence for the
  current lipidomics export). Datasets without a raw workbook still fall
  back to within-batch row order, and that fallback is recorded in the log.
"""

from __future__ import annotations
from enum import Enum,auto
from dataclasses import dataclass,field
import argparse
import logging
import math
import os
import re
import subprocess
import tempfile
import textwrap
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable
from flask import config
import openpyxl
from pycombat import Combat
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

VALID_TIMEPOINTS = set("ABCDE")

#========================
# Universal modality constants
#========================
##Master Switch
class Modality(Enum):
    METABOLOMICS=auto()
    LIPIDOMICS=auto()

##Global Parameter Values
#Modality-specific parameters
METABOLOMICS_DATABASE="HMDB"
METABOLOMICS_MODIFICATION_LIST_CSV="raw/original/MTBL/common_metabolite_modification_list.csv"
METABOLOMICS_ANNOTATION_SYSTEM="schymanski"
METABOLOMICS_POSITIVE_ISTD_NAMES=(
	"D3-Alanine-ISTD", "D3-Alanine ISTD",
	"D3-Creatinine-ISTD", "D3-Creatinine ISTD",
)
METABOLOMICS_NEGATIVE_ISTD_NAMES=(
	"D4-Taurine-ISTD", "D4-Taurine ISTD",
	"D3-Lactate-ISTD", "D3-Lactate ISTD",
)

LIPIDOMICS_DATABASE="LIPID_MAPS"
LIPIDOMICS_MODIFICATION_LIST_CSV="raw/original/LIPD/common_lipid_modification_list.csv"
LIPIDOMICS_ANNOTATION_SYSTEM="lsi+schymanski"
LIPIDOMICS_NORMALIZATION_MODE={
	"POS":"class_matched",
	"NEG":"pooled",
}
LIPIDOMICS_CLASS_TO_ISTD_POS={
	"LPC":"18:1_LPC-d7",
	"LPE":"18:1_LPC-d7",
	"LPI":"18:1_LPC-d7",
	"LPG":"18:1_LPC-d7",
	"LPS":"18:1_LPC-d7",
	"LPA":"18:1_LPC-d7",
	"PC":"15:0/18:1_PC-d7",
	"PE":"15:0/18:1_PC-d7",
	"PI":"15:0/18:1_PC-d7",
	"PG":"15:0/18:1_PC-d7",
	"PS":"15:0/18:1_PC-d7",
	"PA":"15:0/18:1_PC-d7",
	"SM":"18:1_SM-d9",
	"Cer":"18:1_SM-d9",
	"HexCer":"18:1_SM-d9",
	"DG":"15:0/18:1_DG-d7",
	"MG":"15:0/18:1_DG-d7",
	"TG":"15:0/18:1/15:0_TG-d7",
	"CE":"18:1_CE-d7",
}
LIPIDOMICS_CLASS_ADDUCT_MAP={
    "POS":{
        "PC":"[M+H]+",
        "LPC":"[M+H]+",
        "PE":"[M+H]+",
        "LPE":"[M+H]+",
        "PI":"[M+H]+",
        "PG":"[M+H]+",
        "PS":"[M+H]+",
        "SM":"[M+H]+",
        "Cer":"[M+H]+",
        "HexCer":"[M+H]+",
        "TG":"[M+NH4]+",
        "DG":"[M+NH4]+",
        "MG":"[M+NH4]+",
        "CE":"[M+NH4]+",
    },
    "NEG":{
        "PC":"[M+CH3COO]-",
        "LPC":"[M+CH3COO]-",
        "SM":"[M+CH3COO]-",
        "PE":"[M-H]-",
        "LPE":"[M-H]-",
        "PI":"[M-H]-",
        "LPI":"[M-H]-",
        "PG":"[M-H]-",
        "LPG":"[M-H]-",
        "PS":"[M-H]-",
        "LPS":"[M-H]-",
        "PA":"[M-H]-",
        "LPA":"[M-H]-",
        "FA":"[M-H]-",
        "Cer":"[M-H]-",
        "HexCer":"[M-H]-",
    },
}
LIPIDOMICS_POSITIVE_ISTD_NAMES=(
	"18:1_LPC-d7",
	"15:0/18:1_PC-d7",
	"18:1_SM-d9",
	"15:0/18:1_DG-d7",
	"15:0/18:1/15:0_TG-d7",
	"18:1_CE-d7",
)
LIPIDOMICS_NEGATIVE_ISTD_NAMES=(
    "18:1_LPE-d7",
    "15:0-18:1_PG-d7",
)

DATABASE_BY_MODALITY={
	Modality.METABOLOMICS:METABOLOMICS_DATABASE,
	Modality.LIPIDOMICS:LIPIDOMICS_DATABASE,
}

MODIFICATION_LIST_CSV_BY_MODALITY={
	Modality.METABOLOMICS:METABOLOMICS_MODIFICATION_LIST_CSV,
	Modality.LIPIDOMICS:LIPIDOMICS_MODIFICATION_LIST_CSV,
}

ANNOTATION_SYSTEM_BY_MODALITY={
	Modality.METABOLOMICS:METABOLOMICS_ANNOTATION_SYSTEM,
	Modality.LIPIDOMICS:LIPIDOMICS_ANNOTATION_SYSTEM,
}

POSITIVE_ISTD_NAMES_BY_MODALITY={
	Modality.METABOLOMICS:METABOLOMICS_POSITIVE_ISTD_NAMES,
	Modality.LIPIDOMICS:LIPIDOMICS_POSITIVE_ISTD_NAMES,
}

NEGATIVE_ISTD_NAMES_BY_MODALITY={
	Modality.METABOLOMICS:METABOLOMICS_NEGATIVE_ISTD_NAMES,
	Modality.LIPIDOMICS:LIPIDOMICS_NEGATIVE_ISTD_NAMES,
}

DEFAULT_NORMALIZATION_MODE_BY_MODALITY={
	Modality.LIPIDOMICS:LIPIDOMICS_NORMALIZATION_MODE,
}

CLASS_TO_ISTD_POS_BY_MODALITY={
	Modality.LIPIDOMICS:LIPIDOMICS_CLASS_TO_ISTD_POS,
}

CLASS_ADDUCT_MAP_BY_MODALITY={
	Modality.LIPIDOMICS:LIPIDOMICS_CLASS_ADDUCT_MAP,
}

def _require_supported_modality(modality:Modality)->None:
	if modality not in DATABASE_BY_MODALITY:
		raise ValueError(f"Unsupported modality: {modality}")


def get_istd_names(modality:Modality,polarity:str)->tuple[str,...]:
	_require_supported_modality(modality)
	polarity=polarity.upper()

	if polarity=="POS":
		if modality not in POSITIVE_ISTD_NAMES_BY_MODALITY:
			raise ValueError(f"No POS ISTD names configured for modality: {modality}")
		return POSITIVE_ISTD_NAMES_BY_MODALITY[modality]

	if polarity=="NEG":
		if modality not in NEGATIVE_ISTD_NAMES_BY_MODALITY:
			raise ValueError(f"No NEG ISTD names configured for modality: {modality}")
		return NEGATIVE_ISTD_NAMES_BY_MODALITY[modality]

	raise ValueError(f"Unsupported polarity: {polarity}")

#Non-modality-specific parameters
PROTON_MASS = 1.007276
AMMONIUM_MASS=18.033823
SODIUM_MASS=22.989218
ACETATE_MASS=59.013304
FORMATE_MASS=44.997654
MASS_TOLERANCE_PARENT=0.02
MASS_TOLERANCE_NON_PARENT=0.10
MASS_TOLERANCE_DEDUP_NO_FORMULA=0.02
RT_TOLERANCE=0.2
PPM_TOLERANCE_ANNOTATION=5.0
SAMPLE_MAD_THRESHOLD=5.0
SAMPLE_MISSING_THRESHOLD=0.50
FEATURE_MISSING_THRESHOLD=0.20
RSD_THRESHOLD=30.0
IQR_LOW_PERCENTILE=5.0
IQR_HIGH_PERCENTILE=95.0
IQR_FLOOR=0.1
IQR_CEILING=5.0
MZCLOUD_L2_THRESHOLD=80.0
ANNOT_SOURCE_L2_THRESHOLD=3
RT_ARTIFACT_MAX=0.5
RT_ISOMER_MAX=3.0

@dataclass(frozen=True)
class DatasetConfig:
    dataset_id:str
    modality:Modality
    tissue:str
    meta_sheet:str
    meta_sample_col:str
    input_dir:Path
    output_dir:Path
    pca_dir: Path
    trajectory_dir: Path
    positive_istd_names:tuple[str,...]=()
    negative_istd_names:tuple[str,...]=()
    bridge_expected:bool=True
    raw_workbook:Path|None=None
    raw_sheet_pos:str|None=None
    raw_sheet_neg:str|None=None
    raw_sample_row:int|None=None
    raw_file_row:int|None=None
    database:str=""
    modification_list_csv:str=""
    annotation_system:str=""
    normalization_mode:str|dict[str,str]="pooled"
    class_to_istd_pos:dict[str,str]=field(default_factory=dict)
    class_adduct_map:dict[str,str]=field(default_factory=dict)

    def __post_init__(self):
        _require_supported_modality(self.modality)

        if not self.database:
            object.__setattr__(
                self,
                "database",
                DATABASE_BY_MODALITY[self.modality],
            )

        if not self.modification_list_csv:
            object.__setattr__(
                self,
                "modification_list_csv",
                MODIFICATION_LIST_CSV_BY_MODALITY[self.modality],
            )

        if not self.annotation_system:
            object.__setattr__(
                self,
                "annotation_system",
                ANNOTATION_SYSTEM_BY_MODALITY[self.modality],
            )

        if self.normalization_mode=="pooled" and self.modality in DEFAULT_NORMALIZATION_MODE_BY_MODALITY:
            object.__setattr__(
                self,
                "normalization_mode",
                DEFAULT_NORMALIZATION_MODE_BY_MODALITY[self.modality],
            )

        if not self.class_to_istd_pos and self.modality in CLASS_TO_ISTD_POS_BY_MODALITY:
            object.__setattr__(
                self,
                "class_to_istd_pos",
                CLASS_TO_ISTD_POS_BY_MODALITY[self.modality],
            )

        if not self.class_adduct_map and self.modality in CLASS_ADDUCT_MAP_BY_MODALITY:
            object.__setattr__(
                self,
                "class_adduct_map",
                CLASS_ADDUCT_MAP_BY_MODALITY[self.modality],
            )

#Modifications List
#Note: no longer hard-coded in (since files are now available in repo)
def _load_modifications(config:DatasetConfig)->pd.DataFrame:
	path=Path(config.modification_list_csv)
	if not path.is_absolute():
		path=Path(__file__).resolve().parent/"data"/path
	if not path.exists():
		raise FileNotFoundError(f"Modification list not found: {path}")

	df=pd.read_csv(path)
	required={"Type","Name","Delta_m/z","Description"}
	missing=required-set(df.columns)
	if missing:
		raise ValueError(
			f"{path} is missing required columns: {sorted(missing)}"
		)

	return df[["Type","Name","Delta_m/z","Description"]].copy()

##Container of data objects for one polarity run
@dataclass
class PolarityRun:
    polarity: str
    modality: Modality
    expression: pd.DataFrame
    sample_info: pd.DataFrame
    feature_meta: pd.DataFrame
    raw_istd: pd.DataFrame
    class_to_istd_pos:dict[str,str]=field(default_factory=dict)
    injection_order_source: str = "row_order_proxy"
    retained_feature_meta: pd.DataFrame | None = None
    retained_expression: pd.DataFrame | None = None
    dropped_features: list[dict] = field(default_factory=list)

##Log and artifact tracking for the entire pipeline run
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
    # Step 35: structured per-step tracking
    # Each entry: {polarity, step, label, n_features, n_samples}
    step_counts: list[dict] = field(default_factory=list)
    step_execution: list[dict] = field(default_factory=list)
    # polarity -> list of ISTD feature names that were removed during normalization
    istd_names: dict[str, list[str]] = field(default_factory=dict)
    # polarity -> batch -> count of biological samples at pipeline entry
    initial_sample_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    # Comprehensive per-entity drop log (written to CSV; never printed to terminal).
    # Each row captures one dropped feature or sample with full context.
    drop_log: list[dict] = field(default_factory=list)

##Log for dropped events
def _record_drop(
    artifacts: PipelineArtifacts,
    entity_type: str,
    entity_id: str,
    step: int,
    step_name: str,
    polarity: str,
    reason: str,
    *,
    annotation_name: str = "",
    formula: str = "",
    mz: float | None = None,
    rt_min: float | None = None,
    metric_value: float | None = None,
    metric_threshold: float | None = None,
    batch: str = "",
    timepoint: str = "",
    dedup_phase: str = "",
    representative_feature: str = "",
) -> None:
    """Append one drop event to artifacts.drop_log.  Never prints or logs."""
    artifacts.drop_log.append(
        {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "step": step,
            "step_name": step_name,
            "polarity": polarity,
            "reason": reason,
            "annotation_name": annotation_name,
            "formula": formula,
            "mz": mz,
            "rt_min": rt_min,
            "metric_value": metric_value,
            "metric_threshold": metric_threshold,
            "batch": batch,
            "timepoint": timepoint,
            "dedup_phase": dedup_phase,
            "representative_feature": representative_feature,
        }
    )
#Building parser (arg to read, modify, process files)
def _build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="DP3 SOP-native omics preprocessing pipeline.")
    parser.add_argument(
        "--inputs-root",
        type=Path,
        default=root / "data" / "raw" / "extracted",
        help="Root directory containing extracted input CSV files.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=root / "data" / "raw" / "original" / "dp3 master table v2.xlsx",
        help="Path to the master metadata workbook.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "data" / "processed",
        help="Root directory for SOP-native outputs.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["MTBL_plasma", "MTBL_placenta", "MTBL_urine", "LIPD_plasma", "LIPD_placenta"],
        help="Datasets to process.",
    )
    return parser

def _build_configs(repo_root: Path, inputs_root: Path, output_root: Path) -> dict[str, DatasetConfig]:
    results_root = repo_root / "04_results_and_figures"
    return {
        "MTBL_plasma": DatasetConfig(
            dataset_id="MTBL_plasma",
            modality=Modality.METABOLOMICS,
            tissue="plasma",
            meta_sheet="n=133 metabolomics",
            meta_sample_col="Sample ID",
            input_dir=inputs_root / "MTBL" / "plasma",
            output_dir=output_root / "MTBL" / "plasma",
            pca_dir=(results_root / "pre_post_combat_pca"/ "MTBL" / "plasma"),
            trajectory_dir=(results_root/ "trajectory_plots"/ "MTBL"/ "plasma"),
            raw_workbook=repo_root / "data" / "raw"/ "original"/ "MTBL" / "plasma" / "050725_Sadovsky DP3 Plasma Polar Untargeted_ALL copy.xlsx",
            raw_sheet_pos="POS Compounds",
            raw_sheet_neg="NEG Compounds",
            raw_sample_row=2,
            raw_file_row=3,
        ),
        "MTBL_placenta": DatasetConfig(
            dataset_id="MTBL_placenta",
            modality=Modality.METABOLOMICS,
            tissue="placenta",
            meta_sheet="n=133 placenta",
            meta_sample_col="ID",
            input_dir=inputs_root / "MTBL" / "placenta",
            output_dir=output_root / "MTBL" / "placenta",
            pca_dir=(results_root/ "pre_post_combat_pca"/ "MTBL"/ "placenta"),
            trajectory_dir=(results_root/ "trajectory_plots"/ "MTBL"/ "placenta"), 
            raw_workbook=repo_root / "data" / "raw"/ "original"/ "MTBL" / "placenta" / "050725_Sadovsky DP3 Placenta Polar Untargeted_ALL copy.xlsx",
            raw_sheet_pos="POS Compounds",
            raw_sheet_neg="NEG Compounds",
            raw_sample_row=2,
            raw_file_row=3,
        ),
        "LIPD_plasma": DatasetConfig(
            dataset_id="LIPD_plasma",
            modality=Modality.LIPIDOMICS,
            tissue="plasma",
            meta_sheet="n=133 metabolomics",
            meta_sample_col="Sample ID",
            input_dir=inputs_root / "LIPD" / "plasma",
            output_dir=output_root / "LIPD" / "plasma",
            pca_dir=(results_root/ "pre_post_combat_pca"/ "LIPD"/ "plasma"),
            trajectory_dir=(results_root/ "trajectory_plots"/ "LIPD"/ "plasma"),
            raw_workbook=repo_root / "data" / "raw"/ "original"/ "LIPD" / "plasma" / "072925 Sadovsky Plasma Lipids Untargeted ALL.xlsx",
            raw_sheet_pos="Plasma POS Lipids",
            raw_sheet_neg="Plasma NEG Lipids",
            raw_sample_row=3,
            raw_file_row=4,
        ),
        "LIPD_placenta": DatasetConfig(
            dataset_id="LIPD_placenta",
            modality=Modality.LIPIDOMICS,
            tissue="placenta",
            meta_sheet="n=133 placenta",
            meta_sample_col="ID",
            input_dir=inputs_root / "LIPD" / "placenta",
            output_dir=output_root / "LIPD" / "placenta",
            pca_dir=(results_root/ "pre_post_combat_pca"/ "LIPD"/ "placenta"),
            trajectory_dir=(results_root/ "trajectory_plots"/ "LIPD"/ "placenta"),
            raw_workbook=repo_root / "data" / "raw"/ "original"/ "LIPD" / "placenta" / "072925 Sadovsky Placenta Lipids Untargeted ALL.xlsx",
            raw_sheet_pos="Placenta POS Lipids",
            raw_sheet_neg="Placenta NEG Lipids",
            raw_sample_row=3,
            raw_file_row=4,
        ),
        "MTBL_urine": DatasetConfig(
            dataset_id="MTBL_urine",
            modality=Modality.METABOLOMICS,
            tissue="urine",
            meta_sheet="n=133 metabolomics",
            meta_sample_col="Sample ID",
            input_dir=inputs_root / "MTBL" / "urine",
            output_dir=output_root / "MTBL" / "urine",
            pca_dir=results_root / "pre_post_combat_pca" / "MTBL" / "urine",
            trajectory_dir=results_root / "trajectory_plots" / "MTBL" / "urine",
            raw_workbook=(repo_root / "data" / "raw" / "original" / "MTBL" / "urine" / "112525 Sadovsky Urine ALL_Compounds copy.xlsx"),
            # Confirm these sheet names in the workbook:
            raw_sheet_pos="POS Compounds",
            raw_sheet_neg="NEG Compounds",
            # Confirm these header rows in the workbook:
            raw_sample_row=2,
            raw_file_row=3,
        ),
    }

def _load_metadata(meta_path: Path, config: DatasetConfig) -> pd.DataFrame:
    cols = [
        config.meta_sample_col,
        "group",
        "subgroup",
        "gest age del",
        "omics set#",
    ]
    if config.tissue in {"plasma", "urine"}:
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
    if config.tissue == "placenta":
        # Placenta is a single cross-sectional specimen collected at delivery.
        meta["SampleGestAge"] = meta["GestAgeDelivery"]
        meta["SampleTimepoint"] = "Delivery"
    else:
        meta["SampleTimepoint"] = meta.apply(
            lambda row: categorize_sample_time_with_delivery(
                row["SampleGestAge"],
                row["GestAgeDelivery"],
            ),
            axis=1,
        )

    #Pull information from the clinical dataset
    clinical=pd.read_excel(
        meta_path,
        sheet_name="clinical data",
        header=1,
        usecols=["ID","LABOR_ONSET","INDICATED_ONSET (1=INDICATED; 0=NOT INDICATED)"],
    )
    clinical=clinical.rename(
        columns={
            "LABOR_ONSET":"labor_onset",
            "INDICATED_ONSET (1=INDICATED; 0=NOT INDICATED)":"indicated_onset",
        }
    )
    clinical["SampleID"]=clinical["ID"].map(_canonical_sample_name)
    clinical=clinical.set_index("SampleID")
    clinical=clinical.reindex(meta.index)
    meta=meta.join(clinical[["labor_onset","indicated_onset"]])

    #Labor onset/indicated onset flags
    meta["spont_labor_flag"]=(meta["labor_onset"]=="SPONTANEOUS").astype("Int64")
    meta["indicated_onset_flag"]=(meta["indicated_onset"]==1).astype("Int64")
        #if spontaneous or NOT indicated 
    meta["cat_labor_onset_flag"]=(
        (meta["labor_onset"]=="SPONTANEOUS") | (meta["indicated_onset_flag"]==0)).astype("Int64")
    # Indicator if sample was taken within 0.1 weeks of spontaneous/non-indicated delivery
    if "SampleGestAge" in meta.columns:
        diff=(meta["GestAgeDelivery"]-meta["SampleGestAge"]).abs()
        within_01=(diff<=0.1)
        has_onset=(meta["labor_onset"].notna()) | (meta["indicated_onset"].notna())
        meta["within_0_1wk_delivery_flag"]=(within_01 & has_onset).astype("Int64")
        meta["post_birth_sample_flag"]=(meta["SampleGestAge"]>meta["GestAgeDelivery"]).astype("Int64")
    else:
        meta["within_0_1wk_delivery_flag"]=pd.NA
        meta["post_birth_sample_flag"]=pd.NA
    
    #Indicator if sample is taken after the birth period
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

"""
Part 1: Normalization and Drift Correction
"""

### Step 1: Standardize Missing Values
def _safe_float(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if out <= 0:
        return float("nan")
    return out

### STEP 2: Identify Sample Types, Batches, and Injection Order

##Pre-step helpers: sample type, batch, and injection order parsing

def _normalise_name(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").strip()).lower() 

def _canonical_sample_name(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").strip())

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

##(a) Batch identity from date prefix
def _canonical_batch_label(value: str) -> str:
    digits = re.sub(r"\D", "", str(value or "").strip())
    if digits:
        try:
            return str(int(digits))
        except ValueError:
            return digits.lstrip("0") or "0"
    return str(value or "").strip()

def _read_xlsx_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    """Read a full sheet from an XLSX workbook into a DataFrame using openpyxl."""
    return pd.read_excel(path, sheet_name=sheet_name, header=0, engine="openpyxl")

def _read_xlsx_selected_rows(
    path: Path,
    sheet_name: str,
    target_rows: set[int],
) -> dict[int, dict[int, str]]:
    """
    Return only the requested XLSX rows as {row_number: {col_idx: value}}.

    Uses openpyxl in read-only mode so only rows up to max(target_rows) are
    streamed — keeps memory usage low for large untargeted-export workbooks.
    col_idx is 0-based (column A = 0) to match the rest of the pipeline.
    """
    if not target_rows:
        return {}
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        ws = wb[sheet_name]
        result: dict[int, dict[int, str]] = {}
        max_target = max(target_rows)
        for row_num, row in enumerate(ws.iter_rows(values_only=False), start=1):
            if row_num in target_rows:
                result[row_num] = {
                    cell.column - 1: str(cell.value) if cell.value is not None else ""
                    for cell in row
                    if cell.value is not None
                }
            if row_num >= max_target and target_rows.issubset(result.keys()):
                break
    finally:
        wb.close()
    return result

def _extract_batch_label_from_text(*values: str) -> str:
    for value in values:
        text = str(value or "")
        match = re.search(r"(?<!\d)(\d{5,6})(?!\d)", text)
        if match:
            return _canonical_batch_label(match.group(1))
    return ""

#Date parsing for batch labels
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


##(b) Identify Sample Type (QC vs. biological vs. bridge)
#ID Sample Type
def _infer_sample_type(sample_label:str)->str:
    name=_normalise_name(sample_label)
    if any(x in name for x in ("pooled","pool","negpool","pospool","qc")):
        return "qc_pool"
    return "biological"

#Builds the raw aquisition map 
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
        sample_type = _infer_sample_type(sample_label)
        order = _extract_injection_order_value(config.modality,raw_file,sample_label)
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

#Distinguishes biological vs. QC samples vs. bridge samples, identifies injection order
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
    sample_info["sample_type"] = sample_info["sample_name"].map(_infer_sample_type)
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
        parsed_order=[
            _extract_injection_order_value(
                config.modality,
                raw_sample_labels.iloc[i],
                batch_index_labels[i],
                raw_file_values[i],
            )
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

    wanted = {_normalise_name(x) for x in istd_names}
    raw_istd_ids = [
        fid
        for fid in comp.index
        if (
            "annotation_name" in comp.columns
            and _normalise_name(str(comp.at[fid, "annotation_name"])) in wanted
        )
        or _normalise_name(fid) in wanted
    ]
    raw_istd = exp[raw_istd_ids].copy()
    return PolarityRun(
        polarity=polarity.upper(),
        modality = config.modality,
        expression=exp,
        sample_info=sample_info,
        feature_meta=comp,
        raw_istd=raw_istd,
        class_to_istd_pos=dict(config.class_to_istd_pos),
        injection_order_source=order_source,
    )

##(c) Injection order from the compound discoverer file index
def _extract_injection_order_value(modality:Modality,*values: str) -> float:
    #Patterns preferred for metabolomics: CD F-number notation and file index
    metab_patterns=[
        r"\(F\s*([0-9]+)\)",
        r"\bF\s*([0-9]+)\b",
        r"file[_\s-]*index[_\s-]*([0-9]+)",
    ]
    #Patterns preferred for lipidomics: batch_Neg/Pos_injectionNumber, etc.
    lipid_patterns=[
        r"\d{5,8}_(?:set\d+_)?(?:neg|pos)_([0-9]+)(?:\.raw)?$",
        r"(?:raw|pool|set[0-9]+|_pos_|_neg_|\bpos\b|\bneg\b).*?([0-9]+)(?:\.raw)?$",
        r"file[_\s-]*index[_\s-]*([0-9]+)",
    ]
    #Generic fallback if modality is missing or patterns fail
    fallback_patterns=[
        r"\(F\s*([0-9]+)\)",
        r"\bF\s*([0-9]+)\b",
        r"file[_\s-]*index[_\s-]*([0-9]+)",
        r"\d{5,8}_(?:set\d+_)?(?:neg|pos)_([0-9]+)(?:\.raw)?$",
        r"(?:raw|pool|set[0-9]+|_pos_|_neg_|\bpos\b|\bneg\b).*?([0-9]+)(?:\.raw)?$",
    ]

    if modality is Modality.METABOLOMICS:
        patterns=metab_patterns+fallback_patterns
    elif modality is Modality.LIPIDOMICS:
        patterns=lipid_patterns+fallback_patterns
    else:
        patterns=fallback_patterns

    for value in values:
        text=str(value or "")
        for pattern in patterns:
            match=re.search(pattern,text,flags=re.IGNORECASE)
            if match:
                return float(match.group(1))
    return float("nan")

#Identify reference batch for normalization by earliest acquisition date
def _pick_reference_batch(candidate_batches: Iterable[str]) -> str:
    batches = [str(batch) for batch in candidate_batches]
    if not batches:
        raise ValueError("No candidate batches were provided.")
    try:
        return min(batches, key=_parse_batch_date)
    except Exception:
        return sorted(batches)[0]

### Step 3: Drift Assessment

## Pre-Step 3 helpers:

# Directory - Output folder for CSV/plots
def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

# Total ion current (TIC) calculation for drift plots
def _tic(df: pd.DataFrame) -> pd.Series:
    return df.sum(axis=1, skipna=True)

## (a - c): Plot raw ISTD signals, scaled ISTD concordance, and TIC by injection order
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

    if run.modality is Modality.METABOLOMICS and run.polarity.upper() == "POS":
        wanted = {_normalise_name(n) for n in METABOLOMICS_POSITIVE_ISTD_NAMES}
    elif run.modality is Modality.METABOLOMICS and run.polarity.upper() == "NEG":
        wanted = {_normalise_name(n) for n in METABOLOMICS_NEGATIVE_ISTD_NAMES}
    elif run.modality is Modality.LIPIDOMICS and run.polarity.upper() == "POS":
        wanted = {_normalise_name(n) for n in LIPIDOMICS_POSITIVE_ISTD_NAMES}
    elif run.modality is Modality.LIPIDOMICS and run.polarity.upper() == "NEG":
        wanted = {_normalise_name(n) for n in LIPIDOMICS_NEGATIVE_ISTD_NAMES}
    else:
        wanted = set()

    if "annotation_name" in run.feature_meta.columns:
        istd_columns = [
            c for c in run.raw_istd.columns
            if (
                not wanted
                or _normalise_name(str(run.feature_meta.at[c, "annotation_name"])) in wanted
                or _normalise_name(c) in wanted
            )
        ]
    else:
        istd_columns = [
            c for c in run.raw_istd.columns
            if not wanted or _normalise_name(c) in wanted
        ]

    for batch, batch_rows in biological_or_qc.groupby("batch"):
        batch_rows = batch_rows.sort_values("injection_order")
        row_ids = batch_rows.index.tolist()
        if not row_ids:
            continue
        fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

        # (a) Raw ISTD signal vs injection order
        for column in istd_columns:
            axes[0].plot(
                batch_rows["injection_order"],
                run.raw_istd.loc[row_ids, column].values,
                marker="o",
                label=column,
            )
        axes[0].set_title(f"{prefix} batch {batch}: raw ISTD signal")
        axes[0].set_ylabel("raw intensity")
        if istd_columns:
            axes[0].legend(fontsize=8)

        # (b) ISTD concordance
        for column in istd_columns:
            values = pd.to_numeric(run.raw_istd.loc[row_ids, column], errors="coerce")
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
        if istd_columns:
            axes[1].legend(fontsize=8)

        # (c) TIC vs injection order
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

        if len(istd_columns) >= 2:
            # ISTD concordance flag (min pairwise r across ISTDs)
            corr = run.raw_istd.loc[row_ids, istd_columns].corr(min_periods=3).values
            if corr.size and np.isfinite(corr).any():
                finite = corr[np.isfinite(corr)]
                if finite.size and np.nanmin(finite) < 0.50:
                    flags.append(
                        f"{prefix} batch {batch}: ISTD concordance is weak "
                        f"(minimum pairwise r={np.nanmin(finite):.2f})."
                    )
    return flags

## Pre-Step 4 helpers:

# Computing per-sample ISTD geometric means
def _geometric_mean(values:Iterable[float])->float:
    arr=np.asarray(list(values),dtype=float)
    arr=arr[np.isfinite(arr)&(arr>0)]
    if arr.size==0:
        return float("nan")
    return float(np.exp(np.mean(np.log(arr))))

###Step 4: ISTD Normalization
def _istd_normalize(
    run: PolarityRun,
    artifacts: PipelineArtifacts,
) -> tuple[pd.DataFrame, list[str]]:
    istd_columns=run.raw_istd.columns.tolist()
    feature_ids = run.feature_meta.index.tolist()
    non_istd_features = [c for c in feature_ids if c not in istd_columns]
    normalized = run.expression[feature_ids].copy()
    
    if not istd_columns or run.raw_istd.empty:
        artifacts.qc_warnings.append(
            f"{run.polarity}: no internal standards were detected; ISTD normalization was skipped."
        )
        return normalized[non_istd_features], []

    ## (a): Pooled Normalization (metabolomics all modes, lipidomics NEG mode)
    # (ai): pooled normalization (Metabolomics all modes, Lipidomics NEG mode)
    if (
        run.modality is Modality.METABOLOMICS
        or (run.modality is Modality.LIPIDOMICS and run.polarity.upper()=="NEG")
    ):
        geo=run.raw_istd.apply(_geometric_mean,axis=1)
        finite_geo=geo[np.isfinite(geo)&(geo>0)]
        if finite_geo.empty:
            artifacts.qc_warnings.append(
                f"{run.polarity}: all internal-standard geometric means were missing; "
                "ISTD normalization was skipped."
            )
            return normalized[non_istd_features],list(istd_columns)

        geo_filled=geo.copy()
        batch_geo=geo.groupby(run.sample_info["batch"]).transform(
            lambda s:s[np.isfinite(s)&(s>0)].median()
        )
        geo_filled=geo_filled.where(
            np.isfinite(geo_filled)&(geo_filled>0),
            batch_geo,
        )
        global_geo=float(finite_geo.median())
        geo_filled=geo_filled.fillna(global_geo)
        imputed_rows=int(((~np.isfinite(geo))|(geo<=0)).sum())
        if imputed_rows:
            artifacts.method_log.append(
                f"{run.polarity}: filled missing sample-level ISTD geometric means for "
                f"{imputed_rows} rows using batch/global medians."
            )

        normalized=normalized.divide(geo_filled.replace(0,np.nan),axis=0)
        normalized=normalized.replace([np.inf,-np.inf],np.nan)
        artifacts.method_log.append(
            f"{run.polarity}: ISTD normalization (Step 4(ai)) used geometric mean of {list(istd_columns)}."
        )
    # (aii): Class-matched normalization (Lipidomics POS mode)
    elif run.modality is Modality.LIPIDOMICS and run.polarity.upper()=="POS":
        pooled_geo=run.raw_istd.apply(_geometric_mean,axis=1)
        finite_geo=pooled_geo[np.isfinite(pooled_geo)&(pooled_geo>0)]
        if finite_geo.empty:
            artifacts.qc_warnings.append(
                f"{run.polarity}: all POS internal-standard geometric means were missing; "
                "ISTD normalization was skipped."
            )
            return normalized[non_istd_features],list(istd_columns)

        batch_geo=pooled_geo.groupby(run.sample_info["batch"]).transform(
            lambda s:s[np.isfinite(s)&(s>0)].median()
        )
        pooled_geo_filled=pooled_geo.where(
            np.isfinite(pooled_geo)&(pooled_geo>0),
            batch_geo,
        ).fillna(float(finite_geo.median()))

        class_to_istd_pos=run.class_to_istd_pos or {}

        for feature_id in non_istd_features:
            feature_class=str(run.feature_meta.at[feature_id,"Class"]) if "Class" in run.feature_meta.columns else ""
            istd_name=class_to_istd_pos.get(feature_class,"")

            if istd_name in run.raw_istd.columns:
                factor=run.raw_istd[istd_name].replace(0,np.nan)
                factor=factor.where(np.isfinite(factor)&(factor>0),pooled_geo_filled)
            else:
                factor=pooled_geo_filled
                artifacts.method_log.append(
                    f"{run.polarity}: {feature_id} class {feature_class} had no class-matched ISTD; "
                    "used pooled POS ISTD geometric mean (semi-quantitative)."
                )

            normalized[feature_id]=normalized[feature_id].divide(factor)

        normalized=normalized.replace([np.inf,-np.inf],np.nan)
        artifacts.method_log.append(
            f"{run.polarity}: ISTD normalization (Step 4(aii)) used class-matched POS ISTDs where available."
        )

    else:
        artifacts.qc_warnings.append(
            f"{run.polarity}: unsupported modality/polarity for ISTD normalization; step skipped."
        )
        return normalized[non_istd_features],[]
    
    ## (b): Remove ISTD features from the dataset and log their removal
    removed_istds = list(istd_columns)
    for istd_id in removed_istds:
        _record_drop(
            artifacts, "feature", istd_id,
            step=4, step_name="Step 4(b) — ISTD Removal (normalization reference)",
            polarity=run.polarity,
            reason="Internal standard removed after normalization; not a biological analyte",
        )
    return normalized[non_istd_features], removed_istds    

#Appending suffix for duplicate features
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

#Sample matching: canonical sample name for biological samples, empty string for QC pools
def _sample_match_key(sample_name: str, sample_type: str) -> str:
    return "" if sample_type == "qc_pool" else _canonical_sample_name(sample_name)

#Changes values to floats where possible, non-convertible values become NaN
def _standardize_expression(exp: pd.DataFrame) -> pd.DataFrame:
    return exp.apply(lambda col: col.map(_safe_float))

# Return standardized compound metadata 
# Consistent columns for annotation, formula, m/z, RT
# Handles both lipidomics (LipidID-based) and metabolomics (Name-based) metadata formats
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
        comp["annotation_not_top_hit"] = 0
        comp["annotation_partial_matches"] = 0
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
            not_top_hit = (comp[annot_cols] == "Not the top hit").sum(axis=1)
            partial_matches = (comp[annot_cols] == "Partial match").sum(axis=1)
        else:
            full_matches = pd.Series(0, index=comp.index)
            not_top_hit = pd.Series(0, index=comp.index)
            partial_matches = pd.Series(0, index=comp.index)

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
        comp["annotation_not_top_hit"] = not_top_hit
        comp["annotation_partial_matches"] = partial_matches

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

#Creates CSV report summarizing sample metadata audit results
# presence in POS/NEG runs, batch distribution, and metadata matching status
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

###Step 5: Batch Normalization
def _median_fold_change_batch_normalize(
    df: pd.DataFrame,
    sample_info: pd.DataFrame,
    ref_batch: str,
    artifacts: PipelineArtifacts,
) -> pd.DataFrame:
    ##(a)Select reference batch and compute per-feature medians
    corrected = df.copy()
    qc_mask = sample_info["sample_type"] == "qc_pool"
    ref_mask = qc_mask & (sample_info["batch"] == ref_batch)
    ref_qc = corrected.loc[ref_mask]
    if ref_qc.empty:
        artifacts.qc_warnings.append(
            f"Reference batch {ref_batch} has no QC pools; batch scaling factors default to 1.0."
        )
        return corrected
    ##(b) Compute batch-specific per-feature ratio for each non-reference batch
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
    ##(c) Take median of per-feature ratios to get batch-specific scaling factor for batch k
        finite_ratio = per_feature_ratio.replace([np.inf, -np.inf], np.nan).dropna()
        scaling_factor = float(finite_ratio.median()) if not finite_ratio.empty else float("nan")
        if not np.isfinite(scaling_factor) or scaling_factor == 0:
            scaling_factor = 1.0
    ##(d) Divide all sample values in batch k by the batch-specific scaling factor
        corrected.loc[sample_info["batch"] == batch] = (
            corrected.loc[sample_info["batch"] == batch] / scaling_factor
        )
        artifacts.batch_scaling_factors[batch] = scaling_factor
    return corrected

###Step 6: Post-Normalization Drift Check
# (a) Picks IDs of 3-5 high-abundance features
def _select_feature_traces(
    run:PolarityRun,
    df: pd.DataFrame, 
    n_features: int = 5,
) -> list[str]:
    if run.modality is Modality.METABOLOMICS:
        means = df.mean(axis=0, skipna=True).sort_values(ascending=False)
        return means.head(n_features).index.tolist()
    if run.modality is Modality.LIPIDOMICS:
        if "Class" not in run.feature_meta.columns:
            means = df.mean(axis=0, skipna=True).sort_values(ascending=False)
            return means.head(n_features).index.tolist()
        means = df.mean(axis=0, skipna=True)
        meta = run.feature_meta.reindex(df.columns).copy()
        meta["mean_intensity"] = means.reindex(meta.index)
        meta = meta.sort_values("mean_intensity", ascending=False)

        selected = []
        seen_classes = set()

        for idx,row in meta.iterrows():
            cls=str(row.get("Class","") or "")
            if cls and cls not in seen_classes:
                selected.append(idx)
                seen_classes.add(cls)
                if len(selected)>=n_features:
                    break
        
        if len(selected)<n_features:
            for fid in meta.index:
                if fid in selected:
                    continue
                selected.append(fid)
                if len(selected)>=n_features:
                    break

        return selected

    means=df.mean(axis=0,skipna=True).sort_values(ascending=False)
    return means.head(n_features).index.tolist()

##(b) Plots TIC and top feature traces vs injection order, per batch and across batches
def _plot_post_normalization_diagnostics(
    run: PolarityRun,
    normalized: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    istd_columns:Iterable[str],
) -> list[str]:
    flags: list[str] = []
    _ensure_dir(output_dir)
    
    non_istd=normalized.drop(columns=list(istd_columns),errors="ignore")
    traces = _select_feature_traces(run, non_istd)

    x_label = (
        "Injection order (F#)"
        if "f_number" in run.injection_order_source
        else "Injection order"
        if "raw_workbook" in run.injection_order_source or "file_sequence" in run.injection_order_source
        else "Injection order proxy"
    )
    ##per-batch plots of TIC and top feature traces vs injection order
    for batch, batch_rows in run.sample_info.groupby("batch"):
        batch_rows = batch_rows.sort_values("injection_order")
        row_ids = batch_rows.index.tolist()
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        tic = _tic(normalized.loc[row_ids])
        axes[0].plot(batch_rows["injection_order"], tic.values, marker="o", color="black")
        axes[0].set_title(f"{prefix} batch {batch}: post-normalization TIC")
        axes[0].set_ylabel("TIC")
    ##per-batch TIC and top-feature traces
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
        #Flag non-flat TIC trends
        if len(tic) >= 3:
            slope = np.polyfit(batch_rows["injection_order"], tic.values, deg=1)[0]
            if np.isfinite(slope):
                scale = np.nanmean(np.abs(tic.values)) or 1.0
                if abs(slope) / scale > 0.05:
                    flags.append(
                        f"{prefix} batch {batch}: post-normalization TIC still trends with "
                        f"injection order (relative slope={abs(slope)/scale:.3f})."
                    )
    ##Overlay plots of TIC and top feature traces across batches
    if traces:
        batch_frames = [
            (batch, batch_rows.sort_values("injection_order"))
            for batch, batch_rows in run.sample_info.groupby("batch")
        ]
        fig, axes = plt.subplots(len(traces) + 1, 1, figsize=(11, 3.1 * (len(traces) + 1)))

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

"""
Part 2: Data Cleaning and Batch Correction
"""

### Step 7: Feature Missingness Filter
def categorize_sample_time(gest_age_samp):
 if gest_age_samp is None:
  return None
 if gest_age_samp<6.0:
  return None
 if gest_age_samp<=13.9:
  return '1'
 if gest_age_samp<=21.9:
  return '2'
 if gest_age_samp<=31.9:
  return '3'
 if gest_age_samp<=36.9:
  return '4'
 return '5'

def categorize_sample_time_with_delivery(
    gest_age_samp,
    gest_age_deliv,
    delivery_window_weeks: float = 0.1,
) -> str | None:
    """
    Assign longitudinal timepoint 1–5, except samples collected at/within
    `delivery_window_weeks` before delivery, which are labeled Delivery.
    """
    try:
        samp = float(gest_age_samp)
        delivery = float(gest_age_deliv)
    except (TypeError, ValueError):
        return categorize_sample_time(gest_age_samp)

    # Delivery sample: collected from 0 to 0.1 weeks before/at delivery.
    if 0 <= (delivery - samp) <= delivery_window_weeks:
        return "Delivery"

    return categorize_sample_time(samp)

def gest_age_diff(gest_age_deliv,gest_age_samp):
 if gest_age_deliv in ("",None)or gest_age_samp in ("",None):
  return None
 try:
  return float(gest_age_deliv)-float(gest_age_samp)
 except ValueError:
  return None

def _feature_missingness_filter(
    df: pd.DataFrame,
    sample_info: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    artifacts: PipelineArtifacts,
    polarity: str,
    tissue: str,
    feature_meta: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    biological = sample_info["sample_type"] == "biological"
    bio_df = df.loc[biological].copy()
    meta = sample_metadata.reindex(sample_info.loc[biological, "sample_name"]).copy()
    groups = meta["Group"].fillna("Unknown").values

    if tissue in {"plasma", "urine"}:
        age_for_timepoint = meta["SampleGestAge"]
    elif tissue == "placenta":
        age_for_timepoint = meta["GestAgeDelivery"]
    timepoints = age_for_timepoint.map(categorize_sample_time).values
    rows = []
    drop_map = {col: False for col in df.columns}

    for tpt in [t for t in pd.unique(timepoints) if t is not None]:
        tmask = (timepoints == tpt)
        if not tmask.any():
            continue

        t_bio = bio_df.loc[tmask]
        t_groups = groups[tmask]
        control_mask = (t_groups == "Control")
        case_mask = ~control_mask

        for col in t_bio.columns:
            series = t_bio[col]
            miss_any = float(series.isna().mean())
            control_missing = float(series[control_mask].isna().mean()) if control_mask.any() else float("nan")
            case_missing = float(series[case_mask].isna().mean()) if case_mask.any() else float("nan")

            high_control = np.isfinite(control_missing) and control_missing > FEATURE_MISSING_THRESHOLD
            high_case = np.isfinite(case_missing) and case_missing > FEATURE_MISSING_THRESHOLD
            drop = high_control and high_case

            rows.append({
                "feature_id": col,
                "polarity": polarity,
                "timepoint": tpt,
                "miss_any": miss_any,
                "control_missing": control_missing,
                "case_missing": case_missing,
                "dropped": drop,
            })

            if drop and not drop_map[col]:
                drop_map[col] = True

                reason_str = (
                    f"timepoint={tpt}, "
                    f"missingness control={control_missing:.3f}, "
                    f"case={case_missing:.3f}"
                )

                artifacts.feature_filter_log.append(
                    {
                        "step": 7,
                        "feature_id": col,
                        "polarity": polarity,
                        "reason": reason_str,
                    }
                )

                _fm = (
                    feature_meta.loc[col]
                    if feature_meta is not None and col in feature_meta.index
                    else None
                )

                _record_drop(
                    artifacts,
                    "feature",
                    col,
                    step=7,
                    step_name="Step 7 — Feature Missingness",
                    polarity=polarity,
                    reason=reason_str,
                    annotation_name=(str(_fm.get("annotation_name", "")) if _fm is not None else ""),
                    formula=(str(_fm.get("formula", "")) if _fm is not None else ""),
                    mz=(float(_fm["mz"]) if _fm is not None and pd.notna(_fm.get("mz")) else None),
                    rt_min=(float(_fm["rt"]) if _fm is not None and pd.notna(_fm.get("rt")) else None),
                    metric_value=round(miss_any, 4),
                    metric_threshold=FEATURE_MISSING_THRESHOLD,
                )

    kept_cols = [col for col in df.columns if not drop_map[col]]
    return df.loc[:, kept_cols].copy(), pd.DataFrame(rows)

###STEP 8: Sample-Level Missingness Filter
def _sample_missingness_filter(
    df:pd.DataFrame,
    sample_info:pd.DataFrame,
    artifacts:PipelineArtifacts,
    polarity:str,
    tissue:str,
    )->tuple[pd.DataFrame,pd.DataFrame]:
    keep_mask=pd.Series(True,index=df.index)

    ##(a) Missingness: fraction missing per biological sample across remaining features
    biological=sample_info["sample_type"]=="biological"
    #miss=df.loc[biological].isna().mean(axis=1)
    miss=df.isna().mean(axis=1)
    # Drop if sample missingness is greater than threshold
    for row_id,frac in miss.items():
        if frac>SAMPLE_MISSING_THRESHOLD:
            keep_mask.loc[row_id]=False
            sname=sample_info.at[row_id,"sample_name"]
            batch=str(sample_info.at[row_id,"batch"]) if "batch" in sample_info.columns else ""
            tp = (
                str(sample_info.at[row_id, "SampleTimepoint"])
                if "SampleTimepoint" in sample_info.columns
                and pd.notna(sample_info.at[row_id, "SampleTimepoint"])
                else ""
            )
            reason_str=f"missingness={frac:.3f}"
            artifacts.sample_filter_log.append(
                {
                    "step":8,
                    "row_id":row_id,
                    "sample_name":sname,
                    "polarity":polarity,
                    "reason":reason_str,
                }
            )
            _record_drop(
                artifacts,"sample",sname,
                step=8,step_name="Step 8 — Sample Missingness",
                polarity=polarity,reason=reason_str,
                metric_value=round(frac,4),
                metric_threshold=SAMPLE_MISSING_THRESHOLD,
                batch=batch,timepoint=tp,
            )

    ##(b) Time-Point Specific Flags: at / within 0.1 weeks of delivery, and post-delivery
    if (tissue in {"plasma", "urine"}
        and "GestAgeDelivery" in sample_info.columns 
        and "SampleGestAge" in sample_info.columns
    ):
        for row_id in df.index:
            if not keep_mask.loc[row_id]:
                continue

            ga_deliv=sample_info.at[row_id,"GestAgeDelivery"]
            ga_samp=sample_info.at[row_id,"SampleGestAge"]
            diff=gest_age_diff(ga_deliv,ga_samp)
            if diff is None:
                continue

            sname=sample_info.at[row_id,"sample_name"]
            batch=str(sample_info.at[row_id,"batch"]) if "batch" in sample_info.columns else ""
            tp_match=re.search(r"([A-E])$",str(sname).strip())
            tp=tp_match.group(1) if tp_match else ""

            # Flag at or within 0.1 wks before delivery: 0 <= diff <= 0.1
            # Delivery window: retain sample and label it Delivery upstream in metadata.
            # Record a flag only; this is not a sample-removal event.
            if 0 <= diff <= 0.1:
                reason_str = (
                    f"within_0_1wk_before_or_at_delivery, GA_diff={diff:.3f}; "
                    "SampleTimepoint=Delivery; flagged_only_not_removed"
                )

                artifacts.method_log.append(
                    f"{polarity} Step 8: {sname}; {reason_str}"
                )
            elif diff < 0:
                reason_str = (
                    f"post_birth_sample, GA_diff={diff:.3f}; "
                    "flagged_only_not_removed"
                )

                artifacts.method_log.append(
                    f"{polarity} Step 8: {sname}; {reason_str}"
                )
    #(c): Multiple Samples Within Each Timepoint: keep sample closest to median GA within that timepoint
    if (
        tissue in {"plasma", "urine"}
        and {"SubjectID", "SampleTimepoint"}.issubset(sample_info.columns)
    ):
        kept_idx = sample_info.index[keep_mask]

        duplicate_df = pd.DataFrame(
            {
                "SubjectID": sample_info.loc[kept_idx, "SubjectID"],
                "SampleTimepoint": sample_info.loc[
                    kept_idx, "SampleTimepoint"
                ].astype("string"),
                "sample_name": sample_info.loc[kept_idx, "sample_name"].astype(str),
            },
            index=kept_idx,
        ).dropna(subset=["SubjectID", "SampleTimepoint"])

        for (subject_id, timepoint), subset in duplicate_df.groupby(
            ["SubjectID", "SampleTimepoint"],
            sort=False,
        ):
            sample_names = subset["sample_name"].unique().tolist()

            # A repeated identical sample ID is a technical/bridge replicate,
            # not multiple biological specimens.
            if len(sample_names) <= 1:
                continue

            reason_str = (
                "multiple_distinct_samples_same_subject_timepoint; "
                f"SubjectID={subject_id}; "
                f"timepoint={timepoint}; "
                f"n_distinct_samples={len(sample_names)}; "
                f"samples={';'.join(sample_names)}; "
                "flagged_only_not_removed"
            )

            artifacts.method_log.append(
                f"{polarity} Step 8: {reason_str}"
            )
    return df.loc[keep_mask].copy(),sample_info.loc[keep_mask].copy()

###STEP 9: Applying log2 transformation
def log2_transform(df:pd.DataFrame)->pd.DataFrame:
    return np.log2(df+1.0)

###STEP 10: Imputation of missing values with half min observed
def half_min_impute_wide(X: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values on log2 scale using (minimum observed - 1) per assay.

    Subtracting 1 in log2 space is equivalent to halving in linear space,
    placing imputed values just below the minimum detected concentration.
    """
    Xi = X.copy()
    for col in Xi.columns:
        col_min = Xi[col].min(skipna=True)
        
        if pd.isna(col_min):
            # If the column is 100% NaN, min() is NaN. 
            # We leave it as NaN here and drop it after the loop.
            continue
        else:
            Xi[col] = Xi[col].fillna(col_min - 1.0)
            
    # Drop any features that are still completely NaN to protect downstream steps
    Xi = Xi.dropna(axis=1, how="all")
    return Xi

###STEP 11/14: PCA Plot Pre and Post Correction
def _pca_plot(
    df: pd.DataFrame,
    labels: pd.Series,
    title: str,
    out_path: Path,
) -> None:
    X = df.copy()
    #X = X.apply(lambda col: col.fillna(col.median(skipna=True)), axis=0)
    #X = X.dropna(axis=1, how="all")
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

###STEP 12: Verify batch confounding
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
    #(a) Categorical covariates: chi-square test of independence between batch and Group/Subgroup
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
    #(b) Continuous covariates: ANOVA across batches
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

###STEP 13: ComBat Batch Correction
##(a) Construct design matrix for ComBat
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

##(b) Apply ComBat batch correction with reference batch
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

#(b) Fallback 1: Using pycombat as a fallback when R ComBat with ref.batch fails
def _combat_pycombat(
    df_log2_imputed: pd.DataFrame,
    batch_labels: pd.Series,
    covariates: pd.DataFrame | None,
) -> pd.DataFrame:
    model = Combat()
    X = covariates.to_numpy(dtype=float) if covariates is not None and not covariates.empty else None
    corrected = model.fit_transform(
        df_log2_imputed.to_numpy(dtype=float),
        batch_labels.astype(str).to_numpy(),
        X=X,
    )
    return pd.DataFrame(corrected, index=df_log2_imputed.index, columns=df_log2_imputed.columns)

#(b) Fallback 2: Reference batch location/scale correction when true ComBat is unavailable
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

##(c) Returning batch-corrected log2-imputed matrix
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

###STEP 15: Post-ComBat intensity diagnostics
def _post_combat_intensity_check(
    run: PolarityRun,
    corrected: pd.DataFrame,
    sample_info: pd.DataFrame,
    output_dir: Path,
    prefix: str,
) -> list[str]:
    """Step 15 (SOP v4): Re-plot injection-order intensity diagnostics on the
    batch-corrected matrix to verify ComBat did not introduce new trends.

    Generates the same three plots as the post-normalization diagnostics
    (per-batch TIC, per-batch feature traces, cross-batch overlay) but on the
    corrected data.  Any new systematic trends relative to Step 6 should be
    investigated.
    """
    flags: list[str] = []
    diag_dir = output_dir / "post_combat_intensity_check"
    _ensure_dir(diag_dir)

    if run.modality is Modality.METABOLOMICS and run.polarity.upper()=="POS":
        wanted={_normalise_name(n) for n in METABOLOMICS_POSITIVE_ISTD_NAMES}
    elif run.modality is Modality.METABOLOMICS and run.polarity.upper()=="NEG":
        wanted={_normalise_name(n) for n in METABOLOMICS_NEGATIVE_ISTD_NAMES}
    elif run.modality is Modality.LIPIDOMICS and run.polarity.upper()=="POS":
        wanted={_normalise_name(n) for n in LIPIDOMICS_POSITIVE_ISTD_NAMES}
    elif run.modality is Modality.LIPIDOMICS and run.polarity.upper()=="NEG":
        wanted={_normalise_name(n) for n in LIPIDOMICS_NEGATIVE_ISTD_NAMES} 
    else:
        wanted=set()

    if "annotation_name" in run.feature_meta.columns:
        istd_columns=[
            c for c in run.raw_istd.columns
            if (
				not wanted
				or _normalise_name(str(run.feature_meta.at[c,"annotation_name"])) in wanted
				or _normalise_name(c) in wanted
			)
		]
    else:
        istd_columns=[
		 c for c in run.raw_istd.columns
		 if not wanted or _normalise_name(c) in wanted
		]

    # Select representative high-abundance features (excluding ISTDs)
    bio_mask = sample_info["sample_type"] == "biological"
    bio_corrected = corrected.loc[bio_mask]
    traces = _select_feature_traces(run, bio_corrected.drop(columns=istd_columns, errors="ignore"))

    x_label = (
        "Injection order (F#)"
        if "f_number" in run.injection_order_source
        else "Injection order"
        if "raw_workbook" in run.injection_order_source or "file_sequence" in run.injection_order_source
        else "Injection order proxy"
    )

    ##(a) Per-batch plots of TIC and feature traces on corrected data
    for batch, batch_rows in sample_info.groupby("batch"):
        batch_rows = batch_rows.sort_values("injection_order")
        row_ids = batch_rows.index.tolist()
        if len(row_ids) < 2:
            continue

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        tic = _tic(corrected.loc[row_ids])
        axes[0].plot(batch_rows["injection_order"], tic.values, marker="o", color="steelblue")
        axes[0].set_title(f"{prefix} batch {batch}: post-ComBat mean intensity")
        axes[0].set_ylabel("Mean intensity (log2)")
    ##(b) Plot high-abundance feature traces on corrected data
        for feature in traces:
            if feature not in corrected.columns:
                continue
            axes[1].plot(
                batch_rows["injection_order"],
                corrected.loc[row_ids, feature].values,
                marker="o",
                label=feature,
            )
        axes[1].set_title("Post-ComBat high-abundance feature traces")
        axes[1].set_xlabel(x_label)
        axes[1].set_ylabel("corrected intensity (log2)")
        axes[1].legend(fontsize=7, ncol=2)

        out_path = diag_dir / f"{prefix.lower()}_batch_{batch}_postcombat.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)

        # Flag any residual trend
        if len(tic) >= 3:
            slope = np.polyfit(batch_rows["injection_order"], tic.values, deg=1)[0]
            if np.isfinite(slope):
                scale = np.nanmean(np.abs(tic.values)) or 1.0
                if abs(slope) / scale > 0.05:
                    flags.append(
                        f"{prefix} batch {batch}: post-ComBat intensity still trends with "
                        f"injection order (relative slope={abs(slope)/scale:.3f}). "
                        "ComBat may have introduced artifacts — inspect correction parameters."
                    )

    ## (c) Cross-batch overlay of TIC and feature traces on corrected data
    if traces:
        batch_frames = [
            (batch, batch_rows.sort_values("injection_order"))
            for batch, batch_rows in sample_info.groupby("batch")
        ]
        n_panels = len(traces) + 1
        fig, axes = plt.subplots(n_panels, 1, figsize=(11, 3.1 * n_panels))
        if n_panels == 1:
            axes = [axes]

        for batch, batch_rows in batch_frames:
            row_ids = batch_rows.index.tolist()
            axes[0].plot(
                batch_rows["injection_order"],
                _tic(corrected.loc[row_ids]).values,
                marker="o",
                label=batch,
            )
        axes[0].set_title(f"{prefix} cross-batch post-ComBat mean intensity overlay")
        axes[0].set_ylabel("Mean intensity (log2)")
        axes[0].legend(fontsize=8, ncol=2)
        # Feature overlays
        for ax, feature in zip(axes[1:], traces):
            for batch, batch_rows in batch_frames:
                row_ids = batch_rows.index.tolist()
                if feature not in corrected.columns:
                    continue
                ax.plot(
                    batch_rows["injection_order"],
                    corrected.loc[row_ids, feature].values,
                    marker="o",
                    label=batch,
                )
            ax.set_title(feature)
            ax.set_ylabel("corrected intensity (log2)")
        axes[-1].set_xlabel(x_label)
        if len(axes) > 1:
            axes[1].legend(fontsize=8, ncol=2)

        overlay_path = diag_dir / f"{prefix.lower()}_cross_batch_postcombat_overlay.png"
        fig.tight_layout()
        fig.savefig(overlay_path, dpi=180)
        plt.close(fig)

    return flags


###STEP 16: Sample-Level QC via Internal Standards
def sample_istd_mad_filter(
    run: PolarityRun,
    sample_info: pd.DataFrame,
    artifacts: PipelineArtifacts,
    modality: Modality,
) -> set[str]:
    bad_row_ids: set[str] = set()
    biological_mask = sample_info["sample_type"].eq("biological")

    istd_columns = [
        col for col in run.raw_istd.columns
        if col in run.expression.columns
    ]

    if not istd_columns:
        message = (
            f"{run.polarity} Step 16 could not be run: no configured "
            f"{run.modality.name.lower()} ISTD features were detected."
        )
        artifacts.qc_warnings.append(message)
        raise RuntimeError(message)

    for batch, batch_info in sample_info.loc[biological_mask].groupby("batch"):
        row_ids = batch_info.index.tolist()

        for istd in istd_columns:
            values = pd.to_numeric(
                run.raw_istd.loc[row_ids, istd],
                errors="coerce",
            )

            valid_values = values.dropna()
            if valid_values.size < 3:
                artifacts.qc_warnings.append(
                    f"{run.polarity} Step 16 batch {batch}, ISTD {istd}: "
                    f"only {valid_values.size} non-missing biological samples; "
                    "MAD QC not estimable."
                )
                continue

            median_value = valid_values.median()
            mad_value = stats.median_abs_deviation(
                valid_values,
                nan_policy="omit",
                scale=1.0,
            )

            if not np.isfinite(mad_value) or mad_value <= 0:
                artifacts.qc_warnings.append(
                    f"{run.polarity} Step 16 batch {batch}, ISTD {istd}: "
                    "MAD was zero or non-finite; no outlier threshold applied."
                )
                continue

            lower_bound = median_value - SAMPLE_MAD_THRESHOLD * mad_value
            upper_bound = median_value + SAMPLE_MAD_THRESHOLD * mad_value

            failed = values[
                (values < lower_bound) | (values > upper_bound)
            ]

            for row_id, observed_value in failed.items():
                sample_name = str(sample_info.at[row_id, "sample_name"])

                bad_row_ids.add(row_id)

                reason = (
                    f"{istd}: raw ISTD={observed_value:.3e}; "
                    f"batch median={median_value:.3e}; "
                    f"MAD={mad_value:.3e}; "
                    f"limits=[{lower_bound:.3e}, {upper_bound:.3e}]"
                )

                artifacts.sample_filter_log.append(
                    {
                        "step": 16,
                        "row_id": row_id,
                        "sample_name": sample_name,
                        "polarity": run.polarity,
                        "batch": str(batch),
                        "reason": reason,
                    }
                )

                _record_drop(
                    artifacts,
                    entity_type="sample",
                    entity_id=sample_name,
                    step=16,
                    step_name="Step 16 ISTD MAD Sample QC",
                    polarity=run.polarity,
                    reason=reason,
                    metric_value=float(observed_value),
                    metric_threshold=float(SAMPLE_MAD_THRESHOLD),
                    batch=str(batch),
                )

    artifacts.step_execution.append(
        {
            "step": 16,
            "polarity": run.polarity,
            "modality": run.modality.name.lower(),
            "status": "completed",
            "n_istds_tested": len(istd_columns),
            "n_samples_flagged": len(bad_row_ids),
        }
    )

    return bad_row_ids

###STEP 17: RSD Filter on QC Pools
def _rsd_filter(
    df: pd.DataFrame,
    sample_info: pd.DataFrame,
    artifacts: PipelineArtifacts,
    polarity: str,
    feature_meta: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Drop features whose QC-pool RSD exceeds the threshold in ANY batch.

    RSD is computed separately for each batch's QC pools, and a feature is removed 
    if it fails in at least one batch.  Computing a single pooled RSD across all batches
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

    # Feature fails if it exceeds the threshold in ANY batch
    fail_mask = pd.Series(False, index=df.columns)
    for rsd_series in batch_rsd.values():
        fail_mask |= rsd_series.reindex(df.columns).fillna(np.inf) > RSD_THRESHOLD

    for col in fail_mask.index[fail_mask]:
        per_batch_str = ", ".join(
            f"batch{b}={batch_rsd[b].get(col, float('nan')):.1f}%"
            for b in sorted(batch_rsd)
            if col in batch_rsd[b].index
        )
        reason_str = f"QC RSD >30% in ≥1 batch ({per_batch_str})"
        artifacts.feature_filter_log.append(
            {
                "step": 17,
                "feature_id": col,
                "polarity": polarity,
                "reason": reason_str,
            }
        )
        _fm = feature_meta.loc[col] if (feature_meta is not None and col in feature_meta.index) else None
        # worst (highest) RSD across batches as the representative metric value
        worst_rsd = max(
            (batch_rsd[b][col] for b in batch_rsd if col in batch_rsd[b].index),
            default=float("nan"),
        )
        _record_drop(
            artifacts, "feature", col,
            step=17, step_name="Step 17 — QC-Pool RSD Filter",
            polarity=polarity, reason=reason_str,
            annotation_name=str(_fm["annotation_name"]) if _fm is not None else "",
            formula=str(_fm["formula"]) if _fm is not None else "",
            mz=float(_fm["mz"]) if _fm is not None and pd.notna(_fm.get("mz")) else None,
            rt_min=float(_fm["rt"]) if _fm is not None and pd.notna(_fm.get("rt")) else None,
            metric_value=round(worst_rsd, 2) if np.isfinite(worst_rsd) else None,
            metric_threshold=RSD_THRESHOLD,
        )

    keep = ~fail_mask
    return df.loc[:, keep.index[keep]].copy()

###STEP 18: IQR Filter on Biological Samples within Timepoints
def _iqr_filter(
    df: pd.DataFrame,
    sample_info: pd.DataFrame,
    artifacts: PipelineArtifacts,
    polarity: str,
    feature_meta: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Step 18 (SOP v4): Remove near-constant or hypervariable features using
    within-timepoint IQR on the batch-corrected log2 data.

    IQR is computed separately within each timepoint (A–E) for biological
    samples only.  The median of the per-timepoint IQRs is used as the
    feature's representative spread, so longitudinal trends do not inflate it.

    A feature is removed only when BOTH conditions hold:
    * Constant: median IQR < IQR_LOW_PERCENTILE AND median IQR < IQR_FLOOR
    * Hypervariable: median IQR > IQR_HIGH_PERCENTILE AND median IQR > IQR_CEILING

    This dual-threshold (relative AND absolute) ensures no features are removed
    if the data is well-behaved.
    """
    #Biological only subset
    bio_mask = sample_info["sample_type"] == "biological"
    bio_si = sample_info.loc[bio_mask]
    bio_df = df.loc[bio_mask]

    # Extract timepoint from sample_name
    def _tp(name: str) -> str | None:
        m = re.search(r"([A-E])$", str(name).strip())
        return m.group(1) if m else None
    
    bio_si = bio_si.copy()
    bio_si["_tp"] = bio_si["sample_name"].map(_tp)
    timepoints = [t for t in sorted(bio_si["_tp"].dropna().unique()) if t in VALID_TIMEPOINTS]
    #Skip if no valid timepoints found
    if not timepoints:
        artifacts.qc_warnings.append(
            f"{polarity}: no timepoint labels (A–E) found in sample names; IQR filter skipped."
        )
        return df

    ### (a) Compute per-timepoint IQR for each feature 
    tp_iqrs: dict[str, pd.Series] = {}
    for tp in timepoints:
        idx = bio_si.index[bio_si["_tp"] == tp]
        if len(idx) < 4:
            # Need at least 4 samples to get a meaningful IQR
            continue
        sub = bio_df.loc[idx]
        q75 = sub.quantile(0.75, numeric_only=True)
        q25 = sub.quantile(0.25, numeric_only=True)
        tp_iqrs[tp] = (q75 - q25).clip(lower=0.0)

    if not tp_iqrs:
        artifacts.qc_warnings.append(
            f"{polarity}: no timepoint had ≥4 biological samples for IQR calculation; "
            "IQR filter skipped."
        )
        return df
    ### (b) Take median of per-timepoint IQRs (cross-sectional variability)
    iqr_table = pd.DataFrame(tp_iqrs)  # features × timepoints
    median_iqr = iqr_table.median(axis=1, skipna=True)  # per-feature

    ### (c) Compute percentile rank of each feature's median IQR
    n_features = len(median_iqr)
    low_pct_val = float(np.nanpercentile(median_iqr.values, IQR_LOW_PERCENTILE))
    high_pct_val = float(np.nanpercentile(median_iqr.values, IQR_HIGH_PERCENTILE))

    # Log whether the filter is active
    artifacts.qc_warnings.append(
        f"{polarity} Step 18 IQR filter: "
        f"{IQR_LOW_PERCENTILE}th pct IQR = {low_pct_val:.4f} "
        f"(floor = {IQR_FLOOR}; filter active = {low_pct_val < IQR_FLOOR}), "
        f"{IQR_HIGH_PERCENTILE}th pct IQR = {high_pct_val:.4f} "
        f"(ceiling = {IQR_CEILING}; filter active = {high_pct_val > IQR_CEILING})."
    )

    ### (d) Remove features where BOTH relative/absolute threshold are constant or hypervariable 
    to_drop: list[str] = []
    for feat, miqr in median_iqr.items():
        if pd.isna(miqr):
            continue
        _fm = feature_meta.loc[feat] if (feature_meta is not None and feat in feature_meta.index) else None
        if miqr < low_pct_val and miqr < IQR_FLOOR:
            # Near-constant: fails both relative and absolute threshold
            reason_str = (
                f"near-constant: median IQR={miqr:.4f} < "
                f"{IQR_LOW_PERCENTILE}th pct ({low_pct_val:.4f}) "
                f"AND < floor ({IQR_FLOOR})"
            )
            artifacts.feature_filter_log.append(
                {
                    "step": 18,
                    "feature_id": feat,
                    "polarity": polarity,
                    "reason": reason_str,
                }
            )
            _record_drop(
                artifacts, "feature", feat,
                step=18, step_name="Step 18 — IQR Filter (near-constant)",
                polarity=polarity, reason=reason_str,
                annotation_name=str(_fm["annotation_name"]) if _fm is not None else "",
                formula=str(_fm["formula"]) if _fm is not None else "",
                mz=float(_fm["mz"]) if _fm is not None and pd.notna(_fm.get("mz")) else None,
                rt_min=float(_fm["rt"]) if _fm is not None and pd.notna(_fm.get("rt")) else None,
                metric_value=round(float(miqr), 4),
                metric_threshold=IQR_FLOOR,
            )
            to_drop.append(feat)
        elif miqr > high_pct_val and miqr > IQR_CEILING:
            # Hypervariable: fails both relative and absolute threshold
            reason_str = (
                f"hypervariable: median IQR={miqr:.4f} > "
                f"{IQR_HIGH_PERCENTILE}th pct ({high_pct_val:.4f}) "
                f"AND > ceiling ({IQR_CEILING})"
            )
            artifacts.feature_filter_log.append(
                {
                    "step": 18,
                    "feature_id": feat,
                    "polarity": polarity,
                    "reason": reason_str,
                }
            )
            _record_drop(
                artifacts, "feature", feat,
                step=18, step_name="Step 18 — IQR Filter (hypervariable)",
                polarity=polarity, reason=reason_str,
                annotation_name=str(_fm["annotation_name"]) if _fm is not None else "",
                formula=str(_fm["formula"]) if _fm is not None else "",
                mz=float(_fm["mz"]) if _fm is not None and pd.notna(_fm.get("mz")) else None,
                rt_min=float(_fm["rt"]) if _fm is not None and pd.notna(_fm.get("rt")) else None,
                metric_value=round(float(miqr), 4),
                metric_threshold=IQR_CEILING,
            )
            to_drop.append(feat)

    ### (e) Log each removed feature (feature ID, per-timepoint IQR values, median IQR, percentile, classification)
    if to_drop:
        LOGGER.info(
            "%s Step 18 IQR filter: removed %d/%d features "
            "(%d near-constant, %d hypervariable).",
            polarity,
            len(to_drop),
            n_features,
            sum(1 for f in to_drop if "near-constant" in next(
                e["reason"] for e in artifacts.feature_filter_log
                if e.get("feature_id") == f and e.get("step") == 18
            )),
            sum(1 for f in to_drop if "hypervariable" in next(
                e["reason"] for e in artifacts.feature_filter_log
                if e.get("feature_id") == f and e.get("step") == 18
            )),
        )
    else:
        LOGGER.info(
            "%s Step 18 IQR filter: no features removed (data is well-behaved).", polarity
        )

    keep = [c for c in df.columns if c not in set(to_drop)]
    return df[keep].copy()

##STEPS 19, 29-31, 33
def _merge_polarities(
    runs: list[tuple[PolarityRun, pd.DataFrame, pd.DataFrame]],
    sample_metadata: pd.DataFrame,
    config: DatasetConfig,
    artifacts: PipelineArtifacts,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_tables: list[pd.DataFrame] = []
    matrices: list[pd.DataFrame] = []

    istd_names = {
        _normalise_name(name)
        for name in get_istd_names(config.modality, "POS") + get_istd_names(config.modality, "NEG")
    }

    for run, corrected_before_average, sample_info in runs:
        # STEP 19: Average bridge samples after correction, on the log scale.
        biological_mask = sample_info["sample_type"] == "biological"
        biological_names = sample_info.loc[biological_mask, "sample_name"]
        counts=biological_names.value_counts()
        n_bridges=int((counts>1).sum())
        if n_bridges>0:
            artifacts.qc_warnings.append(
                f"{config.dataset_id} {run.polarity}: averaged {n_bridges} biological bridge samples."  # NEW
            )
        averaged = corrected_before_average.loc[biological_mask].groupby(biological_names).mean()
        #STEPS 29 - 31: Finalize feature IDs, merge polarities, assignment of confidence levels
        meta = run.retained_feature_meta.loc[corrected_before_average.columns].copy()
        meta = _finalize_feature_ids(meta, config.database, istd_names)
        averaged.columns = meta["final_feature_id"].values
        meta.index = averaged.columns
        feature_tables.append(meta)
        matrices.append(averaged)

    merged = pd.concat(matrices, axis=1, join="inner")
    feature_meta = pd.concat(feature_tables, axis=0)
    feature_meta = feature_meta[~feature_meta.index.duplicated(keep="first")]
    #STEP 33: Add metadata columns and construct final matrix
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
        "SampleTimepoint",
    ]
    present_meta_cols = [c for c in ordered_meta_cols if c in merged.columns]
    analyte_cols = [c for c in merged.columns if c not in present_meta_cols]
    final_matrix = pd.concat(
        [subject_ids.rename("SubjectID"), merged[present_meta_cols], merged[analyte_cols]],
        axis=1,
    )
    return final_matrix, feature_meta

"""
Part 3: Feature Deduplication
"""

###STEP 20: Expected Parent Ion Calculation; STEP 21: Feature Classification
def _candidate_classifications(
    group_meta:pd.DataFrame,
    modality: Modality, 
    polarity:str,
    modifications:pd.DataFrame,
)->pd.DataFrame:
    group_meta=group_meta.copy()
    group_meta["expected_parent_mz"]=np.nan
    group_meta["classification"]="Unknown/Measurement Error"
    group_meta["modification_name"]=""
    group_meta["modification_type"]=""

    if modality is Modality.METABOLOMICS:
        expected=(
            group_meta["calc_mw"]+PROTON_MASS
            if polarity=="POS"
            else group_meta["calc_mw"]-PROTON_MASS
        )

    if modality is Modality.LIPIDOMICS:
        if "AdductIon" not in group_meta.columns:
            raise ValueError(
                f"{polarity} lipidomics classification requires AdductIon."
            )

        if "Class" not in group_meta.columns:
            raise ValueError(
                f"{polarity} lipidomics classification requires Class."
            )

        adduct_col=(
            group_meta["AdductIon"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        class_col=(
            group_meta["Class"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        for idx in group_meta.index:
            observed_adduct=adduct_col.at[idx]
            lipid_class=class_col.at[idx]

            canonical_adduct=LIPIDOMICS_CLASS_ADDUCT_MAP.get(polarity,{},).get(lipid_class,"",)

            if not lipid_class:
                group_meta.at[idx,"modification_name"]="Missing lipid Class"
                continue

            if not canonical_adduct:
                group_meta.at[idx,"modification_name"]=(
                    f"No canonical adduct configured for Class={lipid_class}"
                )
                continue

            if not observed_adduct:
                group_meta.at[idx,"modification_name"]="Missing AdductIon"
                continue

            if observed_adduct==canonical_adduct:
                group_meta.at[idx,"classification"]="Parent"
                group_meta.at[idx,"expected_parent_mz"]=group_meta.at[idx,"mz"]
            else:
                group_meta.at[idx,"classification"]="non-Parent"
                group_meta.at[idx,"modification_type"]="Adduct"
                group_meta.at[idx,"modification_name"]=(
                    f"Observed {observed_adduct}; canonical {canonical_adduct}"
                )

        return group_meta

    if modality is not Modality.METABOLOMICS:
        raise ValueError(f"Unsupported modality: {modality}")
    
    valid_mw=group_meta["calc_mw"].notna()

    if not valid_mw.any():
        return group_meta

    valid_expected=expected.notna()
    if not valid_expected.any():
        return group_meta

    delta=group_meta["mz"]-expected
    group_meta.loc[valid_expected,"expected_parent_mz"]=expected.loc[valid_expected]
    group_meta.loc[valid_expected,"classification"]="Unknown/Measurement Error"

    parent_mask=valid_expected & (delta.abs()<=MASS_TOLERANCE_PARENT)
    group_meta.loc[parent_mask,"classification"]="Parent"

    for idx,value in delta.loc[valid_expected & ~parent_mask].items():
        matched=modifications.loc[
            (modifications["Delta_m/z"]-value).abs()<=MASS_TOLERANCE_NON_PARENT
        ]
        if not matched.empty:
            group_meta.at[idx,"classification"]="non-Parent"
            group_meta.at[idx,"modification_type"]=matched.iloc[0]["Type"]
            group_meta.at[idx,"modification_name"]=matched.iloc[0]["Name"]

    return group_meta

###STEP 22-23: Feature Filter and Flag
def _filter_and_flag_dedup_unit(
    group:pd.DataFrame,
    modality: Modality,
    polarity:str,
    modifications:pd.DataFrame,
)->tuple[pd.DataFrame,list[str],str,dict[str,str]]:
    """
    Steps 22-23 for one dedup unit.

    Metabolomics:
    - One compound group per dedup unit.

    Lipidomics:
    - One LipidID per dedup unit.
    - A LipidGroup containing multiple LipIDs must be split upstream before
      entering this function and logged as an isobaric_cluster.

    Returns:
        candidate_meta: DataFrame of candidates that proceed to quality scoring
        dropped: list of feature_ids dropped in Step 22
        rt_flag: RT spread classification for the retained candidates
        drop_reasons: Step-22 removal reason by dropped feature_id
    """
    classified=_candidate_classifications(group, modality, polarity, modifications)

    # Lipidomics must already have been split by LipidID before Step 22.
    if modality is Modality.LIPIDOMICS:
        lipid_id_col=next(
            (
                col for col in ["LipidID","lipid_id_norm","lipidid"]
                if col in classified.columns
            ),
            None,
        )

        if lipid_id_col is None:
            raise ValueError(
                "Lipidomics Step 22 requires LipidID, lipid_id_norm, "
                "or lipidid in feature metadata."
            )

        lipid_ids=(
            classified[lipid_id_col]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("",np.nan)
            .dropna()
            .unique()
        )

        if len(lipid_ids)>1:
            raise ValueError(
                "Lipidomics Step 22 received multiple LipidIDs in one "
                "dedup unit. Split the isobaric cluster by LipidID before "
                "calling _filter_and_flag_dedup_unit()."
            )

    valid_classifications={
        "Parent",
        "non-Parent",
        "Unknown/Measurement Error",
    }

    unexpected_classifications=set(
        classified["classification"].dropna().astype(str)
    )-valid_classifications

    if unexpected_classifications:
        raise ValueError(
            f"{polarity} Step 22 received unrecognized classifications: "
            f"{sorted(unexpected_classifications)}. "
            "Ensure _candidate_classifications() assigns Parent, non-Parent, "
            "or Unknown/Measurement Error before Step 22."
        )

    parent_ids=classified.index[
        classified["classification"]=="Parent"
    ].tolist()

    dropped=[]
    drop_reasons={}

    # Step 22, Case 1:
    # Exactly one Parent -> remove all non-Parents.
    # This is the final selection; do not run Step 23 or Step 24.
    if len(parent_ids)==1:
        candidate_ids=parent_ids

        for idx in classified.index:
            if idx not in candidate_ids:
                dropped.append(idx)
                drop_reasons[idx]=(
                    f"Step 22: one Parent candidate ({candidate_ids[0]}) "
                    "identified; non-parent candidate removed."
                )

        candidate_meta=classified.loc[candidate_ids].copy()

        return (
            candidate_meta,
            dropped,
            "single_parent_final_selection",
            drop_reasons,
        )

    # Step 22, Case 2:
    # Two or more Parents -> remove all non-Parents and continue to Step 23.
    if len(parent_ids)>=2:
        candidate_ids=parent_ids

        for idx in classified.index:
            if idx not in candidate_ids:
                dropped.append(idx)
                drop_reasons[idx]=(
                    "Step 22: multiple Parent candidates identified; "
                    "non-parent candidate removed."
                )

    # Step 22, Case 3:
    # No Parents -> remove Unknown/Measurement Errors, retain non-Parents.
    else:
        candidate_ids=classified.index[
            classified["classification"]!="Unknown/Measurement Error"
        ].tolist()

        for idx in classified.index:
            if idx not in candidate_ids:
                dropped.append(idx)
                drop_reasons[idx]=(
                    "Step 22: no Parent candidate identified; "
                    "Unknown/Measurement Error removed."
                )

        # Do NOT restore Unknown/Measurement Error features.
        if not candidate_ids:
            return (
                classified.iloc[0:0].copy(),
                dropped,
                "no_valid_nonparent_candidates",
                drop_reasons,
            )

    candidate_meta=classified.loc[candidate_ids].copy()

    # Step 23: RT flag applies only to:
    # (a) 2+ Parent candidates, or
    # (b) retained non-Parent candidates when no Parent was identified.
    rt_spread=(
        candidate_meta["rt"].max(skipna=True)
        -candidate_meta["rt"].min(skipna=True)
    )

    if len(candidate_meta)>1 and np.isfinite(rt_spread):
        if rt_spread<RT_ARTIFACT_MAX:
            rt_flag="chromatographic_artifact"
        elif rt_spread<=RT_ISOMER_MAX:
            rt_flag="potential_structural_isomers"
        else:
            rt_flag="likely_annotation_error"
    else:
        rt_flag="single_candidate"

    return candidate_meta,dropped,rt_flag,drop_reasons



###STEP 24: Quality Annotation Scoring
def _annotation_score(
    meta:pd.DataFrame,
    expression:pd.DataFrame,
    modality:Modality,
)->pd.Series:
    """
    Return the Step-24 composite quality score (0-50) for one dedup unit.

    Components:
    - Peak quality: 0-10
    - Reproducibility: 0-10
    - MS2 availability: 0-10
    - Signal intensity: 0-10
    - Annotation confidence: 0-10
    """
    if modality not in {Modality.METABOLOMICS,Modality.LIPIDOMICS}:
        raise ValueError(
            "modality must be Modality.METABOLOMICS or Modality.LIPIDOMICS"
        )

    def _numeric_score(value)->float:
        value=pd.to_numeric(value,errors="coerce")
        return float(value) if pd.notna(value) else 0.0

    def _peak_score(value)->float:
        value=pd.to_numeric(value,errors="coerce")

        if pd.isna(value):
            return 0.0
        if value>=7:
            return 10.0
        if value>=5:
            return 7.0
        if value>=3:
            return 4.0
        return 1.0

    def _rsd_score(value)->float:
        value=pd.to_numeric(value,errors="coerce")

        if pd.isna(value):
            return 0.0
        if value<10:
            return 10.0
        if value<15:
            return 8.0
        if value<20:
            return 6.0
        if value<25:
            return 4.0
        if value<30:
            return 2.0
        return 0.0

    def _ms2_score(value)->float:
        value=str(value or "").strip().lower()

        if value=="dda for preferred ion":
            return 10.0
        if value=="dda for other ion":
            return 6.0
        if value=="dda available":
            return 4.0
        return 0.0

    def _continuous_rank_score(values:pd.Series)->pd.Series:
        """
        Dense rank scoring within a dedup unit:
        lowest observed intensity = 0, highest = 10.
        Tied values receive the same score.
        """
        values=pd.to_numeric(values,errors="coerce")
        scores=pd.Series(0.0,index=values.index)

        observed=values.notna()
        if not observed.any():
            return scores

        n_unique=values.loc[observed].nunique()

        if n_unique==1:
            scores.loc[observed]=10.0
            return scores

        ranks=values.loc[observed].rank(
            method="dense",
            ascending=True,
        )

        scores.loc[observed]=10.0*(ranks-1)/(n_unique-1)
        return scores

    def _mzcloud_score(value)->float:
        value=pd.to_numeric(value,errors="coerce")

        if pd.isna(value):
            return np.nan
        if value>=90:
            return 10.0
        if value>=80:
            return 9.0
        if value>=70:
            return 8.0
        if value>=60:
            return 7.0
        if value>=50:
            return 6.0
        if value>=40:
            return 5.0
        if value>=30:
            return 4.0
        if value>=20:
            return 3.0
        if value>=10:
            return 2.0
        if value>0:
            return 1.0
        return 0.0

    def _source_count(row:pd.Series,column:str)->int:
        value=pd.to_numeric(row.get(column,0),errors="coerce")
        return int(value) if pd.notna(value) else 0

    def _annot_source_score(row:pd.Series)->float:
        full=_source_count(row,"annotation_source_matches")

        if full>=6:
            return 10.0
        if full==5:
            return 9.0
        if full==4:
            return 8.0
        if full==3:
            return 7.0
        if full==2:
            return 6.0
        if full==1:
            return 5.0

        not_top_hit=_source_count(row,"annotation_not_top_hit")
        if not_top_hit>=1:
            return 4.0

        partial=_source_count(row,"annotation_partial_matches")
        if partial>=3:
            return 3.0
        if partial==2:
            return 2.0
        if partial==1:
            return 1.0

        return 0.0

    def _lipid_lsi_score(value)->float:
        """
        Update these regular expressions if Appendix F uses a different
        LipidID convention.

        Examples:
        PC(16:0/18:1(9Z))  -> structure-defined: 10
        PC(16:0/18:1)      -> sn-position: 9
        PC(16:0_18:1)      -> molecular species: 7
        PC(34:1) or PC 34:1 -> species: 5
        PC                 -> class only: 2
        """
        lipid_id=str(value or "").strip()

        if not lipid_id:
            return 0.0

        normalized=re.sub(r"\s+","",lipid_id)

        # Structure-defined: acyl chains plus double-bond geometry/location.
        if re.search(r"\(\d+[ZE]\)",normalized,re.IGNORECASE):
            return 10.0

        # sn-position-resolved species.
        if "/" in normalized:
            return 9.0

        # Molecular species, chains known but sn positions unresolved.
        if "_" in normalized:
            return 7.0

        # Sum-composition / species-level notation, e.g. PC(34:1).
        if re.search(r"\d+:\d+",normalized):
            return 5.0

        # Lipid class/category but no acyl composition.
        if re.match(r"^[A-Za-z]+$",normalized):
            return 2.0

        return 0.0

    peak_values=meta.get(
        "peak_rating",
        pd.Series(np.nan,index=meta.index),
    )
    rsd_values=meta.get(
        "rsd_qc",
        pd.Series(np.nan,index=meta.index),
    )
    ms2_values=meta.get(
        "ms2",
        pd.Series("",index=meta.index),
    )
    area_values=meta.get(
        "area_max",
        pd.Series(np.nan,index=meta.index),
    )
    mzcloud_values=meta.get(
        "mzcloud_confidence",
        pd.Series(np.nan,index=meta.index),
    )

    peak=peak_values.map(_peak_score)
    rsd=rsd_values.map(_rsd_score)
    ms2=ms2_values.map(_ms2_score)
    intensity=_continuous_rank_score(area_values)

    mzcloud=mzcloud_values.map(_mzcloud_score)
    source_scores=meta.apply(_annot_source_score,axis=1).astype(float)

    if modality is Modality.METABOLOMICS:
        # Use source evidence only if mzCloud is unavailable.
        annotation=mzcloud.where(mzcloud.notna(),source_scores)

    else:
        lipid_id_col=next(
            (
                col for col in ["lipid_id_norm","LipidID","lipidid"]
                if col in meta.columns
            ),
            None,
        )

        lipid_ids=(
            meta[lipid_id_col]
            if lipid_id_col is not None
            else pd.Series("",index=meta.index)
        )

        lsi=lipid_ids.map(_lipid_lsi_score)

        # Lipidomics annotation confidence is MAX, never sum.
        annotation=pd.concat(
            [
                lsi.rename("lsi"),
                mzcloud.rename("mzcloud"),
                source_scores.rename("source"),
            ],
            axis=1,
        ).max(axis=1)

    total=peak+rsd+ms2+intensity+annotation

    return total.clip(lower=0.0,upper=50.0)

###STEP 25: Feature Selection (Based on Feature Classification, Filtering, and Quality Scores)
def _resolve_named_groups(
    run: PolarityRun,
    expression: pd.DataFrame,
    modifications: pd.DataFrame,
    artifacts: PipelineArtifacts,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = run.feature_meta.loc[expression.columns].copy()

    named = meta[meta["is_named"]].copy()
    unnamed = meta[~meta["is_named"]].copy()

    keep_ids,dropped_ids,named_units_meta=_run_part3_named_dedup(
        feature_meta=named,
        expression=expression,
        modality=run.modality,
        polarity=run.polarity,
        modifications=modifications,
        artifacts=artifacts,
    )
    defaults={
        "quality_score":(np.nan,float),
        "rt_flag":("",object),
        "isobaric_cluster":(False,bool),
        "isobaric_cluster_lipid_ids":("",object),
    }

    for col,(default,dtype) in defaults.items():
        if col in named_units_meta.columns and col not in meta.columns:
            meta[col]=pd.Series(default,index=meta.index,dtype=dtype)
    meta.update(named_units_meta)
    named_kept=meta.loc[sorted(keep_ids)].copy() if keep_ids else meta.iloc[0:0].copy()
    return named_kept, unnamed.copy()

#####PART 3B: Cross-Feature Deduplication for Unannotated Features (Metabolomics Only)
###Steps 26/27: Formula-Based Grouping and Mass Based Grouping for Unannotated Features with Formulas
#Grouping features close in m/z and RT
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

#Select best representative within cluster based on quality score, area, RSD, and RT
def _collapse_group_by_quality(
    group_ids: list[str],
    meta: pd.DataFrame,
    expression: pd.DataFrame,
    modality: Modality,
    artifacts: PipelineArtifacts,
    phase: str,
    representative_reason: str,
) -> str:
    candidate_meta=meta.loc[group_ids].copy()
    candidate_expr=expression.reindex(columns=candidate_meta.index)

    candidate_meta["quality_score"]=_annotation_score(
        candidate_meta,
        candidate_expr,
        modality=modality,
    )

    meta.loc[candidate_meta.index,"quality_score"]=(
        candidate_meta["quality_score"]
    )

    ranked=candidate_meta.sort_values(
        ["quality_score","area_max","rsd_qc","rt"],
        ascending=[False,False,True,True],
    )
    winner = ranked.index[0]
    # Map phase label to SOP step name
    _phase_step_map = {
        "unnamed_formula": (26, "Step 26 — Unnamed Formula+RT Deduplication"),
        "unnamed_mass_rt": (27, "Step 27 — Unnamed Mass+RT Deduplication"),
        "unnamed_adduct_graph": (28, "Step 28 — Unnamed Adduct/Isotope Graph Collapse"),
    }
    step_num, step_label = _phase_step_map.get(phase, (25, f"Deduplication ({phase})"))
    for loser in ranked.index[1:]:
        artifacts.dedup_log.append(
            {
                "phase": phase,
                "representative": winner,
                "dropped_feature": loser,
                "reason": representative_reason,
            }
        )
        _loser_row = meta.loc[loser] if loser in meta.index else None
        _record_drop(
            artifacts, "feature", loser,
            step=step_num, step_name=step_label,
            polarity=str(meta.at[loser, "polarity"]) if "polarity" in meta.columns else "",
            reason=f"{representative_reason}; representative='{winner}'",
            annotation_name=str(_loser_row["annotation_name"]) if _loser_row is not None and "annotation_name" in _loser_row.index else "",
            formula=str(_loser_row["formula"]) if _loser_row is not None and "formula" in _loser_row.index else "",
            mz=float(_loser_row["mz"]) if _loser_row is not None and "mz" in _loser_row.index and pd.notna(_loser_row["mz"]) else None,
            rt_min=float(_loser_row["rt"]) if _loser_row is not None and "rt" in _loser_row.index and pd.notna(_loser_row["rt"]) else None,
            dedup_phase=phase,
            representative_feature=winner,
        )
    return winner

#Applying the grouping/selection steps
    #Formula-Based, Mass-Based, then Adduct-Based
def _resolve_unnamed_groups(
    run: PolarityRun,
    unnamed_meta: pd.DataFrame,
    expression: pd.DataFrame,
    modifications: pd.DataFrame,
    artifacts: PipelineArtifacts,
) -> pd.DataFrame:
    if run.modality is not Modality.METABOLOMICS:
        raise ValueError(
            "Steps 26-28 are metabolomics-only and must not run for lipidomics."
        )

    if unnamed_meta.empty:
        return unnamed_meta
    meta = unnamed_meta.copy()
    meta["quality_score"] = _annotation_score(meta, expression, modality = run.modality)

    keep: set[str] = set()

    #Step 26: Formula-based grouping (unnamed features with formulas)
    formula_meta = meta[meta["formula"].astype(str).str.strip() != ""].copy()
    for cluster in _cluster_indices(
        formula_meta,
        mz_tol=MASS_TOLERANCE_NON_PARENT,
        rt_tol=RT_TOLERANCE,
        require_formula=True,
    ):
        keep.add(
            _collapse_group_by_quality(
                cluster, meta, expression, run.modality, artifacts, "unnamed_formula", "formula+RT collapse"
            )
        )

    #Step 27: Mass+RT Grouping (unnamed features without formulas)
    no_formula = meta[meta["formula"].astype(str).str.strip() == ""].copy()
    for cluster in _cluster_indices(
        no_formula,
        mz_tol=MASS_TOLERANCE_DEDUP_NO_FORMULA,
        rt_tol=RT_TOLERANCE,
        require_formula=False,
    ):
        keep.add(
            _collapse_group_by_quality(
                cluster, meta, expression, run.modality, artifacts, "unnamed_mass_rt", "mz+RT collapse"
            )
        )

    #Construct reduced meta table of remaining features
    reduced = meta.loc[sorted(keep)] if keep else meta.iloc[0:0].copy()
    if reduced.empty:
        return reduced

    #STEP 28: Adduct/Isotope/Fragment Collapse on remaining unnamed features
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
            if right_rt - left_rt > RT_TOLERANCE:
                break
            if abs(right_rt - left_rt) > RT_TOLERANCE:
                continue
            delta = abs(right_mz - left_mz)
            matched = modifications.loc[
                (modifications["Delta_m/z"].abs() - delta).abs()
                <= MASS_TOLERANCE_NON_PARENT
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
                component, reduced, expression, run.modality, artifacts, "unnamed_adduct_graph", "adduct/isotope/fragment graph collapse",
            )
        )
    ordered_winners = sorted(set(winners), key=winners.index)
    return reduced.loc[ordered_winners].copy()

def _build_dedup_units(
    feature_meta:pd.DataFrame,
    modality:Modality,
    polarity:str,
    artifacts:PipelineArtifacts,
)->pd.DataFrame:
    meta=feature_meta.copy()

    # Metabolomics: one named compound group per CD Name.
    if modality is Modality.METABOLOMICS:
        if "annotation_name" not in meta.columns:
            raise ValueError(
                "Metabolomics dedup requires 'annotation_name'."
            )

        names=meta["annotation_name"].fillna("").astype(str).str.strip()

        valid=(
            names.ne("")
            &~names.str.lower().isin({"not named","nan"})
        )

        meta["dedup_unit_id"]=np.where(
            valid,
            names,
            meta.index.astype(str),
        )

    # Lipidomics: preserve LipidGroups, but deduplicate independently by LipidID.
    elif modality is Modality.LIPIDOMICS:
        lipid_id_col=next(
            (
                col for col in [
                    "LipidID",
                    "lipid_id_norm",
                    "lipidid",
                    "annotation_name",
                ]
                if col in meta.columns
            ),
            None,
        )

        lipid_group_col=next(
            (
                col for col in [
                    "LipidGroup",
                    "lipid_group",
                    "compound_group_name",
                ]
                if col in meta.columns
            ),
            None,
        )

        if lipid_id_col is None:
            raise ValueError(
                "Lipidomics dedup requires LipidID or equivalent lipid ID."
            )

        if lipid_group_col is None:
            raise ValueError(
                "Lipidomics dedup requires LipidGroup, lipid_group, "
                "or compound_group_name."
            )

        meta["lipid_id_norm"]=(
            meta[lipid_id_col]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        meta["LipidGroup"]=(
            meta[lipid_group_col]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        meta["isobaric_cluster"]=False
        meta["isobaric_cluster_lipid_ids"]=""

        # Flag parent LipidGroups containing multiple unique LipidIDs.
        for lipid_group,isobaric_group in meta.groupby(
            "LipidGroup",
            dropna=False,
        ):
            lipid_ids=(
                isobaric_group["lipid_id_norm"]
                .replace("",np.nan)
                .dropna()
                .unique()
            )

            if len(lipid_ids)>1:
                lipid_id_text=";".join(sorted(lipid_ids.tolist()))

                meta.loc[
                    isobaric_group.index,
                    "isobaric_cluster",
                ]=True

                meta.loc[
                    isobaric_group.index,
                    "isobaric_cluster_lipid_ids",
                ]=lipid_id_text

                artifacts.method_log.append(
                    f"{polarity} {modality.name}: "
                    f"LipidGroup '{lipid_group}' isobaric_cluster; "
                    f"LipidIDs={lipid_id_text}. "
                    "Each LipidID is processed independently."
                )

        # One unique LipidID equals one Steps 20-25 dedup unit.
        valid=(
            meta["lipid_id_norm"].ne("")
            &~meta["lipid_id_norm"].str.lower().isin(
                {"not named","nan"}
            )
        )

        meta["dedup_unit_id"]=np.where(
            valid,
            meta["lipid_id_norm"],
            meta.index.astype(str),
        )

    else:
        raise ValueError(f"Unsupported modality: {modality}")

    return meta

def _run_part3_named_dedup(
    feature_meta:pd.DataFrame,
    expression:pd.DataFrame,
    modality:Modality,
    polarity:str,
    modifications:pd.DataFrame,
    artifacts:PipelineArtifacts,
):
    meta=_build_dedup_units(
        feature_meta,
        modality,
        polarity,
        artifacts,
    ).copy()

    kept=[]
    dropped=[]

    meta["quality_score"]=np.nan
    meta["rt_flag"]=""

    for unit_id,group in meta.groupby("dedup_unit_id",dropna=False):

        if modality is Modality.LIPIDOMICS:
            lipid_id_col=next(
                (
                    col for col in ["lipid_id_norm","LipidID","lipidid"]
                    if col in group.columns
                ),
                None,
            )

            if lipid_id_col is None:
                raise ValueError(
                    f"{polarity} lipidomics unit '{unit_id}' is missing LipidID."
                )

            lipid_ids=(
                group[lipid_id_col]
                .fillna("")
                .astype(str)
                .str.strip()
                .replace("",np.nan)
                .dropna()
                .unique()
            )

            if len(lipid_ids)!=1:
                raise ValueError(
                    f"{polarity} lipidomics unit '{unit_id}' contains "
                    f"{len(lipid_ids)} LipidIDs: {sorted(lipid_ids.tolist())}. "
                    "Split the isobaric cluster by LipidID before Step 22."
                )

        candidate_meta,early_drops,rt_flag,drop_reasons=(
            _filter_and_flag_dedup_unit(
                group=group,
                modality=modality,
                polarity=polarity,
                modifications=modifications,
            )
        )

        group_name=(
            str(group["dedup_unit_id"].iloc[0])
            if not group.empty
            else str(unit_id)
        )

        # Step 23 log for all units that proceed beyond Step 22.
        if rt_flag not in {
            "single_parent_final_selection",
            "no_valid_nonparent_candidates",
        }:
            artifacts.dedup_log.append(
                {
                    "phase":"step23_rt_flag",
                    "polarity":polarity,
                    "group_name":group_name,
                    "representative":"",
                    "dropped_feature":"",
                    "rt_flag":rt_flag,
                }
            )

        # Step 22 removal logs.
        for loser in early_drops:
            row=meta.loc[loser] if loser in meta.index else None
            reason=drop_reasons.get(loser,"Step 22: candidate removed.")

            artifacts.dedup_log.append(
                {
                    "phase":"step22_feature_filtering",
                    "polarity":polarity,
                    "group_name":group_name,
                    "representative":"",
                    "dropped_feature":loser,
                    "rt_flag":rt_flag,
                    "reason":reason,
                }
            )

            _record_drop(
                artifacts,
                "feature",
                loser,
                step=22,
                step_name="Step 22 — Feature Filtering",
                polarity=polarity,
                reason=reason,
                annotation_name=(
                    str(row.get("annotation_name",""))
                    if row is not None
                    else ""
                ),
                formula=(
                    str(row.get("formula",""))
                    if row is not None
                    else ""
                ),
                mz=(
                    float(row["mz"])
                    if row is not None and pd.notna(row.get("mz"))
                    else None
                ),
                rt_min=(
                    float(row["rt"])
                    if row is not None and pd.notna(row.get("rt"))
                    else None
                ),
                dedup_phase="step22_feature_filtering",
            )

        dropped.extend(early_drops)

        # All candidates were Unknown/Measurement Error.
        if candidate_meta.empty:
            artifacts.qc_warnings.append(
                f"{polarity}: dedup unit '{unit_id}' had no valid "
                "candidates after Step 22."
            )
            continue

        # Exactly one Parent is the final Step-22 selection.
        if rt_flag=="single_parent_final_selection":
            best_id=candidate_meta.index[0]

            kept.append(best_id)
            meta.loc[best_id,"rt_flag"]=rt_flag

            continue

        candidate_expr=expression.reindex(columns=candidate_meta.index)

        scores=_annotation_score(
            meta=candidate_meta,
            expression=candidate_expr,
            modality=modality,
        )

        candidate_meta["quality_score"]=scores
        candidate_meta["rt_flag"]=rt_flag

        meta.loc[candidate_meta.index,"quality_score"]=scores
        meta.loc[candidate_meta.index,"rt_flag"]=rt_flag

        ranked=candidate_meta.sort_values(
            ["quality_score","area_max","rsd_qc","rt"],
            ascending=[False,False,True,True],
        )

        best_id=ranked.index[0]
        kept.append(best_id)

        # These are valid Step-22 survivors that lose at Steps 24-25.
        unit_drops=[
            idx for idx in candidate_meta.index
            if idx!=best_id
        ]

        dropped.extend(unit_drops)

        for loser in unit_drops:
            row=meta.loc[loser] if loser in meta.index else None

            artifacts.dedup_log.append(
                {
                    "phase":"named_within_compound",
                    "polarity":polarity,
                    "group_name":group_name,
                    "representative":best_id,
                    "dropped_feature":loser,
                    "rt_flag":rt_flag,
                }
            )

            _record_drop(
                artifacts,
                "feature",
                loser,
                step=24,
                step_name="Step 24 — Quality Scoring",
                polarity=polarity,
                reason=(
                    f"Lower quality than representative '{best_id}' "
                    f"in named group '{group_name}' (rt_flag={rt_flag})"
                ),
                annotation_name=(
                    str(row.get("annotation_name",""))
                    if row is not None
                    else ""
                ),
                formula=(
                    str(row.get("formula",""))
                    if row is not None
                    else ""
                ),
                mz=(
                    float(row["mz"])
                    if row is not None and pd.notna(row.get("mz"))
                    else None
                ),
                rt_min=(
                    float(row["rt"])
                    if row is not None and pd.notna(row.get("rt"))
                    else None
                ),
                dedup_phase="named_within_compound",
                representative_feature=best_id,
            )

    return kept,dropped,meta

"""
Part 4: Metadata Integration and Annotation
"""
###STEPS 29-31: Final Feature ID and Schymanski Level Assignment
def _finalize_feature_ids(
    meta: pd.DataFrame,
    database: str,
    istd_names: set[str],
) -> pd.DataFrame:
    meta = meta.copy()
    final_ids: list[str] = []
    schymanski: list[str] = []

    #STEP 29: Replacing placeholder feature IDs with final feature IDs
    for fid, row in meta.iterrows():
        polarity = str(row.get("polarity","")).strip()

        raw_name = row.get("compound_group_name", row.get("annotation_name",""))
        if pd.isna(raw_name):
            raw_name = row.get("annotation_name","")
        name = "" if pd.isna(raw_name) else str(raw_name).strip()

        raw_formula = row.get("formula","")
        formula = "" if pd.isna(raw_formula) else str(raw_formula).strip()

        mz = pd.to_numeric(row.get("mz",np.nan), errors="coerce")
        rt = pd.to_numeric(row.get("rt",np.nan), errors="coerce")
        mzcloud_confidence = pd.to_numeric(row.get("mzcloud_confidence",np.nan), errors="coerce")
        annot_matches = pd.to_numeric(row.get("annotation_source_matches",0), errors="coerce")
        if pd.isna(annot_matches):
            annot_matches = 0
        mass_list_matches = "" if pd.isna(row.get("mass_list_matches","")) else str(row.get("mass_list_matches","")).strip()

        if name and name.lower() not in {"not named","nan"}:
            final_id = f"{name}_{polarity}"
        else:
            mz_text = f"{mz:.4f}" if np.isfinite(mz) else "nan"
            rt_text = f"{rt:.2f}" if np.isfinite(rt) else "nan"
            #Encode Polarity into final_feature_id
            final_id = f"unk_{fid}_{polarity}_{mz_text}_RT{rt_text}"
        final_ids.append(final_id)

        #STEP 31: Assigning Schymanski levels
        norm_name = _normalise_name(name)
        if norm_name in istd_names or "istd" in norm_name:
            schymanski.append("Level 1")
        elif name and name.lower() not in {"not named","nan"}:
            if (
                pd.notna(mzcloud_confidence)
                and mzcloud_confidence >= MZCLOUD_L2_THRESHOLD
            ) or annot_matches >= ANNOT_SOURCE_L2_THRESHOLD:
                schymanski.append("Level 2")
            else:
                schymanski.append("Level 3")
        elif formula and formula.lower() != "nan":
            candidates = [
                token.strip()
                for token in re.split(r"[;|]", mass_list_matches)
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

"""
Part 5: Pipeline Validation and Sanity Checks
"""
###STEP 34: Biological Trajectory Plots for Longitudinal Tissues
def _trajectory_plots(
    final_matrix: pd.DataFrame,
    feature_meta: pd.DataFrame,
    output_dir: Path,
    config: DatasetConfig,
    artifacts: PipelineArtifacts,
) -> None:
    if config.tissue.lower() == "placenta":
        artifacts.method_log.append(
            "Step 34 trajectory plots skipped: placenta is a delivery-only tissue."
        )
        return
    
    if "SampleGestAge" not in final_matrix.columns:
        return

    modality = config.modality.name.lower() if hasattr(config.modality,"name") else str(config.modality).lower()

    analyte_names = {
        fid: str(
            feature_meta.at[fid,"annotation_name"]
            if "annotation_name" in feature_meta.columns
            and fid in feature_meta.index
            and pd.notna(feature_meta.at[fid,"annotation_name"])
            and str(feature_meta.at[fid,"annotation_name"]).strip() != ""
            else feature_meta.at[fid,"lipidid"]
            if "lipidid" in feature_meta.columns
            and fid in feature_meta.index
            and pd.notna(feature_meta.at[fid,"lipidid"])
            else ""
        ).lower()
        for fid in feature_meta.index
    }

    if modality == "lipidomics":
        targets = (
            "lpc(16:0", # lysophosphatidylcholine 16:0
            "cer(d18:1_16:0", # ceramide d18:1_16:0
        )
        warn_msg = f"{config.dataset_id}: no curated pregnancy-trajectory lipids were retained."
    else:
        targets = ("l-threonine", "l-kynurenine", "palmitoylcarnitine")
        warn_msg = f"{config.dataset_id}: no curated pregnancy-trajectory metabolites were retained."

    selected = [
        fid for fid, name in analyte_names.items() if any(keyword in name for keyword in targets)
    ]
    if not selected:
        artifacts.qc_warnings.append(warn_msg)
        return

    traj_dir = output_dir / "trajectory_plots"
    _ensure_dir(traj_dir)
    #continuous gest age plots
    # Continuous gestational-age plots
    x = pd.to_numeric(final_matrix["SampleGestAge"], errors="coerce")
    if "SampleTimepoint" not in final_matrix.columns:
        artifacts.qc_warnings.append(
            f"{config.dataset_id}: SampleTimepoint is missing; "
            "binned trajectory plots were skipped."
        )
        return
    timepoint = final_matrix["SampleTimepoint"].astype("string")

    for fid in selected[:6]:
        if fid not in final_matrix.columns:
            continue
        y = pd.to_numeric(final_matrix[fid], errors="coerce")

        # 1) continuous scatter
        valid_cont = x.notna() & y.notna()
        if valid_cont.sum() >= 3:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.scatter(x[valid_cont], y[valid_cont], alpha=0.7)
            ax.set_title(f"{analyte_names[fid]} trajectory (continuous)")
            ax.set_xlabel("Gestational age at collection (weeks)")
            ax.set_ylabel("Processed abundance")
            fig.tight_layout()
            fig.savefig(traj_dir / f"{fid}_continuous.png", dpi=180)
            plt.close(fig)

        # 2) Binned boxplot by longitudinal bins 1–5 and Delivery
        valid_bin = timepoint.notna() & y.notna()
        if valid_bin.sum() < 3:
            continue

        tp_values = timepoint.loc[valid_bin].astype(str)
        y_values = y.loc[valid_bin]

        timepoint_order = ["1", "2", "3", "4", "5", "Delivery"]
        unique_tps = [
            tp for tp in timepoint_order
            if tp in set(tp_values)
        ]

        if not unique_tps:
            continue

        data_by_tp = [
            y_values[tp_values == tp]
            for tp in unique_tps
        ]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.boxplot(data_by_tp, tick_labels=unique_tps)
        ax.set_title(f"{analyte_names[fid]} trajectory (binned)")
        ax.set_xlabel("Gestational timepoint")
        ax.set_ylabel("Processed abundance")
        fig.tight_layout()
        fig.savefig(traj_dir / f"{fid}_binned.png", dpi=180)
        plt.close(fig)

#STEP 35: Writing the SOP Pipeline Log
def _write_plain_text_log(
    output_dir: Path,
    config: DatasetConfig,
    artifacts: PipelineArtifacts,
    sample_matrix: pd.DataFrame,
    feature_meta: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    missingness_reports: dict[str, pd.DataFrame],
    confounding_rows: list[dict],
) -> None:
    """Write the SOP Step 35 pipeline log.

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

    # ── Step 4(b): remove ISTD features after normalization ───────────────────
    lines.append("")
    lines.append("ISTD REMOVAL (Step 4b)")
    lines.append("-" * 40)

    istd_drops = [
        entry
        for entry in artifacts.drop_log
        if entry.get("step") == 4
        and entry.get("entity_type") == "feature"
    ]

    if istd_drops:
        drops_by_polarity: dict[str, list[str]] = {}
        for entry in istd_drops:
            polarity = entry.get("polarity", "?")
            feature_id = entry.get("entity_id", "?")
            drops_by_polarity.setdefault(polarity, []).append(feature_id)

        for polarity, feature_ids in sorted(drops_by_polarity.items()):
            lines.append(
                f"  {polarity}: removed {len(feature_ids)} ISTD feature(s) "
                f"after normalization ({', '.join(feature_ids)})."
            )
    else:
        lines.append("  No ISTD features were removed.")

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
    lines.append("PER-STEP FEATURE AND SAMPLE COUNTS (Steps 7–18, Parts 3A–3B, Step 30)")
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
    
    # ── Delivery-timing flags ─────────────────────────────────────────────────
    lines.append("")
    lines.append("DELIVERY-TIMING FLAGS (metadata audit)")
    lines.append("-" * 40)

    flag_cols = [
        "within_0_1wk_delivery_flag",
        "post_birth_sample_flag",
    ]

    for flag in flag_cols:
        if flag not in sample_metadata.columns:
            lines.append(f"  {flag}: not available")
            continue

        flagged_ids = sample_metadata.index[
            sample_metadata[flag].eq(1).fillna(False)
        ].tolist()

        lines.append(f"  {flag}: {len(flagged_ids)} flagged")
        for sample_id in flagged_ids:
            lines.append(f"    - {sample_id}")

    # ── Samples flagged per filtering step ───────────────────────────────────
    lines.append("")
    lines.append("SAMPLES FLAGGED BY STEP 8: SAMPLE-LEVEL MISSINGNESS")
    lines.append("-" * 40)
    sample_steps: dict[tuple[str, int], list[str]] = {}
    for entry in artifacts.sample_filter_log:
        key = (entry.get("polarity", "?"), int(entry.get("step", 0)))
        sample_steps.setdefault(key, []).append(
            f"{entry.get('sample_name', entry.get('row_id', '?'))} "
            f"({entry.get('reason', '')})"
        )
    if sample_steps:
        step_labels = {8: "Step 8 — sample missingness", 16: "Step 16 — ISTD MAD (post-ComBat)"}
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
            7: "Step 7  — feature missingness",
            17: "Step 17 — QC RSD (per-batch, post-ComBat)",
            18: "Step 18 — IQR filter (within-timepoint)",
        }
        for (pol, step), count in sorted(feat_by_step.items()):
            label = step_labels_feat.get(step, f"Step {step}")
            lines.append(f"  {pol} {label}: {count:,} feature(s) removed")
    else:
        lines.append("  No features were removed during filtering.")

    # ── Deduplication ─────────────────────────────────────────────────────────
    lines.append("")
    lines.append("DEDUPLICATION (Steps 20–28)")
    lines.append("-" * 40)
    lines.append(f"  Total deduplication drop events: {len(artifacts.dedup_log):,}")

    # ── Batch correction ──────────────────────────────────────────────────────
    lines.append("")
    lines.append("BATCH CORRECTION (Step 13)")
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
        lines.append("BATCH-CONFOUNDING CHECKS (Step 12)")
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

"""
PROCESSING FUNCTION FOR SINGLE RUN
"""
def _process_polarity(
    run: PolarityRun,
    sample_metadata: pd.DataFrame,
    ref_batch: str,
    modifications: pd.DataFrame,
    config: DatasetConfig, 
    artifacts: PipelineArtifacts,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    diag_dir = config.pca_dir / "diagnostics" / run.polarity.lower()
    _ensure_dir(diag_dir)

    artifacts.drift_flags.extend(
        _plot_istd_diagnostics(run, diag_dir / "pre_norm", f"{run.polarity}_pre")
    )

    # ── Step 35 tracking: initial counts before any filtering ────────────────
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
    # ─────────────────────────────────────────────────────────────────────────

    normalized, removed_istds = _istd_normalize(run, artifacts)
    artifacts.istd_names[pol] = removed_istds  # Step 35: record ISTD names used
    _snap("entry (post-ISTD removal)", normalized, run.sample_info)

    '''
    NOTE: Step 5 is removed for DP3, since ISTD normalization is being performed.
    '''
    artifacts.drift_flags.extend(
        _plot_post_normalization_diagnostics(
            run, normalized, diag_dir / "post_norm", f"{run.polarity}_post", removed_istds,
        )
    )
    # Add analysis metadata to pipeline-row sample information.
    # The A–E terminal suffix denotes repeated collection within participant.
    sample_info = run.sample_info.copy()

    sample_info["SubjectID"] = sample_info["sample_name"].astype(str).str.replace(
        r"[A-E]$",
        "",
        regex=True,
    )

    meta_for_samples = sample_metadata.reindex(
        sample_info["sample_name"]
    ).copy()
    meta_for_samples.index = sample_info.index

    metadata_cols = [
        "GestAgeDelivery",
        "SampleGestAge",
        "SampleTimepoint",
    ]

    available_metadata_cols = [
        col for col in metadata_cols
        if col in meta_for_samples.columns
    ]

    sample_info = sample_info.join(
        meta_for_samples[available_metadata_cols],
        how="left",
    )
    filtered, missingness_report = _feature_missingness_filter(
        normalized,
        sample_info,
        sample_metadata,
        artifacts,
        run.polarity,
        config.tissue,
        feature_meta=run.feature_meta,
    )
    _snap("after Step 7 (feature missingness)", filtered, sample_info)

    # ── Step 8: Sample missingness filter (moved before log2 in SOP v4) ──────
    filtered, sample_info = _sample_missingness_filter(
        filtered, sample_info, artifacts, run.polarity, tissue = config.tissue
    )
    _snap("after Step 8 (sample missingness)", filtered, sample_info)

    # ── Step 9: Log2 transform  ───────────────────────────────────────────────
    log2_df=log2_transform(filtered)
    
    # -- Step 10: Half-minimum imputation ──────────────────────────────────────
    imputed = half_min_impute_wide(log2_df)

    # ── Step 11: Pre-correction PCA ───────────────────────────────────────────
    batch_labels = sample_info["batch"]
    _pca_plot(
        imputed,
        batch_labels,
        f"{run.polarity} pre-correction PCA (batch)",
        diag_dir / "pca_pre_correction_batch.png",
    )

    # ── Step 12: Batch-confounding checks ─────────────────────────────────────
    # Build group labels for biological samples only.
    # QC pools have no Group metadata so restricting to biological rows prevents
    # them from appearing as "Unknown" in the group PCA.
    bio_mask = sample_info["sample_type"] == "biological"
    bio_sample_info = sample_info.loc[bio_mask]
    groups = sample_metadata.reindex(bio_sample_info["sample_name"])["Group"].fillna("Unknown")
    groups.index = bio_sample_info.index

    confounding_rows = _batch_confounding_checks(sample_info, sample_metadata, artifacts)

    # ── Step 13: ComBat batch correction ──────────────────────────────────────
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

    # ── Step 14: Post-correction PCA ──────────────────────────────────────────
    _pca_plot(
        corrected,
        batch_labels,
        f"{run.polarity} post-correction PCA (batch)",
        diag_dir / "pca_post_correction_batch.png",
    )
    # Group PCA restricted to biological samples so QC pools are excluded
    _pca_plot(
        corrected.loc[bio_mask],
        groups,
        f"{run.polarity} post-correction PCA (group)",
        diag_dir / "pca_post_correction_group.png",
    )

    # ── Step 15: Post-ComBat intensity check (SOP v4 new step) ───────────────
    post_combat_flags = _post_combat_intensity_check(
        run, corrected, sample_info, diag_dir, f"{run.polarity}_postcombat"
    )
    artifacts.drift_flags.extend(post_combat_flags)

    # ── Step 16: ISTD MAD filter (moved post-ComBat in SOP v4) ───────────────
    bad_samples = sample_istd_mad_filter(run, sample_info, artifacts, run.modality)
    if bad_samples:
        keep_mask = ~sample_info.index.isin(bad_samples)
        corrected = corrected.loc[keep_mask]#.copy()
        sample_info = sample_info.loc[keep_mask]#.copy()
    _snap("after Step 16 (sample ISTD MAD, post-ComBat)", corrected, sample_info)

    # ── Step 17: RSD filter on QC pools (post-ComBat in SOP v4) ─────────────
    corrected = _rsd_filter(corrected, sample_info, artifacts, run.polarity, feature_meta=run.feature_meta)
    _snap("after Step 17 (QC RSD filter, post-ComBat)", corrected, sample_info)

    # ── Step 18: IQR filter within-timepoint (SOP v4 new step) ───────────────
    corrected = _iqr_filter(corrected, sample_info, artifacts, run.polarity, feature_meta=run.feature_meta)
    _snap("after Step 18 (IQR filter, within-timepoint)", corrected, sample_info)

    named_kept, unnamed = _resolve_named_groups(
        run, corrected, modifications, artifacts
    )
    n_bio = int((sample_info["sample_type"] == "biological").sum())

    # Step 35 count: features remaining after Part 3A = named survivors + all unnamed (not yet processed)
    artifacts.step_counts.append(
        {
            "polarity": pol,
            "label": "after Part 3A (named-compound dedup)",
            "n_features": len(named_kept) + len(unnamed),
            "n_samples": n_bio,
        }
    )

    if run.modality is Modality.METABOLOMICS:
        unnamed_kept = _resolve_unnamed_groups(
            run,
            unnamed,
            corrected,
            modifications,
            artifacts,
        )
    else:
        unnamed_kept = unnamed.copy()

        artifacts.method_log.append(
            f"{run.polarity} lipidomics: Steps 26-28 skipped; "
            "unnamed-feature cross-deduplication is metabolomics-only."
        )

    retained_meta = pd.concat([named_kept, unnamed_kept], axis=0)
    retained_meta = retained_meta[~retained_meta.index.duplicated(keep="first")]

    # Step 35 count: features remaining after Part 3B = named survivors + unnamed survivors
    artifacts.step_counts.append({
        "polarity": pol,
        "label": "after Part 3B (unnamed dedup)",
        "n_features": len(retained_meta),
        "n_samples": n_bio,
    })

    retained_expression = corrected[retained_meta.index.tolist()].copy()
    run.retained_feature_meta = retained_meta
    run.retained_expression = retained_expression
    return retained_expression, retained_meta, sample_info, missingness_report, confounding_rows

def run_dataset(
    config: DatasetConfig,
    meta_path: Path,
) -> None:
    artifacts = PipelineArtifacts()
    _ensure_dir(config.output_dir)
    _ensure_dir(config.pca_dir)
    _ensure_dir(config.trajectory_dir)
    sample_metadata = _load_metadata(meta_path, config)
    modifications = _load_modifications(config)
    ref_batch = ""

    if config.raw_workbook and config.raw_sheet_pos and config.raw_sheet_neg:
        pos_files = [
            config.input_dir / "pos_expression.csv",
            config.input_dir / "pos_batch.csv",
            config.input_dir / "pos_compounds.csv",
        ]
        neg_files = [
            config.input_dir / "neg_expression.csv",
            config.input_dir / "neg_batch.csv",
            config.input_dir / "neg_compounds.csv",
        ]

    pos_run = _load_polarity_run(config.input_dir, "pos", get_istd_names(config.modality,"POS"), config)
    neg_run = _load_polarity_run(config.input_dir, "neg", get_istd_names(config.modality,"NEG"), config)

    for run in (pos_run, neg_run):
        expected_istds = {
            _normalise_name(name)
            for name in get_istd_names(config.modality, run.polarity)
        }

        detected_istds = set()

        for feature_id in run.raw_istd.columns:
            detected_istds.add(_normalise_name(feature_id))

            if feature_id in run.feature_meta.index:
                detected_istds.add(
                    _normalise_name(
                        run.feature_meta.at[feature_id, "annotation_name"]
                    )
                )

        missing_istds = expected_istds - detected_istds

        if missing_istds:
            artifacts.qc_warnings.append(
                f"{config.dataset_id} {run.polarity}: configured ISTD feature(s) "
                f"not detected in the input export: {sorted(missing_istds)}. "
                f"Proceeding with detected ISTDs only. "
                f"Detected ISTD labels: {sorted(detected_istds)}"
            )
        
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
        modifications,
        config,
        artifacts,
    )
    neg_expression, neg_meta, neg_samples, neg_missing, neg_conf = _process_polarity(
        neg_run,
        sample_metadata,
        ref_batch,
        modifications,
        config,
        artifacts,
    )

    final_matrix, feature_meta = _merge_polarities(
        [
            (pos_run, pos_expression, pos_samples),
            (neg_run, neg_expression, neg_samples),
        ],
        sample_metadata,
        config,
        artifacts,
    )
    # Step 35 count: features after POS/NEG merge (Step 30)
    artifacts.step_counts.append({
        "polarity": "MERGED",
        "label": "after POS/NEG merge (Step 30)",
        "n_features": feature_meta.shape[0],
        "n_samples": len(final_matrix),
    })

    # -------------------------------------------------------------------------
    # Export final abundance matrix and matching feature-level metadata.
    # The index of feature_meta must exactly match analyte column names in
    # final_matrix so downstream differential analysis can filter by
    # schymanski_level after preprocessing is complete.
    # -------------------------------------------------------------------------
    feature_meta.index = feature_meta.index.astype(str)
    feature_meta.index.name = "analyte_id"

    sample_metadata_columns = {
        "SubjectID",
        "SampleID",
        "Group",
        "Subgroup",
        "Batch",
        "GestAgeDelivery",
        "SampleGestAge",
        "SampleTimepoint",
        "MetadataCanonicalID",

        # Clinical / derived sample-level metadata.
        "labor_onset",
        "indicated_onset",
        "spont_labor_flag",
        "indicated_onset_flag",
        "cat_labor_onset_flag",
        "within_0_1wk_delivery_flag",
        "post_birth_sample_flag",
    }

    matrix_analytes = [
        str(column)
        for column in final_matrix.columns
        if column not in sample_metadata_columns
    ]

    metadata_analytes = feature_meta.index.tolist()

    missing_from_feature_metadata = sorted(
        set(matrix_analytes) - set(metadata_analytes)
    )

    missing_from_final_matrix = sorted(
        set(metadata_analytes) - set(matrix_analytes)
    )

    if missing_from_feature_metadata or missing_from_final_matrix:
        raise ValueError(
            f"{config.dataset_id}: final abundance matrix and feature metadata "
            "are not aligned. "
            f"Analytes in matrix but absent from feature metadata "
            f"(first 20): {missing_from_feature_metadata[:20]}; "
            f"features in metadata but absent from matrix "
            f"(first 20): {missing_from_final_matrix[:20]}."
        )

    if "schymanski_level" not in feature_meta.columns:
        raise KeyError(
            f"{config.dataset_id}: final feature metadata does not contain "
            "'schymanski_level'. Cannot perform downstream confidence-level "
            "filtering."
        )

    feature_meta["schymanski_level"] = pd.to_numeric(
        feature_meta["schymanski_level"]
        .astype("string")
        .str.extract(r"(\d+)")[0],
        errors="coerce",
    ).astype("Int64")

    full_path = (
        config.output_dir
        / f"{config.dataset_id}_cleaned_with_metadata.csv"
    )

    feature_path = (
        config.output_dir
        / f"{config.dataset_id}_feature_metadata.csv"
    )

    final_matrix.to_csv(full_path)
    feature_meta.to_csv(feature_path, index=True)

    LOGGER.info(
        "[%s] exported %d analytes and matching feature metadata with "
        "'schymanski_level' -> %s",
        config.dataset_id,
        len(matrix_analytes),
        feature_path,
    )

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
    if artifacts.drop_log:
        # Comprehensive per-entity drop log — never printed to terminal.
        # One row per dropped feature or sample, with full context fields.
        _drop_col_order = [
            "entity_type", "entity_id", "step", "step_name", "polarity", "reason",
            "annotation_name", "formula", "mz", "rt_min",
            "metric_value", "metric_threshold",
            "batch", "timepoint",
            "dedup_phase", "representative_feature",
        ]
        pd.DataFrame(artifacts.drop_log)[_drop_col_order].to_csv(
            config.output_dir / f"{config.dataset_id}_comprehensive_drop_log.csv", index=False
        )

    sample_meta_for_final = sample_metadata.reindex(final_matrix.index)

    # Plasma and urine: export prenatal timepoint-specific files.
    # Placenta: delivery-only cross-sectional tissue; export only the already
    # written <dataset>_cleaned_with_metadata.csv and no T1–T5 files.
    if config.tissue.lower() in {"plasma", "urine"}:
        if "SampleGestAge" not in sample_meta_for_final.columns:
            raise KeyError(
                f"{config.dataset_id}: SampleGestAge is required to export "
                "plasma/urine timepoint-specific matrices."
            )

        sample_gest_age = pd.to_numeric(
            sample_meta_for_final["SampleGestAge"],
            errors="coerce",
        )

        sample_timepoint = sample_gest_age.map(
            categorize_sample_time
        )

        for tp in sorted(sample_timepoint.dropna().unique()):
            tp_mask = sample_timepoint.eq(tp)

            if not tp_mask.any():
                continue

            tp_df = final_matrix.loc[tp_mask].copy()
            tp_df.index.name = "SampleID"

            tp_path = (
                config.output_dir
                / f"{config.dataset_id}_T{int(tp)}.csv"
            )

            tp_df.to_csv(tp_path)

            LOGGER.info(
                "[%s] wrote timepoint T%d matrix: %d samples -> %s",
                config.dataset_id,
                int(tp),
                len(tp_df),
                tp_path,
            )

    elif config.tissue.lower() == "placenta":
        LOGGER.info(
            "[%s] placenta is delivery-only: no prenatal T1–T5 "
            "timepoint matrices written. Use %s for cross-sectional "
            "delivery analysis.",
            config.dataset_id,
            full_path,
        )

    else:
        LOGGER.warning(
            "[%s] unrecognized tissue %r: no timepoint-specific "
            "matrices written.",
            config.dataset_id,
            config.tissue,
        )   

    _trajectory_plots(final_matrix, feature_meta, config.trajectory_dir, config, artifacts)

    _write_plain_text_log(config.output_dir, config, artifacts, final_matrix, feature_meta, sample_metadata, 
                          {"POS": pos_missing,"NEG": neg_missing,}, pos_conf + neg_conf,
    )

    LOGGER.info(
        "[%s] wrote %s and %s",
        config.dataset_id,
        full_path,
        feature_path,
    )

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    repo_root = Path(__file__).resolve().parent
    inputs_root = Path(args.inputs_root).resolve()
    output_root = Path(args.output_root).resolve()
    metadata = Path(args.metadata).resolve()
    configs = _build_configs(repo_root, inputs_root, output_root)

    for dataset_id in args.datasets:
        if dataset_id not in configs:
            raise SystemExit(f"Unknown dataset '{dataset_id}'.")
        config = configs[dataset_id]
        LOGGER.info("Starting SOP pipeline for %s", dataset_id)
        run_dataset(config, metadata)


if __name__ == "__main__":
    main()