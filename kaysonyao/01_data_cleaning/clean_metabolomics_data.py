from utilities import (
    CUTOFF_PERCENT_MISSING,
    collect_olink_files,
    load_metadata_with_batch,
    process_single_file,
    combine_batches,
    apply_panel_normalization_long,
    is_olink_control_sample,
    is_olink_control_assay,
    missingness_filter_and_group_check,
    combat_normalize_wide,
    half_min_impute_wide
)

from clean_proteomics_data import process_all_files
import os



wd = os.getcwd()
data_dir = f"{wd}/data/cleaned/metabolomics/"
data_paths = [p for p in os.listdir(data_dir) if p.endswith(".csv")]

output_dir = f"{wd}/data/cleaned/metabolomics/normalized/"
meta_path = f"{wd}/data/dp3 master table v2.xlsx"

if __name__ == "__main__":
    process_all_files(data_paths, output_dir, meta_path, "metabolomics")


