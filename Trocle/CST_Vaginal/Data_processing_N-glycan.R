#' *CST Vaginal Data*
#' *N-Glycan Preprocessing - Combined CRIB & SPEC*
#' *Ashley Trocle | Piekos Lab*
#' *Spring 2026 Rotation*

# Load libraries
library(readxl)
library(tidyverse)

## Step 1: Data Ingestion and Structural Parsing

# Discard row setting (same for SPEC and CRIB)
rows_to_discard <- c(
  "Total Glycan Signal", "Total (Clinical)", "Total (Research)", "Total (All)",
  "Hex4/Hex9", "Hex5/Hex9", "Man7/Man9", "Man3/4", "Man5/6", "Man6/9", 
  "Mono-sialo/Di-sialo", "Calculated Di-sialo Concentration",
  "Clinical", "Research", "2+"
)

# --- CRIB Processing ---
# path
file_path <- "CRIB cervical samples N-glycan 1-14-2025_MH.xlsx"
raw_df <- read_excel(file_path, sheet = "Summary abundance", col_names = FALSE)

# Extract metadata (Row 3 = Sample ID, Row 4 = Stats)
metadata_crib <- tibble(
  Column_Index   = 1:ncol(raw_df), 
  Birth_Outcome  = as.character(raw_df[2, ]),
  Sample_ID      = as.character(raw_df[3, ]),
  Stats_Label    = as.character(raw_df[4, ])
) |> 
  filter(Column_Index > 3) |>
  filter(!Stats_Label %in% c("Medium", "SD")) |>
  filter(!is.na(Sample_ID), Birth_Outcome != "NA", Birth_Outcome != "")

# Clean abundance matrix
abundance_matrix_crib <- raw_df |> 
  select(all_of(c(1, 3, metadata_crib$Column_Index))) |> 
  rename(mz = ...1, glycan_name = ...3) |> 
  slice(-(1:4)) |> 
  filter(!(glycan_name %in% rows_to_discard)) |> 
  # Keyword filter to remove Internal Standard (1271)
  filter(!str_detect(glycan_name, "1271|IS|13C6")) |> 
  filter(!is.na(mz)) |> 
  mutate(
    mz = round(as.numeric(mz), 4),
    across(-c(mz, glycan_name), as.numeric)
  )

# Set clean names and format final objects
colnames(abundance_matrix_crib)[3:ncol(abundance_matrix_crib)] <- metadata_crib$Sample_ID

metadata_final_crib <- metadata_crib |> 
  select(Sample_ID, Birth_Outcome) |> 
  mutate(Birth_Outcome = case_when(
    tolower(Birth_Outcome) == "sptb" ~ "sPTB",
    tolower(Birth_Outcome) == "term" ~ "Term",
    TRUE ~ Birth_Outcome
  ))

glycan_matrix_crib <- abundance_matrix_crib

# --- SPEC Processing ---
# path
file_path_spec <- "SPEC 25-NGLY41-42_CervicalSwab MH expanded panel.xlsm"
raw_df_spec <- read_excel(file_path_spec, sheet = "Abundance", col_names = FALSE)

# Extract metadata (Row 3 = Outcome, Row 4 = Sample ID)
metadata_spec <- tibble(
  Column_Index   = 1:ncol(raw_df_spec),
  Birth_Outcome  = as.character(raw_df_spec[3, ]),
  Sample_ID      = as.character(raw_df_spec[4, ])
) |> 
  filter(Column_Index > 4) |> 
  filter(!Sample_ID %in% c("Avg", "STDEV", "3SD min", "3SD max", "Actual Min", 
                           "Actual Max", "2SD min", "2SD max", "Median", "QCN", "QCA")) |>
  filter(!is.na(Sample_ID))

# Clean abundance matrix
abundance_matrix_spec <- raw_df_spec |> 
  select(all_of(c(2, 4, metadata_spec$Column_Index))) |> 
  rename(mz = ...2, glycan_name = ...4) |> 
  slice(-(1:6)) |> 
  filter(!(glycan_name %in% rows_to_discard)) |> 
  # Keyword filter to remove Internal Standard (1271)
  filter(!str_detect(glycan_name, "1271|IS|13C6")) |> 
  filter(!is.na(mz)) |> 
  mutate(
    mz = round(as.numeric(mz), 4),
    across(-c(mz, glycan_name), ~ round(as.numeric(.), 2))
  )

# Set clean names and format final objects
colnames(abundance_matrix_spec)[3:ncol(abundance_matrix_spec)] <- metadata_spec$Sample_ID

metadata_final_spec <- metadata_spec |> 
  select(Sample_ID, Birth_Outcome) |>
  mutate(Birth_Outcome = case_when(
    tolower(Birth_Outcome) == "sptb" ~ "sPTB",
    tolower(Birth_Outcome) == "term" ~ "Term",
    TRUE ~ Birth_Outcome
  ))

glycan_matrix_spec <- abundance_matrix_spec


## Step 2: Zero Replacement with Missing

replace_zeros <- function(df) {
  df |> mutate(across(-c(mz, glycan_name), ~ ifelse(. == 0, NA, .)))
}

matrix_crib_na <- replace_zeros(glycan_matrix_crib)
matrix_spec_na <- replace_zeros(glycan_matrix_spec)


## Step 3: Missingness Filtering with Group-Stratified Fisher’s Exact Test

run_missingness_filter <- function(matrix_na, metadata, cohort) {
  sample_cols <- setdiff(colnames(matrix_na), c("mz", "glycan_name"))
  
  # 3a. Calculate overall proportion
  matrix_stats <- matrix_na |> 
    mutate(missing_prop = rowSums(is.na(matrix_na[sample_cols])) / length(sample_cols))
  
  # 3b. Fisher's Test for failed analytes (> 20%)
  failed <- matrix_stats |> filter(missing_prop > 0.20)
  fisher_df <- data.frame()
  
  if (nrow(failed) > 0) {
    long <- failed |> 
      pivot_longer(cols = all_of(sample_cols), names_to = "Sample_ID", values_to = "Val") |> 
      left_join(metadata, by = "Sample_ID") |> 
      mutate(is_missing = is.na(Val))
    
    # Calculate counts and run Fisher
    fisher_df <- long |> 
      group_by(mz, glycan_name) |> 
      summarize(
        m_sPTB = sum(is_missing & Birth_Outcome == "sPTB"),
        o_sPTB = sum(!is_missing & Birth_Outcome == "sPTB"),
        m_Term = sum(is_missing & Birth_Outcome == "Term"),
        o_Term = sum(!is_missing & Birth_Outcome == "Term"),
        .groups = "drop"
      ) |> 
      rowwise() |> 
      mutate(
        p_val_raw = fisher.test(matrix(c(m_sPTB, m_Term, o_sPTB, o_Term), nrow = 2))$p.value
      ) |> 
      ungroup() |> 
      # Apply Benjamini-Hochberg across all failed glycans
      mutate(p_adj_bh = p.adjust(p_val_raw, method = "BH")) |> 
      # Calculate percentages for the final output
      mutate(
        overall_perc_missing = (m_sPTB + m_Term) / (m_sPTB + m_Term + o_sPTB + o_Term),
        perc_missing_sPTB    = m_sPTB / (m_sPTB + o_sPTB),
        perc_missing_Control  = m_Term / (m_Term + o_Term),
        significance_flag    = ifelse(p_adj_bh < 0.05, "Significant", "Non-Significant")
      ) |> 
      # Rename and select columns to match Step 7 specification exactly
      select(
        `Glycan m/z` = mz,
        `Glycan name` = glycan_name,
        `Overall % missing` = overall_perc_missing,
        `% missing in sPTB` = perc_missing_sPTB,
        `% missing in Control` = perc_missing_Control,
        `Fisher’s exact test p-value (raw)` = p_val_raw,
        `Benjamini-Hochberg adjusted p-value` = p_adj_bh,
        `Significance flag` = significance_flag
      )
    
    # Write the Secondary Output
    write_csv(fisher_df, paste0(cohort, "_Fisher_Missingness_Results.csv"))
  }
  
  # 3c. Return the filtered matrix for Step 4
  filtered_mat <- matrix_stats |> filter(missing_prop <= 0.20) |> select(-missing_prop)
  return(list(mat = filtered_mat, fisher = fisher_df))
}

crib_step3 <- run_missingness_filter(matrix_crib_na, metadata_final_crib, "CRIB")
spec_step3 <- run_missingness_filter(matrix_spec_na, metadata_final_spec, "SPEC")


## Step 4: Missing Value Imputation (1/2 Minimum)

impute_half_min <- function(filtered_matrix) {
  sample_cols <- setdiff(colnames(filtered_matrix), c("mz", "glycan_name"))
  
  imputed_df <- filtered_matrix |> 
    rowwise() |> 
    mutate(across(all_of(sample_cols), ~ {
      if(is.na(.)) {
        obs_vals <- c_across(all_of(sample_cols))
        return(min(obs_vals, na.rm = TRUE) / 2)
      } else { . }
    })) |> 
    ungroup()
  
  # Ensure no zeros / NAs remain
  if (any(imputed_df[sample_cols] <= 0) | any(is.na(imputed_df[sample_cols]))) {
    stop("Non-positive or NA values")
  }
  
  return(imputed_df)
}

matrix_crib_imputed <- impute_half_min(crib_step3$mat)
matrix_spec_imputed <- impute_half_min(spec_step3$mat)


## Step 5: CLR Transformation

apply_clr <- function(imputed_df) {
  sample_cols <- setdiff(colnames(imputed_df), c("mz", "glycan_name"))
  # Formula: ln(xi) - mean(ln(x1...xD))
  clr_df <- imputed_df |> 
    mutate(across(all_of(sample_cols), ~ log(.) - mean(log(.))))
  
  return(clr_df)
}

matrix_crib_clr <- apply_clr(matrix_crib_imputed)
matrix_spec_clr <- apply_clr(matrix_spec_imputed)


## Step 6: Matrix Transposition and Metadata Integration

integrate_metadata <- function(clr_df, metadata) {
  clr_prepared <- clr_df |> 
    mutate(mz_label = paste0(round(mz, 4), "_", glycan_name)) |> 
    select(-mz, -glycan_name)
  
  transposed_df <- clr_prepared |> 
    pivot_longer(cols = -mz_label, names_to = "Sample_ID", values_to = "Abundance") |> 
    pivot_wider(names_from = mz_label, values_from = Abundance)
  
  final_df <- transposed_df |> 
    inner_join(metadata, by = "Sample_ID") |> 
    relocate(Sample_ID, Birth_Outcome)
  
  return(final_df)
}

final_crib_ready <- integrate_metadata(matrix_crib_clr, metadata_final_crib)
final_spec_ready <- integrate_metadata(matrix_spec_clr, metadata_final_spec)


## Step 7: Final Outputs

write_csv(final_crib_ready, "CRIB_Analysis_Ready_CLR.csv")
write_csv(final_spec_ready, "SPEC_Analysis_Ready_CLR.csv")

generate_processing_log <- function(step3_out, final_df, cohort) {
  log_path   <- paste0(cohort, "_preprocessing_log.txt")

  kept_count      <- nrow(step3_out$mat)
  discarded_df    <- step3_out$fisher
  discarded_count <- nrow(discarded_df)
  input_count     <- kept_count + discarded_count
    final_matrix_count <- ncol(final_df) - 2
  
  sink(log_path)
  
  cat("Input glycan count:", input_count, "\n\n")
  cat("Number of glycans discarded in Step 3:", discarded_count, "\n")
  
  if(discarded_count > 0) {
    cat("Discarded m/z values:", paste(discarded_df$`Glycan m/z`, collapse = ", "), "\n")
  } else {
    cat("Discarded m/z values: None\n")
  }
  
  cat("Final glycan count:", final_matrix_count, "\n")
  cat("Total samples included:", nrow(final_df), "\n\n")
  sink()
}

# Execute for both cohorts
generate_processing_log(crib_step3, final_crib_ready, "CRIB")
generate_processing_log(spec_step3, final_spec_ready, "SPEC")