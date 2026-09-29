#' *CST Vaginal Data*
#' *Microbiome Preprocessing*
#' *Ashley Trocle | Piekos Lab *
#' *Spring 2026 Rotation*

#load libraries
library(dplyr)
library(tidyr)
library(readr)
library(stringr)
library(ggplot2)

# Processing log (Appendix A): per-cohort sample/feature counts at each step.
# Populated as the pipeline runs; written to disk in Step 8.
processing_log <- list(
  CRIB = list(),
  SPEC = list()
)

## Step 1: Load Data and Initial Inspection

raw_data <- read_csv("all_samples.csv")

otu_col     <- "#OTU ID"
lineage_col <- "Consensus Lineage"

# Non-sample (metadata) columns carried alongside the count/CLR/RA matrices.
# Sample columns are always derived as setdiff(colnames(x), meta_cols).
meta_cols <- c("#OTU ID", "Kingdom", "Phylum", "Class", "Order",
               "Family", "Genus", "Species", "Consensus Lineage")

# Define Controls
pos_controls <- c("LCPelletA-taxa", "LCPelletB-taxa", "LCPelletC-taxa",
                  "GVPelletA-taxa", "GVPelletB-taxa", "GVPelletC-taxa",
                  "mockdna1.20250630-taxa")

neg_controls <- c("DNAfreewater1.20250630-taxa",
                  "Extractblankswab1.20250630-taxa",
                  "Extractemptywell1.20250630-taxa")

all_sample_cols <- setdiff(colnames(raw_data), c(otu_col, lineage_col))
study_cols <- setdiff(all_sample_cols, c(pos_controls, neg_controls))

# Negative-control check (decontam-style): flag taxa reaching >= 1% relative
# abundance in at least one negative control AND >= 1% in at least one study
# sample. Flagged taxa are written out for review with CHOP and are NOT removed
# here. Appropriate for the moderate-to-high biomass of vaginal swabs; this is a
# sanity check, not a full decontamination step.
actual_neg_cols <- intersect(colnames(raw_data), neg_controls)
actual_study_cols <- intersect(colnames(raw_data), study_cols)

neg_mat   <- as.matrix(raw_data[, actual_neg_cols])
study_mat <- as.matrix(raw_data[, actual_study_cols])

# Relative abundance
neg_RA   <- sweep(neg_mat, 2, colSums(neg_mat), "/")
study_RA <- sweep(study_mat, 2, colSums(study_mat), "/")

# Taxa present at >= 1% in both negative controls and study samples (flag only)
flag_idx <- rowSums(neg_RA >= 0.01, na.rm = TRUE) > 0 & 
  rowSums(study_RA >= 0.01, na.rm = TRUE) > 0

flagged_taxa <- raw_data[flag_idx, c(otu_col, lineage_col)]
write_csv(flagged_taxa, "taxa_flagged_from_negative_controls.csv")

# Drop QC-failed samples 
qc_fail_samples <- c("CRIB.1071.SPTB-taxa", "CRIB.1105.SPTB-taxa") 
final_sample_cols <- setdiff(actual_study_cols, qc_fail_samples)

# Study samples
data_clean <- raw_data |>
  dplyr::select(dplyr::all_of(c(otu_col, final_sample_cols, lineage_col)))

# CRIB/ SPEC Cohorts
crib_cols <- final_sample_cols[str_starts(final_sample_cols, "CRIB")]
spec_cols <- final_sample_cols[str_starts(final_sample_cols, "SPEC")]

crib <- data_clean |> 
  dplyr::select(dplyr::all_of(c(otu_col, crib_cols, lineage_col)))
spec <- data_clean |> 
  dplyr::select(dplyr::all_of(c(otu_col, spec_cols, lineage_col)))

# Log post-control-removal / post-QC-removal counts (QC-failed samples were
# already dropped above via final_sample_cols; features unchanged at this stage).
processing_log[["CRIB"]][["post_qc_samples"]]  <- length(crib_cols)
processing_log[["CRIB"]][["post_qc_features"]] <- nrow(crib)
processing_log[["SPEC"]][["post_qc_samples"]]  <- length(spec_cols)
processing_log[["SPEC"]][["post_qc_features"]] <- nrow(spec)

## Step 2: Sample-Level Read Depth Filter
# Filter by depth and plot
filter_read_depth <- function(cohort_data, cohort_name) {
  
  # Sample columns
  sample_cols <- setdiff(colnames(cohort_data), c(otu_col, lineage_col))
  
  # Total non-host reads per sample
  sample_totals <- colSums(cohort_data[, sample_cols])
  depth_df <- data.frame(Sample = sample_cols, Total_Reads = sample_totals)
  
  # Visualize BEFORE filtering (full per-sample read-total distribution)
  p_before <- ggplot(depth_df, aes(x = Total_Reads)) +
    geom_histogram(bins = 30, fill = "steelblue", color = "black") +
    geom_vline(xintercept = 10000, color = "red", linetype = "dashed") +
    labs(title = paste(cohort_name, "- Read Depth (before filter)"),
         x = "Total Reads", y = "Count") +
    theme_minimal()
  print(p_before)
  
  # Minimum  
  dropped_samples <- depth_df |>
    dplyr::filter(Total_Reads < 10000)
  kept_samples <- depth_df |>
    dplyr::filter(Total_Reads >= 10000)
  
  # Visualize AFTER filtering (retained samples only)
  p_after <- ggplot(kept_samples, aes(x = Total_Reads)) +
    geom_histogram(bins = 30, fill = "seagreen", color = "black") +
    geom_vline(xintercept = 10000, color = "red", linetype = "dashed") +
    labs(title = paste(cohort_name, "- Read Depth (after filter)"),
         x = "Total Reads", y = "Count") +
    theme_minimal()
  print(p_after)
  
  # Excluded samples
  if(nrow(dropped_samples) > 0) {
    message(paste("Dropped samples in", cohort_name, ":"))
    print(dropped_samples)
    write_csv(dropped_samples, paste0(cohort_name, "_dropped_samples_step2.csv"))
  } else {
    message(paste("No samples dropped due to read depth in", cohort_name))
  }
  filtered_data <- cohort_data |> 
    dplyr::select(dplyr::all_of(c(otu_col, kept_samples$Sample, lineage_col)))
  
  processing_log[[cohort_name]][["after_read_depth_samples"]] <<- nrow(kept_samples)
  
  return(filtered_data)
}

# Filter independently per cohort
crib_filtered <- filter_read_depth(crib, "CRIB")
spec_filtered <- filter_read_depth(spec, "SPEC")

# No sample dropped due to depth


## Step 3: Taxonomic Resolution Filter

filter_taxonomic_resolution <- function(cohort_data, cohort_name) {
  
  # Feature count
  total_features <- nrow(cohort_data)
  
  # Lineage column into kingdom - species 
  parsed_data <- cohort_data |>
    separate(all_of(lineage_col), 
             into = c("Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"),
             sep = "; ",
             remove = FALSE, 
             fill = "right",
             extra = "warn")
  
  # Extract genus
  parsed_data <- parsed_data|>
    dplyr::mutate(
      genus_clean = str_remove(Genus, "^g__"),
      has_genus = !is.na(genus_clean) & str_trim(genus_clean) != ""
    )
  
  # Filter to retain only features with a resolved genus
  filtered_data <- parsed_data |>
    dplyr::filter(has_genus)
  
  # Feature counts
  retained_features <- nrow(filtered_data)
  dropped_features  <- total_features - retained_features
  message(sprintf("[%s] Taxonomic Filter - Total: %d | Retained: %d | Dropped: %d", 
                 cohort_name, total_features, retained_features, dropped_features))
  
  processing_log[[cohort_name]][["after_taxonomic_resolution_features"]] <<- retained_features
  
  final_data <- filtered_data |>
    dplyr::select(-genus_clean, -has_genus)
  return(final_data)
}

# Apply independently per cohort
crib_genus <- filter_taxonomic_resolution(crib_filtered, "CRIB")
spec_genus <- filter_taxonomic_resolution(spec_filtered, "SPEC")


## Step 3.5: Taxonomic Agglomeration
# Collapse features to a consistent analytic rank before filtering/CLR.
#
# Standard rule: every taxon is agglomerated to GENUS (all species within a
# genus are summed into one feature). This removes mixed-rank redundancy
# (a genus and its own species appearing as separate rows) and stabilizes the
# composition for CLR.
#
# Two carve-outs keep specific clades at species level:
#
#  (a) GARDNERELLA - ALL species kept. The mucin-glycobiology hypothesis
#      concerns cleaved glycan residues, and sialidase (nanH) capacity differs
#      across Gardnerella species; all four DB species (vaginalis, swidsinskii,
#      leopoldii, piotii) are highly prevalent and well powered to test
#      separately. Represented by its species rows plus a genus-only remainder;
#      NO genus total, so the clade is not double-counted.
#
#  (b) LACTOBACILLUS - only the CST-defining species kept. Downstream analyses
#      require separate CST I vs CST III (and related) contrasts, which hinge on
#      L. crispatus (CST I), L. iners (CST III), L. gasseri (CST II), and
#      L. jensenii (CST V). Only these four are resolved to species; all other
#      Lactobacillus (predominantly non-vaginal gut/environmental species in a
#      broad reference DB) are pooled into a single "Lactobacillus (other)"
#      feature so contaminant species do not each become their own noisy feature.
#      NO genus total is created.
#
# The other three targets follow the standard genus rule (no special-casing):
# Prevotella (speciose, mostly non-vaginal species -> genus aggregates the
# vaginal guild), Sneathia (only S. vaginalis resolved -> genus), and Mobiluncus
# (M. mulieris absent from the DB and reported collapsed in the sPTB literature
# -> genus).

# Carve-out configuration.
#  - all_species_genera : keep every species as its own feature (+ genus-only remainder).
#  - named_species_carveouts : keep only the listed species; pool the rest of the
#    genus into "<Genus> (other)".
all_species_genera <- c("Gardnerella")
named_species_carveouts <- list(
  Lactobacillus = c("Lactobacillus crispatus",   # CST I
                    "Lactobacillus iners",       # CST III
                    "Lactobacillus gasseri",     # CST II
                    "Lactobacillus jensenii")    # CST V
)

agglomerate_taxa <- function(cohort_data, cohort_name) {

  meta_rank_cols <- c("Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species")
  sample_cols <- setdiff(colnames(cohort_data),
                         c(otu_col, lineage_col, meta_rank_cols))

  # Normalized genus/species labels (strip g__/s__, trim) for carve-out tests.
  genus_norm   <- str_trim(str_remove(cohort_data$Genus,   "^g__"))
  species_norm <- str_trim(str_remove(cohort_data$Species, "^s__"))

  # Canonical Species value: a full "s__<name>" where a species is resolved, or
  # "" where the row is genus-only (raw token "s__" strips to empty). Using this
  # everywhere avoids the literal "s__" leaking into keys, IDs, and annotations.
  species_canon <- ifelse(species_norm == "", "", paste0("s__", species_norm))

  # Classify each feature's agglomeration mode.
  is_all_species   <- genus_norm %in% all_species_genera
  is_named_genus   <- genus_norm %in% names(named_species_carveouts)
  # Within a named-carveout genus, is THIS species one of the kept ones?
  is_named_kept <- mapply(function(g, s) {
    g %in% names(named_species_carveouts) && s %in% named_species_carveouts[[g]]
  }, genus_norm, species_norm)

  # A feature is represented at species level if it is an all-species genus, or a
  # kept species within a named-carveout genus.
  is_species_level <- is_all_species | is_named_kept

  # Grouping key:
  #  - all-species genera            -> Genus + Species (per species + genus-only remainder)
  #  - named carve-out, kept species -> Genus + Species (that species on its own)
  #  - named carve-out, other species-> Genus + "(other)" (pooled remainder)
  #  - everything else               -> Genus (all species summed)
  # For an all-species genus-only row, species_canon is "" so the key is
  # "<Genus>|", giving that remainder its own group distinct from resolved species.
  group_key <- ifelse(
    is_species_level,
    paste(cohort_data$Genus, species_canon, sep = "|"),
    ifelse(is_named_genus,
           paste0(cohort_data$Genus, "|(other)"),
           cohort_data$Genus)
  )

  agg_df <- cohort_data
  agg_df$.group_key <- group_key
  agg_df$.is_species_level <- is_species_level
  agg_df$.is_named_other <- is_named_genus & !is_species_level
  agg_df$.species_canon <- species_canon

  # Sum sample counts within each group; take the first lineage/rank values as
  # the representative annotation for the collapsed feature.
  summed <- agg_df |>
    dplyr::group_by(.group_key) |>
    dplyr::summarise(
      dplyr::across(dplyr::all_of(sample_cols), ~ sum(.x, na.rm = TRUE)),
      Kingdom = dplyr::first(Kingdom),
      Phylum  = dplyr::first(Phylum),
      Class   = dplyr::first(Class),
      Order   = dplyr::first(Order),
      Family  = dplyr::first(Family),
      Genus   = dplyr::first(Genus),
      # Species: the canonical species for a species-level carve-out (or "" for
      # its genus-only remainder); "s__(other)" for the pooled named-carveout
      # remainder; "" for genus-collapsed features so each honestly reports rank.
      Species = dplyr::case_when(
        dplyr::first(.is_species_level) ~ dplyr::first(.species_canon),
        dplyr::first(.is_named_other)   ~ "s__(other)",
        TRUE                            ~ ""
      ),
      .groups = "drop"
    )

  # Rebuild a representative Consensus Lineage from the (possibly blanked) ranks.
  summed <- summed |>
    dplyr::mutate(
      `Consensus Lineage` = paste(Kingdom, Phylum, Class, Order, Family, Genus,
                                  ifelse(Species == "", "s__", Species),
                                  sep = "; ")
    )

  # Synthetic feature ID: genus, or genus + species for carve-outs / remainders.
  summed <- summed |>
    dplyr::mutate(
      `#OTU ID` = ifelse(Species == "" | is.na(Species),
                         Genus,
                         paste(Genus, Species, sep = "|"))
    )

  # Reassemble in the same column contract downstream steps expect.
  out <- summed |>
    dplyr::select(dplyr::all_of(c(otu_col, sample_cols,
                                  meta_rank_cols, lineage_col)))

  message(sprintf("[%s] Agglomeration - features: %d -> %d (Gardnerella all spp.; Lactobacillus CST spp.)",
                  cohort_name, nrow(cohort_data), nrow(out)))
  processing_log[[cohort_name]][["after_agglomeration_features"]] <<- nrow(out)

  return(out)
}

crib_agg <- agglomerate_taxa(crib_genus, "CRIB")
spec_agg <- agglomerate_taxa(spec_genus, "SPEC")


## Step 4: Feature-Level Detection and Prevalence Filters
# Mean-RA filter (former 4c) intentionally removed: on this vaginal-microbiome
# data a 0.5% mean-RA cut discards prevalent low-abundance taxa of interest
# (e.g. Mobiluncus, Sneathia, Prevotella). Feature filtering is now presence +
# prevalence only, applied identically and independently per cohort.

# Target features that must survive preprocessing in every cohort (biological
# allowlist), defined at the AGGLOMERATED feature level from Step 3.5:
#   - Gardnerella vaginalis : species-level carve-out (the pre-registered
#     Gardnerella target; other Gardnerella species are retained as separate
#     features and analyzed as exploratory).
#   - Mobiluncus / Prevotella / Sneathia : genus-level features.
#   - Lactobacillus crispatus / iners / gasseri / jensenii : species-level
#     carve-outs required for CST I/III/II/V contrasts downstream.
# Matched against the synthetic #OTU ID produced by agglomeration.
target_features <- c(
  "g__Gardnerella|s__Gardnerella vaginalis",   # species carve-out
  "g__Mobiluncus",
  "g__Prevotella",
  "g__Sneathia",
  "g__Lactobacillus|s__Lactobacillus crispatus",  # CST I
  "g__Lactobacillus|s__Lactobacillus iners",      # CST III
  "g__Lactobacillus|s__Lactobacillus gasseri",    # CST II
  "g__Lactobacillus|s__Lactobacillus jensenii"    # CST V
)

filter_features <- function(cohort_data, cohort_name) {
  
  # Sample columns and matrices (meta_cols defined once at top)
  sample_cols <- setdiff(colnames(cohort_data), meta_cols)
  count_mat <- as.matrix(cohort_data[, sample_cols])
  total_samples <- ncol(count_mat)
  initial_features <- nrow(count_mat)
  
  # 4a. Minimum Count Threshold (Per-Sample Detection): present if >= 3 reads
  presence_mat <- count_mat >= 3
  feat_after_count <- sum(rowSums(presence_mat) > 0)
  
  # 4b. Prevalence Filter: retain features present in >= 10% of cohort samples
  prevalence <- rowSums(presence_mat) / total_samples
  prev_keep <- prevalence >= 0.10
  feat_after_prev <- sum(prev_keep)
  
  # Final feature set (presence + prevalence only)
  final_keep <- prev_keep
  filtered_data <- cohort_data[final_keep, ]
  
  # Target-feature survival check (match on the agglomerated #OTU ID)
  surviving_ids  <- filtered_data[[otu_col]]
  present_targets <- target_features[target_features %in% surviving_ids]
  missing_targets <- setdiff(target_features, present_targets)
  
  # Summary
  message(sprintf("%s Summary", cohort_name))
  message(sprintf("Starting features : %d", initial_features))
  message(sprintf("Features passing Min Count (>= 3 reads): %d", feat_after_count))
  message(sprintf("Features passing Prevalence (>= 10%%): %d", feat_after_prev))
  if (length(missing_targets) == 0) {
    message(sprintf("[%s] All target features survived: %s",
                    cohort_name, paste(target_features, collapse = ", ")))
  } else {
    warning(sprintf("[%s] Target features MISSING after filtering: %s",
                    cohort_name, paste(missing_targets, collapse = ", ")))
  }
  
  # Record counts for the processing log (Appendix A)
  processing_log[[cohort_name]][["after_min_count"]]  <<- feat_after_count
  processing_log[[cohort_name]][["after_prevalence"]] <<- feat_after_prev
  processing_log[[cohort_name]][["final_features"]]   <<- nrow(filtered_data)
  processing_log[[cohort_name]][["targets_present"]]  <<- paste(present_targets, collapse = ";")
  processing_log[[cohort_name]][["targets_missing"]]  <<- paste(missing_targets, collapse = ";")
  
  return(filtered_data)
}

# Apply filters independently per cohort (on the agglomerated features)
crib_filtered_features <- filter_features(crib_agg, "CRIB")
spec_filtered_features <- filter_features(spec_agg, "SPEC")

## Step 5: Relative Abundance Conversion
# NOTE: A standalone non-imputed RA matrix is not produced here because it is not
# consumed downstream. The RA matrix that IS exported (Step 8) is the CZM-imputed
# RA from Step 6, which is what alpha/beta-diversity and any RA-based reporting
# should use. Per-sample RA closure and its sum-to-1 check are therefore performed
# inside Step 6 on the imputed counts rather than duplicated here.

##Step 6:
library(zCompositions)
library(compositions)

transform_clr <- function(count_data, cohort_name) {
  
  # Count matrix and metadata (meta_cols defined once at top)
  sample_cols <- setdiff(colnames(count_data), meta_cols)
  count_mat <- as.matrix(count_data[, sample_cols])
  
  # 6a. Zero Imputation
  # z.delete = FALSE, z.warning = 1: do NOT let cmultRepl silently drop sparse
  # features or samples. Feature inclusion is decided solely by the Step 4
  # prevalence filter, and no sample may be removed here. (By default cmultRepl
  # deletes any row/column that is >80% zeros, which on this deliberately
  # permissive feature set would drop both sparse features and whole samples.)
  count_mat_t <- t(count_mat)   # samples in rows, features in columns
  imputed_counts_t <- cmultRepl(count_mat_t, method = "CZM", output = "p-counts",
                                z.delete = FALSE, z.warning = 1,
                                suppress.print = TRUE)
  
  # Guard: cmultRepl must return the exact same shape it was given. If dimensions
  # changed, something was dropped silently -> stop rather than misalign data.
  if (!all(dim(imputed_counts_t) == dim(count_mat_t))) {
    stop(sprintf("[%s] cmultRepl changed matrix dimensions (in: %d x %d, out: %d x %d). Aborting to avoid silent sample/feature loss.",
                 cohort_name, nrow(count_mat_t), ncol(count_mat_t),
                 nrow(imputed_counts_t), ncol(imputed_counts_t)))
  }
  
  # Counts to relative abundance 
  imputed_ra_t <- sweep(imputed_counts_t, 1, rowSums(imputed_counts_t), "/")
  imputed_ra_mat <- t(imputed_ra_t)
  
  # 6b. CLR Transformation
  clr_mat_t <- clr(imputed_ra_t)
  clr_mat <- t(clr_mat_t)
  
  # Verify output (columns should sum to ~0)
  col_sums_check <- colSums(clr_mat)
  all_sum_to_zero <- all(abs(col_sums_check) < 1e-5)
  
  if (all_sum_to_zero) {
    message(sprintf("%s All CLR sample columns sum to 0.", cohort_name))
  } else {
    warning(sprintf("%s CLR sample columns do not sum to 0. Max deviation: %f", 
                    cohort_name, max(abs(col_sums_check))))
  }
  
  # 5. Recombine the CLR matrix with the original metadata columns
  clr_data <- count_data
  clr_data[, sample_cols] <- clr_mat
  
  ra_imputed_data <- count_data
  ra_imputed_data[, sample_cols] <- imputed_ra_mat
  
  # Return both datasets
  return(list(CLR = clr_data, RA_Imputed = ra_imputed_data))
}

# Apply filters independently per cohort & separate matrices 
crib_transformed <- transform_clr(crib_filtered_features, "CRIB")
spec_transformed <- transform_clr(spec_filtered_features, "SPEC")

crib_clr <- crib_transformed$CLR
crib_ra_imputed <- crib_transformed$RA_Imputed

spec_clr <- spec_transformed$CLR
spec_ra_imputed <- spec_transformed$RA_Imputed

## Step 7: Distributional QC and Outlier Detection

library(moments)

run_distributional_qc <- function(clr_data, metadata, cohort_name) {
  
  # CLR matrix and sample columns
  sample_cols <- setdiff(colnames(clr_data), meta_cols)
  clr_mat <- as.matrix(clr_data[, sample_cols])
  
  # Transpose
  clr_mat_t <- t(clr_mat)
  
  # PCA on CLR-transformed data
  pca_res <- prcomp(clr_mat_t, center = TRUE, scale. = FALSE)
  
  # PC1 and PC2
  pca_df <- as.data.frame(pca_res$x[, 1:2])
  pca_df$Sample <- rownames(pca_df)
  
  # Merge with metadata to get phenotype (SPTB/TB)
  if (!missing(metadata) && !is.null(metadata)) {
    pca_df <- left_join(pca_df, metadata[, c("Sample", "Phenotype")], by = "Sample")
  } else {
    pca_df$Phenotype <- ifelse(grepl("SPTB", pca_df$Sample), "SPTB", 
                               ifelse(grepl("TB", pca_df$Sample), "TB", "Unknown"))
  }
  
  # Centroid and distance for outlier flagging (> 3SD)
  centroid_pc1 <- mean(pca_df$PC1)
  sd_pc1 <- sd(pca_df$PC1)
  centroid_pc2 <- mean(pca_df$PC2)
  sd_pc2 <- sd(pca_df$PC2)
  
  pca_df$Outlier_PCA <- (abs(pca_df$PC1 - centroid_pc1) > 3 * sd_pc1) | 
    (abs(pca_df$PC2 - centroid_pc2) > 3 * sd_pc2)
  
  flagged_pca_samples <- pca_df$Sample[pca_df$Outlier_PCA]
  if (length(flagged_pca_samples) > 0) {
    message(sprintf("%s Flagged %d samples >3 SD from PCA centroid: %s", 
                    cohort_name, length(flagged_pca_samples), paste(flagged_pca_samples, collapse=", ")))
  }
  
  # Plot 
  p_pca <- ggplot(pca_df, aes(x = PC1, y = PC2, color = Phenotype)) + 
    geom_point(size = 3, alpha = 0.8) +
    stat_ellipse(level = 0.95, linetype = "dashed") +
    geom_text(aes(label = Sample), data = subset(pca_df, Outlier_PCA), vjust = -1, size = 3, color = "red") + 
    labs(title = paste(cohort_name, "- PCA of CLR Transformed Data"),
         x = paste0("PC1 (", round(summary(pca_res)$importance[2,1]*100, 1), "%)"),
         y = paste0("PC2 (", round(summary(pca_res)$importance[2,2]*100, 1), "%)")) +
    theme_minimal()
  print(p_pca)
  
  
  # Aitchison distance matrix 
  aitchison_dist <- dist(clr_mat_t, method = "euclidean")
  
  # Samples with  high mean distance
  dist_mat <- as.matrix(aitchison_dist)
  mean_dists <- rowMeans(dist_mat)
  dist_threshold <- mean(mean_dists) + 3 * sd(mean_dists)
  
  flagged_dist_samples <- names(mean_dists)[mean_dists > dist_threshold]
  if (length(flagged_dist_samples) > 0) {
    message(sprintf("%s Flagged %d samples with high mean Aitchison distance: %s", 
                    cohort_name, length(flagged_dist_samples), paste(flagged_dist_samples, collapse=", ")))
  }
  
  # Per-feature normality check
  # Skewness
  feature_skewness <- apply(clr_mat, 1, skewness, na.rm = TRUE)
  
  # Flag features with |skewness| > 3
  flagged_features_idx <- which(abs(feature_skewness) > 3)
  flagged_features <- clr_data$`#OTU ID`[flagged_features_idx]
  
  if (length(flagged_features) > 0) {
    message(sprintf("%s Flagged %d features with |skewness| > 3. ", 
                    cohort_name, length(flagged_features)))
  }
  
  # Final counts for summary
  total_samples_passing <- length(sample_cols)
  total_features_retained <- nrow(clr_data)
  
  # Summary list
  return(list(
    Samples_Passing_Filters = total_samples_passing,
    Features_Retained = total_features_retained,
    PCA_Plot = p_pca,
    Flagged_PCA_Samples = flagged_pca_samples,
    Flagged_Dist_Samples = flagged_dist_samples,
    Flagged_Skewed_Features = flagged_features
  ))
}

# QC independently per cohort
crib_qc <- run_distributional_qc(crib_clr, metadata = NULL, "CRIB")
spec_qc <- run_distributional_qc(spec_clr, metadata = NULL, "SPEC")

## Step 8 

save_outputs <- function(clr_data, ra_data, cohort_name) {
  
  # meta_cols defined once at top
  sample_cols <- setdiff(colnames(clr_data), meta_cols)
  
  # Save processed feature matrix (CLR-transformed)
  clr_out <- clr_data[, c("#OTU ID", sample_cols)]
  write_tsv(clr_out, paste0(cohort_name, "_CLR_matrix.tsv"))
  
  # Save feature metadata table
  feature_metadata <- clr_data[, meta_cols]
  write_tsv(feature_metadata, paste0(cohort_name, "_Feature_Metadata.tsv"))
  
  # Save RA matrix
  ra_out <- ra_data[, c("#OTU ID", sample_cols)]
  write_tsv(ra_out, paste0(cohort_name, "_RA_matrix.tsv"))
}

# Export independently per cohort using the Step 6 imputed RA data
save_outputs(crib_clr, crib_ra_imputed, "CRIB")
save_outputs(spec_clr, spec_ra_imputed, "SPEC")

# Record final post-CLR feature counts (unchanged from Step 4 final set)
processing_log[["CRIB"]][["final_clr_features"]] <- nrow(crib_clr)
processing_log[["SPEC"]][["final_clr_features"]] <- nrow(spec_clr)

# --- Processing log (Appendix A) ---
# One row per pipeline step, columns per cohort, matching the SOP template.
write_processing_log <- function(plog, path = "processing_log.tsv") {
  step_order <- c(
    post_qc_samples                     = "Samples (post control + QC removal)",
    post_qc_features                    = "Features (initial load)",
    after_read_depth_samples            = "Samples after read-depth filter (>=10,000)",
    after_taxonomic_resolution_features = "Features after taxonomic resolution (genus+)",
    after_agglomeration_features        = "Features after agglomeration (genus; Gardnerella + Lactobacillus spp.)",
    after_min_count                     = "Features after min-count detection (>=3 reads)",
    after_prevalence                    = "Features after prevalence filter (>=10%)",
    final_features                      = "Features after filtering (final set)",
    final_clr_features                  = "Features after CLR transformation",
    targets_present                     = "Target features present",
    targets_missing                     = "Target features missing"
  )
  cohorts <- names(plog)
  log_df <- data.frame(Step = unname(step_order), stringsAsFactors = FALSE)
  for (ch in cohorts) {
    log_df[[ch]] <- vapply(names(step_order), function(k) {
      v <- plog[[ch]][[k]]
      if (is.null(v)) NA_character_ else as.character(v)
    }, character(1))
  }
  write_tsv(log_df, path)
  message(sprintf("Processing log written to %s", path))
  print(log_df)
}

write_processing_log(processing_log, "processing_log.tsv")
