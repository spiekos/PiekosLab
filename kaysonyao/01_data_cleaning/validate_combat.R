# Validation harness for combat_fit_transform.R
#
# Run this FIRST, before wiring anything into the pipeline. It answers two
# questions that cannot be answered without R:
#
#   TEST 1  Does combat_fit() + combat_apply(), fitted and applied to the SAME
#           samples, reproduce sva::ComBat() exactly? If yes, the split-out
#           estimator is faithful and ref.batch anchoring is correct.
#
#   TEST 2  Does fitting on a subset and applying to everything differ from
#           fitting on everything? It must - that difference is the whole point.
#
# It also writes reference output that the Python implementation
# (01_data_cleaning/combat_ref.py) can be checked against, since that
# implementation currently disagrees with inmoose (corr 0.82-0.92) for reasons
# I could not isolate without R.
#
#   Rscript validate_combat.R

source("combat_fit_transform.R")
set.seed(42)

n_feat <- 300; n_per <- 40
batch  <- rep(c("51223", "110123", "112524"), each = n_per)
n      <- length(batch)

# Batch effects: a feature-specific additive offset centred on the batch shift,
# plus a batch-wide multiplicative scale. NOTE: the shift is the MEAN of rnorm,
# not its sd - passing a negative number as sd yields NAs, which then make
# ComBat's standardisation singular.
batch_shift <- c(0, 1.5, -1.0)   # log2 units
batch_scale <- c(1.0, 1.3, 0.85)

dat <- matrix(rnorm(n_feat * n, mean = 5, sd = 1), nrow = n_feat)
for (i in seq_along(unique(batch))) {
  idx <- which(batch == unique(batch)[i])
  offset <- rnorm(n_feat, mean = batch_shift[i], sd = 0.5)  # one per feature
  dat[, idx] <- (dat[, idx] + offset) * batch_scale[i]
}
rownames(dat) <- paste0("F", 1:n_feat)
colnames(dat) <- paste0("S", 1:n)

stopifnot(!any(is.na(dat)), all(is.finite(dat)))
cat(sprintf("simulated %d features x %d samples, %d batches, no missing values\n\n",
            n_feat, n, length(unique(batch))))

cat("== TEST 1: agreement with sva::ComBat (fit and apply on all samples) ==\n")
ref <- "51223"
sva_out <- ComBat(dat = dat, batch = batch, mod = NULL, par.prior = TRUE, ref.batch = ref)
fit     <- combat_fit(dat, batch, ref.batch = ref)
mine    <- combat_apply(fit, dat, batch)
d <- abs(sva_out - mine)
cat(sprintf("   max |diff| = %.3e   mean |diff| = %.3e\n", max(d), mean(d)))
cat(sprintf("   VERDICT: %s\n\n",
            if (max(d) < 1e-8) "MATCHES sva::ComBat" else "DIVERGES - do not use in the pipeline"))

cat("== TEST 2: reference batch left untouched ==\n")
ridx <- which(batch == ref)
cat(sprintf("   max |Y_adj - Y| on reference batch = %.3e (expect ~0)\n\n",
            max(abs(mine[, ridx] - dat[, ridx]))))

cat("== TEST 3: fit on dev only, apply to everyone ==\n")
dev <- rep(FALSE, n)
for (b in unique(batch)) {                    # 70% of each batch into dev
  idx <- which(batch == b); dev[sample(idx, floor(0.7 * length(idx)))] <- TRUE
}
fit_dev  <- combat_fit(dat[, dev, drop = FALSE], batch[dev], ref.batch = ref)
applied  <- combat_apply(fit_dev, dat, batch)
cat(sprintf("   fitted on %d of %d samples\n", sum(dev), n))
cat(sprintf("   mean |dev-fit vs all-fit| on held-out samples = %.4f  (must be > 0)\n",
            mean(abs(applied[, !dev] - mine[, !dev]))))

cat("\n== TEST 4: unseen batch is refused ==\n")
res <- try(combat_apply(fit_dev, dat[, 1:5], rep("NEWBATCH", 5)), silent = TRUE)
cat(sprintf("   %s\n", if (inherits(res, "try-error")) "correctly refused" else "NOT REFUSED - bug"))

# reference output for cross-checking the Python implementation
write.csv(dat,  "validate_combat_input.csv")
write.csv(mine, "validate_combat_R_output.csv")
write.csv(data.frame(SampleID = colnames(dat), Batch = batch,
                     split = ifelse(dev, "dev", "test")),
          "validate_combat_meta.csv", row.names = FALSE)
cat("\nwrote validate_combat_{input,R_output,meta}.csv for Python cross-check\n")
