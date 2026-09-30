# ComBat with reference batch, split into fit and apply.
#
# WHY THIS FILE EXISTS
# --------------------
# sva::ComBat() returns only the corrected matrix. It never exposes gamma.star
# / delta.star, so it cannot be applied to samples it did not see. SOP Step 13
# requires ref.batch anchoring; Aim F.2 requires held-out participants to
# influence no parameter. Both together need the estimator split in two.
#
# combat_fit()   estimates parameters on the development split
# combat_apply() applies them to any samples whose batches were seen at fit
#
# The estimation below follows sva::ComBat exactly (parametric priors,
# ref.batch handling, and the final restoration of the reference batch's
# original values). It is deliberately a thin re-expression of that code
# rather than a reimplementation, so results should match ComBat() exactly -
# validate_combat.R checks that.
#
# Orientation: FEATURES x SAMPLES, the Bioconductor convention. The Python
# pipeline holds samples x features, so transpose on the way in and out.

suppressPackageStartupMessages({
  library(sva)
})

.aprior <- function(d) { m <- mean(d); s2 <- var(d); (2*s2 + m^2)/s2 }
.bprior <- function(d) { m <- mean(d); s2 <- var(d); (m*s2 + m^3)/s2 }
.postmean <- function(g.hat, g.bar, n, d.star, t2) {
  (t2*n*g.hat + d.star*g.bar) / (t2*n + d.star)
}
.postvar <- function(sum2, n, a, b) (0.5*sum2 + b) / (n/2 + a - 1)

.it.sol <- function(sdat, g.hat, d.hat, g.bar, t2, a, b, conv = 1e-4) {
  n <- rowSums(!is.na(sdat))
  g.old <- g.hat; d.old <- d.hat; change <- 1
  while (change > conv) {
    g.new  <- .postmean(g.hat, g.bar, n, d.old, t2)
    sum2   <- rowSums((sdat - g.new %*% t(rep(1, ncol(sdat))))^2, na.rm = TRUE)
    d.new  <- .postvar(sum2, n, a, b)
    change <- max(abs(g.new - g.old)/g.old, abs(d.new - d.old)/d.old)
    g.old  <- g.new; d.old <- d.new
  }
  list(gamma.star = g.old, delta.star = d.old)
}

# Internal: the original, validated estimator. Call combat_fit() instead.
.combat_fit_core <- function(dat, batch, mod = NULL, ref.batch = NULL) {
  dat <- as.matrix(dat)
  batch <- as.character(batch)
  batches_lvl <- unique(batch)
  n.batch <- length(batches_lvl)
  n.array <- ncol(dat)
  batches <- lapply(batches_lvl, function(b) which(batch == b))
  n.batches <- sapply(batches, length)

  batchmod <- model.matrix(~ -1 + factor(batch, levels = batches_lvl))
  ref <- NULL
  if (!is.null(ref.batch)) {
    if (!(ref.batch %in% batches_lvl)) stop("ref.batch not found among batches")
    ref <- which(batches_lvl == ref.batch)
    batchmod[, ref] <- 1          # sva: reference column becomes an intercept
  }
  design <- cbind(batchmod, mod)

  B.hat <- solve(crossprod(design), tcrossprod(t(design), as.matrix(dat)))

  if (!is.null(ref)) {
    grand.mean <- t(B.hat[ref, ])
    ref.idx <- batches[[ref]]
    var.pooled <- ((dat[, ref.idx] - t(design[ref.idx, ] %*% B.hat))^2) %*%
                    rep(1/n.batches[ref], n.batches[ref])
  } else {
    grand.mean <- crossprod(n.batches/n.array, B.hat[1:n.batch, ])
    var.pooled <- ((dat - t(design %*% B.hat))^2) %*% rep(1/n.array, n.array)
  }

  stand.mean <- t(grand.mean) %*% t(rep(1, n.array))
  if (!is.null(design)) {
    tmp <- design; tmp[, 1:n.batch] <- 0
    stand.mean <- stand.mean + t(tmp %*% B.hat)
  }
  s.data <- (dat - stand.mean) / (sqrt(var.pooled) %*% t(rep(1, n.array)))

  batch.design <- design[, 1:n.batch, drop = FALSE]
  gamma.hat <- solve(crossprod(batch.design),
                     tcrossprod(t(batch.design), as.matrix(s.data)))
  delta.hat <- do.call(rbind, lapply(batches, function(i)
    apply(s.data[, i, drop = FALSE], 1, var, na.rm = TRUE)))

  gamma.bar <- rowMeans(gamma.hat)
  t2        <- apply(gamma.hat, 1, var)
  a.prior   <- apply(delta.hat, 1, .aprior)
  b.prior   <- apply(delta.hat, 1, .bprior)

  gamma.star <- delta.star <- NULL
  for (i in 1:n.batch) {
    temp <- .it.sol(s.data[, batches[[i]], drop = FALSE], gamma.hat[i, ],
                    delta.hat[i, ], gamma.bar[i], t2[i], a.prior[i], b.prior[i])
    gamma.star <- rbind(gamma.star, temp$gamma.star)
    delta.star <- rbind(delta.star, temp$delta.star)
  }
  if (!is.null(ref)) { gamma.star[ref, ] <- 0; delta.star[ref, ] <- 1 }

  list(batches_lvl = batches_lvl, n.batch = n.batch, ref = ref,
       ref.batch = ref.batch, B.hat = B.hat, grand.mean = grand.mean,
       var.pooled = var.pooled, gamma.star = gamma.star,
       delta.star = delta.star, n.cov = if (is.null(mod)) 0 else ncol(mod),
       features = rownames(dat))
}

# Internal: the original, validated transform. Call combat_apply() instead.
.combat_apply_core <- function(fit, dat, batch, mod = NULL) {
  dat <- as.matrix(dat); batch <- as.character(batch)
  unseen <- setdiff(unique(batch), fit$batches_lvl)
  if (length(unseen)) {
    stop(sprintf("Batch(es) %s were not present at fit time; no parameters exist for them.",
                 paste(unseen, collapse = ", ")))
  }
  n.array <- ncol(dat)
  # Build the dummy matrix explicitly rather than via model.matrix(): a batch
  # with zero rows in `dat` would otherwise lose its column, silently
  # misaligning against gamma.star / delta.star.
  bm <- sapply(fit$batches_lvl, function(b) as.numeric(batch == b))
  bm <- matrix(bm, nrow = n.array, ncol = fit$n.batch,
               dimnames = list(NULL, fit$batches_lvl))
  if (!is.null(fit$ref)) bm[, fit$ref] <- 1
  stopifnot(ncol(bm) == nrow(fit$gamma.star))

  stand.mean <- t(fit$grand.mean) %*% t(rep(1, n.array))
  if (fit$n.cov > 0 && !is.null(mod)) {
    Bcov <- fit$B.hat[(fit$n.batch + 1):nrow(fit$B.hat), , drop = FALSE]
    stand.mean <- stand.mean + t(mod %*% Bcov)
  }
  s.data <- (dat - stand.mean) / (sqrt(fit$var.pooled) %*% t(rep(1, n.array)))

  bayesdata <- s.data - t(bm %*% fit$gamma.star)
  for (i in seq_along(fit$batches_lvl)) {
    idx <- which(batch == fit$batches_lvl[i])
    if (!length(idx)) next
    bayesdata[, idx] <- bayesdata[, idx] /
      (sqrt(fit$delta.star[i, ]) %*% t(rep(1, length(idx))))
  }
  out <- bayesdata * (sqrt(fit$var.pooled) %*% t(rep(1, n.array))) + stand.mean

  # sva restores the reference batch to its original values
  if (!is.null(fit$ref)) {
    ridx <- which(batch == fit$ref.batch)
    if (length(ridx)) out[, ridx] <- dat[, ridx]
  }
  out
}

#' Fit ComBat on the training split.
#'
#' Features with zero variance within any batch of more than one sample are
#' excluded from estimation and returned unadjusted by combat_apply(), exactly
#' as sva::ComBat does ("Found N genes with uniform expression within a single
#' batch ... these will not be adjusted for batch"). "Uniform" is tested as
#' max == min, which is exact everywhere; sva's var(x) == 0 gives the same
#' answer only when R's compensated variance returns exactly zero.
#'
#' Without it, a feature that is constant in the reference batch gets a pooled
#' variance of ~0; standardising by it produces enormous values that corrupt
#' the empirical-Bayes priors shared by every feature, and the non-reference
#' batches collapse to one value per feature. That happened to LIPD_placenta.
#' @param dat      features x samples matrix (log2 scale, complete)
#' @param batch    character vector of batch labels, length = ncol(dat)
#' @param mod      optional model matrix of covariates to preserve (samples x k)
#' @param ref.batch label of the reference batch
combat_fit <- function(dat, batch, mod = NULL, ref.batch = NULL) {
  dat <- as.matrix(dat); batch <- as.character(batch)
  zero.rows.lst <- lapply(unique(batch), function(batch_level) {
    if (sum(batch == batch_level) > 1) {
      which(apply(dat[, batch == batch_level, drop = FALSE], 1,
                  function(x) max(x) == min(x)))
    } else integer(0)
  })
  zero.rows <- Reduce(union, zero.rows.lst)
  keep.rows <- setdiff(seq_len(nrow(dat)), zero.rows)
  if (length(zero.rows) > 0) {
    cat(sprintf(paste0("Found %d features with uniform expression within a single batch; ",
                       "these will not be adjusted for batch.\n"), length(zero.rows)))
  }
  fit <- .combat_fit_core(dat[keep.rows, , drop = FALSE], batch, mod = mod,
                          ref.batch = ref.batch)
  fit$keep.rows  <- keep.rows
  fit$n.features <- nrow(dat)
  fit
}

#' Apply fitted ComBat parameters to new samples. Features excluded at fit time
#' (zero variance within a batch) pass through unchanged.
combat_apply <- function(fit, dat, batch, mod = NULL) {
  dat <- as.matrix(dat)
  keep <- if (is.null(fit$keep.rows)) seq_len(nrow(dat)) else fit$keep.rows
  if (!is.null(fit$n.features) && nrow(dat) != fit$n.features) {
    stop(sprintf("dat has %d features; fitted on %d.", nrow(dat), fit$n.features))
  }
  out <- dat
  out[keep, ] <- .combat_apply_core(fit, dat[keep, , drop = FALSE], batch, mod = mod)
  out
}

# --------------------------------------------------------------------------
# CLI: fit on the rows flagged as 'dev', apply to everyone, write the result.
#
#   Rscript combat_fit_transform.R <matrix.csv> <meta.csv> <out.csv> [ref_batch]
#
# matrix.csv : features x samples, first column = feature id, header = sample ids
# meta.csv   : columns SampleID, Batch, split   (split in {dev, test})
# --------------------------------------------------------------------------
if (sys.nframe() == 0) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) < 3) stop("usage: <matrix.csv> <meta.csv> <out.csv> [ref_batch]")
  dat  <- as.matrix(read.csv(args[1], row.names = 1, check.names = FALSE))
  meta <- read.csv(args[2], stringsAsFactors = FALSE)
  meta <- meta[match(colnames(dat), meta$SampleID), ]
  stopifnot(!any(is.na(meta$SampleID)))
  refb <- if (length(args) >= 4) args[4] else NULL

  dev <- meta$split == "dev"
  cat(sprintf("fitting on %d dev samples; applying to all %d\n", sum(dev), ncol(dat)))
  fit <- combat_fit(dat[, dev, drop = FALSE], meta$Batch[dev], ref.batch = refb)
  out <- combat_apply(fit, dat, meta$Batch)
  write.csv(out, args[3])
  cat(sprintf("wrote %s\n", args[3]))
}
