"""ComBat with reference-batch anchoring AND fit/transform separation.

Why this exists
---------------
SOP Step 13 requires reference-batch ComBat ("use the reference-batch
('ref.batch') mode ... which anchors batch parameter estimation to the bridge
samples"). Aim F.2 requires the held-out participants to influence nothing,
i.e. parameters estimated on the development split and *applied* to the test
split. No available implementation offers both:

    pycombat            fit/transform, but no ref.batch at all
    inmoose             ref_batch, but a single-shot function
    sva::ComBat (R)     ref.batch, but also single-shot - it returns only the
                        corrected matrix and never exposes gamma*/delta*, so it
                        cannot be applied out-of-sample either. R is also not
                        installable in this environment (no root).

This module implements the parametric empirical-Bayes ComBat of Johnson,
Li & Rabinovic (2007) as sva does it, exposing the fitted parameters so new
samples can be corrected with constants derived without them.

Model
-----
    Y_ijg = alpha_g + X_j beta_g + gamma_ig + delta_ig * eps_ijg

fit() estimates, on the fit rows only:
    alpha_g        feature grand mean (the reference batch's mean when
                   ref_batch is given)
    beta_g         covariate coefficients
    sigma_g        pooled SD (reference batch only when ref_batch is given)
    gamma*_ig      shrunken additive batch effect
    delta*_ig      shrunken multiplicative batch effect

transform() applies:
    Y_adj = sigma_g * (Z - gamma*_ig) / delta*_ig + alpha_g + X beta_g

With ref_batch, the reference batch carries gamma*=0 and delta*=1, so its
samples pass through unchanged and every other batch is mapped onto its scale.
That is what makes the correction transferable: a new sample only needs its
batch label, and the constants for that batch already exist.

Orientation: samples x features, matching the rest of the pipeline.
"""

from __future__ import annotations

import numpy as np


def _aprior(delta_hat):
    m, s2 = np.mean(delta_hat), np.var(delta_hat, ddof=1)
    return (2 * s2 + m ** 2) / s2


def _bprior(delta_hat):
    m, s2 = np.mean(delta_hat), np.var(delta_hat, ddof=1)
    return (m * s2 + m ** 3) / s2


def _postmean(g_hat, g_bar, n, d_star, t2):
    return (t2 * n * g_hat + d_star * g_bar) / (t2 * n + d_star)


def _postvar(sum2, n, a, b):
    return (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)


def _it_sol(s_data, g_hat, d_hat, g_bar, t2, a, b, conv=1e-4, max_iter=500):
    """Iterative empirical-Bayes solution, as in sva's it.sol."""
    n = (~np.isnan(s_data)).sum(axis=1)
    g_old, d_old = g_hat.copy(), d_hat.copy()
    for _ in range(max_iter):
        g_new = _postmean(g_hat, g_bar, n, d_old, t2)
        sum2 = np.nansum((s_data - g_new[:, None]) ** 2, axis=1)
        d_new = _postvar(sum2, n, a, b)
        change = max(
            np.max(np.abs(g_new - g_old) / np.where(np.abs(g_old) > 0, np.abs(g_old), 1)),
            np.max(np.abs(d_new - d_old) / np.where(np.abs(d_old) > 0, np.abs(d_old), 1)),
        )
        g_old, d_old = g_new, d_new
        if change < conv:
            break
    return g_old, d_old


class ComBatRef:
    """Parametric ComBat with optional reference batch and out-of-sample transform."""

    def __init__(self, ref_batch=None, conv: float = 1e-4):
        self.ref_batch = None if ref_batch is None else str(ref_batch)
        self.conv = conv

    # ---------------------------------------------------------------- fit
    def fit(self, Y, batch, X=None):
        """Estimate parameters. Y is (n_samples, n_features)."""
        Y = np.asarray(Y, dtype=float)
        batch = np.asarray([str(b) for b in batch])
        n_samp, n_feat = Y.shape

        self.batches_ = list(dict.fromkeys(batch))
        if self.ref_batch is not None and self.ref_batch not in self.batches_:
            raise ValueError(
                f"ref_batch={self.ref_batch!r} not present in the fit data "
                f"(batches: {self.batches_})"
            )

        # Design: batch dummies + covariates.
        #
        # With a reference batch, sva sets that batch's column to all ones
        # (`batchmod[, ref] <- 1`), turning it into an intercept so the other
        # batch columns encode offsets *relative to the reference* and
        # B_hat[ref] is the reference batch's mean. Using plain dummies here
        # instead silently changes the estimand - it over-corrects, removing
        # the full batch shift rather than sva's shrunken estimate.
        D = np.column_stack([(batch == b).astype(float) for b in self.batches_])
        self.n_batch_ = D.sum(axis=0)
        if self.ref_batch is not None:
            D[:, self.batches_.index(self.ref_batch)] = 1.0
        design = D if X is None else np.column_stack([D, np.asarray(X, dtype=float)])
        self.n_cov_ = 0 if X is None else np.asarray(X).shape[1]

        # OLS on standardised model
        coef, *_ = np.linalg.lstsq(design, Y, rcond=None)
        self.beta_ = coef[len(self.batches_):] if self.n_cov_ else None
        B_hat = coef[: len(self.batches_)]

        if self.ref_batch is not None:
            ri = self.batches_.index(self.ref_batch)
            self.alpha_ = B_hat[ri]
            ref_mask = batch == self.ref_batch
            resid = Y[ref_mask] - design[ref_mask] @ coef
            self.sigma_ = np.sqrt(np.mean(resid ** 2, axis=0))
        else:
            self.alpha_ = (self.n_batch_ / n_samp) @ B_hat
            resid = Y - design @ coef
            self.sigma_ = np.sqrt(np.mean(resid ** 2, axis=0))

        self.sigma_ = np.where(self.sigma_ <= 0, 1e-8, self.sigma_)

        # standardise
        stand_mean = np.tile(self.alpha_, (n_samp, 1))
        if self.n_cov_:
            stand_mean = stand_mean + np.asarray(X, dtype=float) @ self.beta_
        Z = (Y - stand_mean) / self.sigma_

        # gamma_hat by REGRESSION on the batch design, not per-batch means.
        # With the all-ones reference column this makes gamma_hat[ref] the
        # standardised intercept and gamma_hat[k] the offset of batch k from
        # the reference - and it means every sample later has gamma_star[ref]
        # subtracted, which per-batch means would not reproduce.
        self.batch_design_ = D
        gamma_hat = np.linalg.solve(D.T @ D, D.T @ Z)

        self.gamma_star_, self.delta_star_ = {}, {}
        for bi, b in enumerate(self.batches_):
            m = batch == b
            Zb = Z[m]
            g_hat = gamma_hat[bi]
            d_hat = np.nanvar(Zb, axis=0, ddof=1)
            d_hat = np.where(~np.isfinite(d_hat) | (d_hat <= 0), 1e-8, d_hat)
            g_bar, t2 = np.mean(g_hat), np.var(g_hat, ddof=1)
            a, bpr = _aprior(d_hat), _bprior(d_hat)
            gs, ds = _it_sol(Zb.T, g_hat, d_hat, g_bar, t2, a, bpr, conv=self.conv)
            self.gamma_star_[b] = gs
            self.delta_star_[b] = np.where(ds <= 0, 1e-8, ds)

        self.n_features_in_ = n_feat
        return self

    # ----------------------------------------------------------- transform
    def transform(self, Y, batch, X=None):
        """Apply fitted parameters. Batches must have been seen during fit."""
        Y = np.asarray(Y, dtype=float)
        batch = np.asarray([str(b) for b in batch])
        unseen = set(batch) - set(self.batches_)
        if unseen:
            raise ValueError(
                f"Batch(es) {sorted(unseen)} were not present at fit time, so no "
                "gamma*/delta* exist for them. Every batch must appear in the fit split."
            )

        stand_mean = np.tile(self.alpha_, (Y.shape[0], 1))
        if self.n_cov_:
            stand_mean = stand_mean + np.asarray(X, dtype=float) @ self.beta_
        Z = (Y - stand_mean) / self.sigma_

        # Subtract batch_design @ gamma_star, so a sample also carries
        # gamma_star[ref] when the reference column is all ones.
        G = np.vstack([self.gamma_star_[b] for b in self.batches_])
        Dn = np.column_stack([(batch == b).astype(float) for b in self.batches_])
        if self.ref_batch is not None:
            Dn[:, self.batches_.index(self.ref_batch)] = 1.0

        out = (Z - Dn @ G)
        for bi, b in enumerate(self.batches_):
            m = batch == b
            if m.any():
                out[m] = out[m] / np.sqrt(self.delta_star_[b])
        out = out * self.sigma_ + stand_mean

        # sva restores the reference batch to its original values, so the
        # reference defines the target scale exactly rather than approximately.
        if self.ref_batch is not None:
            ref_m = batch == self.ref_batch
            if ref_m.any():
                out[ref_m] = Y[ref_m]
        return out

    def fit_transform(self, Y, batch, X=None):
        return self.fit(Y, batch, X).transform(Y, batch, X)
