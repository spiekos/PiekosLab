"""Cross-check combat_ref.py (Python) against the R reference output.

Run AFTER `Rscript validate_combat.R`, from the same directory. It compares the
Python implementation to sva's answer on identical input and reports where they
diverge, so the Python version can either be corrected or abandoned.

Background: combat_ref.py currently disagrees with inmoose (correlation
0.82-0.92 on synthetic data) for reasons that could not be isolated without a
ground truth. This script provides that ground truth.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from combat_ref import ComBatRef  # noqa: E402

NEEDED = ["validate_combat_input.csv", "validate_combat_R_output.csv",
          "validate_combat_meta.csv"]


def main() -> None:
    missing = [f for f in NEEDED if not os.path.exists(f)]
    if missing:
        raise SystemExit(f"Missing {missing}. Run `Rscript validate_combat.R` first.")

    dat = pd.read_csv("validate_combat_input.csv", index_col=0)      # features x samples
    r_out = pd.read_csv("validate_combat_R_output.csv", index_col=0)
    meta = pd.read_csv("validate_combat_meta.csv")
    meta = meta.set_index("SampleID").loc[dat.columns]
    batch = meta["Batch"].astype(str).values
    ref = "51223"

    # Python side works samples x features
    Y = dat.T.to_numpy(dtype=float)
    mine = ComBatRef(ref_batch=ref).fit_transform(Y, batch)
    theirs = r_out.T.to_numpy(dtype=float)

    d = np.abs(mine - theirs)
    print("=== combat_ref.py vs sva::ComBat ===")
    print(f"  max |diff|  = {d.max():.3e}")
    print(f"  mean |diff| = {d.mean():.3e}")
    print(f"  correlation = {np.corrcoef(mine.ravel(), theirs.ravel())[0, 1]:.8f}")
    print(f"  VERDICT: {'MATCHES' if d.max() < 1e-8 else 'DIVERGES'}")

    if d.max() >= 1e-8:
        print("\n=== where does it diverge? ===")
        per_batch = pd.Series(
            {b: d[batch == b].mean() for b in dict.fromkeys(batch)}
        ).sort_values(ascending=False)
        print("  mean |diff| by batch:")
        for b, v in per_batch.items():
            tag = "  <- reference" if b == ref else ""
            print(f"    {b}: {v:.4e}{tag}")
        print("\n  If the reference batch is ~0 and others are not, the error is in")
        print("  gamma*/delta* estimation. If the reference batch also differs, it is")
        print("  in var.pooled or the standardisation.")

        col = d.mean(axis=0)
        print(f"\n  features with mean |diff| > 0.1: {(col > 0.1).sum()} / {len(col)}")

    # TEST 5 case from validate_combat.R: features constant within a batch.
    zin, zout = "validate_combat_zerovar_input.csv", "validate_combat_zerovar_sva_output.csv"
    if os.path.exists(zin) and os.path.exists(zout):
        dz = pd.read_csv(zin, index_col=0)
        rz = pd.read_csv(zout, index_col=0)
        Yz = dz.T.to_numpy(dtype=float)
        model = ComBatRef(ref_batch=ref).fit(Yz, batch)
        mz = model.transform(Yz, batch)
        dd = np.abs(mz - rz.T.to_numpy(dtype=float))
        print("\n=== constant-within-a-batch features (sva zero-variance rule) ===")
        print(f"  features excluded by combat_ref.py: {model.n_zero_var_} (expect 2)")
        print(f"  max |diff| vs sva::ComBat = {dd.max():.3e}")
        print(f"  VERDICT: {'MATCHES' if dd.max() < 1e-8 else 'DIVERGES'}")
    else:
        print("\n(zero-variance case not found - rerun `Rscript validate_combat.R` to create it)")


if __name__ == "__main__":
    main()
