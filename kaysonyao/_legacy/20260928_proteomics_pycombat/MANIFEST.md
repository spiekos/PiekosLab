# Archived 2026-09-28: proteomics outputs corrected with pycombat (no reference batch)

Produced 2026-08-11 by clean_proteomics_data.py using pycombat, which has no
reference-batch mode (SOP Step 13 requires one). The plasma file here already
carries the SubjectID fix from _legacy/20260928_proteomics_subjectid/.

Superseded by reference-batch ComBat (combat_ref.ComBatRef, batch 1 as
reference), which matches sva::ComBat(ref.batch=...) to 4.97e-14. Same raw
inputs and same master table (md5 54262754...); only the batch correction differs.

## Follow-up (same day): zero-variance guard added to combat_ref.ComBatRef

The first reference-batch regeneration collapsed placenta batches 2-3 to one value
per protein. Cause: proteins absent from a batch are median-filled, so they are
constant within that batch; a constant protein in the reference batch has pooled
SD ~1e-16, and standardising by it wrecked the empirical-Bayes priors for every
protein. sva::ComBat excludes such features ("uniform expression within a single
batch") and passes them through unadjusted; the port lacked this. Added, using an
exact max == min test (numpy's var() == 0 misses most constant columns through
rounding). Plasma: 88 proteins passed through; placenta: 117. After the fix, batch
spread vs reference is 0.99/0.99 (plasma) and 0.99/0.96 (placenta), and still
matches sva on the saved validation data to 4.97e-14.
