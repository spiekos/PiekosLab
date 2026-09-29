# Archived 2026-09-28: proteomics plasma file with wrong SubjectIDs

`proteomics_plasma_cleaned_with_metadata.csv` as produced 2026-08-11.

`clean_proteomics_data.py` derived SubjectID with `r"\s*[A-Z]+$"`, stripping every
trailing capital. Late-enrolment IDs lost their `E` (DP3-0233EA -> DP3-0233 instead
of DP3-0233E). 19 participants / 80 samples got IDs that exist nowhere in the master
table or locked_split.csv and were silently dropped from all proteomics modelling.
All 19 are complications.

Fix: drop whitespace, then only the final visit letter [A-E]. SubjectID is inserted
after ComBat, filtering and imputation, so the corrected file differs from this one
in the SubjectID column only (verified: protein matrix identical).
