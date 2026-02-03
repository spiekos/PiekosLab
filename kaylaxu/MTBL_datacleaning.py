### Data Cleaning and Quality Control for Pregnancy Deep Phenotyping Metabolomics Data
##### All data cleaning and quality control steps are outlined in the data/README.md
### Kayla Xu, Piekos Lab
### 01/28/2026

# set up environment
import pandas as pd
import numpy as np
import scipy.stats as sp
from scipy import ndimage as nd
import logging
import sys

##############################################################
# DATA CLEANING OVERVIEW 
# 1. Convert improper values to standard missing value.
# 2. Separate QC and biological samples.
# 3. Calculate median absolute deviation (MAD) for the standard compound runs (01 and 02), then filter out biological samples with a standard compound expression more than 5*MAD away from median (i.e. threshold = median +- (5 x MAD)). Track any samples that fail this test in a log file.
# 4. Filter out biological samples with >50% missingness. Track any samples that fail this test in a log file.
# 5. Calculate relative standard deviation (RSD = SD/Mean * 100) of all compounds in the QC pools, split by batch. Remove any metabolites with an RSD > 30%.
# 6. Remove any metabolites with >20% missingness. Double-check that no group has significantly differential patterns of missingness for discard analyses.
# 7. Generate a non-batch corrected PCA plot.
# 8. Batch normalization using the biological replicates (better than quantile normalization):
#       1. Identify and log the number of batch replicates (samples that are present in both batches).
#       2. Calculate the ratio for each sample replicate pair (batch 2 value / batch 1 value) for each metabolite. Store in a list for each metabolite, making sure to exclude any outliers. (what counts as an outlier?)
#       3. Take the median ratio for each metabolite.
#       4. Calculate the correction factor = 1 / median ratio for each metabolite.
#       5. Apply the correction factors to all batch 2 samples
# 9. Generate a batch-corrected PCA plot in which the color of the sample dots match the batch they were in.
# 10. For the samples ran in both batches - average metabolic expression.
# 11. Perform median normalization
# 12. Perform log2 transformation
# 13. Combine the POS and NEG compounds:
#       1. Identify compounds present in both
#       2. Check the correlation between the POS and NEG values
#               1. If it is r >= 0.9, then average the two samples (or if one is missing, take the non-missing vaue)
#               2. If r < 0.9, keep both separately and append _POS or _NEG to the compound name respectively.
# 14. Perform 1/2 minimum imputation of missing data.
# 15. Handle all additional formatting considerations described above. 

# converts improper or missing expression data to np.nan values
def convert_missing(x):
    try:
        val = float(x)
        if val == 0:
            return np.nan
        else:
            return val
    except:
        return np.nan





def main():
    print("blank")

if __name__ == "__main__":
    main()
