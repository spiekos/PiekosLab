from clean_proteomics_data import *
import scipy.stats as stats

def test(result_path, verbose=False):
    df = pd.read_csv(result_path, index_col=0)
    if verbose:
        print(f'Dataframe shape: {df.shape}')
    
    X = np.log2(df)
    
    # Test 1: Basic structure checks
    assert df.shape[0] > 0 and df.shape[1] > 0, "Resulting dataframe is empty."
    assert X.index.is_unique, "Row indices are not unique."
    assert X.columns.is_unique, "Column indices are not unique."
    assert not X.isnull().any().any(), "Dataframe contains NaN values."
    
    # Test 2: Check that distributions are similar across samples
    # Compare distribution statistics across samples
    sample_means = X.mean(axis=1)
    sample_stds = X.std(axis=1)
    sample_medians = X.median(axis=1)
    
    # After quantile normalization, these should be very similar (but not identical)
    cv_means = sample_means.std() / sample_means.mean()  # Coefficient of variation
    cv_stds = sample_stds.std() / sample_stds.mean()
    cv_medians = sample_medians.std() / sample_medians.mean()
    
    if verbose:
        print(f"Sample means - mean: {sample_means.mean():.4f}, std: {sample_means.std():.4f}, CV: {cv_means:.4f}")
        print(f"Sample stds - mean: {sample_stds.mean():.4f}, std: {sample_stds.std():.4f}, CV: {cv_stds:.4f}")
        print(f"Sample medians - mean: {sample_medians.mean():.4f}, std: {sample_medians.std():.4f}, CV: {cv_medians:.4f}")
    
    # These CVs should be small if quantile normalization worked
    assert cv_means < 0.15, f"Sample means too variable (CV={cv_means:.4f}), quantile normalization may have failed."
    assert cv_stds < 0.05, f"Sample stds too variable (CV={cv_stds:.4f}), quantile normalization may have failed."
    assert cv_medians < 0.20, f"Sample medians too variable (CV={cv_medians:.4f}), quantile normalization may have failed."
    
    # Test 3: Check that sample distributions are approximately the same shape
    # Use Kolmogorov-Smirnov test between pairs of samples
    n_samples = X.shape[0]
    ks_stats = []
    
    # Test first 10 pairs (or fewer if less samples)
    n_tests = min(10, n_samples - 1)
    for i in range(n_tests):
        stat, _ = stats.ks_2samp(X.iloc[i].values, X.iloc[i+1].values)
        ks_stats.append(stat)
    
    mean_ks = np.mean(ks_stats)
    if verbose:
        print(f"Mean KS statistic between consecutive samples: {mean_ks:.4f}")
    
    # KS statistic should be small if distributions are similar
    assert mean_ks < 0.15, f"Sample distributions too different (mean KS={mean_ks:.4f}), quantile normalization may have failed."
    
    # Test 4: Check that quantiles are approximately aligned across samples
    # For each quantile, check variance across samples
    quantiles = [0.25, 0.5, 0.75]
    for q in quantiles:
        q_values = X.quantile(q, axis=1)
        q_std = q_values.std()
        q_mean = q_values.mean()
        q_cv = q_std / abs(q_mean) if q_mean != 0 else float('inf')
        
        if verbose:
            print(f"Quantile {q}: mean={q_mean:.4f}, std={q_std:.4f}, CV={q_cv:.4f}")
        
        assert q_cv < 0.4, f"Quantile {q} too variable across samples (CV={q_cv:.4f}), quantile normalization may have failed."
    
    # Test 5: Check that the overall distribution shape is reasonable
    # All samples should have similar skewness and kurtosis
    skewness = X.apply(stats.skew, axis=1)
    kurtosis = X.apply(stats.kurtosis, axis=1)
    
    if verbose:
        print(f"Skewness - mean: {skewness.mean():.4f}, std: {skewness.std():.4f}")
        print(f"Kurtosis - mean: {kurtosis.mean():.4f}, std: {kurtosis.std():.4f}")
    
    # These should have low variance across samples
    assert skewness.std() < 1.0, f"Skewness too variable (std={skewness.std():.4f}), distributions may not be normalized."
    assert kurtosis.std() < 2.0, f"Kurtosis too variable (std={kurtosis.std():.4f}), distributions may not be normalized."
    
    print("All tests passed.")

if __name__ == "__main__":
    wkdir = os.getcwd()
    result_path = os.path.join(wkdir, "data", "cleaned", "proteomics", "proteomics_cleaned_panelnorm_quantilenorm_imputed.csv")
    test(result_path, verbose=True)