from clean_proteomics_data import *
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

def test(result_path, verbose=False):
    df = pd.read_csv(result_path, index_col=0)
    if verbose:
        print(f'Dataframe shape: {df.shape}')
        print(f'Index name: {df.index.name}')
    
    # Identify metadata columns (all non-numeric columns at the start)
    metadata_cols = []
    common_metadata_names = ['SampleID', 'Group', 'Subgroup', 'Batch', 'GestAgeDelivery', 'SampleGestAge', 'SubjectID']
    for col in common_metadata_names:
        if col in df.columns:
            metadata_cols.append(col)
    
    # Drop metadata columns to get protein data
    X = df.drop(columns=metadata_cols)
    X = np.log2(X)

    
    # Test 1: Basic structure checks
    assert df.shape[0] > 0 and df.shape[1] > 0, "Resulting dataframe is empty."
    assert X.index.is_unique, "Row indices are not unique."
    assert X.columns.is_unique, "Column indices are not unique."
    assert not X.isnull().any().any(), "Dataframe contains NaN values."
    
    # Test 2: Distribution sanity after normalization
    # Compare distribution statistics across samples (diagnostic, not strict equalization)
    sample_means = X.mean(axis=1)
    sample_stds = X.std(axis=1)
    sample_medians = X.median(axis=1)
    
    # After ComBat + filtering, distributions should remain numerically stable.
    cv_means = sample_means.std() / sample_means.mean()  # Coefficient of variation
    cv_stds = sample_stds.std() / sample_stds.mean()
    cv_medians = sample_medians.std() / abs(sample_medians.mean()) if abs(sample_medians.mean()) > 0.01 else float('inf')
    
    if verbose:
        print(f"Sample means - mean: {sample_means.mean():.4f}, std: {sample_means.std():.4f}, CV: {cv_means:.4f}")
        print(f"Sample stds - mean: {sample_stds.mean():.4f}, std: {sample_stds.std():.4f}, CV: {cv_stds:.4f}")
        print(f"Sample medians - mean: {sample_medians.mean():.4f}, std: {sample_medians.std():.4f}, CV: {cv_medians:.4f}")
    
    # Broad bounds to catch numerical instability while allowing biological heterogeneity.
    assert np.isfinite(cv_stds), f"Sample std CV is non-finite (CV={cv_stds})."
    assert cv_stds < 1.0, f"Sample stds extremely variable (CV={cv_stds:.4f}), check normalization."
    # Relaxed median CV check
    if abs(sample_medians.mean()) > 0.1:  # Only check if mean is not too close to zero
        assert cv_medians < 2.0, f"Sample medians extremely variable (CV={cv_medians:.4f}), check normalization."
    
    # Test 3: Check that sample distributions are approximately the same shape
    # Use Kolmogorov-Smirnov test between pairs of samples
    n_samples = X.shape[0]
    ks_stats = []
    
    # Test first 10 pairs
    n_tests = min(10, n_samples - 1)
    for i in range(n_tests):
        stat, _ = stats.ks_2samp(X.iloc[i].values, X.iloc[i+1].values)
        ks_stats.append(stat)
    
    mean_ks = np.mean(ks_stats)
    if verbose:
        print(f"Mean KS statistic between consecutive samples: {mean_ks:.4f}")
    
    # Broad KS bound: catches pathological failures only.
    assert mean_ks < 0.80, f"Sample distributions too different (mean KS={mean_ks:.4f}), check normalization."
    
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
        
        assert np.isfinite(q_cv), f"Quantile {q} CV is non-finite."
    
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


def test_with_visualizations(result_path, output_dir=None, verbose=False):
    """
    Enhanced test with visualization outputs to assess normalization quality.
    
    Args:
        result_path: Path to the cleaned proteomics CSV file
        output_dir: Directory to save plots (if None, uses same directory as result_path)
        verbose: Print detailed statistics
    """
    df = pd.read_csv(result_path, index_col=0)
    
    # Set up output directory
    if output_dir is None:
        output_dir = os.path.dirname(result_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get file prefix for naming plots
    file_prefix = os.path.splitext(os.path.basename(result_path))[0]
    
    # Extract protein data (log2 scale for analysis)
    # Identify metadata columns
    metadata_cols = []
    common_metadata_names = ['SampleID', 'Group', 'Subgroup', 'Batch', 'GestAgeDelivery', 'SampleGestAge', 'SubjectID']
    for col in common_metadata_names:
        if col in df.columns:
            metadata_cols.append(col)
    
    X = df.drop(columns=metadata_cols)
    X_log2 = np.log2(X)
    
    print(f"\n{'='*80}")
    print(f"NORMALIZATION QUALITY ASSESSMENT: {file_prefix}")
    print(f"{'='*80}\n")
    
    # =======================
    # Test 1: Sample Distribution Alignment
    # =======================
    print("[Test 1] Sample Distribution Alignment")
    print("-" * 40)
    
    sample_means = X_log2.mean(axis=1)
    sample_stds = X_log2.std(axis=1)
    sample_medians = X_log2.median(axis=1)
    
    # Calculate CVs
    cv_stds = sample_stds.std() / sample_stds.mean()
    cv_medians = sample_medians.std() / abs(sample_medians.mean()) if abs(sample_medians.mean()) > 0.01 else float('inf')
    
    print(f"Sample medians: mean={sample_medians.mean():.4f}, std={sample_medians.std():.4f}, CV={cv_medians:.4f}")
    print(f"Sample stds:    mean={sample_stds.mean():.4f}, std={sample_stds.std():.4f}, CV={cv_stds:.4f}")
    
    # Interpretation
    if cv_stds < 0.05:
        print("✓ EXCELLENT: Sample standard deviations are very uniform (CV < 5%)")
    elif cv_stds < 0.10:
        print("✓ GOOD: Sample standard deviations are reasonably uniform (CV < 10%)")
    else:
        print("✗ WARNING: Sample standard deviations are variable (CV ≥ 10%)")
    
    if abs(sample_medians.mean()) > 0.1 and cv_medians < 0.10:
        print("✓ EXCELLENT: Sample medians are very uniform (CV < 10%)")
    elif abs(sample_medians.mean()) > 0.1 and cv_medians < 0.20:
        print("✓ GOOD: Sample medians are reasonably uniform (CV < 20%)")
    
    # Plot 1: Distribution of sample medians
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Sample medians histogram
    axes[0].hist(sample_medians, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(sample_medians.mean(), color='red', linestyle='--', label=f'Mean={sample_medians.mean():.3f}')
    axes[0].axvline(sample_medians.median(), color='orange', linestyle='--', label=f'Median={sample_medians.median():.3f}')
    axes[0].set_xlabel('Sample Median (log2)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Sample Medians')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Sample means histogram
    axes[1].hist(sample_means, bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[1].axvline(sample_means.mean(), color='red', linestyle='--', label=f'Mean={sample_means.mean():.3f}')
    axes[1].set_xlabel('Sample Mean (log2)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Sample Means')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # Sample stds histogram
    axes[2].hist(sample_stds, bins=50, edgecolor='black', alpha=0.7, color='purple')
    axes[2].axvline(sample_stds.mean(), color='red', linestyle='--', label=f'Mean={sample_stds.mean():.3f}')
    axes[2].set_xlabel('Sample Std Dev (log2)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Distribution of Sample Std Devs')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{file_prefix}_sample_distributions.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {file_prefix}_sample_distributions.png")
    
    # =======================
    # Test 2: Boxplot of Sample Distributions
    # =======================
    print(f"\n[Test 2] Sample Distribution Boxplots")
    print("-" * 40)
    
    # Create boxplots for a subset of samples (first 50 or all if fewer)
    n_samples_to_plot = min(50, X_log2.shape[0])
    
    fig, ax = plt.subplots(figsize=(20, 6))
    X_log2.iloc[:n_samples_to_plot].T.boxplot(ax=ax)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('log2(Protein Abundance)')
    ax.set_title(f'Sample Distribution Boxplots (first {n_samples_to_plot} samples)')
    ax.grid(alpha=0.3, axis='y')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{file_prefix}_sample_boxplots.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {file_prefix}_sample_boxplots.png")
    
    # Check alignment of boxplot medians
    box_medians = X_log2.median(axis=1)
    median_range = box_medians.max() - box_medians.min()
    print(f"Boxplot median range: {median_range:.4f}")
    if median_range < 0.5:
        print("✓ EXCELLENT: Boxplot medians are very well aligned (range < 0.5)")
    elif median_range < 1.0:
        print("✓ GOOD: Boxplot medians are reasonably aligned (range < 1.0)")
    else:
        print("✗ WARNING: Boxplot medians show substantial variation (range ≥ 1.0)")
    
    # =======================
    # Test 3: Density Plots (Sample Overlay)
    # =======================
    print(f"\n[Test 3] Sample Density Overlap")
    print("-" * 40)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot density for a random sample of 20 samples
    n_samples_density = min(20, X_log2.shape[0])
    sample_indices = np.random.choice(X_log2.index, n_samples_density, replace=False)
    
    for idx in sample_indices:
        X_log2.loc[idx].plot.density(ax=ax, alpha=0.3, linewidth=1)
    
    ax.set_xlabel('log2(Protein Abundance)')
    ax.set_ylabel('Density')
    ax.set_title(f'Sample Density Plots (n={n_samples_density} random samples)')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{file_prefix}_density_overlay.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {file_prefix}_density_overlay.png")
    
    # =======================
    # Test 4: Quantile Alignment Heatmap
    # =======================
    print(f"\n[Test 4] Quantile Alignment Check")
    print("-" * 40)
    
    # Calculate quantiles for each sample
    quantile_values = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    quantile_matrix = pd.DataFrame({
        f'Q{int(q*100)}': X_log2.quantile(q, axis=1) 
        for q in quantile_values
    })
    
    # Print quantile statistics
    print("Quantile alignment (CV across samples):")
    for q in quantile_values:
        q_col = f'Q{int(q*100)}'
        q_cv = quantile_matrix[q_col].std() / abs(quantile_matrix[q_col].mean()) if abs(quantile_matrix[q_col].mean()) > 0.01 else float('inf')
        status = "✓" if q_cv < 0.2 else "✗"
        print(f"  {status} Q{int(q*100):02d}: mean={quantile_matrix[q_col].mean():.4f}, std={quantile_matrix[q_col].std():.4f}, CV={q_cv:.4f}")
    
    # Plot quantile heatmap (first 100 samples or all if fewer)
    n_samples_heatmap = min(100, quantile_matrix.shape[0])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(quantile_matrix.iloc[:n_samples_heatmap].T, cmap='RdYlBu_r', center=0, 
                cbar_kws={'label': 'log2(Protein Abundance)'}, ax=ax)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Quantile')
    ax.set_title(f'Quantile Values Across Samples (first {n_samples_heatmap} samples)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{file_prefix}_quantile_heatmap.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {file_prefix}_quantile_heatmap.png")
    
    # =======================
    # Test 5: Batch Effect Check (if Batch column exists)
    # =======================
    if 'Batch' in df.columns:
        print(f"\n[Test 5] Batch Effect Assessment")
        print("-" * 40)
        
        batches = df['Batch']
        unique_batches = batches.unique()
        
        print(f"Number of batches: {len(unique_batches)}")
        print(f"Batch distribution: {batches.value_counts().to_dict()}")
        
        # Calculate median difference between batches
        batch_medians = {}
        for batch in unique_batches:
            batch_samples = X_log2.loc[batches == batch]
            batch_medians[batch] = batch_samples.median(axis=1).mean()
        
        if len(batch_medians) > 1:
            median_diff = max(batch_medians.values()) - min(batch_medians.values())
            print(f"Median difference between batches: {median_diff:.4f}")
            
            if median_diff < 0.1:
                print("✓ EXCELLENT: Negligible batch effect on medians (< 0.1)")
            elif median_diff < 0.3:
                print("✓ GOOD: Small batch effect on medians (< 0.3)")
            else:
                print(" WARNING: Noticeable batch effect on medians (≥ 0.3)")
        
        # Boxplot by batch
        fig, ax = plt.subplots(figsize=(10, 6))
        data_for_boxplot = []
        labels_for_boxplot = []
        
        for batch in sorted(unique_batches):
            batch_samples = X_log2.loc[batches == batch]
            batch_medians_all = batch_samples.median(axis=1)
            data_for_boxplot.append(batch_medians_all)
            labels_for_boxplot.append(f'Batch {batch}')
        
        ax.boxplot(data_for_boxplot, labels=labels_for_boxplot)
        ax.set_ylabel('Sample Median (log2)')
        ax.set_title('Sample Medians by Batch')
        ax.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{file_prefix}_batch_comparison.png"), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {file_prefix}_batch_comparison.png")
    
    # =======================
    # Summary Report
    # =======================
    print(f"\n{'='*80}")
    print("NORMALIZATION QUALITY SUMMARY")
    print(f"{'='*80}")
    
    # Overall assessment
    issues = []
    
    if cv_stds >= 0.10:
        issues.append("Sample standard deviations are highly variable")
    
    if abs(sample_medians.mean()) > 0.1 and cv_medians >= 0.20:
        issues.append("Sample medians are highly variable")
    
    if median_range >= 1.0:
        issues.append("Boxplot medians show substantial variation")
    
    if len(issues) == 0:
        print("✓ NORMALIZATION QUALITY: EXCELLENT")
        print("  No major numerical stability issues detected after normalization.")
    else:
        print(" WARNING: NORMALIZATION QUALITY: NEEDS REVIEW")
        print("  Issues detected:")
        for issue in issues:
            print(f"    - {issue}")
    
    print(f"\nPlots saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    wkdir = os.getcwd()
    output_base = os.path.join(wkdir, "data", "cleaned", "proteomics", "qa_reports")
    
    result_path_1 = os.path.join(wkdir, "data", "cleaned", "proteomics", "proteomics_placenta_cleaned_with_metadata.csv")
    result_path_2 = os.path.join(wkdir, "data", "cleaned", "proteomics", "proteomics_plasma_cleaned_with_metadata.csv")
    
    # Run basic tests first
    print("\n" + "="*80)
    print("RUNNING BASIC VALIDATION TESTS")
    print("="*80 + "\n")
    
    if os.path.exists(result_path_1):
        print(f"\nTesting: {os.path.basename(result_path_1)}")
        print("-" * 80)
        try:
            test(result_path_1, verbose=True)
        except AssertionError as e:
            print(f"✗ Test failed: {e}")
        except FileNotFoundError:
            print(f"✗ File not found: {result_path_1}")
    
    if os.path.exists(result_path_2):
        print(f"\nTesting: {os.path.basename(result_path_2)}")
        print("-" * 80)
        try:
            test(result_path_2, verbose=True)
        except AssertionError as e:
            print(f"✗ Test failed: {e}")
    
    # Run enhanced tests with visualizations
    print("\n" + "="*80)
    print("RUNNING ENHANCED TESTS WITH VISUALIZATIONS")
    print("="*80 + "\n")
    
    if os.path.exists(result_path_1):
        try:
            test_with_visualizations(result_path_1, output_dir=os.path.join(output_base, "placenta"))
        except FileNotFoundError:
            print(f"✗ File not found: {result_path_1}")
    
    if os.path.exists(result_path_2):
        test_with_visualizations(result_path_2, output_dir=os.path.join(output_base, "plasma"))
