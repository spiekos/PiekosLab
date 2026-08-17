import sys
import warnings
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pylimma

# Suppress warnings for cleaner output during batch processing
warnings.filterwarnings('ignore', category=FutureWarning)


def process_pca(adata, color_by, output_path):
    """Runs PCA and saves a dynamically named scatter plot."""
    # PCA on samples (centre proteins)
    Xc = adata.X - adata.X.mean(axis=0, keepdims=True)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = U * s
    var_explained = (s ** 2) / (s ** 2).sum()

    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Dynamically generate the group palette
    unique_groups = adata.obs[color_by].unique()
    num_groups = len(unique_groups)
    cmap = plt.get_cmap('tab10' if num_groups <= 10 else 'rainbow')
    colors = [cmap(i / max(1, num_groups - 1)) if num_groups > 10 else cmap(i) for i in range(num_groups)]
    group_palette = dict(zip(unique_groups, colors))

    for grp, colour in group_palette.items():
        m = (adata.obs[color_by] == grp).values
        ax.scatter(pcs[m, 0], pcs[m, 1], s=80, color=colour, edgecolor='black', label=grp)

    ax.set_xlabel(f'PC1 ({var_explained[0]:.0%})')
    ax.set_ylabel(f'PC2 ({var_explained[1]:.0%})')
    ax.set_title(f'{output_path.stem} - PCA ({color_by})')
    
    # Move legend outside the plot area
    ax.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_limma_pipeline(adata, feature, target_group, control_group, output_csv, output_volcano):
    """
    Runs pylimma on the anndata object, saves significant results, 
    and generates a volcano plot.
    """
    # 1. Setup Design Matrix (Example: target vs control)
    # Filter dataset to only include the two groups for a clean comparison
  
    group = adata.obs[feature]

    try: 
        design = pd.DataFrame({
                'Intercept':     1.0,
                target_group: group.str.contains(target_group, regex=True, na=False).astype(float).values,
                }, index=adata.obs_names)
        contrasts = pylimma.make_contrasts(CompvsCtrl=target_group, levels=design)
    except:
        if target_group == "PE|HELLP":
            design = pd.DataFrame({
                            'Intercept':     1.0,
                            "PE": group.str.contains(target_group, regex=True, na=False).astype(float).values,
                            }, index=adata.obs_names)
            contrasts = pylimma.make_contrasts(CompvsCtrl="PE", levels=design)
        else:
            design = pd.DataFrame({
                                'Intercept':     1.0,
                                "Complication": group.str.contains(target_group, regex=True, na=False).astype(float).values,
                                }, index=adata.obs_names)
            contrasts = pylimma.make_contrasts(CompvsCtrl="Complication", levels=design)
        
    try:
        pylimma.lm_fit(adata, design)
        pylimma.contrasts_fit(adata, contrasts=contrasts)
        pylimma.e_bayes(adata)

        results_df = pylimma.top_table(adata, coef= 'CompvsCtrl',
                            number=np.inf, sort_by='p').join(adata.var[['uniprot_id']], how='left')
        
        
        # Calculate adjusted p-values (FDR using Benjamini-Hochberg)
        #from statsmodels.stats.multitest import multipletests
        #results_df['adj.P.Val'] = multipletests(results_df['P.Value'], method='fdr_bh')[1]
        
    except Exception as e:
        print(f"Limma execution failed for {output_csv.stem}: {e}")
        return

    # 2. Filter and Save Significant Results
    #p_val_threshold = 0.05
    #logfc_threshold = 1.0
    
    top = pylimma.top_table(adata, coef='CompvsCtrl',
                            number=np.inf, sort_by='p')
    
    top = top.join(adata.var[['uniprot_id']], how='left')

    pvalue_threshold = 0.05
    logfc_threshold = 0

    significant_df = top[
        (top['adj_p_value'] < pvalue_threshold) & 
        (top['log_fc'].abs() >= logfc_threshold)
    ]
    significant_df.to_csv(output_csv, index=False)
    print(f"Saved {len(significant_df)} significant proteins to {output_csv.name}")

    n_sig_05 = int((top['adj_p_value'] < 0.05).sum())
    n_sig_01 = int((top['adj_p_value'] < 0.01).sum())
    print(f'differentially abundant proteins at adj_p_value < 0.05: {n_sig_05:,}')
    print(f'differentially abundant proteins at adj_p_value < 0.01: {n_sig_01:,}')

    display_cols = ['uniprot_id', 'log_fc', 'ave_expr', 't',
                    'p_value', 'adj_p_value', 'b']
    top.loc[:, display_cols].head(15).round(3)

    fig, ax = plt.subplots(figsize=(7, 6))

    sig_up   = (top['adj_p_value'] < pvalue_threshold) & (top['log_fc'] > logfc_threshold)
    sig_down = (top['adj_p_value'] < pvalue_threshold) & (top['log_fc'] < logfc_threshold)
    ns       = ~(sig_up | sig_down)

    nlp = -np.log10(np.maximum(top['p_value'].values, 1e-300))

    ax.scatter(top.loc[ns, 'log_fc'],   nlp[ns.values],
            s=8, color='lightgrey', alpha=0.6, edgecolor='none')
    ax.scatter(top.loc[sig_down, 'log_fc'], nlp[sig_down.values],
            s=12, color='#4c78a8', alpha=0.8,
            edgecolor='none', label= 'down in complication')
    ax.scatter(top.loc[sig_up, 'log_fc'],   nlp[sig_up.values],
            s=12, color='#e45756', alpha=0.8,
            edgecolor='none', label='up in complication')


    sig = top[top['adj_p_value'] < 0.05].copy()
    top_up   = sig.sort_values('log_fc', ascending=False).head(5)
    top_down = sig.sort_values('log_fc', ascending=True ).head(5)
    for _, row in pd.concat([top_up, top_down]).iterrows():
        label = row['uniprot_id']
        if not isinstance(label, str) or ';' in label or label == '':
            continue
        ax.annotate(label, (row['log_fc'], -np.log10(max(row['p_value'], 1e-300))),
                    fontsize=8, ha='left', va='bottom')

    ax.axhline(-np.log10(0.05), color='black', lw=0.5, linestyle='--')
    ax.axvline(0, color='black', lw=0.5)
    ax.set_xlabel('log2 fold-change (Complication - control)')
    ax.set_ylabel('-log10 p-value')
    ax.set_title(f'{complication} vs control - placenta proteome ')
    ax.legend(frameon=False, loc='lower right')
    fig.tight_layout()
    plt.savefig(output_volcano)
    plt.close()
    '''
    # 3. Create Volcano Plot
    results_df['-log10(p-value)'] = -np.log10(results_df['p_value'])
    
    # Categorize for coloring
    results_df['Significance'] = 'Not Sig'
    results_df.loc[(results_df['log_fc'] >= logfc_threshold) & (results_df['adj_p_value'] < p_val_threshold), 'Significance'] = 'Up'
    results_df.loc[(results_df['log_fc'] <= -logfc_threshold) & (results_df['adj_p_value'] < p_val_threshold), 'Significance'] = 'Down'
    
    plt.figure(figsize=(8, 6))
    palette = {'Not Sig': 'lightgrey', 'Up': '#e45756', 'Down': '#4c78a8'}
    
    sns.scatterplot(
        data=results_df, 
        x='log_fc', 
        y='-log10(p-value)', 
        hue='Significance',
        palette=palette,
        alpha=0.7,
        edgecolor=None
    )
    
    # Add threshold lines
    plt.axhline(-np.log10(0.05), color='black', linestyle='--', lw=1, alpha=0.5)
    plt.axvline(logfc_threshold, color='black', linestyle='--', lw=1, alpha=0.5)
    plt.axvline(-logfc_threshold, color='black', linestyle='--', lw=1, alpha=0.5)
    
    plt.title(f'{output_volcano.stem} - Volcano Plot ({target_group} vs {control_group})')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_volcano, dpi=300)
    plt.close()
    '''

def process_proteomics_files(file_list, output_dir, meta_columns, feature = "Group", target_group = "HDP", control_group="Control"):
    """
    Main loop to process a list of CSV files.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for file in file_list:
        file_path = Path(file)
        print(file)
        if not file_path.exists():
            print(f"File not found: {file_path}. Skipping...")
            continue

        if "plasma" in file:
            if "1.csv" in file:
                base_name = f"{target_group}_v_{control_group}_plasma_1"
            elif "2.csv" in file:
                base_name = f"{target_group}_v_{control_group}_plasma_2"
            elif "3.csv" in file:
                base_name = f"{target_group}_v_{control_group}_plasma_3"
            elif "4.csv" in file:
                base_name = f"{target_group}_v_{control_group}_plasma_4"
            elif "5.csv" in file:
                base_name = f"{target_group}_v_{control_group}_plasma_5"
            else:
                base_name = f"{target_group}_v_{control_group}_plasma"
        elif "placenta" in file:
            base_name = f"{target_group}_v_{control_group}_placenta"
        else:
            print(f"Tissue must be plasma or placenta - no valid tissue specified by file: {file}")
            return
        
        print(f"--- Processing {base_name} ---")
        
        # Load Data
        df = pd.read_csv(file_path)
        
        # Dynamically separate metadata (targets) and expression (proteome)
        current_meta_cols = [c for c in meta_columns if c in df.columns]
        obs_df = df[current_meta_cols].copy()
        obs_df.index = df['SampleID'] if 'SampleID' in df.columns else df.index
        
        proteome_cols = [c for c in df.columns if c not in current_meta_cols]
        X = df[proteome_cols].copy()
        X.index = obs_df.index
        
        var_df = pd.DataFrame(index=proteome_cols)
        var_df['uniprot_id'] = proteome_cols
        
        # Create AnnData
        adata = ad.AnnData(X=X, obs=obs_df, var=var_df)
        
        # Define Output Paths
        pca_out = out_path / f"batch_correction/PROT/corrected_{base_name}_PCA.jpeg"
        limma_csv_out = out_path / f"differential_expression/PROT/results/{base_name}_significant_results.csv"
        volcano_out = out_path / f"differential_expression/PROT/volcano_plots/{base_name}_volcano.jpeg"
        
        # Execute Tasks
        process_pca(adata, color_by='Group', output_path=pca_out)
        run_limma_pipeline(adata, feature, target_group, control_group, limma_csv_out, volcano_out)


if __name__ == "__main__":
    # Example Usage Configuration
    
    # 1. Define the metadata columns based on your notebook structure
    metadata_cols_to_extract = [
        'SampleID', 'SubjectID', 'SampleGestAge', 
        'Batch', 'Group', 'Subgroup', 'GestAgeDelivery', 'Timepoint'
    ]
    
    # 2. Provide the list of files you want to loop over
    files_to_process = [
        "/Users/kaylaxu/Desktop/dp3_project/data/processed/PROT/normalized_full_results/PROT_placenta.csv",
        "/Users/kaylaxu/Desktop/dp3_project/data/processed/PROT/normalized_full_results/PROT_plasma.csv",
        "/Users/kaylaxu/Desktop/dp3_project/data/processed/PROT/normalized_sliced_by_suffix/proteomics_plasma_formatted_suffix_1.csv",
        "/Users/kaylaxu/Desktop/dp3_project/data/processed/PROT/normalized_sliced_by_suffix/proteomics_plasma_formatted_suffix_2.csv",
        "/Users/kaylaxu/Desktop/dp3_project/data/processed/PROT/normalized_sliced_by_suffix/proteomics_plasma_formatted_suffix_3.csv",
        "/Users/kaylaxu/Desktop/dp3_project/data/processed/PROT/normalized_sliced_by_suffix/proteomics_plasma_formatted_suffix_4.csv",
        "/Users/kaylaxu/Desktop/dp3_project/data/processed/PROT/normalized_sliced_by_suffix/proteomics_plasma_formatted_suffix_5.csv"
    ]
    
    # 3. Define the destination folder for plots and tables
    output_directory = "/Users/kaylaxu/Desktop/dp3_project/04_results_and_figures"

    comps = {"FGR": "Group", 
             "HDP": "Group",
             "sPTB": "Group",
             "PE|HELLP": "Subgroup",
             "HDP|FGR|sPTB": "Group"}
    # 4. Execute the pipeline
    for complication in comps.keys():
        process_proteomics_files(
            file_list=files_to_process,
            output_dir=output_directory,
            meta_columns=metadata_cols_to_extract,
            feature=comps[complication],
            target_group=complication,      
            control_group="Control"
        )
    