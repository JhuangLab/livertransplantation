#!/usr/bin/env python3
"""
Multi-omics integrated clustered heatmap.

- Integrate consensus protein and metabolite features for joint clustering.
- Two metabolite names are shortened to:
    CMPF  = 3-carboxy-4-methyl-5-pentyl-2-furanpropanoic acid
    C21H34O6S = 5.alpha.-Pregnan-3.alpha., 17-diol-20-one 3-sulfate
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from ..extract_consensus_features.load_data import (
    load_original_data,
    load_feature_mappings,
    map_feature_names,
    get_group_mapping,
)

plt.rcParams['font.family'] = 'Arial'

# Group colors (using mapped group names)
GROUP_MAPPING = get_group_mapping()
GROUP_COLORS = {
    'Control': '#90BFF9',
    'Rejection-treat-pre': '#F2B77C',
    'Rejection-treat-post': '#F59092',
}

OMICS_COLORS = {
    'Protein': '#800080',
    'Metabolite': '#228B22',
}

# Font sizes for heatmap
HEATMAP_SAMPLE_FONT_SIZE = 28
HEATMAP_FEATURE_FONT_SIZE = 30
LEGEND_FONT_SIZE = 26
LEGEND_TITLE_FONT_SIZE = 26
COLORBAR_TICK_FONT_SIZE = 28
COLORBAR_LABEL_FONT_SIZE = 28
ANNOTATION_FONT_SIZE = 30


def load_consensus_features(matrix_file, min_votes=5):
    """
    Load high-consensus features from a binary matrix.

    Parameters
    ----------
    matrix_file : str
        CSV file with columns: Feature, method1, method2, ...
    min_votes : int
        Minimum number of methods voting for a feature.

    Returns
    -------
    list of str
        Consensus feature IDs.
    """
    df = pd.read_csv(matrix_file)
    method_cols = [col for col in df.columns if col != 'Feature']
    df['total_votes'] = df[method_cols].sum(axis=1)
    consensus_df = df[df['total_votes'] >= min_votes].copy()
    return consensus_df['Feature'].tolist()


def calculate_feature_discriminative_power(data_df, y_all):
    """
    Compute F-test values for each feature (ANOVA F-test).

    Parameters
    ----------
    data_df : pd.DataFrame
        Samples × features.
    y_all : list / array
        Class labels.

    Returns
    -------
    F_values : np.ndarray
    p_values : np.ndarray
    """
    from sklearn.feature_selection import f_classif

    X = data_df.values
    F_values, p_values = f_classif(X, y_all)
    return F_values, p_values


def plot_multiomics_heatmap(
    input_dir,
    save_dir,
    comparison_name,
    min_votes_protein=5,
    min_votes_metabolite=6,
    emphasize_discriminative=True,
    clustering_method='auto',
    use_robust_scaler=False,
):
    """
    Plot an integrated multi-omics clustered heatmap for a two-class comparison.

    Parameters
    ----------
    input_dir : str
        Directory containing feature binary matrices.
    save_dir : str
        Base output directory.
    comparison_name : {'Control_vs_Pre', 'Pre_vs_Post'}
    min_votes_protein : int
        Minimum votes for protein features.
    min_votes_metabolite : int
        Minimum votes for metabolite features.
    emphasize_discriminative : bool
        If True, weight features by discriminative power.
    clustering_method : str
        One of predefined clustering configs (or 'auto').
    use_robust_scaler : bool
        If True, use RobustScaler; else StandardScaler.
    """
    print(f"\nCreating multi-omics clustered heatmap for {comparison_name}...")

    # Load full original data (no manual sample removal)
    protein_df, metabolite_df, y_all, sample_ids, gender = load_original_data()
    feature_mappings = load_feature_mappings()

    # Select samples for the given comparison
    if comparison_name == 'Control_vs_Pre':
        target_groups = ['Control', 'Rejection-treat-pre']
    elif comparison_name == 'Pre_vs_Post':
        target_groups = ['Rejection-treat-pre', 'Rejection-treat-post']
    else:
        raise ValueError(f"Unknown comparison: {comparison_name}")

    comp_mask = np.array([y in target_groups for y in y_all])
    comp_protein_df = protein_df[comp_mask]
    comp_metabolite_df = metabolite_df[comp_mask]
    comp_y_all = [y_all[i] for i in range(len(y_all)) if comp_mask[i]]
    comp_sample_ids = [sample_ids[i] for i in range(len(sample_ids)) if comp_mask[i]]

    print(f"Samples in comparison: {len(comp_sample_ids)}")
    unique_labels, counts = np.unique(comp_y_all, return_counts=True)
    print(f"Class distribution: {dict(zip(unique_labels, counts))}")

    # Load consensus protein features from comparison-specific directory
    comparison_dir = os.path.join(input_dir, comparison_name)
    protein_matrix_file = os.path.join(comparison_dir, 'protein_feature_binary_matrix.csv')
    if not os.path.exists(protein_matrix_file):
        print(f"Protein matrix file not found: {protein_matrix_file}")
        return

    protein_features = load_consensus_features(
        protein_matrix_file, min_votes=min_votes_protein
    )
    protein_features = [f for f in protein_features if f in comp_protein_df.columns]

    # Load consensus metabolite features
    metabolite_matrix_file = os.path.join(comparison_dir, 'metabolite_feature_binary_matrix.csv')
    if not os.path.exists(metabolite_matrix_file):
        print(f"Metabolite matrix file not found: {metabolite_matrix_file}")
        return

    metabolite_features = load_consensus_features(
        metabolite_matrix_file, min_votes=min_votes_metabolite
    )
    metabolite_features = [
        f for f in metabolite_features if f in comp_metabolite_df.columns
    ]

    print(
        f"Protein features: {len(protein_features)} (min_votes ≥ {min_votes_protein})"
    )
    print(
        f"Metabolite features: {len(metabolite_features)} (min_votes ≥ {min_votes_metabolite})"
    )
    print(
        f"Total features: {len(protein_features) + len(metabolite_features)}"
    )

    if len(protein_features) == 0 and len(metabolite_features) == 0:
        print("  No consensus features found; skipping.")
        return

    # Subset data
    protein_subset = (
        comp_protein_df[protein_features].copy()
        if protein_features
        else pd.DataFrame(index=comp_protein_df.index)
    )
    metabolite_subset = (
        comp_metabolite_df[metabolite_features].copy()
        if metabolite_features
        else pd.DataFrame(index=comp_metabolite_df.index)
    )

    # Numeric conversion and missing value handling
    if protein_features:
        protein_subset = (
            protein_subset.apply(pd.to_numeric, errors='coerce')
            .fillna(protein_subset.mean())
        )
    if metabolite_features:
        metabolite_subset = (
            metabolite_subset.apply(pd.to_numeric, errors='coerce')
            .fillna(metabolite_subset.mean())
        )

    # Scaling
    if use_robust_scaler:
        print("  Using RobustScaler.")
        scaler_protein = RobustScaler() if protein_features else None
        scaler_metabolite = RobustScaler() if metabolite_features else None
    else:
        scaler_protein = StandardScaler() if protein_features else None
        scaler_metabolite = StandardScaler() if metabolite_features else None

    n_samples = len(comp_sample_ids)
    protein_scaled = (
        scaler_protein.fit_transform(protein_subset)
        if protein_features
        else np.empty((n_samples, 0))
    )
    metabolite_scaled = (
        scaler_metabolite.fit_transform(metabolite_subset)
        if metabolite_features
        else np.empty((n_samples, 0))
    )

    if protein_features and metabolite_features:
        combined_scaled = np.hstack([protein_scaled, metabolite_scaled])
    elif protein_features:
        combined_scaled = protein_scaled
    elif metabolite_features:
        combined_scaled = metabolite_scaled
    else:
        return

    combined_features = protein_features + metabolite_features
    combined_df = pd.DataFrame(
        combined_scaled,
        index=comp_protein_df.index if protein_features else comp_metabolite_df.index,
        columns=combined_features,
    )

    # Emphasize discriminative features by F-test weights
    if emphasize_discriminative:
        print("  Applying discriminative weighting.")
        F_values, p_values = calculate_feature_discriminative_power(
            combined_df, comp_y_all
        )
        F_normalized = F_values / np.max(F_values)
        weights = 0.3 + 1.7 * F_normalized  # range: 0.3–2.0
        combined_df = combined_df * weights

        mapped_all = (
            map_feature_names(protein_features, 'protein', feature_mappings)
            + map_feature_names(metabolite_features, 'metabolite', feature_mappings)
        )
        top_idx = np.argsort(F_values)[-5:][::-1]
        print("  Top 5 discriminative features:")
        for idx in top_idx:
            omics_type = 'Protein' if idx < len(protein_features) else 'Metabolite'
            print(
                f"[{omics_type}] {mapped_all[idx][:50]}: "
                f"F={F_values[idx]:.2f}, p={p_values[idx]:.2e}"
            )

    # Column colors (sample groups)
    y_series = pd.Series(comp_y_all, index=combined_df.index)
    if comparison_name == 'Control_vs_Pre':
        group_color_map = {
            'Control': GROUP_COLORS['Control'],
            'Rejection-treat-pre': GROUP_COLORS['Rejection-treat-pre'],
        }
    else:  # Pre_vs_Post
        group_color_map = {
            'Rejection-treat-pre': GROUP_COLORS['Rejection-treat-pre'],
            'Rejection-treat-post': GROUP_COLORS['Rejection-treat-post'],
        }
    col_colors = y_series.map(group_color_map)

    # Row colors (feature type)
    feature_types = ['Protein'] * len(protein_features) + ['Metabolite'] * len(
        metabolite_features
    )
    row_colors = pd.Series(
        [OMICS_COLORS[ft] for ft in feature_types],
        index=combined_df.columns,
    )

    # Map feature names to human-readable names
    mapped_names_all = (
        map_feature_names(protein_features, 'protein', feature_mappings)
        + map_feature_names(metabolite_features, 'metabolite', feature_mappings)
    )

    # Shorten two specific metabolite names
    def _shorten_metabolite_name(name: str) -> str:
        long_cmpf = '3-carboxy-4-methyl-5-pentyl-2-furanpropanoic acid'
        if long_cmpf in name:
            return 'CMPF'
        if ('Pregnan' in name or 'Pregnan-3' in name) and '17-diol-20-one 3-sulfate' in name:
            return 'C21H34O6S'
        return name

    mapped_names_all = [_shorten_metabolite_name(n) for n in mapped_names_all]

    # In Control_vs_Pre, rename the higher-expression Arg-Thr feature to Arg-Thr-2
    if comparison_name == 'Control_vs_Pre':
        arg_thr_idx = [i for i, n in enumerate(mapped_names_all) if n == 'Arg-Thr']
        if len(arg_thr_idx) >= 2:
            means = [combined_df.iloc[:, idx].mean() for idx in arg_thr_idx]
            max_pos = int(np.argmax(means))
            mapped_names_all[arg_thr_idx[max_pos]] = 'Arg-Thr-2'

    mapped_names_short = [
        name[:55] + '...' if len(name) > 55 else name for name in mapped_names_all
    ]

    # Clustering options
    clustering_configs = {
        'correlation': {'method': 'average', 'metric': 'correlation'},
        'euclidean': {'method': 'ward', 'metric': 'euclidean'},
        'cosine': {'method': 'average', 'metric': 'cosine'},
        'manhattan': {'method': 'complete', 'metric': 'cityblock'},
        'correlation_complete': {'method': 'complete', 'metric': 'correlation'},
        'euclidean_complete': {'method': 'complete', 'metric': 'euclidean'},
        'correlation_single': {'method': 'single', 'metric': 'correlation'},
        'euclidean_average': {'method': 'average', 'metric': 'euclidean'},
    }

    if clustering_method in clustering_configs:
        cfg = clustering_configs[clustering_method]
        method = cfg['method']
        metric = cfg['metric']
    else:
        method = 'average'
        metric = 'correlation'

    n_clusters = 2
    print(f"Clustering: method={method}, metric={metric}, k={n_clusters}")

    # Heatmap
    print("  Generating heatmap...")
    g = sns.clustermap(
        combined_df.T,
        cmap='RdBu_r',
        center=0,
        vmin=-3,
        vmax=3,
        row_cluster=True,
        col_cluster=True,
        method=method,
        metric=metric,
        yticklabels=mapped_names_short,
        xticklabels=True,
        figsize=(20, max(12, len(combined_features) * 0.55)),
        cbar_kws={'shrink': 0.6, 'aspect': 20},
        dendrogram_ratio=(0.12, 0.15),
        colors_ratio=0.04,
        linewidths=0,
        col_colors=col_colors,
        row_colors=row_colors,
        cbar_pos=(0.02, 0.83, 0.03, 0.12),
    )

    # Add group labels on top of column color bar
    g.ax_col_colors.set_xticks([])
    g.ax_col_colors.set_yticks([])

    clustered_xticks = g.ax_heatmap.get_xticklabels()
    clustered_sample_names = [t.get_text() for t in clustered_xticks]

    clustered_groups = []
    for sample_name in clustered_sample_names:
        idx = comp_sample_ids.index(sample_name)
        clustered_groups.append(comp_y_all[idx])

    group_positions = {}
    current_group = None
    start_idx = 0
    for idx, grp in enumerate(clustered_groups):
        if grp != current_group:
            if current_group is not None:
                group_positions[current_group] = (start_idx, idx - 1)
            current_group = grp
            start_idx = idx
    group_positions[current_group] = (start_idx, len(clustered_groups) - 1)

    for grp, (start, end) in group_positions.items():
        center_x = (start + end) / 2 + 0.5
        g.ax_col_colors.text(
            x=center_x,
            y=0.5,
            s=grp,
            ha='center',
            va='center',
            fontsize=ANNOTATION_FONT_SIZE,
            fontweight='bold',
            color='black',
        )

    # Legend for feature type
    from matplotlib.patches import Patch

    omics_legend_elems = [
        Patch(
            facecolor=OMICS_COLORS['Protein'],
            label='Protein',
            edgecolor='black',
            linewidth=0.5,
        ),
        Patch(
            facecolor=OMICS_COLORS['Metabolite'],
            label='Metabolite',
            edgecolor='black',
            linewidth=0.5,
        ),
    ]

    legend = g.fig.legend(
        handles=omics_legend_elems,
        title='Feature Type',
        loc='upper left',
        bbox_to_anchor=(0.97, 1.0),
        borderaxespad=0.5,
        frameon=True,
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_TITLE_FONT_SIZE,
        edgecolor='black',
    )
    legend.get_title().set_fontweight('bold')

    # Colorbar font sizes
    if hasattr(g, 'cax') and g.cax is not None:
        g.cax.tick_params(labelsize=COLORBAR_TICK_FONT_SIZE)
        if hasattr(g.cax, 'yaxis'):
            g.cax.yaxis.label.set_fontsize(COLORBAR_LABEL_FONT_SIZE)

    plt.setp(
        g.ax_heatmap.get_yticklabels(),
        rotation=0,
        fontsize=HEATMAP_FEATURE_FONT_SIZE,
    )
    plt.setp(
        g.ax_heatmap.get_xticklabels(),
        rotation=90,
        fontsize=HEATMAP_SAMPLE_FONT_SIZE,
        ha='center',
    )

    # Save figure
    suffix = f"_{clustering_method}_{method}_{metric}"
    if use_robust_scaler:
        suffix += "_robust"
    if emphasize_discriminative:
        suffix += "_weighted"

    comparison_save_dir = os.path.join(save_dir, comparison_name)
    os.makedirs(comparison_save_dir, exist_ok=True)
    output_file = os.path.join(
        comparison_save_dir,
        f'multiomics_heatmap_{comparison_name}_vote{min_votes_protein}_{min_votes_metabolite}{suffix}.png',
    )
    g.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap to: {output_file}")

    # Cluster assignment vs. labels (simple mismatch report)
    distances = pdist(combined_df, metric=metric)
    linkage_matrix = linkage(distances, method=method)
    clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

    misclassified = []
    for sample_id, true_group, cluster_id in zip(
        comp_sample_ids, comp_y_all, clusters
    ):
        group_mask = [y == true_group for y in comp_y_all]
        group_clusters = [c for c, m in zip(clusters, group_mask) if m]
        most_common_cluster = max(set(group_clusters), key=group_clusters.count)

        if cluster_id != most_common_cluster:
            misclassified.append(
                {
                    'sample_id': sample_id,
                    'true_group': true_group,
                    'assigned_cluster': cluster_id,
                    'expected_cluster': most_common_cluster,
                }
            )

    if misclassified:
        print(f"{len(misclassified)} samples have cluster–group mismatch.")
        misclass_file = os.path.join(
            comparison_save_dir,
            f'multiomics_misclassified_samples_{comparison_name}_vote{min_votes_protein}_{min_votes_metabolite}.txt',
        )
        with open(misclass_file, 'w') as f:
            f.write(
                f"Multi-omics: samples with clustering–group mismatch ({comparison_name})\n"
            )
            f.write("=" * 60 + "\n")
            for item in misclassified:
                f.write(
                    f"{item['sample_id']}\t{item['true_group']}\t"
                    f"Cluster {item['assigned_cluster']}\t"
                    f"(Expected: {item['expected_cluster']})\n"
                )
        print(f"Mismatch list saved to: {misclass_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Multi-omics integrated clustered heatmap (two-group comparisons)'
    )
    parser.add_argument('--input_dir', default='top50_features_two_comparisons')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument(
        '--min_votes_protein',
        type=int,
        default=5,
        help='Minimum votes for protein features',
    )
    parser.add_argument(
        '--min_votes_metabolite',
        type=int,
        default=6,
        help='Minimum votes for metabolite features',
    )
    parser.add_argument('--robust', action='store_true', help='Use RobustScaler')
    parser.add_argument(
        '--no-weight', action='store_true', help='Disable discriminative weighting'
    )
    parser.add_argument(
        '--method',
        choices=[
            'correlation',
            'euclidean',
            'cosine',
            'manhattan',
            'correlation_complete',
            'euclidean_complete',
            'correlation_single',
            'euclidean_average',
            'all',
        ],
        default='all',
        help='Clustering method (or "all" to try all)',
    )

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, args.input_dir)
    output_dir = args.output_dir if args.output_dir else input_dir

    print("=" * 80)
    print("Multi-omics integrated clustered heatmap (two-group comparisons)")
    print("=" * 80)
    print(f"Protein min_votes: {args.min_votes_protein}")
    print(f"Metabolite min_votes: {args.min_votes_metabolite}")
    print("  Comparisons: Control_vs_Pre, Pre_vs_Post")
    print(f"Discriminative weighting: {'OFF' if args.no_weight else 'ON'}")
    print(f"Scaler: {'RobustScaler' if args.robust else 'StandardScaler'}")
    print("=" * 80)

    comparisons = ['Control_vs_Pre', 'Pre_vs_Post']
    all_methods = [
        'correlation',
        'euclidean',
        'cosine',
        'manhattan',
        'correlation_complete',
        'euclidean_complete',
        'correlation_single',
        'euclidean_average',
    ]
    clustering_methods = all_methods if args.method == 'all' else [args.method]

    for comparison_name in comparisons:
        print(f"\n=== {comparison_name} ===")
        for clustering_method in clustering_methods:
            print(f"Method: {clustering_method}")
            try:
                plot_multiomics_heatmap(
                    input_dir,
                    output_dir,
                    comparison_name,
                    min_votes_protein=args.min_votes_protein,
                    min_votes_metabolite=args.min_votes_metabolite,
                    emphasize_discriminative=not args.no_weight,
                    clustering_method=clustering_method,
                    use_robust_scaler=args.robust,
                )
            except Exception as e:
                print(f"Error with {clustering_method}: {e}")

    print("\nDone.")


if __name__ == '__main__':
    main()