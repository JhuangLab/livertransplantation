#!/usr/bin/env python3
"""
Cluster heatmap for two-group comparisons.
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

# Global font
plt.rcParams['font.family'] = 'Arial'

# Group mapping and colors (using mapped names)
GROUP_MAPPING = get_group_mapping()
GROUP_COLORS = {
    'Control': '#90BFF9',
    'Rejection-treat-pre': '#F2B77C',
    'Rejection-treat-post': '#F59092',
}

# Font sizes
HEATMAP_SAMPLE_FONT_SIZE = 30
HEATMAP_FEATURE_FONT_SIZE = 30
LEGEND_FONT_SIZE = 29
LEGEND_TITLE_FONT_SIZE = 32
COLORBAR_TICK_FONT_SIZE = 26
COLORBAR_LABEL_FONT_SIZE = 26
ANNOTATION_FONT_SIZE = 20


def load_consensus_features(matrix_file, min_votes=5):
    """
    Load high-consensus features from a binary matrix.

    Parameters
    ----------
    matrix_file : str
        CSV file with feature-by-method 0/1 matrix.
    min_votes : int
        Minimum number of methods voting for a feature.

    Returns
    -------
    features : list of str
        Consensus feature IDs.
    votes : dict
        {feature: vote_count}
    """
    if not os.path.exists(matrix_file):
        print(f"File not found: {matrix_file}")
        return [], {}

    df = pd.read_csv(matrix_file, index_col=0)
    method_cols = list(df.columns)
    df['total_votes'] = df[method_cols].sum(axis=1)
    consensus_df = df[df['total_votes'] >= min_votes].copy()
    features = consensus_df.index.tolist()
    votes = dict(zip(consensus_df.index, consensus_df['total_votes']))
    return features, votes


def calculate_feature_discriminative_power(data_df, y_all):
    """
    Compute discriminative power of each feature (ANOVA F-test).

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


def plot_final_heatmap(
    features,
    omics_type,
    save_dir,
    comparison_name,
    use_robust_scaler=False,
    emphasize_discriminative=True,
    clustering_method='correlation',
    min_votes=5,
):
    """
    Final optimized heatmap for one omics and one comparison.

    Parameters
    ----------
    features : list of str
        Consensus feature IDs.
    omics_type : {'protein', 'metabolite'}
    save_dir : str
    comparison_name : {'Control_vs_Pre', 'Pre_vs_Post'}
    use_robust_scaler : bool
    emphasize_discriminative : bool
    clustering_method : str
        One of predefined clustering configs.
    min_votes : int
        Used for output file naming.
    """
    print(f"\nCreating optimized heatmap for {omics_type} - {comparison_name}...")

    protein_df, metabolite_df, y_all, sample_ids, gender = load_original_data()
    feature_mappings = load_feature_mappings()

    # Select samples for this comparison
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

    data_df = comp_protein_df if omics_type == 'protein' else comp_metabolite_df
    available_features = [f for f in features if f in data_df.columns]

    if len(available_features) == 0:
        print("No consensus features present in data; skipping.")
        return

    subset_df = data_df[available_features].copy()
    subset_df = subset_df.apply(pd.to_numeric, errors='coerce')
    subset_df = subset_df.fillna(subset_df.mean())
    print(f"Using {len(available_features)} consensus features.")

    # Scaling
    if use_robust_scaler:
        print("Using RobustScaler.")
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    data_scaled = scaler.fit_transform(subset_df)
    data_scaled_df = pd.DataFrame(
        data_scaled, index=subset_df.index, columns=subset_df.columns
    )

    # Discriminative weighting
    if emphasize_discriminative:
        print("Applying discriminative feature weighting.")
        F_values, p_values = calculate_feature_discriminative_power(
            subset_df, comp_y_all
        )
        F_normalized = F_values / np.max(F_values)
        weights = 0.3 + 1.7 * F_normalized  # in [0.3, 2.0]
        data_scaled_df = data_scaled_df * weights

        mapped_names = map_feature_names(
            available_features, omics_type, feature_mappings
        )
        top_idx = np.argsort(F_values)[-5:][::-1]
        print("Top 5 discriminative features:")
        for idx in top_idx:
            print(
                f"{mapped_names[idx][:50]}: "
                f"F={F_values[idx]:.2f}, p={p_values[idx]:.2e}"
            )

    # Group-only column colors
    y_series = pd.Series(comp_y_all, index=data_scaled_df.index)
    if comparison_name == 'Control_vs_Pre':
        group_color_map = {
            'Control': GROUP_COLORS['Control'],
            'Rejection-treat-pre': GROUP_COLORS['Rejection-treat-pre'],
        }
    else:
        group_color_map = {
            'Rejection-treat-pre': GROUP_COLORS['Rejection-treat-pre'],
            'Rejection-treat-post': GROUP_COLORS['Rejection-treat-post'],
        }

    col_colors = y_series.map(group_color_map)

    # Feature labels
    mapped_names = map_feature_names(
        available_features, omics_type, feature_mappings
    )
    mapped_names_short = [
        n[:60] + '...' if len(n) > 60 else n for n in mapped_names
    ]

    # Clustering config
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
    print(f"Clustering: {method} linkage + {metric} distance (k={n_clusters})")

    # Heatmap
    print("  Generating heatmap...")
    g = sns.clustermap(
        data_scaled_df.T,
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
        figsize=(20, max(12, len(available_features) * 0.3)),
        cbar_kws={'shrink': 0.6, 'aspect': 20},
        dendrogram_ratio=(0.12, 0.15),
        colors_ratio=0.07,
        linewidths=0,
        col_colors=col_colors,
        cbar_pos=(0.02, 0.83, 0.03, 0.12),
    )

    # Add group labels on top of column color bar, once per cluster of same group
    g.ax_col_colors.set_xticks([])
    g.ax_col_colors.set_yticks([])

    clustered_sample_order = g.ax_heatmap.get_xticklabels()
    clustered_sample_names = [tick.get_text() for tick in clustered_sample_order]

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
            fontsize=ANNOTATION_FONT_SIZE + 6,
            fontweight='bold',
            color='black',
        )

    # Colorbar font
    if hasattr(g, 'cax') and g.cax is not None:
        g.cax.tick_params(labelsize=COLORBAR_TICK_FONT_SIZE)
        if hasattr(g.cax, 'yaxis'):
            g.cax.yaxis.label.set_fontsize(COLORBAR_LABEL_FONT_SIZE)

    # Tick label fonts
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

    comparison_dir = os.path.join(save_dir, comparison_name)
    os.makedirs(comparison_dir, exist_ok=True)
    output_file = os.path.join(
        comparison_dir,
        f'{omics_type}_heatmap_{comparison_name}_vote{min_votes}{suffix}.png',
    )
    g.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap: {output_file}")

    # Simple mismatch report between clustering and labels
    distances = pdist(data_scaled_df, metric=metric)
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
        print(
            f"{len(misclassified)} samples have cluster–group mismatch "
            f"({comparison_name}, {omics_type})."
        )
        misclass_file = os.path.join(
            save_dir,
            f'{omics_type}_misclassified_samples_{comparison_name}_vote{min_votes}.txt',
        )
        with open(misclass_file, 'w') as f:
            f.write(
                f"Samples with clustering–group mismatch ({comparison_name}):\n"
            )
            f.write("=" * 60 + "\n")
            for item in misclassified:
                f.write(
                    f"{item['sample_id']}\t{item['true_group']}\t"
                    f"Cluster {item['assigned_cluster']}\t"
                    f"(Expected: {item['expected_cluster']})\n"
                )
        print(f"Saved mismatch list: {misclass_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Final optimized heatmap for two-group comparisons'
    )
    parser.add_argument('--input_dir', default='top50_features_two_comparisons')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument(
        '--omics',
        choices=['protein', 'metabolite', 'both'],
        default='both',
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
    print("Heatmap (two-group comparisons)")
    print("=" * 80)
    print("  Protein: min_votes = 5")
    print("  Metabolite: min_votes = 6")
    print("  Comparisons: Control_vs_Pre, Pre_vs_Post")
    print("  Sample names shown at bottom")
    print(f"Discriminative weighting: {'OFF' if args.no_weight else 'ON'}")
    print(f"Scaler: {'RobustScaler' if args.robust else 'StandardScaler'}")
    print("=" * 80)

    omics_types = ['protein', 'metabolite'] if args.omics == 'both' else [args.omics]
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

    for omics_type in omics_types:
        min_votes = 5 if omics_type == 'protein' else 6

        for comparison_name in comparisons:
            print(f"\n=== {omics_type.upper()} - {comparison_name} ===")
            print(f"min_votes = {min_votes}")

            comparison_dir = os.path.join(input_dir, comparison_name)
            matrix_file = os.path.join(
                comparison_dir, f'{omics_type}_feature_binary_matrix.csv'
            )

            if not os.path.exists(matrix_file):
                print(f"Feature matrix not found: {matrix_file}")
                continue

            features, votes = load_consensus_features(matrix_file, min_votes)
            print(f"Loaded {len(features)} consensus features.")

            if len(features) == 0:
                print("  No features to plot; skipping.")
                continue

            for clustering_method in clustering_methods:
                print(f"Method: {clustering_method}")
                try:
                    plot_final_heatmap(
                        features,
                        omics_type,
                        output_dir,
                        comparison_name,
                        use_robust_scaler=args.robust,
                        emphasize_discriminative=not args.no_weight,
                        clustering_method=clustering_method,
                        min_votes=min_votes,
                    )
                except Exception as e:
                    print(f"Error with {clustering_method}: {e}")

    print("\nCompleted.")


if __name__ == '__main__':
    main()
