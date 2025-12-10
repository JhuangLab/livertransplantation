#!/usr/bin/env python3
"""
Visualize feature selection overlaps using UpSet and intersection matrices.

- Read feature binary matrices from CSV
- Draw UpSet plots (method intersections)
- Draw intersection matrices (pairwise shared feature counts)
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(
    0,
    '/cluster/home/liuyy_jh/projects/liver/analysis/wrr_final/models_votes_classification_two_groups'
)

# Global font
plt.rcParams['font.family'] = 'Arial'

# Global font sizes
TITLE_FONT_SIZE = 26
SUBTITLE_FONT_SIZE = 24
LABEL_FONT_SIZE = 20
STATS_FONT_SIZE = 20
HEATMAP_TICK_FONT_SIZE = 20
HEATMAP_TITLE_FONT_SIZE = 28
UPSET_FONT_SIZE = 20


def load_feature_lists_from_csv(omics_type, save_dir, comparison=None, min_votes=None):
    """
    Load feature lists from binary matrix CSV.

    Parameters
    ----------
    omics_type : {'protein', 'metabolite', 'multi_omics'}
    save_dir : str
    comparison : {'Control_vs_Pre', 'Pre_vs_Post'} or None
    min_votes : int or dict or None
        If provided, filter to consensus features with total_votes >= min_votes.

    Returns
    -------
    dict
        {method_name: [feature_ids]}
    """
    print("\n  Loading feature lists from CSV...")

    # Multi-omics: combine prefixed protein & metabolite features
    if omics_type == 'multi_omics':
        protein_lists = load_feature_lists_from_csv(
            'protein',
            save_dir,
            comparison,
            min_votes={'protein': 5, 'metabolite': 6}['protein'],
        )
        metabolite_lists = load_feature_lists_from_csv(
            'metabolite',
            save_dir,
            comparison,
            min_votes={'protein': 5, 'metabolite': 6}['metabolite'],
        )

        combined_lists = {}
        for method in protein_lists:
            combined_lists[method] = [
                f'protein_{f}' for f in protein_lists[method]
            ]
        for method in metabolite_lists:
            if method not in combined_lists:
                combined_lists[method] = []
            combined_lists[method].extend(
                [f'metabolite_{f}' for f in metabolite_lists[method]]
            )

        return combined_lists

    # Path to binary matrix
    if comparison:
        comparison_dir = os.path.join(save_dir, comparison)
        binary_file = os.path.join(
            comparison_dir, f'{omics_type}_feature_binary_matrix.csv'
        )
    else:
        binary_file = os.path.join(
            save_dir, f'{omics_type}_feature_binary_matrix.csv'
        )

    if not os.path.exists(binary_file):
        print(f"Binary matrix file not found: {binary_file}")
        return {}

    df = pd.read_csv(binary_file, index_col=0)

    feature_lists = {}
    method_cols = [col for col in df.columns if col != 'Feature']

    for col in method_cols:
        selected_features = df.index[df[col] == 1].tolist()
        feature_lists[col] = selected_features
        print(f"{col}: {len(selected_features)} features")

    # Optional consensus filter
    if min_votes is not None:
        print(f"\n  Filtering features with ≥{min_votes} votes...")
        df['total_votes'] = df[method_cols].sum(axis=1)
        consensus_features = df[df['total_votes'] >= min_votes].index.tolist()

        filtered_lists = {}
        for method in feature_lists:
            filtered_lists[method] = [
                f for f in feature_lists[method] if f in consensus_features
            ]
            print(
                f"{method} (after filtering): "
                f"{len(filtered_lists[method])} features"
            )

        return filtered_lists

    return feature_lists


def plot_upset_plot(feature_lists, omics_type, save_dir, comparison=None, min_votes=None):
    """
    Draw UpSet plot showing intersections across methods.

    Parameters
    ----------
    feature_lists : dict
        {method_name: [feature_ids]}
    omics_type : str
    save_dir : str
    comparison : str or None
    min_votes : int or dict or None
        Used only for file naming.
    """
    print("\n  Creating UpSet plot...")

    try:
        from upsetplot import from_memberships, UpSet

        all_features = set()
        for feats in feature_lists.values():
            all_features.update(feats)

        memberships_list = []
        for feat in all_features:
            membership = tuple(
                [m for m in feature_lists.keys() if feat in feature_lists[m]]
            )
            if membership:
                memberships_list.append(membership)

        upset_data = from_memberships(memberships_list)

        with plt.rc_context(
            {
                'font.size': UPSET_FONT_SIZE,
                'axes.titlesize': SUBTITLE_FONT_SIZE,
                'axes.labelsize': SUBTITLE_FONT_SIZE,
            }
        ):
            fig = plt.figure(figsize=(16, 10))
            upset = UpSet(
                upset_data,
                subset_size='count',
                show_counts=True,
                sort_by='cardinality',
                sort_categories_by='cardinality',
                facecolor='#4169E1',
                element_size=40,
                show_percentages=False,
            )
            upset.plot(fig=fig)

        if comparison:
            comparison_dir = os.path.join(save_dir, comparison)
            os.makedirs(comparison_dir, exist_ok=True)
            output_file = os.path.join(
                comparison_dir,
                f'{omics_type}_upset_plot_{comparison}_all_methods.png',
            )
        else:
            output_file = os.path.join(
                save_dir, f'{omics_type}_upset_plot_all_methods.png'
            )

        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"UpSet plot saved: {output_file}")

    except ImportError:
        print("    upsetplot not installed. Creating intersection matrix instead...")
        plot_intersection_matrix(
            feature_lists, omics_type, save_dir, comparison, min_votes
        )


def plot_intersection_matrix(
    feature_lists, omics_type, save_dir, comparison=None, min_votes=None
):
    """
    Draw a pairwise intersection matrix (backup when UpSet is not available).
    """
    print("    Creating intersection matrix...")

    methods = list(feature_lists.keys())
    n_methods = len(methods)

    intersection_matrix = np.zeros((n_methods, n_methods))

    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            s1 = set(feature_lists[m1])
            s2 = set(feature_lists[m2])
            intersection_matrix[i, j] = len(s1 & s2)

    fig, ax = plt.subplots(figsize=(12, 10))

    heat = sns.heatmap(
        intersection_matrix,
        annot=True,
        fmt='.0f',
        cmap='YlOrRd',
        xticklabels=methods,
        yticklabels=methods,
        cbar_kws={'label': 'Number of Shared Features'},
        ax=ax,
        square=True,
        linewidths=1,
        linecolor='white',
    )

    title = f'{omics_type.upper()} - Feature Intersection Matrix'
    if comparison:
        title += f' - {comparison}'
    if min_votes:
        if isinstance(min_votes, dict):
            title += (
                f' (Protein≥{min_votes["protein"]}, '
                f'Metabolite≥{min_votes["metabolite"]})'
            )
        else:
            title += f' (≥{min_votes} votes)'
    ax.set_title(
        title, fontsize=HEATMAP_TITLE_FONT_SIZE, fontweight='bold', pad=18
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha='right',
        fontsize=HEATMAP_TICK_FONT_SIZE,
    )
    plt.setp(
        ax.get_yticklabels(),
        rotation=0,
        fontsize=HEATMAP_TICK_FONT_SIZE,
    )

    if heat and heat.collections:
        cbar = heat.collections[0].colorbar
        if cbar:
            cbar.ax.tick_params(labelsize=HEATMAP_TICK_FONT_SIZE)
            cbar.set_label(
                'Number of Shared Features',
                fontsize=HEATMAP_TITLE_FONT_SIZE - 2,
            )

    plt.tight_layout()

    if comparison:
        comparison_dir = os.path.join(save_dir, comparison)
        os.makedirs(comparison_dir, exist_ok=True)
        vote_str = (
            f'{min_votes["protein"]}_{min_votes["metabolite"]}'
            if isinstance(min_votes, dict)
            else str(min_votes)
            if min_votes
            else ''
        )
        out_name = (
            f'{omics_type}_intersection_matrix_vote{vote_str}.png'
            if vote_str
            else f'{omics_type}_intersection_matrix.png'
        )
        output_file = os.path.join(comparison_dir, out_name)
    else:
        vote_str = str(min_votes) if min_votes else ''
        out_name = (
            f'{omics_type}_intersection_matrix_vote{vote_str}.png'
            if vote_str
            else f'{omics_type}_intersection_matrix.png'
        )
        output_file = os.path.join(save_dir, out_name)

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Intersection matrix saved: {output_file}")


def main():
    print("=" * 80)
    print("FEATURE COMPARISON VISUALIZATION (UpSet & Intersection Matrix)")
    print("=" * 80)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, 'top50_features_two_comparisons')

    # min_votes: protein 5, metabolite 6
    min_votes_config = {
        'protein': 5,
        'metabolite': 6,
        'multi_omics': {'protein': 5, 'metabolite': 6},
    }

    omics_types = ['protein', 'metabolite', 'multi_omics']

    for omics_type in omics_types:
        print(f"\n{'=' * 80}")
        print(f"Processing {omics_type.upper()}")
        print("=" * 80)

        if omics_type == 'multi_omics':
            min_votes = min_votes_config['multi_omics']
        else:
            min_votes = min_votes_config[omics_type]

        print(f"Using min_votes: {min_votes}")

        # Load consensus features (by method) for both comparisons
        print("\n  Loading features for Control_vs_Pre...")
        features_control_pre_dict = load_feature_lists_from_csv(
            omics_type, save_dir, comparison='Control_vs_Pre', min_votes=min_votes
        )

        print("\n  Loading features for Pre_vs_Post...")
        features_pre_post_dict = load_feature_lists_from_csv(
            omics_type, save_dir, comparison='Pre_vs_Post', min_votes=min_votes
        )

        if not features_control_pre_dict or not features_pre_post_dict:
            print(f"No consensus feature lists found for {omics_type}")
            continue

        all_features_control_pre = set()
        for feats in features_control_pre_dict.values():
            all_features_control_pre.update(feats)

        all_features_pre_post = set()
        for feats in features_pre_post_dict.values():
            all_features_pre_post.update(feats)

        print(
            f"\n  Control_vs_Pre: {len(all_features_control_pre)} unique features"
        )
        print(
            f"Pre_vs_Post: {len(all_features_pre_post)} unique features"
        )

        # For each comparison, draw:
        # 1) UpSet plot (all 8 methods, no min_votes filter)
        for comparison in ['Control_vs_Pre', 'Pre_vs_Post']:
            print(f"\n  Creating UpSet plot for {comparison}...")
            print("    (All 8 methods, no min_votes filter)")
            raw_feature_lists = load_feature_lists_from_csv(
                omics_type, save_dir, comparison=comparison, min_votes=None
            )
            if raw_feature_lists:
                plot_upset_plot(
                    raw_feature_lists,
                    omics_type,
                    save_dir,
                    comparison=comparison,
                    min_votes=None,
                )

    print(f"\n{'=' * 80}")
    print("VISUALIZATION COMPLETED")
    print("=" * 80)
    print(f"\nResults saved to: {save_dir}")


if __name__ == '__main__':
    main()