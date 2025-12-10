#!/usr/bin/env python3
"""
Consensus feature visualization

For each omics type (protein, metabolite, multi-omics) and comparison
(Control_vs_Pre, Pre_vs_Post), this script:

- Loads high-consensus features from feature binary matrices.
- Builds omics matrices (single-omics or multi-omics with prefixes).
- Plots:
    1) PCA (2D) of consensus features.
    2) Co-attention (cosine similarity) heatmap grouped by class.
    3) Co-attention heatmap with sample IDs on axes.

Notes:
    For multi-omics:
        - protein features: ≥5 votes
        - metabolite features: ≥6 votes
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
import matplotlib.patheffects as PathEffects

from load_data import (
    load_original_data,
    load_feature_mappings,
    map_feature_names,
    get_group_mapping,
)

# Group mapping and colors
GROUP_MAPPING = get_group_mapping()
GROUP_COLORS = {
    'Control': '#90BFF9',
    'Rejection-treat-pre': '#F2B77C',
    'Rejection-treat-post': '#F59092',
}
# Marker shapes: Control=o, Pre=^, Post=s
GROUP_MARKERS = {
    'Control': 'o',
    'Rejection-treat-pre': '^',
    'Rejection-treat-post': 's',
}

# Global font settings
plt.rcParams['font.family'] = 'Arial'

AXIS_LABEL_FONT_SIZE = 32
TICK_FONT_SIZE = 30
LEGEND_FONT_SIZE = 30
ANNOTATION_FONT_SIZE = 32
COLORBAR_TICK_FONT_SIZE = 26


def load_consensus_features(omics_type, save_dir, comparison=None, min_votes=5):
    """
    Load high-consensus features from feature binary matrix.

    Parameters
    ----------
    omics_type : {'protein', 'metabolite', 'multi_omics'}
    save_dir : str
        Base directory that contains comparison subfolders.
    comparison : {'Control_vs_Pre', 'Pre_vs_Post'} or None
        If None and omics_type != 'multi_omics', loads & merges both comparisons.
    min_votes : int
        Minimum number of methods for a feature to be considered consensus.

    Returns
    -------
    features : list of str
    votes : dict
        For single-omics: {feature: total_votes}
        For multi-omics (with comparison given):
            {prefixed_feature: total_votes}
    extra : object
        For single-omics + specific comparison:
            consensus_df (DataFrame)
        For multi-omics + specific comparison:
            empty DataFrame (not used)
        For single-omics + comparison=None:
            dict of comparison→features
    """
    # Multi-omics: per comparison, combine protein+metabolite with prefixes
    if omics_type == 'multi_omics' and comparison is not None:
        print(f"\nLoading multi-omics consensus features for {comparison}...")

        comparison_dir = os.path.join(save_dir, comparison)
        protein_file = os.path.join(
            comparison_dir, 'protein_feature_binary_matrix.csv'
        )
        metabolite_file = os.path.join(
            comparison_dir, 'metabolite_feature_binary_matrix.csv'
        )

        all_features = []
        all_votes = {}

        # Protein: min_votes=5
        if os.path.exists(protein_file):
            protein_df = pd.read_csv(protein_file, index_col=0)
            method_cols = list(protein_df.columns)
            protein_df['total_votes'] = protein_df[method_cols].sum(axis=1)
            protein_consensus = protein_df[protein_df['total_votes'] >= 5]
            protein_features = protein_consensus.index.tolist()

            for feat in protein_features:
                pf = f'protein_{feat}'
                all_features.append(pf)
                all_votes[pf] = protein_consensus.loc[feat, 'total_votes']

            print(
                f"Protein: {len(protein_features)} consensus features (≥5 votes)"
            )
        else:
            print(f"Protein matrix not found: {protein_file}")

        # Metabolite: min_votes=6
        if os.path.exists(metabolite_file):
            metabolite_df = pd.read_csv(metabolite_file, index_col=0)
            method_cols = list(metabolite_df.columns)
            metabolite_df['total_votes'] = metabolite_df[method_cols].sum(axis=1)
            metabolite_consensus = metabolite_df[metabolite_df['total_votes'] >= 6]
            metabolite_features = metabolite_consensus.index.tolist()

            for feat in metabolite_features:
                mf = f'metabolite_{feat}'
                all_features.append(mf)
                all_votes[mf] = metabolite_consensus.loc[feat, 'total_votes']

            print(
                f"Metabolite: {len(metabolite_features)} consensus features (≥6 votes)"
            )
        else:
            print(f"Metabolite matrix not found: {metabolite_file}")

        print(f"Total multi-omics consensus features: {len(all_features)}")
        return all_features, all_votes, pd.DataFrame()

    # Single-omics: with specific comparison
    if comparison is not None:
        print(f"\nLoading consensus features for {omics_type} - {comparison}...")
        comparison_dir = os.path.join(save_dir, comparison)
        binary_file = os.path.join(
            comparison_dir, f'{omics_type}_feature_binary_matrix.csv'
        )

        if not os.path.exists(binary_file):
            print(f"Binary matrix not found: {binary_file}")
            return [], {}, pd.DataFrame()

        df = pd.read_csv(binary_file, index_col=0)
        method_cols = list(df.columns)
        df['total_votes'] = df[method_cols].sum(axis=1)

        consensus_df = df[df['total_votes'] >= min_votes].copy()
        consensus_features = consensus_df.index.tolist()
        consensus_votes = consensus_df['total_votes'].to_dict()

        print(
            f"Found {len(consensus_features)} consensus features "
            f"(≥{min_votes} votes)"
        )
        return consensus_features, consensus_votes, consensus_df

    # Single-omics: merge both comparisons
    print(f"\nLoading consensus features for {omics_type} (both comparisons)...")

    all_features = []
    all_votes = {}
    comparison_features = {}

    for comp in ['Control_vs_Pre', 'Pre_vs_Post']:
        comparison_dir = os.path.join(save_dir, comp)
        binary_file = os.path.join(
            comparison_dir, f'{omics_type}_feature_binary_matrix.csv'
        )
        if os.path.exists(binary_file):
            df = pd.read_csv(binary_file, index_col=0)
            method_cols = list(df.columns)
            df['total_votes'] = df[method_cols].sum(axis=1)
            cons_df = df[df['total_votes'] >= min_votes].copy()
            feats = cons_df.index.tolist()

            comparison_features[comp] = feats
            all_features.extend(feats)

            for feat, votes in cons_df['total_votes'].to_dict().items():
                if feat not in all_votes:
                    all_votes[feat] = {'Control_vs_Pre': 0, 'Pre_vs_Post': 0}
                all_votes[feat][comp] = votes

            print(
                f"{comp}: {len(feats)} consensus features (≥{min_votes} votes)"
            )
        else:
            print(f"Binary matrix not found: {binary_file}")
            comparison_features[comp] = []

    all_features = sorted(set(all_features))
    print(f"Total unique features across comparisons: {len(all_features)}")

    if (
        'Control_vs_Pre' in comparison_features
        and 'Pre_vs_Post' in comparison_features
    ):
        inter = set(comparison_features['Control_vs_Pre']) & set(
            comparison_features['Pre_vs_Post']
        )
        print(f"Intersection features (both comparisons): {len(inter)}")

    return all_features, all_votes, comparison_features


def plot_ellipse_for_group(ax, group_data, color, sigma=2):
    """
    Draw an ellipse around group points based on covariance (not CI).
    """
    if group_data.shape[0] < 2:
        return

    mean = np.mean(group_data, axis=0)
    cov = np.cov(group_data.T)

    eigenvals, eigenvecs = np.linalg.eigh(cov)
    order = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[order]
    eigenvecs = eigenvecs[:, order]

    width, height = 2 * sigma * np.sqrt(eigenvals)
    angle = np.degrees(np.arctan2(*eigenvecs[:, 0][::-1]))

    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        edgecolor=color,
        facecolor=color,
        alpha=0.2,
        linewidth=2,
    )
    ax.add_patch(ellipse)


def create_dimred_plot(
    data_2d,
    y,
    groups,
    method_name,
    variance_info,
    omics_type,
    save_dir,
    min_votes,
    min_votes_protein=None,
    min_votes_metabolite=None,
):
    """
    Create PCA (or generic 2D) scatter plot with group ellipses.

    Parameters
    ----------
    data_2d : np.ndarray, shape (n_samples, 2)
    y : list
        Labels (mapped group names).
    groups : list of str
        Group order to plot.
    method_name : {'PCA', ...}
    variance_info : list-like of length 2
        Explained variance ratios for PC1 / PC2, or [0, 0] if not applicable.
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    actual_groups = [g for g in groups if g in y]

    for group in actual_groups:
        mask = [label == group for label in y]
        group_data = data_2d[mask]
        ax.scatter(
            group_data[:, 0],
            group_data[:, 1],
            c=GROUP_COLORS[group],
            label=group,
            s=180,
            alpha=0.75,
            edgecolors='black',
            linewidths=1.8,
            marker=GROUP_MARKERS[group],
        )
        plot_ellipse_for_group(ax, group_data, GROUP_COLORS[group])

    if method_name == 'PCA':
        ax.set_xlabel(
            f'PC1 ({variance_info[0]:.1%} variance)',
            fontsize=AXIS_LABEL_FONT_SIZE + 2,
            fontweight='bold',
        )
        ax.set_ylabel(
            f'PC2 ({variance_info[1]:.1%} variance)',
            fontsize=AXIS_LABEL_FONT_SIZE + 2,
            fontweight='bold',
        )
    else:
        ax.set_xlabel(
            f'{method_name} 1',
            fontsize=AXIS_LABEL_FONT_SIZE,
            fontweight='bold',
        )
        ax.set_ylabel(
            f'{method_name} 2',
            fontsize=AXIS_LABEL_FONT_SIZE,
            fontweight='bold',
        )

    if actual_groups:
        ax.legend(loc='best', fontsize=LEGEND_FONT_SIZE + 2)

    ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE + 2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if omics_type == 'multi_omics' and min_votes_protein is not None and min_votes_metabolite is not None:
        vote_str = f'{min_votes_protein}_{min_votes_metabolite}'
    else:
        vote_str = str(min_votes)

    output_file = os.path.join(
        save_dir,
        f'{omics_type}_consensus_{method_name.lower()}_vote{vote_str}.png',
    )
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"{method_name} plot saved: {output_file}")


def create_coattention_matrix(
    subset_df,
    y,
    groups,
    omics_type,
    save_dir,
    min_votes,
    min_votes_protein=None,
    min_votes_metabolite=None,
):
    """
    Co-attention (cosine similarity) matrix grouped by class.
    """
    fig, ax = plt.subplots(figsize=(14, 12))

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(subset_df)
    sim = cosine_similarity(data_scaled)

    y_series = pd.Series(y, index=subset_df.index)
    sorted_indices = []
    for group in groups:
        sorted_indices.extend(y_series[y_series == group].index)

    sorted_pos = [list(subset_df.index).index(idx) for idx in sorted_indices]
    sim_sorted = sim[np.ix_(sorted_pos, sorted_pos)]
    y_sorted = [y_series[idx] for idx in sorted_indices]

    im = ax.imshow(sim_sorted, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, aspect=20)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONT_SIZE + 2)

    actual_groups = [g for g in groups if g in y_sorted]
    group_counts = [sum(1 for g in y_sorted if g == group) for group in actual_groups]
    boundaries = np.cumsum([0] + group_counts)

    # Draw dashed boxes for each group
    for i, group in enumerate(actual_groups):
        x_start = boundaries[i] - 0.5
        x_end = boundaries[i + 1] - 0.5
        y_start = boundaries[i] - 0.5
        y_end = boundaries[i + 1] - 0.5

        ax.plot([x_start, x_end], [y_end, y_end], 'k--', linewidth=2.5, alpha=0.8)
        ax.plot([x_start, x_end], [y_start, y_start], 'k--', linewidth=2.5, alpha=0.8)
        ax.plot([x_start, x_start], [y_start, y_end], 'k--', linewidth=2.5, alpha=0.8)
        ax.plot([x_end, x_end], [y_start, y_end], 'k--', linewidth=2.5, alpha=0.8)

    # Group labels on left and bottom
    for i, group in enumerate(actual_groups):
        mid = (boundaries[i] + boundaries[i + 1]) / 2
        t_left = ax.text(
            -1,
            mid,
            group,
            rotation=90,
            va='center',
            ha='right',
            fontsize=ANNOTATION_FONT_SIZE + 4,
            fontweight='heavy',
            color='black',
        )
        t_bottom = ax.text(
            mid,
            -1,
            group,
            ha='center',
            va='bottom',
            fontsize=ANNOTATION_FONT_SIZE + 4,
            fontweight='heavy',
            color='black',
        )
        for txt in (t_left, t_bottom):
            txt.set_path_effects(
                [PathEffects.withStroke(linewidth=3, foreground='white')]
            )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    plt.tight_layout()

    if omics_type == 'multi_omics' and min_votes_protein is not None and min_votes_metabolite is not None:
        vote_str = f'{min_votes_protein}_{min_votes_metabolite}'
    else:
        vote_str = str(min_votes)

    output_file = os.path.join(
        save_dir,
        f'{omics_type}_consensus_coattention_standalone_vote{vote_str}.png',
    )
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Co-attention matrix (grouped) saved: {output_file}")


def create_coattention_matrix_by_sample_id(
    subset_df,
    y,
    groups,
    omics_type,
    save_dir,
    min_votes,
    min_votes_protein=None,
    min_votes_metabolite=None,
):
    """
    Co-attention matrix with sample IDs on both axes (sorted by group).
    """
    fig, ax = plt.subplots(figsize=(16, 14))

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(subset_df)
    sim = cosine_similarity(data_scaled)

    y_series = pd.Series(y, index=subset_df.index)
    sorted_indices = []
    for group in groups:
        sorted_indices.extend(y_series[y_series == group].index)

    sorted_pos = [list(subset_df.index).index(idx) for idx in sorted_indices]
    sim_sorted = sim[np.ix_(sorted_pos, sorted_pos)]
    y_sorted = [y_series[idx] for idx in sorted_indices]
    sample_ids_sorted = list(sorted_indices)

    im = ax.imshow(sim_sorted, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, aspect=20)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONT_SIZE + 2)

    actual_groups = [g for g in groups if g in y_sorted]
    group_counts = [sum(1 for g in y_sorted if g == group) for group in actual_groups]
    boundaries = np.cumsum([0] + group_counts)

    for i, group in enumerate(actual_groups):
        x_start = boundaries[i] - 0.5
        x_end = boundaries[i + 1] - 0.5
        y_start = boundaries[i] - 0.5
        y_end = boundaries[i + 1] - 0.5

        ax.plot([x_start, x_end], [y_end, y_end], 'k--', linewidth=2.5, alpha=0.8)
        ax.plot([x_start, x_end], [y_start, y_start], 'k--', linewidth=2.5, alpha=0.8)
        ax.plot([x_start, x_start], [y_start, y_end], 'k--', linewidth=2.5, alpha=0.8)
        ax.plot([x_end, x_end], [y_start, y_end], 'k--', linewidth=2.5, alpha=0.8)

    # Dynamic font size for sample IDs
    dynamic_font = 500 // max(1, len(sample_ids_sorted))
    fontsize = min(34, max(28, dynamic_font))

    for i, sid in enumerate(sample_ids_sorted):
        ax.text(
            -0.5,
            i,
            str(sid),
            rotation=0,
            va='center',
            ha='right',
            fontsize=fontsize,
            color='black',
        )
        ax.text(
            i,
            len(sample_ids_sorted) - 0.5,
            str(sid),
            rotation=90,
            ha='left',
            va='top',
            fontsize=fontsize,
            color='black',
        )

    for i, group in enumerate(actual_groups):
        mid = (boundaries[i] + boundaries[i + 1]) / 2
        t_left = ax.text(
            -2,
            mid,
            group,
            rotation=90,
            va='center',
            ha='right',
            fontsize=ANNOTATION_FONT_SIZE + 2,
            fontweight='heavy',
            color=GROUP_COLORS[group],
        )
        t_bottom = ax.text(
            mid,
            len(sample_ids_sorted) + 1,
            group,
            ha='center',
            va='bottom',
            fontsize=ANNOTATION_FONT_SIZE + 2,
            fontweight='heavy',
            color=GROUP_COLORS[group],
        )
        for txt in (t_left, t_bottom):
            txt.set_path_effects(
                [PathEffects.withStroke(linewidth=3, foreground='white')]
            )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    plt.tight_layout()

    if omics_type == 'multi_omics' and min_votes_protein is not None and min_votes_metabolite is not None:
        vote_str = f'{min_votes_protein}_{min_votes_metabolite}'
    else:
        vote_str = str(min_votes)

    output_file = os.path.join(
        save_dir,
        f'{omics_type}_consensus_coattention_by_sample_id_vote{vote_str}.png',
    )
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Co-attention matrix (by sample_id) saved: {output_file}")


def create_consensus_visualizations(
    data_df,
    y,
    gender,
    consensus_features,
    omics_type,
    feature_mappings,
    save_dir,
    min_votes,
    min_votes_protein=None,
    min_votes_metabolite=None,
):
    """
    Run PCA and Co-attention visualizations for the given consensus features.
    """
    print("\n  Creating PCA and co-attention visualizations...")

    groups = ['Control', 'Rejection-treat-pre', 'Rejection-treat-post']

    available_features = [f for f in consensus_features if f in data_df.columns]
    if not available_features:
        print("    No consensus features available in data.")
        return

    subset_df = (
        data_df[available_features]
        .apply(pd.to_numeric, errors='coerce')
        .fillna(data_df[available_features].mean())
    )

    # PCA
    print("    PCA...")
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(subset_df)
    create_dimred_plot(
        pca_result,
        y,
        groups,
        'PCA',
        pca.explained_variance_ratio_,
        omics_type,
        save_dir,
        min_votes,
        min_votes_protein,
        min_votes_metabolite,
    )

    # Co-attention (by group)
    print("    Co-attention (grouped)...")
    create_coattention_matrix(
        subset_df,
        y,
        groups,
        omics_type,
        save_dir,
        min_votes,
        min_votes_protein,
        min_votes_metabolite,
    )

    # Co-attention (by sample_id)
    print("    Co-attention (by sample_id)...")
    create_coattention_matrix_by_sample_id(
        subset_df,
        y,
        groups,
        omics_type,
        save_dir,
        min_votes,
        min_votes_protein,
        min_votes_metabolite,
    )

    print("    Done for this setting.")


def main():
    print("=" * 80)
    print("CONSENSUS FEATURE VISUALIZATION (PCA & Co-attention)")
    print("=" * 80)

    print("\nLoading data...")
    protein_df, metabolite_df, y_all, sample_ids, gender = load_original_data()
    feature_mappings = load_feature_mappings()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, 'top50_features_two_comparisons')

    min_votes_config = {
        'protein': 5,
        'metabolite': 6,
        'multi_omics': {'protein': 5, 'metabolite': 6},
    }

    print(
        f"\nUsing all {len(y_all)} samples "
    )
    print(
        f"Control: {y_all.count('Control')}, "
        f"Rejection-treat-pre: {y_all.count('Rejection-treat-pre')}, "
        f"Rejection-treat-post: {y_all.count('Rejection-treat-post')}"
    )

    # Multi-omics matrix: prefix protein_ / metabolite_
    valid_metabolite_features = []
    for feat in metabolite_df.columns:
        if feat in feature_mappings['metabolite']:
            ms2_name = feature_mappings['metabolite'][feat]
            if ms2_name and ms2_name.strip() and ms2_name != feat:
                valid_metabolite_features.append(feat)
    metabolite_df_filtered = (
        metabolite_df[valid_metabolite_features]
        if valid_metabolite_features
        else metabolite_df
    )

    multi_omics_df = pd.DataFrame(index=protein_df.index)
    for col in protein_df.columns:
        multi_omics_df[f'protein_{col}'] = protein_df[col]
    for col in metabolite_df_filtered.columns:
        multi_omics_df[f'metabolite_{col}'] = metabolite_df_filtered[col]

    omics_data = {
        'protein': protein_df,
        'metabolite': metabolite_df,
        'multi_omics': multi_omics_df,
    }

    analysis_mode = 'separate'  # only per-comparison analysis

    for omics_type, data_df in omics_data.items():
        print(f"\n{'=' * 80}")
        print(f"Analyzing {omics_type.upper()}")
        print(f"{'=' * 80}")

        if omics_type == 'metabolite':
            print("\n  Filtering metabolites with MS2 names...")
            print(f"Original features: {len(data_df.columns)}")
            valid_feats = []
            for feat in data_df.columns:
                if feat in feature_mappings['metabolite']:
                    ms2_name = feature_mappings['metabolite'][feat]
                    if ms2_name and ms2_name.strip() and ms2_name != feat:
                        valid_feats.append(feat)
            if not valid_feats:
                print("    No MS2-named features found; using all features.")
            else:
                print(f"Features with MS2 names: {len(valid_feats)}")
                data_df = data_df[valid_feats]
        elif omics_type == 'multi_omics':
            print("\n  Multi-omics data:")
            print(
                f"Protein features: "
                f"{len([c for c in data_df.columns if c.startswith('protein_')])}"
            )
            print(
                f"Metabolite features: "
                f"{len([c for c in data_df.columns if c.startswith('metabolite_')])}"
            )
            print(f"Total features: {len(data_df.columns)}")

        if analysis_mode == 'separate':
            min_votes = (
                min_votes_config[omics_type]
                if omics_type != 'multi_omics'
                else 5
            )

            for comparison in ['Control_vs_Pre', 'Pre_vs_Post']:
                print(f"\n{'-' * 80}")
                print(f"{omics_type.upper()} - {comparison} (min_votes={min_votes})")
                print(f"{'-' * 80}")

                consensus_features, consensus_votes, _ = load_consensus_features(
                    omics_type,
                    save_dir,
                    comparison=comparison,
                    min_votes=min_votes,
                )
                if not consensus_features:
                    print("  No consensus features for this setting; skipping.")
                    continue

                if comparison == 'Control_vs_Pre':
                    comp_mask = np.array(
                        [y in ['Control', 'Rejection-treat-pre'] for y in y_all]
                    )
                else:
                    comp_mask = np.array(
                        [y in ['Rejection-treat-pre', 'Rejection-treat-post'] for y in y_all]
                    )

                comp_y = [y_all[i] for i in range(len(y_all)) if comp_mask[i]]
                comp_df = data_df[comp_mask]

                if isinstance(gender, (list, np.ndarray)) and len(gender) == len(y_all):
                    comp_gender = [
                        gender[i] for i in range(len(gender)) if comp_mask[i]
                    ]
                else:
                    comp_gender = gender

                comparison_save_dir = os.path.join(save_dir, comparison)
                os.makedirs(comparison_save_dir, exist_ok=True)

                if omics_type == 'multi_omics':
                    create_consensus_visualizations(
                        comp_df,
                        comp_y,
                        comp_gender,
                        consensus_features,
                        omics_type,
                        feature_mappings,
                        comparison_save_dir,
                        min_votes,
                        min_votes_protein=min_votes_config['multi_omics']['protein'],
                        min_votes_metabolite=min_votes_config['multi_omics']['metabolite'],
                    )
                else:
                    create_consensus_visualizations(
                        comp_df,
                        comp_y,
                        comp_gender,
                        consensus_features,
                        omics_type,
                        feature_mappings,
                        comparison_save_dir,
                        min_votes,
                    )

    print(f"\n{'=' * 80}")
    print("VISUALIZATION COMPLETED.")
    print(f"{'=' * 80}")
    print(f"\nResults saved to: {save_dir}")
    print("\nFile patterns:")
    print("  - {omics}_consensus_pca_vote{votes}.png")
    print("  - {omics}_consensus_coattention_standalone_vote{votes}.png")
    print("  - {omics}_consensus_coattention_by_sample_id_vote{votes}.png")


if __name__ == '__main__':
    main()
