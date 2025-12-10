#!/usr/bin/env python3
"""
Run all feature selection methods and save results for two binary comparisons:
- Control vs Pre
- Pre vs Post

For each comparison:
- Run 8 feature selection methods
- Save top N (default 50) features of each method to CSV
- Generate a feature binary matrix for each comparison and omics type
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

import importlib

load_data_module = importlib.import_module('load_data')
feat_sel_module = importlib.import_module('advanced_feature_selection')

load_original_data = load_data_module.load_original_data
load_feature_mappings = load_data_module.load_feature_mappings
method1_anova_f_test = feat_sel_module.method1_anova_f_test
method2_mutual_information = feat_sel_module.method2_mutual_information
method3_random_forest_importance = feat_sel_module.method3_random_forest_importance
method4_gradient_boosting_importance = feat_sel_module.method4_gradient_boosting_importance
method5_lasso_selection = feat_sel_module.method5_lasso_selection
method6_relief_based = feat_sel_module.method6_relief_based
method7_differential_expression = feat_sel_module.method7_differential_expression
method8_lda_projection = feat_sel_module.method8_lda_projection


def prepare_two_group_data(data_df, y_all, comparison='con_pre'):
    """
    Prepare data for a two-group comparison.

    Parameters
    ----------
    data_df : pd.DataFrame
        Omics data (samples × features).
    y_all : list
        Class labels for all samples.
    comparison : {'con_pre', 'pre_post'}

    Returns
    -------
    X_filtered : np.ndarray
        Feature matrix for selected samples.
    y_filtered_simple : np.ndarray
        Binary labels for selected samples.
    feature_names : list of str
        Column names of features.
    sample_mask : np.ndarray (bool)
        Mask over all samples indicating those used in this comparison.
    comparison_name : str
        Human-readable comparison name.
    """
    if comparison == 'con_pre':
        target_groups = ['Control', 'Rejection-treat-pre']
        comparison_name = 'Control_vs_Pre'
    elif comparison == 'pre_post':
        target_groups = ['Rejection-treat-pre', 'Rejection-treat-post']
        comparison_name = 'Pre_vs_Post'
    else:
        raise ValueError(f"Unknown comparison: {comparison}")

    # Select only samples belonging to the target groups
    sample_mask = np.array([y in target_groups for y in y_all])

    X_filtered = data_df.values[sample_mask]
    y_filtered = np.array([y_all[i] for i in range(len(y_all)) if sample_mask[i]])
    feature_names = list(data_df.columns)

    # Map labels to simplified binary labels
    if comparison == 'con_pre':
        label_mapping = {
            'Control': 'Control',
            'Rejection-treat-pre': 'Pre'
        }
    else:  # pre_post
        label_mapping = {
            'Rejection-treat-pre': 'Pre',
            'Rejection-treat-post': 'Post'
        }

    y_filtered_simple = np.array([label_mapping[y] for y in y_filtered])

    return X_filtered, y_filtered_simple, feature_names, sample_mask, comparison_name


def run_all_methods_and_save(
    X,
    y,
    feature_names,
    omics_type,
    comparison_name,
    save_dir,
    n_features=50
):
    """
    Run all feature selection methods and save results.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : array-like
        Binary labels.
    feature_names : list of str
    omics_type : str
        'protein' or 'metabolite'.
    comparison_name : str
        'Control_vs_Pre' or 'Pre_vs_Post'.
    save_dir : str
        Base directory to save results.
    n_features : int
        Top N features requested from each method.

    Returns
    -------
    all_results : dict
        {method_name: {'features': [...], 'scores': [...]}}
    """
    print(f"\nRunning feature selection for {omics_type} - {comparison_name}")

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    print(f"Data shape: {X_imputed.shape}")
    unique_classes, class_counts = np.unique(y, return_counts=True)
    print(f"Classes: {unique_classes.tolist()} with counts {class_counts.tolist()}")

    methods = {
        'ANOVA_F': method1_anova_f_test,
        'MutualInfo': method2_mutual_information,
        'RandomForest': method3_random_forest_importance,
        'GradientBoosting': method4_gradient_boosting_importance,
        'Lasso': method5_lasso_selection,
        'Relief': method6_relief_based,
        'DiffExpr': method7_differential_expression,
        'LDA': method8_lda_projection,
    }

    all_results = {}

    for method_name, method_func in methods.items():
        print(f"- {method_name}")
        try:
            features, scores, _ = method_func(
                X_imputed, y, feature_names, n_features=n_features
            )

            # Remove zero/NaN/inf scores
            valid_mask = np.array([
                (score is not None)
                and (not np.isnan(score))
                and (not np.isinf(score))
                and (score != 0)
                for score in scores
            ])
            features_filtered = [f for f, v in zip(features, valid_mask) if v]
            scores_filtered = [s for s, v in zip(scores, valid_mask) if v]

            all_results[method_name] = {
                'features': features_filtered,
                'scores': scores_filtered
            }
        except Exception as e:
            print(f"Failed: {str(e)[:80]}")
            all_results[method_name] = {'features': [], 'scores': []}

    save_results_to_csv(
        all_results, omics_type, comparison_name, save_dir, n_features
    )

    return all_results


def save_results_to_csv(
    all_results,
    omics_type,
    comparison_name,
    save_dir,
    n_features
):
    """
    Save feature selection results to CSV files.

    Creates:
    - Combined table of top features for all methods.
    - Binary matrix (methods × features).
    - Per-method top feature list CSVs.
    """
    comparison_save_dir = os.path.join(save_dir, comparison_name)
    os.makedirs(comparison_save_dir, exist_ok=True)

    # 1) Combined table
    max_len = max(
        [len(r['features']) for r in all_results.values() if r['features']],
        default=0
    )
    if max_len == 0:
        print("  No valid features found for any method; skipping CSV creation.")
        return

    data_dict = {}
    for method_name, result in all_results.items():
        features = result['features']
        scores = result['scores']

        features_padded = features + [''] * (max_len - len(features))
        scores_padded = list(scores) + [np.nan] * (max_len - len(scores))

        data_dict[f'{method_name}_Feature'] = features_padded
        data_dict[f'{method_name}_Score'] = scores_padded

    df_all = pd.DataFrame(data_dict)
    output_file = os.path.join(
        comparison_save_dir,
        f'{omics_type}_all_methods_top{n_features}_features.csv'
    )
    df_all.to_csv(output_file, index=False)
    print(f"Saved combined feature table: {output_file}")

    # 2) Binary matrix of feature presence across methods
    feature_lists = {
        mname: res['features']
        for mname, res in all_results.items()
        if res['features']
    }
    if not feature_lists:
        print("  No feature lists for binary matrix.")
        return

    all_unique_features = sorted(
        {f for feats in feature_lists.values() for f in feats}
    )

    binary_matrix = {}
    for method_name, features in feature_lists.items():
        feature_set = set(features)
        binary_matrix[method_name] = [
            1 if feat in feature_set else 0 for feat in all_unique_features
        ]

    df_binary = pd.DataFrame(binary_matrix, index=all_unique_features)
    df_binary.index.name = 'Feature'
    output_file2 = os.path.join(
        comparison_save_dir,
        f'{omics_type}_feature_binary_matrix.csv'
    )
    df_binary.to_csv(output_file2)
    print(f"Saved feature binary matrix: {output_file2}")

    # 3) Per-method feature tables
    for method_name, features in feature_lists.items():
        scores = all_results[method_name]['scores'][:len(features)]
        df_method = pd.DataFrame({
            'Rank': range(1, len(features) + 1),
            'Feature': features,
            'Score': scores
        })
        output_file3 = os.path.join(
            comparison_save_dir,
            f'{omics_type}_{method_name}_top{n_features}.csv'
        )
        df_method.to_csv(output_file3, index=False)

    print(f"Saved per-method feature lists in {comparison_save_dir}")


def main():
    print("Running feature selection for two-group comparisons...")

    # Load data (uses all common samples; no manual sample removal)
    protein_df, metabolite_df, y_all, common_samples, gender = load_original_data()
    feature_mappings = load_feature_mappings()

    print(f"Total samples: {len(y_all)}")
    print(
        "Class counts:",
        {
            cls: y_all.count(cls)
            for cls in ['Control', 'Rejection-treat-pre', 'Rejection-treat-post']
            if cls in y_all
        }
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, 'top50_features_two_comparisons')
    os.makedirs(save_dir, exist_ok=True)

    comparisons = ['con_pre', 'pre_post']
    comparison_names = {
        'con_pre': 'Control_vs_Pre',
        'pre_post': 'Pre_vs_Post'
    }

    omics_types = {
        'protein': protein_df,
        'metabolite': metabolite_df,
    }

    for comparison in comparisons:
        comparison_name = comparison_names[comparison]
        print(f"\n=== Comparison: {comparison_name} ===")

        for omics_type, data_df in omics_types.items():
            print(f"\nProcessing {omics_type} for {comparison_name}...")

            # For metabolites, optionally filter by MS2Metabolite names
            if omics_type == 'metabolite':
                print("  Filtering metabolite features by MS2Metabolite names...")
                valid_features = []
                for feat in data_df.columns:
                    if feat in feature_mappings['metabolite']:
                        ms2_name = feature_mappings['metabolite'][feat]
                        if ms2_name and ms2_name.strip() and ms2_name != feat:
                            valid_features.append(feat)

                if len(valid_features) == 0:
                    # No mapping: keep all features
                    data_df_filtered = data_df
                else:
                    data_df_filtered = data_df[valid_features]
                    print(
                        f"Kept {len(valid_features)} metabolite features with MS2 names "
                        f"(from {len(data_df.columns)} original)."
                    )
            else:
                data_df_filtered = data_df

            # Prepare two-group data
            X, y, feature_names, sample_mask, comp_name = prepare_two_group_data(
                data_df_filtered, y_all, comparison=comparison
            )

            print(
                f"Two-group data: {len(y)} samples, "
                f"{len(feature_names)} features, classes {np.unique(y).tolist()}"
            )

            # Run all feature selection methods and save
            run_all_methods_and_save(
                X,
                y,
                feature_names,
                omics_type,
                comp_name,
                save_dir,
                n_features=50
            )

    print("\nFeature selection completed.")
    print(f"Results saved under: {save_dir}")


if __name__ == '__main__':
    main()