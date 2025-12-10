#!/usr/bin/env python3
"""
Binary classification using high-consensus features with 5-fold cross-validation.

- Two binary tasks: Control vs Pre, and Pre vs Post
- Load high-consensus protein and metabolite features
  (e.g., protein ≥5 methods, metabolite ≥6 methods)
- Use 5-fold CV for training and testing
- Compute and plot ROC curves and confusion matrices
- Use classifiers that are different from those used in feature selection
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from ..extract_consensus_features.load_data import load_original_data, get_group_mapping
from ..visualize.visualize_consensus_features import load_consensus_features

np.random.seed(42)

# Group colors (using mapped group names)
GROUP_MAPPING = get_group_mapping()
GROUP_COLORS = {
    'Control': '#90BFF9',
    'Rejection-treat-pre': '#F2B77C',
    'Rejection-treat-post': '#F59092'
}


def train_and_evaluate_5cv(
    X,
    y,
    cv_splits,
    sample_ids,
    classifier_name='LogisticRegression'
):
    """
    Train and evaluate a classifier with 5-fold CV.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : array-like
        Labels.
    cv_splits : list of dict
        Each dict contains 'train_samples' and 'test_samples'.
    sample_ids : list of str
        Sample IDs corresponding to rows in X.
    classifier_name : str
        Classifier name. Supported:
        'LogisticRegression', 'SVC', 'KNN', 'GaussianNB',
        'DecisionTree', 'MLP'.

    Returns
    -------
    results : dict
        Fold-wise results: accuracies, predictions, probabilities, labels, indices.
    classes : np.ndarray
        Unique class labels.
    """
    print(f"\nTraining {classifier_name} with 5-fold CV...")

    # Classifier selection (none of these were used in feature selection)
    if classifier_name == 'LogisticRegression':
        base_clf = LogisticRegression(max_iter=1000, random_state=42)
    elif classifier_name == 'SVC':
        base_clf = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42
        )
    elif classifier_name == 'KNN':
        base_clf = KNeighborsClassifier(
            n_neighbors=5,
            weights='distance'
        )
    elif classifier_name == 'GaussianNB':
        base_clf = GaussianNB()
    elif classifier_name == 'DecisionTree':
        base_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
    elif classifier_name == 'MLP':
        base_clf = MLPClassifier(
            hidden_layer_sizes=(16,),
            activation='relu',
            alpha=0.001,
            max_iter=2000,
            random_state=42
        )
    else:
        raise ValueError(
            f"Unsupported classifier: {classifier_name}. "
            "Supported: LogisticRegression, SVC, KNN, GaussianNB, "
            "DecisionTree, MLP."
        )

    classes = np.unique(y)
    sample_id_to_idx = {sid: idx for idx, sid in enumerate(sample_ids)}

    results = {
        'fold_accuracies': [],
        'fold_predictions': [],
        'fold_probabilities': [],
        'fold_true_labels': [],
        'fold_indices': []
    }

    y = np.array(y)

    for fold_idx, fold_info in enumerate(cv_splits):
        print(f"  Fold {fold_idx + 1}/5")

        train_sample_ids = fold_info['train_samples']
        test_sample_ids = fold_info['test_samples']

        train_idx = np.array(
            [sample_id_to_idx[sid] for sid in train_sample_ids if sid in sample_id_to_idx]
        )
        test_idx = np.array(
            [sample_id_to_idx[sid] for sid in test_sample_ids if sid in sample_id_to_idx]
        )

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = base_clf
        clf.fit(X_train_scaled, y_train)

        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        unique_test, counts_test = np.unique(y_test, return_counts=True)
        class_dist_test = dict(zip(unique_test, counts_test))

        print(f"    Accuracy: {acc:.4f}")
        print(f"    Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")
        print(f"    Test class distribution: {class_dist_test}")

        results['fold_accuracies'].append(acc)
        results['fold_predictions'].append(y_pred)
        results['fold_probabilities'].append(y_prob)
        results['fold_true_labels'].append(y_test)
        results['fold_indices'].append(test_idx)

    mean_acc = np.mean(results['fold_accuracies'])
    std_acc = np.std(results['fold_accuracies'])
    print(f"  Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

    return results, classes


def plot_binary_roc_curve(
    results,
    classes,
    omics_type,
    classifier_name,
    comparison_name,
    save_dir
):
    """
    Plot ROC curve and confusion matrix for a binary classifier.

    Parameters
    ----------
    results : dict
        Output from train_and_evaluate_5cv.
    classes : array-like
        Class labels (length 2 for binary).
    omics_type : str
        'protein', 'metabolite', or 'multi_omics'.
    classifier_name : str
    comparison_name : str
        'Control_vs_Pre' or 'Pre_vs_Post'.
    save_dir : str
        Directory where plots will be saved.

    Returns
    -------
    roc_auc_combined : float
        AUC computed on all folds combined.
    mean_auc : float
        Mean per-fold AUC.
    std_auc : float
        Std of per-fold AUC.
    """
    print("  Plotting ROC and confusion matrix...")

    y_true_all = np.concatenate(results['fold_true_labels'])
    y_prob_all = np.vstack(results['fold_probabilities'])

    label_encoder = {label: idx for idx, label in enumerate(classes)}
    y_true_encoded = np.array([label_encoder[label] for label in y_true_all])

    if len(classes) == 2:
        y_prob_positive = y_prob_all[:, 1]
    else:
        y_prob_positive = y_prob_all[:, 1] if y_prob_all.shape[1] > 1 else y_prob_all[:, 0]

    fpr, tpr, _ = roc_curve(y_true_encoded, y_prob_positive)
    roc_auc_combined = auc(fpr, tpr)

    fold_aucs = []
    for y_true, y_prob in zip(
        results['fold_true_labels'], results['fold_probabilities']
    ):
        if len(classes) == 2:
            y_true_encoded_fold = np.array([label_encoder[label] for label in y_true])
            y_prob_positive_fold = y_prob[:, 1]
            fold_fpr, fold_tpr, _ = roc_curve(
                y_true_encoded_fold, y_prob_positive_fold
            )
            fold_auc = auc(fold_fpr, fold_tpr)
            fold_aucs.append(fold_auc)

    mean_auc = np.mean(fold_aucs) if fold_aucs else roc_auc_combined
    std_auc = np.std(fold_aucs) if fold_aucs else 0.0

    fig, ax = plt.subplots(figsize=(8, 6))
    color = GROUP_COLORS.get(classes[1], '#F2B77C') if len(classes) > 1 else '#F2B77C'
    ax.plot(
        fpr,
        tpr,
        color=color,
        lw=2,
        label=f'{comparison_name} (AUC = {roc_auc_combined:.3f})'
    )
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.3)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{omics_type.upper()} - {classifier_name} - {comparison_name}')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    roc_save_path = os.path.join(
        save_dir,
        f'{omics_type}_{classifier_name}_{comparison_name}_roc_curve_5cv.png'
    )
    plt.savefig(roc_save_path, dpi=300, bbox_inches='tight')
    plt.close()

    y_pred_all = np.concatenate(results['fold_predictions'])
    cm = confusion_matrix(y_true_all, y_pred_all, labels=classes)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={'label': 'Normalized Frequency'},
        ax=ax
    )
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'{omics_type.upper()} - {classifier_name} - {comparison_name}')
    plt.tight_layout()

    cm_save_path = os.path.join(
        save_dir,
        f'{omics_type}_{classifier_name}_{comparison_name}_confusion_matrix_5cv.png'
    )
    plt.savefig(cm_save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    ROC AUC (combined): {roc_auc_combined:.4f}")
    print(f"    ROC AUC (mean ± std): {mean_auc:.4f} ± {std_auc:.4f}")

    print("    Per-fold performance:")
    for i, (acc, fold_auc) in enumerate(
        zip(results['fold_accuracies'], fold_aucs), start=1
    ):
        print(f"      Fold {i}: Acc={acc:.4f}, AUC={fold_auc:.4f}")

    return roc_auc_combined, mean_auc, std_auc


def prepare_two_group_data(data_df, y_all, comparison='Control_vs_Pre'):
    """
    Prepare data for a given two-group comparison.

    Parameters
    ----------
    data_df : pd.DataFrame
        DataFrame with samples as index and features as columns.
    y_all : list
        Labels for all samples (same order as data_df.index).
    comparison : {'Control_vs_Pre', 'Pre_vs_Post'}

    Returns
    -------
    X : np.ndarray
        Feature matrix.
    y_filtered : np.ndarray
        Labels for selected samples.
    feature_names : list of str
    sample_ids : list of str
        Sample IDs for selected samples.
    """
    if comparison == 'Control_vs_Pre':
        target_groups = ['Control', 'Rejection-treat-pre']
    elif comparison == 'Pre_vs_Post':
        target_groups = ['Rejection-treat-pre', 'Rejection-treat-post']
    else:
        raise ValueError(f"Unknown comparison: {comparison}")

    sample_mask = np.array([y in target_groups for y in y_all])

    X = data_df.values[sample_mask]
    y_filtered = np.array([y_all[i] for i in range(len(y_all)) if sample_mask[i]])
    feature_names = list(data_df.columns)
    sample_ids = data_df.index[sample_mask].tolist()

    return X, y_filtered, feature_names, sample_ids


def save_performance_summary(all_results, save_dir):
    """
    Save a summary of classification performance across all settings.
    """
    print("\nSaving performance summary...")

    summary_data = []

    for key, result_data in all_results.items():
        # key: {omics_type}_{classifier}_{comparison}
        parts = key.split('_')
        if len(parts) >= 4 and parts[0] == 'multi':
            omics_type = '_'.join(parts[0:2])
            classifier = parts[2]
            comparison = '_'.join(parts[3:])
        elif len(parts) >= 3:
            omics_type = parts[0]
            classifier = parts[1]
            comparison = '_'.join(parts[2:])
        else:
            continue

        mean_acc = np.mean(result_data['results']['fold_accuracies'])
        std_acc = np.std(result_data['results']['fold_accuracies'])

        auc_combined = result_data.get('auc_combined', 0.0)
        auc_mean = result_data.get('auc_mean', auc_combined)
        auc_std = result_data.get('auc_std', 0.0)

        perfect_classification = (
            mean_acc == 1.0 and (auc_combined == 1.0 or auc_mean == 1.0)
        )

        row = {
            'Omics_Type': omics_type,
            'Classifier': classifier,
            'Comparison': comparison,
            'Mean_Accuracy': f"{mean_acc:.4f}",
            'Std_Accuracy': f"{std_acc:.4f}",
            'AUC_Combined': f"{auc_combined:.4f}",
            'AUC_Mean': f"{auc_mean:.4f}",
            'AUC_Std': f"{auc_std:.4f}",
            'Perfect_Classification': 'Yes' if perfect_classification else 'No'
        }
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(
        save_dir, 'classification_performance_summary_5cv.csv'
    )
    summary_df.to_csv(summary_file, index=False)

    print(f"  Saved: {summary_file}")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))


def create_cv_splits_for_two_groups(y, sample_ids, n_splits=5):
    """
    Create stratified k-fold splits for binary classification.

    Parameters
    ----------
    y : array-like
        Labels.
    sample_ids : list of str
        Sample IDs.
    n_splits : int

    Returns
    -------
    cv_splits : list of dict
        Each dict has 'train_samples' and 'test_samples'.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_splits = []
    for train_idx, test_idx in skf.split(sample_ids, y):
        train_samples = [sample_ids[i] for i in train_idx]
        test_samples = [sample_ids[i] for i in test_idx]
        cv_splits.append(
            {'train_samples': train_samples, 'test_samples': test_samples}
        )

    return cv_splits


def main():
    print("Two-class classification with consensus features (5-fold CV)")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'top50_features_two_comparisons')
    save_dir = os.path.join(script_dir, 'consensus_classification_results_5cv')
    os.makedirs(save_dir, exist_ok=True)

    protein_df, metabolite_df, y_all, sample_ids, gender = load_original_data()

    print(f"Total samples: {len(sample_ids)}")

    comparisons = ['Control_vs_Pre', 'Pre_vs_Post']
    omics_types = ['protein', 'metabolite', 'multi_omics']

    classifiers = [
        'LogisticRegression',
        'SVC',
        'KNN',
        'GaussianNB',
        'DecisionTree',
        'MLP'
    ]

    min_votes = {
        'protein': 5,
        'metabolite': 6,
        'multi_omics': {'protein': 5, 'metabolite': 6}
    }

    all_results = {}

    for comparison_name in comparisons:
        print(f"\n=== Comparison: {comparison_name} ===")

        for omics_type in omics_types:
            print(f"\n--- {omics_type.upper()} ---")

            if omics_type == 'multi_omics':
                protein_features, _, _ = load_consensus_features(
                    'protein',
                    results_dir,
                    comparison=comparison_name,
                    min_votes=min_votes['multi_omics']['protein']
                )
                metabolite_features, _, _ = load_consensus_features(
                    'metabolite',
                    results_dir,
                    comparison=comparison_name,
                    min_votes=min_votes['multi_omics']['metabolite']
                )

                protein_features = [
                    f for f in protein_features if f in protein_df.columns
                ]
                metabolite_features = [
                    f for f in metabolite_features if f in metabolite_df.columns
                ]

                if len(protein_features) == 0 and len(metabolite_features) == 0:
                    print("  No consensus features found, skipping.")
                    continue

                print(
                    f"  Protein features: {len(protein_features)} "
                    f"(≥{min_votes['multi_omics']['protein']} votes)"
                )
                print(
                    f"  Metabolite features: {len(metabolite_features)} "
                    f"(≥{min_votes['multi_omics']['metabolite']} votes)"
                )

                multi_omics_df = pd.DataFrame(index=protein_df.index)
                for col in protein_features:
                    multi_omics_df[f'protein_{col}'] = protein_df[col]
                for col in metabolite_features:
                    multi_omics_df[f'metabolite_{col}'] = metabolite_df[col]

                subset_df = multi_omics_df

            else:
                features, _, _ = load_consensus_features(
                    omics_type,
                    results_dir,
                    comparison=comparison_name,
                    min_votes=min_votes[omics_type]
                )
                if len(features) == 0:
                    print("  No consensus features found, skipping.")
                    continue

                print(
                    f"  Using {len(features)} {omics_type} features "
                    f"(≥{min_votes[omics_type]} votes)"
                )

                data_df = protein_df if omics_type == 'protein' else metabolite_df
                available_features = [
                    f for f in features if f in data_df.columns
                ]
                if len(available_features) == 0:
                    print("  No available features in data, skipping.")
                    continue

                subset_df = data_df[available_features]

            X, y, feature_names, sample_ids_filtered = prepare_two_group_data(
                subset_df, y_all, comparison=comparison_name
            )

            uniq_labels, counts = np.unique(y, return_counts=True)
            print(
                f"  Data: {len(y)} samples, {len(feature_names)} features, "
                f"class distribution: {dict(zip(uniq_labels, counts))}"
            )

            cv_splits = create_cv_splits_for_two_groups(
                y, sample_ids_filtered, n_splits=5
            )

            for classifier_name in classifiers:
                print(f"\n  Classifier: {classifier_name}")
                results, classes = train_and_evaluate_5cv(
                    X, y, cv_splits, sample_ids_filtered, classifier_name
                )

                roc_auc_combined, roc_auc_mean, roc_auc_std = plot_binary_roc_curve(
                    results,
                    classes,
                    omics_type,
                    classifier_name,
                    comparison_name,
                    save_dir
                )

                key = f"{omics_type}_{classifier_name}_{comparison_name}"
                all_results[key] = {
                    'results': results,
                    'auc_combined': roc_auc_combined,
                    'auc_mean': roc_auc_mean,
                    'auc_std': roc_auc_std
                }

    save_performance_summary(all_results, save_dir)
    print("\nAnalysis completed.")
    print(f"Results saved to: {save_dir}")


if __name__ == '__main__':
    main()
