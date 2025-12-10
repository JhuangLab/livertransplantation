#!/usr/bin/env python3
"""
Advanced Feature Selection - Multiple Algorithms Ensemble

Goal:
    - Identify truly discriminative top features from all samples

Supports:
    - Binary classification
    - Multi-class classification

Notes:
    - All methods use LabelEncoder to automatically adapt to the number of classes
    - All methods use all samples (no subsampling by group)
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def method1_anova_f_test(X, y, feature_names, n_features=10):
    """Method 1: ANOVA F-test - classical statistical method"""
    print("  Using Method 1: ANOVA F-test...")

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Compute F-statistics
    selector = SelectKBest(f_classif, k=min(n_features, X.shape[1]))
    selector.fit(X, y_encoded)

    scores = selector.scores_
    top_indices = np.argsort(scores)[-n_features:][::-1]

    top_features = [feature_names[i] for i in top_indices]
    top_scores = [scores[i] for i in top_indices]

    return top_features, top_scores, "ANOVA-F"


def method2_mutual_information(X, y, feature_names, n_features=10):
    """Method 2: Mutual information - captures nonlinear relationships"""
    print("  Using Method 2: Mutual Information...")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    mi_scores = mutual_info_classif(X, y_encoded, random_state=42)

    top_indices = np.argsort(mi_scores)[-n_features:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    top_scores = [mi_scores[i] for i in top_indices]

    return top_features, top_scores, "MutualInfo"


def method3_random_forest_importance(X, y, feature_names, n_features=10):
    """Method 3: Random Forest feature importance (considers feature interactions)"""
    print("  Using Method 3: Random Forest Importance...")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X, y_encoded)

    importances = rf.feature_importances_

    top_indices = np.argsort(importances)[-n_features:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    top_scores = [importances[i] for i in top_indices]

    return top_features, top_scores, "RandomForest"


def method4_gradient_boosting_importance(X, y, feature_names, n_features=10):
    """Method 4: Gradient Boosting feature importance (nonlinear capacity)"""
    print("  Using Method 4: Gradient Boosting Importance...")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        random_state=42,
        learning_rate=0.1,
    )
    gb.fit(X, y_encoded)

    importances = gb.feature_importances_

    top_indices = np.argsort(importances)[-n_features:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    top_scores = [importances[i] for i in top_indices]

    return top_features, top_scores, "GradientBoosting"


def method5_lasso_selection(X, y, feature_names, n_features=10):
    """Method 5: L1-regularized Logistic Regression (Lasso-style sparse selection)"""
    print("  Using Method 5: L1-regularized Logistic Regression...")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lasso = LogisticRegressionCV(
        penalty="l1",
        solver="saga",
        cv=3,
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
    )
    lasso.fit(X_scaled, y_encoded)

    # For multi-class, take max abs coefficient across classes
    if len(lasso.coef_.shape) > 1:
        importances = np.abs(lasso.coef_).max(axis=0)
    else:
        importances = np.abs(lasso.coef_)

    top_indices = np.argsort(importances)[-n_features:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    top_scores = [importances[i] for i in top_indices]

    return top_features, top_scores, "Lasso"


def method6_relief_based(X, y, feature_names, n_features=10):
    """Method 6: Relief-like scoring (instance-based)"""
    print("  Using Method 6: Relief-based scoring...")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    n_samples, n_feat = X.shape
    scores = np.zeros(n_feat)

    # Use all samples (no subsampling)
    for i in range(n_samples):
        sample = X[i]
        label = y_encoded[i]

        same_class_mask = y_encoded == label
        diff_class_mask = y_encoded != label

        if same_class_mask.sum() > 1 and diff_class_mask.sum() > 0:
            same_class_X = X[same_class_mask]
            diff_class_X = X[diff_class_mask]

            # Nearest hit (same class, exclude itself)
            same_distances = np.sum((same_class_X - sample) ** 2, axis=1)
            same_distances[same_distances == 0] = np.inf
            nearest_same = same_class_X[np.argmin(same_distances)]

            # Nearest miss (different class)
            diff_distances = np.sum((diff_class_X - sample) ** 2, axis=1)
            nearest_diff = diff_class_X[np.argmin(diff_distances)]

            scores += np.abs(sample - nearest_diff) - np.abs(sample - nearest_same)

    scores = scores / max(n_samples, 1)

    top_indices = np.argsort(scores)[-n_features:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    top_scores = [scores[i] for i in top_indices]

    return top_features, top_scores, "Relief"


def method7_differential_expression(X, y, feature_names, n_features=10):
    """Method 7: Differential expression-like analysis (effect size × significance)"""
    print("  Using Method 7: Differential Expression-like Analysis...")

    groups = np.unique(y)
    diff_scores = []

    for feat_idx in range(X.shape[1]):
        feature_values = X[:, feat_idx]
        effect_sizes = []

        for i, g1 in enumerate(groups):
            for g2 in groups[i + 1 :]:
                mask1 = y == g1
                mask2 = y == g2

                vals1 = feature_values[mask1]
                vals2 = feature_values[mask2]

                vals1 = vals1[~np.isnan(vals1)]
                vals2 = vals2[~np.isnan(vals2)]

                if len(vals1) > 1 and len(vals2) > 1:
                    mean_diff = np.abs(np.mean(vals1) - np.mean(vals2))
                    pooled_std = np.sqrt((np.var(vals1) + np.var(vals2)) / 2.0)
                    cohens_d = mean_diff / (pooled_std + 1e-9)

                    try:
                        _, p_value = stats.mannwhitneyu(
                            vals1, vals2, alternative="two-sided"
                        )
                        significance = -np.log10(p_value + 1e-10)
                    except Exception:
                        significance = 0.0

                    effect_sizes.append(cohens_d * significance)

        diff_scores.append(max(effect_sizes) if effect_sizes else 0.0)

    diff_scores = np.array(diff_scores)

    top_indices = np.argsort(diff_scores)[-n_features:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    top_scores = [diff_scores[i] for i in top_indices]

    return top_features, top_scores, "DiffExpr"


def method8_lda_projection(X, y, feature_names, n_features=10):
    """Method 8: LDA projection weights (linear discriminant directions)"""
    print("  Using Method 8: LDA Projection Weights...")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_scaled, y_encoded)

    if len(lda.coef_.shape) > 1:
        importances = np.abs(lda.coef_).sum(axis=0)
    else:
        importances = np.abs(lda.coef_)

    top_indices = np.argsort(importances)[-n_features:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    top_scores = [importances[i] for i in top_indices]

    return top_features, top_scores, "LDA"


def ensemble_feature_selection(X, y, feature_names, n_features=10, methods="all"):
    """
    Ensemble feature selection – combine multiple methods.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
    feature_names : list of str
        Names of features (length = n_features)
    n_features : int
        Number of final top features to return
    methods : str or list
        'all'    - use all methods
        'fast'   - use fast methods (ANOVA, MI, RF)
        'robust' - use robust methods (RF, GB, Lasso, LDA)
        list     - specify method IDs, e.g. [1, 3, 4, 8]

    Returns
    -------
    top_features : list
    top_votes : list
    top_scores : list
    method_results : dict
        {method_name: {'features': [...], 'scores': [...]}}
    """
    print(f"\nEnsemble Feature Selection (n={n_features})")

    all_methods = {
        1: method1_anova_f_test,
        2: method2_mutual_information,
        3: method3_random_forest_importance,
        4: method4_gradient_boosting_importance,
        5: method5_lasso_selection,
        6: method6_relief_based,
        7: method7_differential_expression,
        8: method8_lda_projection,
    }

    fast_methods = [1, 2, 3]
    robust_methods = [3, 4, 5, 8]

    if methods == "all":
        selected_methods = list(all_methods.keys())
    elif methods == "fast":
        selected_methods = fast_methods
    elif methods == "robust":
        selected_methods = robust_methods
    elif isinstance(methods, list):
        selected_methods = methods
    else:
        selected_methods = [1, 3, 4, 8]

    method_results = {}
    feature_votes = {}
    feature_scores = {}

    for mid in selected_methods:
        if mid not in all_methods:
            continue
        try:
            feats, scores, mname = all_methods[mid](
                X, y, feature_names, n_features * 2
            )
            method_results[mname] = {"features": feats, "scores": scores}

            max_score = max(scores) if max(scores) > 0 else 1.0
            for f, s in zip(feats[:n_features], scores[:n_features]):
                if f not in feature_votes:
                    feature_votes[f] = 0
                    feature_scores[f] = 0.0
                feature_votes[f] += 1
                feature_scores[f] += s / max_score

            print(f"✓ {mname}: top feature = {feats[0]}")
        except Exception as e:
            print(f"✗ Method {mid} failed: {str(e)[:80]}")

    final_ranking = sorted(
        feature_votes.keys(),
        key=lambda x: (feature_votes[x], feature_scores[x]),
        reverse=True,
    )

    top_features = final_ranking[:n_features]
    top_votes = [feature_votes[f] for f in top_features]
    top_scores = [feature_scores[f] for f in top_features]

    print("  Ensemble selection finished.")
    return top_features, top_votes, top_scores, method_results


if __name__ == "__main__":
    print("Advanced Feature Selection (define 8 methods only, no plotting)")
