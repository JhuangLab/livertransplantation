import sys
import os
import pickle
import numpy as np
import pandas as pd

# Mapping between original group labels and unified labels
GROUP_MAPPING = {
    'Control': 'Control',
    'Rejection-pre': 'Rejection-treat-pre',
    'Rejection-post': 'Rejection-treat-post'
}

# Reverse mapping (from mapped labels back to original labels)
REVERSE_GROUP_MAPPING = {v: k for k, v in GROUP_MAPPING.items()}


def map_group_labels(labels):
    """
    Map group labels using GROUP_MAPPING.

    Parameters
    ----------
    labels : list / np.ndarray / str

    Returns
    -------
    Mapped labels with the same type as input.
    """
    if isinstance(labels, list):
        return [GROUP_MAPPING.get(label, label) for label in labels]
    elif isinstance(labels, np.ndarray):
        return np.array([GROUP_MAPPING.get(label, label) for label in labels])
    else:
        return GROUP_MAPPING.get(labels, labels)


def get_group_mapping():
    """Return a copy of the group mapping dictionary."""
    return GROUP_MAPPING.copy()


def load_feature_mappings():
    """
    Load feature name mappings for protein and metabolite.

    Follows the logic of comprehensive_interpretable_visualization_train_val_test.py.
    """
    print("Loading feature name mappings...")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.join(script_dir, 'raw_data')

    mappings = {}

    # Protein feature mapping
    try:
        protein_file = os.path.join(raw_data_dir, 'pr_exp_dat.tsv')
        protein_df = pd.read_csv(protein_file, sep='\t')

        protein_mapping = {}
        if 'genename' in protein_df.columns:
            for _, row in protein_df.iterrows():
                protein_id = row.iloc[0]  # first column = protein ID
                gene_name = row['genename']
                if pd.notna(gene_name) and gene_name != '':
                    protein_mapping[protein_id] = gene_name
                else:
                    protein_mapping[protein_id] = protein_id

        mappings['protein'] = protein_mapping
        print(f"  Protein mapping loaded: {len(protein_mapping)} features")
    except Exception as e:
        print(f"  Failed to load protein mapping: {e}")
        mappings['protein'] = {}

    # Metabolite feature mapping
    try:
        metabolite_file = os.path.join(raw_data_dir, 'mob_exp_dat.tsv')
        metabolite_df = pd.read_csv(metabolite_file, sep='\t')

        metabolite_mapping = {}
        if 'MS2Metabolite' in metabolite_df.columns:
            for _, row in metabolite_df.iterrows():
                metabolite_id = row.iloc[0]  # first column = metabolite ID
                ms2_name = row['MS2Metabolite']
                if pd.notna(ms2_name) and ms2_name not in ['-', '']:
                    metabolite_mapping[metabolite_id] = ms2_name
                else:
                    metabolite_mapping[metabolite_id] = metabolite_id

        mappings['metabolite'] = metabolite_mapping
        print(f"  Metabolite mapping loaded: {len(metabolite_mapping)} features")
    except Exception as e:
        print(f"  Failed to load metabolite mapping: {e}")
        mappings['metabolite'] = {}

    # Combined mapping for multi-omics
    mappings['multi_omics'] = {
        'protein': mappings['protein'],
        'metabolite': mappings['metabolite']
    }
    print(
        f"  Multi-omics mapping created: "
        f"protein({len(mappings['protein'])}) + metabolite({len(mappings['metabolite'])})"
    )

    return mappings


def map_feature_names(feature_names, omics_type, mappings):
    """
    Map feature names using the provided mappings.

    For multi-omics, features may have 'protein_' or 'metabolite_' prefixes.
    """
    if omics_type not in mappings:
        return feature_names

    mapping = mappings[omics_type]
    mapped_names = []

    for name in feature_names:
        # Multi-omics feature names
        if name.startswith('protein_'):
            original_name = name.replace('protein_', '')
            if omics_type == 'multi_omics':
                mapped_name = mappings['protein'].get(original_name, original_name)
                mapped_names.append(f"Protein: {mapped_name}")
            else:
                mapped_name = mapping.get(original_name, original_name)
                mapped_names.append(mapped_name)

        elif name.startswith('metabolite_'):
            original_name = name.replace('metabolite_', '')
            if omics_type == 'multi_omics':
                mapped_name = mappings['metabolite'].get(original_name, original_name)
                mapped_names.append(f"Metabolite: {mapped_name}")
            else:
                mapped_name = mapping.get(original_name, original_name)
                mapped_names.append(mapped_name)

        else:
            # Single-omics feature
            mapped_name = mapping.get(name, name)
            mapped_names.append(mapped_name)

    return mapped_names


def load_original_data():
    """
    Load original protein and metabolite data and sample information.

    Returns
    -------
    protein_df_common : pd.DataFrame
        Protein data for all common samples.
    metabolite_df_common : pd.DataFrame
        Metabolite data for all common samples (optionally filtered by MS2Metabolite).
    y : list
        Group labels (after GROUP_MAPPING).
    common_samples : list
        Sample IDs used.
    gender : list
        Gender per sample (same order as common_samples).
    """
    print("Loading original data...")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.join(script_dir, 'raw_data')

    # Protein data
    protein_file = os.path.join(raw_data_dir, 'pr_exp_dat.tsv')
    protein_df = pd.read_csv(protein_file, sep='\t')
    protein_df = protein_df.set_index(protein_df.columns[0]).T
    protein_df = protein_df.apply(pd.to_numeric, errors='coerce')

    # Metabolite data
    metabolite_file = os.path.join(raw_data_dir, 'mob_exp_dat.tsv')
    metabolite_df = pd.read_csv(metabolite_file, sep='\t')
    metabolite_df = metabolite_df.set_index(metabolite_df.columns[0]).T
    metabolite_df = metabolite_df.apply(pd.to_numeric, errors='coerce')

    # Sample info
    sample_info_file = os.path.join(raw_data_dir, 'sampleinfo.xlsx')
    df_group = pd.read_excel(sample_info_file)
    df_group_filtered = df_group[
        ~df_group['regroup'].isin(['unknown', 'unknow'])
    ].copy()

    # Common samples across protein, metabolite and sample info
    protein_samples = set(protein_df.index)
    metabolite_samples = set(metabolite_df.index)
    group_samples = set(df_group_filtered['sample_id'])
    common_samples = list(protein_samples & metabolite_samples & group_samples)

    # Filter metabolite features by MS2Metabolite names
    print("  Filtering metabolites with valid MS2Metabolite names (if available)...")
    print(f"  Original metabolite features: {len(metabolite_df.columns)}")

    feature_mappings = load_feature_mappings()
    valid_metabolite_features = []
    for feat in metabolite_df.columns:
        if feat in feature_mappings['metabolite']:
            ms2_name = feature_mappings['metabolite'][feat]
            if ms2_name and ms2_name.strip() and ms2_name != feat:
                valid_metabolite_features.append(feat)

    if len(valid_metabolite_features) == 0:
        # No MS2 names: keep all metabolite features
        metabolite_df_filtered = metabolite_df
    else:
        metabolite_df_filtered = metabolite_df[valid_metabolite_features]

    # Subset data for all common samples
    protein_df_common = protein_df.loc[common_samples]
    metabolite_df_common = metabolite_df_filtered.loc[common_samples]

    # Build label and gender vectors
    sample_to_group = dict(
        zip(df_group_filtered['sample_id'], df_group_filtered['regroup'])
    )
    sample_to_gender = dict(
        zip(df_group_filtered['sample_id'], df_group_filtered['gender'])
    )

    y = [sample_to_group[sid] for sid in common_samples]
    y = map_group_labels(y)
    gender = [sample_to_gender.get(sid, 'Unknown') for sid in common_samples]

    print(f"  Common samples: {len(common_samples)}")
    print(f"  Protein features: {protein_df_common.shape[1]}")
    print(
        f"  Metabolite features: {metabolite_df_common.shape[1]} "
        f"(from {len(metabolite_df.columns)} original features)"
    )

    return protein_df_common, metabolite_df_common, y, common_samples, gender


def load_consensus_features(omics_type, results_dir, min_votes=5):
    """
    Load high-consensus features from a binary matrix file.

    Parameters
    ----------
    omics_type : str
        'protein' or 'metabolite'.
    results_dir : str
        Directory containing the binary matrix CSV.
    min_votes : int
        Minimum number of votes to be considered a consensus feature.

    Returns
    -------
    consensus_features : list of str
    """
    matrix_file = os.path.join(results_dir, f'{omics_type}_feature_binary_matrix.csv')

    if not os.path.exists(matrix_file):
        raise FileNotFoundError(f"Binary matrix file not found: {matrix_file}")

    df = pd.read_csv(matrix_file)

    # Columns: Feature, ANOVA_F, MutualInfo, RandomForest, GradientBoosting,
    #          Lasso, Relief, DiffExpr, LDA, Ensemble_Robust
    method_cols = [col for col in df.columns if col != 'Feature']
    df['total_votes'] = df[method_cols].sum(axis=1)

    consensus_df = df[df['total_votes'] >= min_votes]
    consensus_features = consensus_df['Feature'].tolist()

    print(
        f"{omics_type.capitalize()}: {len(consensus_features)} consensus features "
        f"(≥{min_votes} votes)"
    )

    return consensus_features


def load_cv_splits(cv_splits_file):
    """
    Load precomputed cross-validation splits (e.g., 5-fold CV).

    Returns
    -------
    cv_splits : list or dict
    """
    with open(cv_splits_file, 'rb') as f:
        cv_splits = pickle.load(f)

    print(f"Loaded CV splits from: {cv_splits_file}")
    print(f"Number of folds: {len(cv_splits)}")

    return cv_splits


def prepare_data(omics_type, consensus_features):
    """
    Prepare data matrix X and labels y for model training.

    Parameters
    ----------
    omics_type : str
        'protein', 'metabolite', or 'multi_omics'.
    consensus_features : dict
        For single-omics: dict with key 'protein' or 'metabolite'.
        For multi-omics: dict with keys 'protein' and 'metabolite'.

    Returns
    -------
    X : np.ndarray
        Feature matrix (samples × features).
    y : list
        Labels.
    feature_names : list of str
        Feature names (column order matches X).
    sample_ids : list of str
        Sample IDs.
    """
    protein_df, metabolite_df, y_labels, sample_ids, gender = load_original_data()
    _ = load_feature_mappings()  # kept if needed elsewhere

    print(f"Preparing {omics_type} data...")
    print(f"  Total samples: {len(sample_ids)}")

    if omics_type == 'protein':
        available_features = [
            f for f in consensus_features['protein'] if f in protein_df.columns
        ]
        X = protein_df[available_features].values
        feature_names = available_features
        print(f"  Using {len(available_features)} protein consensus features")

    elif omics_type == 'metabolite':
        available_features = [
            f for f in consensus_features['metabolite'] if f in metabolite_df.columns
        ]
        X = metabolite_df[available_features].values
        feature_names = available_features
        print(f"  Using {len(available_features)} metabolite consensus features")

    elif omics_type == 'multi_omics':
        protein_features = [
            f for f in consensus_features['protein'] if f in protein_df.columns
        ]
        metabolite_features = [
            f for f in consensus_features['metabolite'] if f in metabolite_df.columns
        ]

        X_protein = protein_df[protein_features].values
        X_metabolite = metabolite_df[metabolite_features].values
        X = np.hstack([X_protein, X_metabolite])

        feature_names = (
            [f'protein_{f}' for f in protein_features]
            + [f'metabolite_{f}' for f in metabolite_features]
        )
        print(
            f"  Using {len(protein_features)} protein + "
            f"{len(metabolite_features)} metabolite consensus features"
        )
        print(f"  Total features: {X.shape[1]}")

    else:
        raise ValueError(f"Unknown omics_type: {omics_type}")

    # Replace NaNs with 0
    X = np.nan_to_num(X, nan=0.0)

    return X, y_labels, feature_names, sample_ids