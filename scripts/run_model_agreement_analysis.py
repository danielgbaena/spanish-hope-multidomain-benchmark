from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"

TEST_GOLD_FILE = DATA_DIR / "test_gold.csv"

MODEL_FILES = {
    "SVM": RESULTS_DIR / "svm_predictions.csv",
    "BETO": RESULTS_DIR / "beto_majority_predictions.csv",
    "RoBERTuito": RESULTS_DIR / "robertuito_majority_predictions.csv",
    "GPT-4.1-mini": RESULTS_DIR / "gpt41_mini_predictions.csv",
    "GPT-4o-mini": RESULTS_DIR / "gpt4o_mini_predictions.csv",
    "Qwen2.5-7B": RESULTS_DIR / "qwen25_7b_predictions.csv",
}

REQUIRED_PREDICTION_COLUMNS = {
    "id",
    "gold",
    "prediction",
    "topic",
}

REQUIRED_TEST_GOLD_COLUMNS = {
    "id",
    "text",
    "category",
    "topic",
}

DIFFICULTY_ORDER = [
    "easy",
    "moderate",
    "hard",
]


# =============================================================================
# Prediction-file loading and validation
# =============================================================================

def load_prediction_file(model_name, file_path):
    """
    Load and validate one model prediction file.

    Each prediction file must contain:
        id, gold, prediction, topic

    Returns
    -------
    pandas.DataFrame
        Validated prediction dataframe.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Prediction file for {model_name} not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    missing_columns = (
        REQUIRED_PREDICTION_COLUMNS.difference(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{model_name}: missing required columns "
            f"{sorted(missing_columns)}"
        )

    df = df[
        [
            "id",
            "gold",
            "prediction",
            "topic",
        ]
    ].copy()

    if df["id"].duplicated().any():
        duplicated_ids = df.loc[
            df["id"].duplicated(),
            "id",
        ].tolist()

        raise ValueError(
            f"{model_name}: duplicated instance IDs detected: "
            f"{duplicated_ids[:10]}"
        )

    if df.isnull().any().any():
        null_columns = df.columns[
            df.isnull().any()
        ].tolist()

        raise ValueError(
            f"{model_name}: missing values detected in columns "
            f"{null_columns}"
        )

    valid_labels = {
        "hs",
        "nhs",
    }

    invalid_gold_labels = (
        set(df["gold"].unique()) - valid_labels
    )

    invalid_prediction_labels = (
        set(df["prediction"].unique()) - valid_labels
    )

    if invalid_gold_labels:
        raise ValueError(
            f"{model_name}: invalid gold labels detected: "
            f"{sorted(invalid_gold_labels)}"
        )

    if invalid_prediction_labels:
        raise ValueError(
            f"{model_name}: invalid prediction labels detected: "
            f"{sorted(invalid_prediction_labels)}"
        )

    return df


def validate_common_instances(prediction_dfs):
    """
    Verify that all prediction files contain exactly the same
    test instances and identical gold labels and topic assignments.

    Returns
    -------
    pandas.DataFrame
        Reference dataframe containing id, gold, and topic.
    """

    model_names = list(prediction_dfs.keys())

    reference_model = model_names[0]

    reference_df = (
        prediction_dfs[reference_model]
        .sort_values("id")
        .reset_index(drop=True)
    )

    reference_ids = reference_df["id"].tolist()

    for model_name in model_names[1:]:

        current_df = (
            prediction_dfs[model_name]
            .sort_values("id")
            .reset_index(drop=True)
        )

        current_ids = current_df["id"].tolist()

        if current_ids != reference_ids:
            raise ValueError(
                f"{model_name}: test-instance IDs do not match "
                f"those of {reference_model}."
            )

        if not current_df["gold"].equals(
            reference_df["gold"]
        ):
            raise ValueError(
                f"{model_name}: gold labels do not match "
                f"those of {reference_model}."
            )

        if not current_df["topic"].equals(
            reference_df["topic"]
        ):
            raise ValueError(
                f"{model_name}: topic assignments do not match "
                f"those of {reference_model}."
            )

    return reference_df[
        [
            "id",
            "gold",
            "topic",
        ]
    ].copy()


# =============================================================================
# Gold test-set loading and validation
# =============================================================================

def load_test_gold(file_path):
    """
    Load and validate the official gold test partition.

    The official benchmark file uses:
        id, text, category, topic

    The category column contains the gold labels and is renamed
    internally to gold so that the analysis code uses a consistent
    schema across prediction files and the official test partition.

    Returns
    -------
    pandas.DataFrame
        Validated gold test dataframe with columns:
        id, text, gold, topic
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Gold test file not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    missing_columns = (
        REQUIRED_TEST_GOLD_COLUMNS.difference(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Gold test file: missing required columns "
            f"{sorted(missing_columns)}"
        )

    df = df[
        [
            "id",
            "text",
            "category",
            "topic",
        ]
    ].copy()

    # Rename the official benchmark label column to the internal
    # analysis name used throughout the reproducibility pipeline.
    df = df.rename(
        columns={
            "category": "gold",
        }
    )

    if df["id"].duplicated().any():
        duplicated_ids = df.loc[
            df["id"].duplicated(),
            "id",
        ].tolist()

        raise ValueError(
            f"Gold test file: duplicated instance IDs detected: "
            f"{duplicated_ids[:10]}"
        )

    if df.isnull().any().any():
        null_columns = df.columns[
            df.isnull().any()
        ].tolist()

        raise ValueError(
            f"Gold test file: missing values detected in columns "
            f"{null_columns}"
        )

    valid_labels = {
        "hs",
        "nhs",
    }

    invalid_gold_labels = (
        set(df["gold"].unique()) - valid_labels
    )

    if invalid_gold_labels:
        raise ValueError(
            f"Gold test file: invalid gold labels detected: "
            f"{sorted(invalid_gold_labels)}"
        )

    return df


def validate_test_gold(test_gold_df, reference_df):
    """
    Verify that the official gold test partition contains exactly
    the same IDs, gold labels, and topic assignments as the
    prediction files.
    """

    test_gold_sorted = (
        test_gold_df
        .sort_values("id")
        .reset_index(drop=True)
    )

    reference_sorted = (
        reference_df
        .sort_values("id")
        .reset_index(drop=True)
    )

    if test_gold_sorted["id"].tolist() != (
        reference_sorted["id"].tolist()
    ):
        raise ValueError(
            "Gold test file IDs do not match prediction-file IDs."
        )

    if not test_gold_sorted["gold"].equals(
        reference_sorted["gold"]
    ):
        raise ValueError(
            "Gold labels in test_gold.csv do not match "
            "prediction-file gold labels."
        )

    if not test_gold_sorted["topic"].equals(
        reference_sorted["topic"]
    ):
        raise ValueError(
            "Topic assignments in test_gold.csv do not match "
            "prediction-file topic assignments."
        )


# =============================================================================
# Prediction matrix
# =============================================================================

def build_prediction_matrix(prediction_dfs, reference_df):
    """
    Construct a dataframe containing the gold labels, topics,
    and predictions from every evaluated model.

    Returns
    -------
    pandas.DataFrame
        Combined prediction matrix.
    """

    prediction_matrix = reference_df.copy()

    for model_name, df in prediction_dfs.items():

        prediction_column = f"{model_name}_prediction"

        model_predictions = (
            df[
                [
                    "id",
                    "prediction",
                ]
            ]
            .rename(
                columns={
                    "prediction": prediction_column,
                }
            )
        )

        prediction_matrix = prediction_matrix.merge(
            model_predictions,
            on="id",
            how="inner",
            validate="one_to_one",
        )

    return prediction_matrix


# =============================================================================
# Pairwise model agreement
# =============================================================================

def compute_pairwise_agreement(
    prediction_matrix,
    model_names,
):
    """
    Compute pairwise raw agreement and Cohen's kappa for every
    pair of evaluated systems.

    Returns
    -------
    pandas.DataFrame
        Pairwise agreement results.
    """

    rows = []

    for model_a, model_b in combinations(
        model_names,
        2,
    ):

        column_a = f"{model_a}_prediction"
        column_b = f"{model_b}_prediction"

        predictions_a = prediction_matrix[column_a]
        predictions_b = prediction_matrix[column_b]

        agreement_mask = (
            predictions_a == predictions_b
        )

        n_instances = len(prediction_matrix)

        n_agreements = int(
            agreement_mask.sum()
        )

        n_disagreements = (
            n_instances - n_agreements
        )

        agreement_rate = (
            n_agreements / n_instances
        )

        kappa = cohen_kappa_score(
            predictions_a,
            predictions_b,
        )

        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "n_instances": n_instances,
                "n_agreements": n_agreements,
                "n_disagreements": n_disagreements,
                "agreement_rate": agreement_rate,
                "cohen_kappa": kappa,
            }
        )

    agreement_df = pd.DataFrame(rows)

    agreement_df = agreement_df.sort_values(
        [
            "agreement_rate",
            "cohen_kappa",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(drop=True)

    return agreement_df


def build_agreement_matrix(
    agreement_df,
    model_names,
):
    """
    Build a symmetric matrix of pairwise raw agreement rates.

    Returns
    -------
    pandas.DataFrame
        Symmetric agreement matrix.
    """

    matrix = pd.DataFrame(
        np.eye(len(model_names)),
        index=model_names,
        columns=model_names,
    )

    for _, row in agreement_df.iterrows():

        model_a = row["model_a"]
        model_b = row["model_b"]

        agreement_rate = row[
            "agreement_rate"
        ]

        matrix.loc[
            model_a,
            model_b,
        ] = agreement_rate

        matrix.loc[
            model_b,
            model_a,
        ] = agreement_rate

    matrix.index.name = "model"

    return matrix


# =============================================================================
# Empirical instance difficulty
# =============================================================================

def assign_difficulty_group(n_model_errors):
    """
    Assign an empirical instance-difficulty group relative to
    the six evaluated systems.

    easy:
        0 or 1 model errors.

    moderate:
        2 or 3 model errors.

    hard:
        4, 5, or 6 model errors.
    """

    if n_model_errors <= 1:
        return "easy"

    if n_model_errors <= 3:
        return "moderate"

    return "hard"


def compute_instance_difficulty(
    prediction_matrix,
    model_names,
):
    """
    Characterize empirical instance difficulty according to the
    number of evaluated systems that misclassify each instance.

    Returns
    -------
    pandas.DataFrame
        Instance-level difficulty results.
    """

    difficulty_df = prediction_matrix.copy()

    correctness_columns = []

    for model_name in model_names:

        prediction_column = (
            f"{model_name}_prediction"
        )

        correctness_column = (
            f"{model_name}_correct"
        )

        difficulty_df[correctness_column] = (
            difficulty_df[prediction_column]
            == difficulty_df["gold"]
        ).astype(int)

        correctness_columns.append(
            correctness_column
        )

    difficulty_df["n_model_correct"] = (
        difficulty_df[
            correctness_columns
        ].sum(axis=1)
    )

    difficulty_df["n_model_errors"] = (
        len(model_names)
        - difficulty_df["n_model_correct"]
    )

    difficulty_df["error_rate"] = (
        difficulty_df["n_model_errors"]
        / len(model_names)
    )

    difficulty_df["difficulty_group"] = (
        difficulty_df["n_model_errors"]
        .apply(assign_difficulty_group)
    )

    difficulty_df["difficulty_group"] = pd.Categorical(
        difficulty_df["difficulty_group"],
        categories=DIFFICULTY_ORDER,
        ordered=True,
    )

    difficulty_df = difficulty_df.sort_values(
        [
            "n_model_errors",
            "topic",
            "id",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    ).reset_index(drop=True)

    return difficulty_df


def compute_global_difficulty_distribution(
    difficulty_df,
):
    """
    Compute the global empirical difficulty distribution.

    Returns
    -------
    pandas.DataFrame
        Global difficulty distribution.
    """

    distribution = (
        difficulty_df[
            "difficulty_group"
        ]
        .value_counts(sort=False)
        .rename("n_instances")
        .to_frame()
    )

    distribution["percentage"] = (
        100.0
        * distribution["n_instances"]
        / len(difficulty_df)
    )

    distribution["percentage"] = (
        distribution["percentage"]
        .round(2)
    )

    return distribution


def compute_mean_errors_by_domain(
    difficulty_df,
):
    """
    Compute descriptive statistics for the number of model errors
    within each evaluation domain.

    Returns
    -------
    pandas.DataFrame
        Domain-level error summary.
    """

    summary = (
        difficulty_df
        .groupby("topic")["n_model_errors"]
        .agg(
            [
                "count",
                "mean",
                "std",
                "median",
                "max",
            ]
        )
    )

    return summary


def compute_difficulty_by_domain(
    difficulty_df,
):
    """
    Compute empirical difficulty-group distributions within each
    evaluation domain.

    Returns
    -------
    pandas.DataFrame
        Domain-level difficulty distributions.
    """

    difficulty_by_domain = (
        difficulty_df
        .groupby(
            [
                "topic",
                "difficulty_group",
            ],
            observed=False,
        )
        .size()
        .reset_index(
            name="n_instances"
        )
    )

    domain_sizes = (
        difficulty_df
        .groupby("topic")
        .size()
        .rename("domain_size")
        .reset_index()
    )

    difficulty_by_domain = (
        difficulty_by_domain.merge(
            domain_sizes,
            on="topic",
            how="left",
            validate="many_to_one",
        )
    )

    difficulty_by_domain[
        "percentage_within_domain"
    ] = (
        100.0
        * difficulty_by_domain["n_instances"]
        / difficulty_by_domain["domain_size"]
    )

    difficulty_by_domain[
        "percentage_within_domain"
    ] = (
        difficulty_by_domain[
            "percentage_within_domain"
        ].round(2)
    )

    difficulty_by_domain = (
        difficulty_by_domain[
            [
                "topic",
                "difficulty_group",
                "n_instances",
                "percentage_within_domain",
            ]
        ]
    )

    return difficulty_by_domain


# =============================================================================
# Unanimous-error extraction
# =============================================================================

def extract_unanimous_errors(
    difficulty_df,
    test_gold_df,
    model_names,
):
    """
    Extract instances misclassified by every evaluated system and
    merge them with the original post text.

    The resulting output provides a reproducible basis for
    qualitative inspection of shared cross-model failures.

    Returns
    -------
    pandas.DataFrame
        Unanimously misclassified test instances.
    """

    n_models = len(model_names)

    unanimous_errors = (
        difficulty_df.loc[
            difficulty_df["n_model_errors"]
            == n_models
        ]
        .copy()
    )

    unanimous_errors = unanimous_errors.merge(
        test_gold_df[
            [
                "id",
                "text",
                "gold",
                "topic",
            ]
        ],
        on="id",
        how="left",
        suffixes=(
            "_analysis",
            "_test_gold",
        ),
        validate="one_to_one",
    )

    if unanimous_errors["text"].isnull().any():
        raise ValueError(
            "Missing original text after merging unanimous errors "
            "with test_gold.csv."
        )

    if not (
        unanimous_errors["gold_analysis"]
        == unanimous_errors["gold_test_gold"]
    ).all():
        raise ValueError(
            "Gold-label inconsistency detected while extracting "
            "unanimous errors."
        )

    if not (
        unanimous_errors["topic_analysis"]
        == unanimous_errors["topic_test_gold"]
    ).all():
        raise ValueError(
            "Topic inconsistency detected while extracting "
            "unanimous errors."
        )

    unanimous_errors = unanimous_errors.rename(
        columns={
            "gold_analysis": "gold",
            "topic_analysis": "topic",
        }
    )

    prediction_columns = [
        f"{model_name}_prediction"
        for model_name in model_names
    ]

    output_columns = [
        "id",
        "text",
        "gold",
        "topic",
        "n_model_correct",
        "n_model_errors",
        "error_rate",
        "difficulty_group",
    ] + prediction_columns

    unanimous_errors = unanimous_errors[
        output_columns
    ].copy()

    unanimous_errors = unanimous_errors.sort_values(
        [
            "topic",
            "gold",
            "id",
        ]
    ).reset_index(drop=True)

    return unanimous_errors


def compute_unanimous_error_distribution(
    unanimous_errors,
):
    """
    Compute the domain distribution of unanimous model failures.

    Returns
    -------
    pandas.DataFrame
        Number and percentage of unanimous failures by domain.
    """

    if len(unanimous_errors) == 0:
        return pd.DataFrame(
            columns=[
                "topic",
                "n_instances",
                "percentage_of_unanimous_errors",
            ]
        )

    distribution = (
        unanimous_errors
        .groupby("topic")
        .size()
        .reset_index(
            name="n_instances"
        )
    )

    distribution[
        "percentage_of_unanimous_errors"
    ] = (
        100.0
        * distribution["n_instances"]
        / len(unanimous_errors)
    )

    distribution[
        "percentage_of_unanimous_errors"
    ] = (
        distribution[
            "percentage_of_unanimous_errors"
        ].round(2)
    )

    return distribution


# =============================================================================
# Main analysis
# =============================================================================

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading prediction files...")

    prediction_dfs = {
        model_name: load_prediction_file(
            model_name,
            file_path,
        )
        for model_name, file_path
        in MODEL_FILES.items()
    }

    reference_df = validate_common_instances(
        prediction_dfs
    )

    model_names = list(
        MODEL_FILES.keys()
    )

    print(
        f"Validated {len(model_names)} models on "
        f"{len(reference_df)} common test instances."
    )

    print("Loading official gold test partition...")

    test_gold_df = load_test_gold(
        TEST_GOLD_FILE
    )

    validate_test_gold(
        test_gold_df,
        reference_df,
    )

    print(
        f"Validated test_gold.csv on "
        f"{len(test_gold_df)} test instances."
    )

    prediction_matrix = build_prediction_matrix(
        prediction_dfs,
        reference_df,
    )

    # -------------------------------------------------------------------------
    # Pairwise model agreement
    # -------------------------------------------------------------------------

    agreement_df = compute_pairwise_agreement(
        prediction_matrix,
        model_names,
    )

    agreement_matrix = build_agreement_matrix(
        agreement_df,
        model_names,
    )

    model_agreement_path = (
        RESULTS_DIR
        / "model_agreement.csv"
    )

    model_agreement_matrix_path = (
        RESULTS_DIR
        / "model_agreement_matrix.csv"
    )

    agreement_df.to_csv(
        model_agreement_path,
        index=False,
    )

    agreement_matrix.to_csv(
        model_agreement_matrix_path,
    )

    print("\nPairwise model agreement:")

    print(
        agreement_df.to_string(
            index=False,
            formatters={
                "agreement_rate":
                    "{:.4f}".format,
                "cohen_kappa":
                    "{:.4f}".format,
            },
        )
    )

    # -------------------------------------------------------------------------
    # Empirical instance difficulty
    # -------------------------------------------------------------------------

    difficulty_df = compute_instance_difficulty(
        prediction_matrix,
        model_names,
    )

    instance_difficulty_path = (
        RESULTS_DIR
        / "instance_difficulty.csv"
    )

    difficulty_df.to_csv(
        instance_difficulty_path,
        index=False,
    )

    difficulty_distribution = (
        compute_global_difficulty_distribution(
            difficulty_df
        )
    )

    print("\nInstance difficulty distribution:")

    print(
        difficulty_distribution.to_string(
            formatters={
                "percentage":
                    "{:.2f}".format,
            }
        )
    )

    # -------------------------------------------------------------------------
    # Mean model errors by domain
    # -------------------------------------------------------------------------

    domain_error_summary = (
        compute_mean_errors_by_domain(
            difficulty_df
        )
    )

    print(
        "\nMean number of model errors by domain:"
    )

    print(
        domain_error_summary.to_string(
            formatters={
                "mean":
                    "{:.4f}".format,
                "std":
                    "{:.4f}".format,
                "median":
                    "{:.4f}".format,
            }
        )
    )

    # -------------------------------------------------------------------------
    # Difficulty distribution by domain
    # -------------------------------------------------------------------------

    difficulty_by_domain = (
        compute_difficulty_by_domain(
            difficulty_df
        )
    )

    difficulty_by_domain_path = (
        RESULTS_DIR
        / "instance_difficulty_by_domain.csv"
    )

    difficulty_by_domain.to_csv(
        difficulty_by_domain_path,
        index=False,
    )

    print(
        "\nInstance difficulty distribution by domain:"
    )

    print(
        difficulty_by_domain.to_string(
            index=False,
            formatters={
                "percentage_within_domain":
                    "{:.2f}".format,
            },
        )
    )

    # -------------------------------------------------------------------------
    # Unanimous-error extraction
    # -------------------------------------------------------------------------

    unanimous_errors = extract_unanimous_errors(
        difficulty_df,
        test_gold_df,
        model_names,
    )

    unanimous_errors_path = (
        RESULTS_DIR
        / "unanimous_error_instances.csv"
    )

    unanimous_errors.to_csv(
        unanimous_errors_path,
        index=False,
    )

    unanimous_error_distribution = (
        compute_unanimous_error_distribution(
            unanimous_errors
        )
    )

    print(
        "\nUnanimous model failures:"
    )

    print(
        f"Total unanimously misclassified instances: "
        f"{len(unanimous_errors)}"
    )

    print(
        "\nDistribution of unanimous failures by domain:"
    )

    if unanimous_error_distribution.empty:
        print(
            "No unanimously misclassified instances."
        )
    else:
        print(
            unanimous_error_distribution.to_string(
                index=False,
                formatters={
                    "percentage_of_unanimous_errors":
                        "{:.2f}".format,
                },
            )
        )

    # -------------------------------------------------------------------------
    # Saved outputs
    # -------------------------------------------------------------------------

    print("\nSaved outputs:")

    print(
        f"  {model_agreement_path}"
    )

    print(
        f"  {model_agreement_matrix_path}"
    )

    print(
        f"  {instance_difficulty_path}"
    )

    print(
        f"  {difficulty_by_domain_path}"
    )

    print(
        f"  {unanimous_errors_path}"
    )


if __name__ == "__main__":
    main()