#!/usr/bin/env python3
"""
Run model-agreement analysis, model-relative instance-difficulty analysis,
and extraction of unanimously misclassified instances for manual qualitative
error analysis.

The script evaluates six model-level prediction vectors:

    - SVM
    - BETO majority vote
    - RoBERTuito majority vote
    - Qwen2.5-7B-Instruct
    - GPT-4o-mini
    - GPT-4.1-mini

Canonical benchmark files
-------------------------
data/test.csv
    Columns:
        id, text

data/test_gold.csv
    Columns:
        id, text, category, topic

Prediction files
----------------
results/svm_predictions.csv
results/beto_majority_predictions.csv
results/robertuito_majority_predictions.csv
results/qwen25_7b_predictions.csv
results/gpt4o_mini_predictions.csv
results/gpt41_mini_predictions.csv

Expected prediction columns:
    id, gold, prediction, topic

The script performs the following analyses:

1. Resolves and loads the canonical benchmark test files.
2. Validates test.csv against test_gold.csv by stable instance ID.
3. Loads all six model-level prediction vectors.
4. Aligns predictions to the canonical benchmark order by instance ID.
5. Validates:
       - exactly 400 canonical test instances,
       - no missing IDs,
       - no unexpected IDs,
       - no duplicate IDs,
       - gold-label consistency,
       - domain consistency.
6. Computes pairwise:
       - raw prediction agreement,
       - Cohen's kappa.
7. Computes model-relative empirical instance groupings:
       - easy:     0-1 model errors,
       - moderate: 2-3 model errors,
       - hard:     4-6 model errors.
8. Computes model-relative difficulty summaries by domain.
9. Extracts instances misclassified by all six model-level prediction vectors.
10. Creates a manual qualitative coding template containing original texts.

Important methodological note
-----------------------------
The easy/moderate/hard categories are model-relative empirical groupings.
They must not be described as intrinsic instance difficulty.

The qualitative coding template is an input to manual qualitative analysis.
Generating it does not itself constitute qualitative error analysis.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


# =============================================================================
# CONFIGURATION
# =============================================================================

EXPECTED_N_INSTANCES = 400

MODEL_FILES = {
    "SVM": "svm_predictions.csv",
    "BETO": "beto_majority_predictions.csv",
    "RoBERTuito": "robertuito_majority_predictions.csv",
    "Qwen2.5-7B": "qwen25_7b_predictions.csv",
    "GPT-4o-mini": "gpt4o_mini_predictions.csv",
    "GPT-4.1-mini": "gpt41_mini_predictions.csv",
}

MODEL_ORDER = list(MODEL_FILES.keys())

TEST_FILE = "test.csv"
TEST_GOLD_FILE = "test_gold.csv"

PAIRWISE_AGREEMENT_OUTPUT = "model_agreement.csv"
AGREEMENT_MATRIX_OUTPUT = "model_agreement_matrix.csv"
INSTANCE_DIFFICULTY_OUTPUT = "instance_difficulty.csv"
DIFFICULTY_BY_DOMAIN_OUTPUT = "instance_difficulty_by_domain.csv"
UNANIMOUS_ERRORS_OUTPUT = "unanimous_error_instances.csv"
QUALITATIVE_TEMPLATE_OUTPUT = "qualitative_error_coding_template.csv"


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def print_section(title: str) -> None:
    print()
    print("=" * 40)
    print(title)
    print("=" * 40)


def normalize_id_series(series: pd.Series) -> pd.Series:
    """
    Normalize instance identifiers to strings.

    Reading IDs directly as strings would also be valid, but this explicit
    normalization makes comparisons robust if a dataframe was constructed
    using integer identifiers.
    """
    if series.isna().any():
        raise ValueError("Instance ID column contains missing values.")

    return series.astype(str).str.strip()


def normalize_label(value: object) -> str:
    """
    Normalize benchmark labels to canonical lowercase values:
        hs
        nhs
    """
    normalized = str(value).strip().lower()

    mapping = {
        "hs": "hs",
        "hope speech": "hs",
        "hope_speech": "hs",
        "hope-speech": "hs",
        "nhs": "nhs",
        "non hope speech": "nhs",
        "non_hope_speech": "nhs",
        "non-hope-speech": "nhs",
        "non hope": "nhs",
    }

    if normalized not in mapping:
        raise ValueError(f"Unrecognized label value: {value!r}")

    return mapping[normalized]


def normalize_label_series(series: pd.Series) -> pd.Series:
    if series.isna().any():
        raise ValueError("Label column contains missing values.")

    return series.map(normalize_label)


def normalize_domain(value: object) -> str:
    if pd.isna(value):
        raise ValueError("Domain column contains missing values.")

    return str(value).strip().lower()


def normalize_domain_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_domain)


def validate_unique_ids(
    dataframe: pd.DataFrame,
    source_name: str,
) -> None:
    duplicated = dataframe["id"].duplicated(keep=False)

    if duplicated.any():
        duplicate_ids = (
            dataframe.loc[duplicated, "id"]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            f"Duplicate instance IDs found in {source_name}:\n"
            f"{duplicate_ids[:20]}"
        )


def validate_expected_size(
    dataframe: pd.DataFrame,
    source_name: str,
) -> None:
    if len(dataframe) != EXPECTED_N_INSTANCES:
        raise ValueError(
            f"{source_name} contains {len(dataframe)} rows; "
            f"expected {EXPECTED_N_INSTANCES}."
        )


def require_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    source_name: str,
) -> None:
    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns in {source_name}: {missing}\n"
            f"Available columns: {dataframe.columns.tolist()}"
        )


# =============================================================================
# REPOSITORY PATHS
# =============================================================================

def resolve_repository_paths() -> tuple[Path, Path, Path]:
    script_path = Path(__file__).resolve()
    repository_root = script_path.parent.parent
    data_dir = repository_root / "data"
    results_dir = repository_root / "results"

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}"
        )

    if not results_dir.is_dir():
        raise FileNotFoundError(
            f"Results directory not found: {results_dir}"
        )

    return repository_root, data_dir, results_dir


# =============================================================================
# CANONICAL BENCHMARK LOADING
# =============================================================================

def load_test_input(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Canonical test input file not found: {path}"
        )

    dataframe = pd.read_csv(path, dtype={"id": str})

    require_columns(
        dataframe,
        ["id", "text"],
        str(path),
    )

    dataframe = dataframe[["id", "text"]].copy()

    dataframe["id"] = normalize_id_series(dataframe["id"])

    if dataframe["text"].isna().any():
        raise ValueError(
            f"Missing text values found in {path}."
        )

    validate_expected_size(dataframe, str(path))
    validate_unique_ids(dataframe, str(path))

    return dataframe


def load_test_gold(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Canonical gold test file not found: {path}"
        )

    dataframe = pd.read_csv(path, dtype={"id": str})

    require_columns(
        dataframe,
        ["id", "text", "category", "topic"],
        str(path),
    )

    dataframe = dataframe[
        ["id", "text", "category", "topic"]
    ].copy()

    dataframe["id"] = normalize_id_series(dataframe["id"])
    dataframe["gold"] = normalize_label_series(dataframe["category"])
    dataframe["domain"] = normalize_domain_series(dataframe["topic"])

    dataframe = dataframe[
        ["id", "text", "gold", "domain"]
    ]

    if dataframe["text"].isna().any():
        raise ValueError(
            f"Missing text values found in {path}."
        )

    validate_expected_size(dataframe, str(path))
    validate_unique_ids(dataframe, str(path))

    return dataframe


def validate_test_input_against_gold(
    test_input: pd.DataFrame,
    test_gold: pd.DataFrame,
) -> None:
    input_ids = set(test_input["id"])
    gold_ids = set(test_gold["id"])

    missing_from_input = sorted(gold_ids - input_ids)
    unexpected_in_input = sorted(input_ids - gold_ids)

    if missing_from_input or unexpected_in_input:
        raise ValueError(
            "ID mismatch between data/test.csv and "
            "data/test_gold.csv.\n"
            f"Missing from test.csv: {missing_from_input[:20]}\n"
            f"Unexpected in test.csv: {unexpected_in_input[:20]}"
        )

    input_by_id = test_input.set_index("id")
    gold_by_id = test_gold.set_index("id")

    aligned_input = input_by_id.loc[gold_by_id.index]

    text_match = (
        aligned_input["text"].astype(str)
        ==
        gold_by_id["text"].astype(str)
    )

    if not text_match.all():
        mismatched_ids = text_match.index[~text_match].tolist()

        raise ValueError(
            "Text mismatch between data/test.csv and "
            "data/test_gold.csv.\n"
            f"Mismatched IDs: {mismatched_ids[:20]}"
        )


# =============================================================================
# PREDICTION LOADING AND VALIDATION
# =============================================================================

def load_prediction_file(
    path: Path,
    model_name: str,
    canonical_test: pd.DataFrame,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Prediction file for {model_name} not found: {path}"
        )

    dataframe = pd.read_csv(path, dtype={"id": str})

    require_columns(
        dataframe,
        ["id", "gold", "prediction", "topic"],
        str(path),
    )

    dataframe = dataframe[
        ["id", "gold", "prediction", "topic"]
    ].copy()

    dataframe["id"] = normalize_id_series(dataframe["id"])
    dataframe["gold"] = normalize_label_series(dataframe["gold"])
    dataframe["prediction"] = normalize_label_series(
        dataframe["prediction"]
    )
    dataframe["domain"] = normalize_domain_series(
        dataframe["topic"]
    )

    dataframe = dataframe[
        ["id", "gold", "prediction", "domain"]
    ]

    validate_expected_size(dataframe, str(path))
    validate_unique_ids(dataframe, str(path))

    canonical_ids = set(canonical_test["id"])
    prediction_ids = set(dataframe["id"])

    missing_ids = sorted(canonical_ids - prediction_ids)
    unexpected_ids = sorted(prediction_ids - canonical_ids)

    if missing_ids or unexpected_ids:
        raise ValueError(
            f"ID mismatch for {model_name}.\n"
            f"Missing IDs: {missing_ids[:20]}\n"
            f"Unexpected IDs: {unexpected_ids[:20]}"
        )

    # Align by stable instance ID to canonical benchmark order.
    dataframe = (
        dataframe
        .set_index("id")
        .loc[canonical_test["id"]]
        .reset_index()
    )

    gold_matches = (
        dataframe["gold"].to_numpy()
        ==
        canonical_test["gold"].to_numpy()
    )

    if not gold_matches.all():
        mismatch_positions = np.where(~gold_matches)[0]
        mismatch_ids = canonical_test.iloc[mismatch_positions]["id"].tolist()

        raise ValueError(
            f"Gold-label mismatch for {model_name}.\n"
            f"Mismatched IDs: {mismatch_ids[:20]}"
        )

    domain_matches = (
        dataframe["domain"].to_numpy()
        ==
        canonical_test["domain"].to_numpy()
    )

    if not domain_matches.all():
        mismatch_positions = np.where(~domain_matches)[0]
        mismatch_ids = canonical_test.iloc[mismatch_positions]["id"].tolist()

        raise ValueError(
            f"Domain mismatch for {model_name}.\n"
            f"Mismatched IDs: {mismatch_ids[:20]}"
        )

    return dataframe


# =============================================================================
# PAIRWISE MODEL AGREEMENT
# =============================================================================

def compute_pairwise_agreement(
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []

    for model_1, model_2 in combinations(MODEL_ORDER, 2):
        prediction_1 = predictions[model_1]
        prediction_2 = predictions[model_2]

        agreement_mask = prediction_1 == prediction_2

        n_agree = int(agreement_mask.sum())
        n_disagree = int((~agreement_mask).sum())

        raw_agreement = float(agreement_mask.mean())

        kappa = float(
            cohen_kappa_score(
                prediction_1,
                prediction_2,
                labels=["hs", "nhs"],
            )
        )

        rows.append(
            {
                "model_1": model_1,
                "model_2": model_2,
                "raw_agreement": raw_agreement,
                "cohen_kappa": kappa,
                "n_agree": n_agree,
                "n_disagree": n_disagree,
                "n_instances": len(prediction_1),
            }
        )

    dataframe = pd.DataFrame(rows)

    dataframe = dataframe.sort_values(
        by=["raw_agreement", "cohen_kappa"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return dataframe


def compute_agreement_matrix(
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    matrix = pd.DataFrame(
        index=MODEL_ORDER,
        columns=MODEL_ORDER,
        dtype=float,
    )

    for model_1 in MODEL_ORDER:
        for model_2 in MODEL_ORDER:
            matrix.loc[model_1, model_2] = float(
                np.mean(
                    predictions[model_1]
                    ==
                    predictions[model_2]
                )
            )

    matrix.index.name = "model"

    return matrix


# =============================================================================
# INSTANCE-LEVEL MODEL-RELATIVE ANALYSIS
# =============================================================================

def assign_difficulty_group(n_model_errors: int) -> str:
    if n_model_errors <= 1:
        return "easy"

    if n_model_errors <= 3:
        return "moderate"

    return "hard"


def compute_instance_difficulty(
    canonical_test: pd.DataFrame,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    dataframe = canonical_test.copy()

    prediction_matrix = np.column_stack(
        [predictions[model] for model in MODEL_ORDER]
    )

    gold = dataframe["gold"].to_numpy()

    correctness_matrix = (
        prediction_matrix
        ==
        gold[:, np.newaxis]
    )

    for index, model in enumerate(MODEL_ORDER):
        dataframe[f"{model}_prediction"] = prediction_matrix[:, index]
        dataframe[f"{model}_correct"] = correctness_matrix[:, index]

    dataframe["n_models_correct"] = correctness_matrix.sum(axis=1)
    dataframe["n_model_errors"] = len(MODEL_ORDER) - dataframe["n_models_correct"]
    dataframe["model_error_rate"] = (
        dataframe["n_model_errors"] / len(MODEL_ORDER)
    )

    dataframe["difficulty_group"] = (
        dataframe["n_model_errors"]
        .map(assign_difficulty_group)
    )

    return dataframe


def summarize_difficulty(
    instance_difficulty: pd.DataFrame,
) -> pd.DataFrame:
    group_order = ["easy", "moderate", "hard"]

    counts = (
        instance_difficulty["difficulty_group"]
        .value_counts()
        .reindex(group_order, fill_value=0)
    )

    dataframe = pd.DataFrame(
        {
            "difficulty_group": group_order,
            "n_instances": counts.values,
        }
    )

    dataframe["percentage"] = (
        dataframe["n_instances"]
        / len(instance_difficulty)
        * 100.0
    )

    return dataframe


def compute_difficulty_by_domain(
    instance_difficulty: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for domain, group in instance_difficulty.groupby(
        "domain",
        sort=False,
    ):
        n_instances = len(group)

        n_easy = int((group["difficulty_group"] == "easy").sum())
        n_moderate = int(
            (group["difficulty_group"] == "moderate").sum()
        )
        n_hard = int((group["difficulty_group"] == "hard").sum())

        rows.append(
            {
                "domain": domain,
                "n_instances": n_instances,
                "n_easy": n_easy,
                "pct_easy": n_easy / n_instances,
                "n_moderate": n_moderate,
                "pct_moderate": n_moderate / n_instances,
                "n_hard": n_hard,
                "pct_hard": n_hard / n_instances,
                "mean_models_correct": group[
                    "n_models_correct"
                ].mean(),
                "mean_model_errors": group[
                    "n_model_errors"
                ].mean(),
                "mean_model_error_rate": group[
                    "model_error_rate"
                ].mean(),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# UNANIMOUS ERRORS AND QUALITATIVE TEMPLATE
# =============================================================================

def extract_unanimous_errors(
    instance_difficulty: pd.DataFrame,
) -> pd.DataFrame:
    unanimous_errors = instance_difficulty.loc[
        instance_difficulty["n_model_errors"] == len(MODEL_ORDER)
    ].copy()

    unanimous_errors = unanimous_errors.reset_index(drop=True)

    return unanimous_errors


def create_qualitative_template(
    unanimous_errors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create an empty manual coding template.

    The coding fields are intentionally left blank. They must be completed
    through manual qualitative examination of the unanimously misclassified
    instances.
    """
    template = unanimous_errors[
        [
            "id",
            "text",
            "gold",
            "domain",
            "n_models_correct",
            "n_model_errors",
            "model_error_rate",
            "difficulty_group",
        ]
    ].copy()

    template["primary_error_phenomenon"] = ""
    template["secondary_error_phenomenon"] = ""
    template["implicit_supportive_intent"] = ""
    template["mixed_communicative_stance"] = ""
    template["informational_or_context_dependent_discourse"] = ""
    template["irony_or_sarcasm"] = ""
    template["sociocultural_reference"] = ""
    template["annotation_ambiguity"] = ""
    template["qualitative_notes"] = ""

    return template


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    repository_root, data_dir, results_dir = resolve_repository_paths()

    # -------------------------------------------------------------------------
    # Load and validate canonical benchmark files
    # -------------------------------------------------------------------------

    print_section("LOADING CANONICAL TEST DATA")

    test_input_path = data_dir / TEST_FILE
    test_gold_path = data_dir / TEST_GOLD_FILE

    test_input = load_test_input(test_input_path)
    canonical_test = load_test_gold(test_gold_path)

    print(
        f"test.csv: {len(test_input)} instances loaded "
        f"from {test_input_path.relative_to(repository_root)}"
    )

    print(
        f"test_gold.csv: {len(canonical_test)} instances loaded "
        f"from {test_gold_path.relative_to(repository_root)}"
    )

    # -------------------------------------------------------------------------
    # Validate test.csv against test_gold.csv
    # -------------------------------------------------------------------------

    print_section("VALIDATING CANONICAL TEST FILES")

    validate_test_input_against_gold(
        test_input=test_input,
        test_gold=canonical_test,
    )

    print("test.csv and test_gold.csv contain identical instance IDs.")
    print("Original texts are identical after alignment by instance ID.")
    print("Canonical benchmark order taken from data/test_gold.csv.")

    # -------------------------------------------------------------------------
    # Resolve prediction files
    # -------------------------------------------------------------------------

    print_section("RESOLVING PREDICTION FILES")

    prediction_paths = {}

    for model_name, filename in MODEL_FILES.items():
        path = results_dir / filename

        if not path.is_file():
            raise FileNotFoundError(
                f"Prediction file for {model_name} not found: {path}"
            )

        prediction_paths[model_name] = path

        print(
            f"{model_name}: "
            f"{path.relative_to(repository_root)}"
        )

    # -------------------------------------------------------------------------
    # Load and validate predictions
    # -------------------------------------------------------------------------

    print_section("LOADING AND ALIGNING PREDICTIONS")

    prediction_frames = {}
    predictions = {}

    for model_name in MODEL_ORDER:
        dataframe = load_prediction_file(
            path=prediction_paths[model_name],
            model_name=model_name,
            canonical_test=canonical_test,
        )

        prediction_frames[model_name] = dataframe
        predictions[model_name] = dataframe["prediction"].to_numpy()

        print(
            f"{model_name}: "
            f"{len(dataframe)} predictions loaded, aligned, and validated"
        )

    # -------------------------------------------------------------------------
    # Common-instance validation summary
    # -------------------------------------------------------------------------

    print_section("VALIDATING COMMON TEST INSTANCES")

    print(
        f"All models contain {EXPECTED_N_INSTANCES} predictions."
    )
    print(
        "All prediction files contain exactly the canonical test instance IDs."
    )
    print(
        "All predictions were aligned by stable instance ID."
    )
    print(
        "Gold labels are identical across canonical data and all models."
    )
    print(
        "Domain labels are identical across canonical data and all models."
    )
    print(
        "Original texts were obtained from the canonical benchmark files."
    )

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    print_section("ANALYSIS CONFIGURATION")

    n_pairs = len(list(combinations(MODEL_ORDER, 2)))

    print(f"Number of models: {len(MODEL_ORDER)}")
    print(f"Number of test instances: {len(canonical_test)}")
    print(f"Number of pairwise model comparisons: {n_pairs}")
    print("Pairwise measures: raw agreement and Cohen's kappa")
    print(
        "Instance-difficulty interpretation: "
        "model-relative empirical grouping"
    )
    print(
        "Easy: 0-1 model errors; "
        "Moderate: 2-3 model errors; "
        "Hard: 4-6 model errors"
    )

    # -------------------------------------------------------------------------
    # Pairwise agreement
    # -------------------------------------------------------------------------

    print_section("COMPUTING PAIRWISE MODEL AGREEMENT")

    agreement = compute_pairwise_agreement(predictions)

    print(agreement.to_string(index=False))

    # -------------------------------------------------------------------------
    # Agreement matrix
    # -------------------------------------------------------------------------

    print_section("RAW AGREEMENT MATRIX")

    agreement_matrix = compute_agreement_matrix(predictions)

    print(agreement_matrix.to_string())

    # -------------------------------------------------------------------------
    # Instance-level model-relative difficulty
    # -------------------------------------------------------------------------

    print_section("COMPUTING MODEL-RELATIVE INSTANCE DIFFICULTY")

    instance_difficulty = compute_instance_difficulty(
        canonical_test=canonical_test,
        predictions=predictions,
    )

    difficulty_summary = summarize_difficulty(instance_difficulty)

    print(difficulty_summary.to_string(index=False))

    # -------------------------------------------------------------------------
    # Difficulty by domain
    # -------------------------------------------------------------------------

    print_section("COMPUTING MODEL-RELATIVE DIFFICULTY BY DOMAIN")

    difficulty_by_domain = compute_difficulty_by_domain(
        instance_difficulty
    )

    print(difficulty_by_domain.to_string(index=False))

    # -------------------------------------------------------------------------
    # Unanimous errors
    # -------------------------------------------------------------------------

    print_section("EXTRACTING UNANIMOUS ERRORS")

    unanimous_errors = extract_unanimous_errors(instance_difficulty)

    print(
        "Number of instances misclassified by all "
        f"{len(MODEL_ORDER)} model-level prediction vectors: "
        f"{len(unanimous_errors)}"
    )

    if len(unanimous_errors) > 0:
        unanimous_by_domain = (
            unanimous_errors["domain"]
            .value_counts()
            .rename_axis("domain")
            .reset_index(name="n_unanimous_errors")
        )

        print()
        print(unanimous_by_domain.to_string(index=False))

    # -------------------------------------------------------------------------
    # Qualitative coding template
    # -------------------------------------------------------------------------

    print_section("CREATING QUALITATIVE CODING TEMPLATE")

    qualitative_template = create_qualitative_template(
        unanimous_errors
    )

    print(
        f"Qualitative coding template created with "
        f"{len(qualitative_template)} instances."
    )
    print(
        "Original benchmark texts are included in the coding template."
    )
    print(
        "The coding fields are intentionally empty and must be completed "
        "through manual qualitative analysis."
    )

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------

    agreement_output = results_dir / PAIRWISE_AGREEMENT_OUTPUT
    agreement_matrix_output = results_dir / AGREEMENT_MATRIX_OUTPUT
    instance_difficulty_output = results_dir / INSTANCE_DIFFICULTY_OUTPUT
    difficulty_by_domain_output = (
        results_dir / DIFFICULTY_BY_DOMAIN_OUTPUT
    )
    unanimous_errors_output = results_dir / UNANIMOUS_ERRORS_OUTPUT
    qualitative_template_output = (
        results_dir / QUALITATIVE_TEMPLATE_OUTPUT
    )

    agreement.to_csv(agreement_output, index=False)
    agreement_matrix.to_csv(agreement_matrix_output)
    instance_difficulty.to_csv(instance_difficulty_output, index=False)
    difficulty_by_domain.to_csv(
        difficulty_by_domain_output,
        index=False,
    )
    unanimous_errors.to_csv(
        unanimous_errors_output,
        index=False,
    )
    qualitative_template.to_csv(
        qualitative_template_output,
        index=False,
    )

    print_section("SAVED RESULTS")

    for path in [
        agreement_output,
        agreement_matrix_output,
        instance_difficulty_output,
        difficulty_by_domain_output,
        unanimous_errors_output,
        qualitative_template_output,
    ]:
        print(path.relative_to(repository_root))

    # -------------------------------------------------------------------------
    # Methodological reminder
    # -------------------------------------------------------------------------

    print_section("METHODOLOGICAL REMINDER")

    print(
        "The easy/moderate/hard categories are model-relative empirical "
        "groupings and must not be described as intrinsic instance difficulty."
    )

    print(
        "The qualitative coding template is an input to manual qualitative "
        "analysis. Generating the template does not by itself constitute the "
        "qualitative error analysis."
    )

    print_section("Done.")


if __name__ == "__main__":
    main()
    