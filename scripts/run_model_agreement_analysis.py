#!/usr/bin/env python3

"""
run_model_agreement_analysis.py

Model-agreement, model-relative instance-difficulty, unanimous-error
extraction, and qualitative-error coding-template generation for the
SpanishHopeMultidomain benchmark.

The script performs the following analyses:

1. Resolves the fixed prediction files used by the benchmark analyses.
2. Loads one model-level prediction vector for each evaluated system:
   - SVM
   - BETO majority-vote predictions
   - RoBERTuito majority-vote predictions
   - Qwen2.5-7B predictions
   - GPT-4o-mini predictions
   - GPT-4.1-mini predictions
3. Validates that all models refer to the same 400 test instances in
   identical order, with identical gold labels and domain assignments.
4. Computes pairwise raw prediction agreement and Cohen's kappa.
5. Writes:
   - model_agreement.csv
   - model_agreement_matrix.csv
6. Computes, for every test instance:
   - number of correct model predictions
   - number of model errors
   - proportion of model errors
   - model-relative instance-difficulty group
7. Writes:
   - instance_difficulty.csv
   - instance_difficulty_by_domain.csv
8. Extracts instances misclassified by all six model-level prediction
   vectors.
9. Writes:
   - unanimous_error_instances.csv
10. Creates a reproducible qualitative-error coding template containing
    all unanimous errors and empty coding fields for two independent
    coders and final consensus adjudication.
11. Writes:
    - qualitative_error_coding_template.csv

Important methodological note
-----------------------------
The easy/moderate/hard categories produced by this script are
MODEL-RELATIVE empirical groupings based on the number of errors among
the six evaluated model-level prediction vectors. They must not be
interpreted as intrinsic properties of the benchmark instances.

The qualitative coding template is an input to a subsequent manual
qualitative analysis. This script does not assign linguistic or pragmatic
error categories automatically and does not claim to perform the manual
qualitative analysis itself.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


# =============================================================================
# CONFIGURATION
# =============================================================================

EXPECTED_N_INSTANCES = 400

MODEL_ORDER = [
    "SVM",
    "BETO",
    "RoBERTuito",
    "Qwen2.5-7B",
    "GPT-4o-mini",
    "GPT-4.1-mini",
]

PREDICTION_FILES = {
    "SVM": "results/svm_predictions.csv",
    "BETO": "results/beto_majority_predictions.csv",
    "RoBERTuito": "results/robertuito_majority_predictions.csv",
    "Qwen2.5-7B": "results/qwen25_7b_predictions.csv",
    "GPT-4o-mini": "results/gpt4o_mini_predictions.csv",
    "GPT-4.1-mini": "results/gpt41_mini_predictions.csv",
}

OUTPUT_MODEL_AGREEMENT = "results/model_agreement.csv"
OUTPUT_MODEL_AGREEMENT_MATRIX = "results/model_agreement_matrix.csv"
OUTPUT_INSTANCE_DIFFICULTY = "results/instance_difficulty.csv"
OUTPUT_INSTANCE_DIFFICULTY_BY_DOMAIN = (
    "results/instance_difficulty_by_domain.csv"
)
OUTPUT_UNANIMOUS_ERRORS = "results/unanimous_error_instances.csv"
OUTPUT_QUALITATIVE_TEMPLATE = (
    "results/qualitative_error_coding_template.csv"
)


# Candidate column names supported by the loader.

ID_COLUMN_CANDIDATES = [
    "id",
    "instance_id",
    "post_id",
    "index",
    "idx",
]

TEXT_COLUMN_CANDIDATES = [
    "text",
    "tweet",
    "post",
    "sentence",
    "content",
]

DOMAIN_COLUMN_CANDIDATES = [
    "domain",
    "topic",
    "category",
]

GOLD_COLUMN_CANDIDATES = [
    "gold",
    "gold_label",
    "label",
    "true_label",
    "y_true",
    "target",
]

PREDICTION_COLUMN_CANDIDATES = [
    "prediction",
    "predicted_label",
    "pred",
    "y_pred",
    "output",
]


# =============================================================================
# GENERAL UTILITIES
# =============================================================================


def print_section(title: str) -> None:
    print()
    print("=" * 40)
    print(title)
    print("=" * 40)


def find_repository_root() -> Path:
    """
    Locate the repository root.

    Expected repository structure:

        repository/
            scripts/
                run_model_agreement_analysis.py
            results/
                ...

    The function first uses the script location and then falls back to
    the current working directory.
    """

    script_path = Path(__file__).resolve()
    candidate = script_path.parent.parent

    if (candidate / "results").exists():
        return candidate

    cwd = Path.cwd().resolve()

    if (cwd / "results").exists():
        return cwd

    raise FileNotFoundError(
        "Could not locate the repository root. "
        "Expected a repository containing a 'results' directory."
    )


def normalize_column_name(column: str) -> str:
    return str(column).strip().lower()


def find_column(
    dataframe: pd.DataFrame,
    candidates: Sequence[str],
) -> Optional[str]:
    """
    Find the first dataframe column matching one of the candidate names.
    Matching is case-insensitive.
    """

    normalized_to_original = {
        normalize_column_name(column): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        normalized_candidate = normalize_column_name(candidate)

        if normalized_candidate in normalized_to_original:
            return normalized_to_original[normalized_candidate]

    return None


def require_column(
    dataframe: pd.DataFrame,
    candidates: Sequence[str],
    description: str,
    file_path: Path,
) -> str:
    column = find_column(dataframe, candidates)

    if column is None:
        raise ValueError(
            f"Could not identify the {description} column in "
            f"{file_path}.\n"
            f"Available columns: {list(dataframe.columns)}"
        )

    return column


def normalize_label(value) -> str:
    """
    Normalize common binary label representations to HS/NHS.

    The function intentionally fails on unknown values instead of silently
    guessing labels.
    """

    if pd.isna(value):
        raise ValueError("Encountered a missing label.")

    raw = str(value).strip()
    normalized = raw.lower()

    compact = (
        normalized
        .replace("_", " ")
        .replace("-", " ")
        .replace('"', "")
        .replace("'", "")
    )

    compact = " ".join(compact.split())

    hs_values = {
        "hs",
        "hope speech",
        "hope",
        "1",
        "true",
    }

    nhs_values = {
        "nhs",
        "non hope speech",
        "nonhope speech",
        "non hope",
        "0",
        "false",
    }

    if compact in hs_values:
        return "HS"

    if compact in nhs_values:
        return "NHS"

    raise ValueError(
        f"Unsupported label value: {raw!r}. "
        "Expected a recognizable HS/NHS label."
    )


def normalize_domain(value) -> str:
    """
    Normalize benchmark domain names while preserving unknown non-empty
    domain labels.
    """

    if pd.isna(value):
        raise ValueError("Encountered a missing domain value.")

    raw = str(value).strip()
    normalized = raw.lower()

    mapping = {
        "lgbt": "LGBT",
        "lgbtq": "LGBT",
        "lgbtq+": "LGBT",
        "obesity": "Obesity",
        "obesidad": "Obesity",
        "racism": "Racism",
        "racismo": "Racism",
    }

    return mapping.get(normalized, raw)


def resolve_prediction_files(
    repository_root: Path,
) -> Dict[str, Path]:
    resolved = {}

    for model in MODEL_ORDER:
        relative_path = PREDICTION_FILES[model]
        absolute_path = repository_root / relative_path

        if not absolute_path.exists():
            raise FileNotFoundError(
                f"Prediction file for {model} was not found:\n"
                f"  {absolute_path}"
            )

        resolved[model] = absolute_path

    return resolved


# =============================================================================
# PREDICTION LOADING
# =============================================================================


def load_prediction_file(
    model_name: str,
    file_path: Path,
) -> pd.DataFrame:
    """
    Load and standardize one prediction file.

    Returned columns:

        instance_id
        text               (if available)
        domain             (if available)
        gold
        prediction
    """

    dataframe = pd.read_csv(file_path)

    if len(dataframe) != EXPECTED_N_INSTANCES:
        raise ValueError(
            f"{model_name}: expected {EXPECTED_N_INSTANCES} rows, "
            f"found {len(dataframe)} in {file_path}."
        )

    gold_column = require_column(
        dataframe,
        GOLD_COLUMN_CANDIDATES,
        "gold-label",
        file_path,
    )

    prediction_column = require_column(
        dataframe,
        PREDICTION_COLUMN_CANDIDATES,
        "prediction",
        file_path,
    )

    id_column = find_column(
        dataframe,
        ID_COLUMN_CANDIDATES,
    )

    text_column = find_column(
        dataframe,
        TEXT_COLUMN_CANDIDATES,
    )

    domain_column = find_column(
        dataframe,
        DOMAIN_COLUMN_CANDIDATES,
    )

    standardized = pd.DataFrame()

    if id_column is not None:
        standardized["instance_id"] = dataframe[id_column]
    else:
        standardized["instance_id"] = np.arange(len(dataframe))

    if text_column is not None:
        standardized["text"] = dataframe[text_column].fillna("").astype(str)

    if domain_column is not None:
        standardized["domain"] = dataframe[domain_column].map(
            normalize_domain
        )

    standardized["gold"] = dataframe[gold_column].map(normalize_label)

    standardized["prediction"] = dataframe[prediction_column].map(
        normalize_label
    )

    if standardized["instance_id"].duplicated().any():
        duplicated = standardized.loc[
            standardized["instance_id"].duplicated(keep=False),
            "instance_id",
        ].tolist()

        raise ValueError(
            f"{model_name}: duplicate instance IDs found: {duplicated[:10]}"
        )

    return standardized


def validate_common_test_instances(
    predictions: Dict[str, pd.DataFrame],
) -> None:
    """
    Validate identical instance ordering, gold labels, domain assignments,
    and texts whenever those fields are available.
    """

    reference_model = MODEL_ORDER[0]
    reference = predictions[reference_model]

    for model in MODEL_ORDER[1:]:
        current = predictions[model]

        if len(current) != len(reference):
            raise ValueError(
                f"{model}: number of predictions differs from "
                f"{reference_model}."
            )

        if not np.array_equal(
            reference["instance_id"].to_numpy(),
            current["instance_id"].to_numpy(),
        ):
            raise ValueError(
                f"{model}: instance ordering differs from "
                f"{reference_model}."
            )

        if not np.array_equal(
            reference["gold"].to_numpy(),
            current["gold"].to_numpy(),
        ):
            raise ValueError(
                f"{model}: gold-label ordering differs from "
                f"{reference_model}."
            )

        if "domain" in reference.columns and "domain" in current.columns:
            if not np.array_equal(
                reference["domain"].to_numpy(),
                current["domain"].to_numpy(),
            ):
                raise ValueError(
                    f"{model}: domain ordering differs from "
                    f"{reference_model}."
                )

        if "text" in reference.columns and "text" in current.columns:
            if not np.array_equal(
                reference["text"].to_numpy(),
                current["text"].to_numpy(),
            ):
                raise ValueError(
                    f"{model}: text ordering differs from "
                    f"{reference_model}."
                )


def select_metadata_source(
    predictions: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Select the richest prediction dataframe as metadata source.

    Preference is given to a dataframe containing both text and domain.
    """

    for model in MODEL_ORDER:
        dataframe = predictions[model]

        if "text" in dataframe.columns and "domain" in dataframe.columns:
            return dataframe.copy()

    for model in MODEL_ORDER:
        dataframe = predictions[model]

        if "domain" in dataframe.columns:
            return dataframe.copy()

    return predictions[MODEL_ORDER[0]].copy()


# =============================================================================
# PAIRWISE MODEL AGREEMENT
# =============================================================================


def compute_pairwise_model_agreement(
    predictions: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    for model_1, model_2 in combinations(MODEL_ORDER, 2):
        prediction_1 = predictions[model_1]["prediction"].to_numpy()
        prediction_2 = predictions[model_2]["prediction"].to_numpy()

        raw_agreement = float(
            np.mean(prediction_1 == prediction_2)
        )

        kappa = float(
            cohen_kappa_score(prediction_1, prediction_2)
        )

        n_agree = int(
            np.sum(prediction_1 == prediction_2)
        )

        n_disagree = int(
            np.sum(prediction_1 != prediction_2)
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

    result = pd.DataFrame(rows)

    return result.sort_values(
        by=["raw_agreement", "cohen_kappa"],
        ascending=[False, False],
    ).reset_index(drop=True)


def compute_agreement_matrix(
    predictions: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    matrix = pd.DataFrame(
        index=MODEL_ORDER,
        columns=MODEL_ORDER,
        dtype=float,
    )

    for model_1 in MODEL_ORDER:
        prediction_1 = predictions[model_1]["prediction"].to_numpy()

        for model_2 in MODEL_ORDER:
            prediction_2 = predictions[model_2]["prediction"].to_numpy()

            matrix.loc[model_1, model_2] = np.mean(
                prediction_1 == prediction_2
            )

    matrix.index.name = "model"

    return matrix


# =============================================================================
# MODEL-RELATIVE INSTANCE DIFFICULTY
# =============================================================================


def assign_model_relative_difficulty(
    n_model_errors: int,
) -> str:
    """
    Empirical grouping relative to the six evaluated model-level
    prediction vectors.

    Easy:
        0-1 model errors

    Moderate:
        2-3 model errors

    Hard:
        4-6 model errors
    """

    if n_model_errors <= 1:
        return "easy"

    if n_model_errors <= 3:
        return "moderate"

    return "hard"


def compute_instance_difficulty(
    predictions: Dict[str, pd.DataFrame],
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    gold = predictions[MODEL_ORDER[0]]["gold"].to_numpy()

    result = pd.DataFrame(
        {
            "instance_id": metadata["instance_id"],
        }
    )

    if "text" in metadata.columns:
        result["text"] = metadata["text"]

    if "domain" in metadata.columns:
        result["domain"] = metadata["domain"]

    result["gold"] = gold

    correctness_columns = []

    for model in MODEL_ORDER:
        prediction = predictions[model]["prediction"].to_numpy()

        prediction_column = f"{model}_prediction"
        correctness_column = f"{model}_correct"

        result[prediction_column] = prediction

        result[correctness_column] = (
            prediction == gold
        ).astype(int)

        correctness_columns.append(correctness_column)

    result["n_models_correct"] = result[
        correctness_columns
    ].sum(axis=1)

    result["n_model_errors"] = (
        len(MODEL_ORDER) - result["n_models_correct"]
    )

    result["model_error_rate"] = (
        result["n_model_errors"] / len(MODEL_ORDER)
    )

    result["model_relative_difficulty"] = result[
        "n_model_errors"
    ].map(assign_model_relative_difficulty)

    return result


def compute_instance_difficulty_by_domain(
    instance_difficulty: pd.DataFrame,
) -> pd.DataFrame:
    if "domain" not in instance_difficulty.columns:
        raise ValueError(
            "Domain information is required to compute "
            "instance_difficulty_by_domain.csv."
        )

    domain_order = [
        domain
        for domain in ["LGBT", "Obesity", "Racism"]
        if domain in set(instance_difficulty["domain"])
    ]

    additional_domains = sorted(
        set(instance_difficulty["domain"]) - set(domain_order)
    )

    domain_order.extend(additional_domains)

    rows = []

    for domain in domain_order:
        subset = instance_difficulty[
            instance_difficulty["domain"] == domain
        ]

        n_instances = len(subset)

        counts = subset[
            "model_relative_difficulty"
        ].value_counts()

        n_easy = int(counts.get("easy", 0))
        n_moderate = int(counts.get("moderate", 0))
        n_hard = int(counts.get("hard", 0))

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
                "mean_models_correct": subset[
                    "n_models_correct"
                ].mean(),
                "mean_model_errors": subset[
                    "n_model_errors"
                ].mean(),
                "mean_model_error_rate": subset[
                    "model_error_rate"
                ].mean(),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# UNANIMOUS ERRORS
# =============================================================================


def extract_unanimous_errors(
    instance_difficulty: pd.DataFrame,
) -> pd.DataFrame:
    unanimous = instance_difficulty[
        instance_difficulty["n_models_correct"] == 0
    ].copy()

    return unanimous.reset_index(drop=True)


# =============================================================================
# QUALITATIVE ERROR CODING TEMPLATE
# =============================================================================


def create_qualitative_coding_template(
    unanimous_errors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a coding template containing every unanimously misclassified
    instance.

    The fields are intentionally empty. They must be completed through
    manual qualitative analysis.

    Recommended workflow:

    1. Two authors independently inspect all unanimous errors.
    2. Each coder assigns:
       - one primary phenomenon;
       - zero or more secondary phenomena;
       - a short analytical rationale.
    3. The coders compare their annotations.
    4. Disagreements are resolved through discussion.
    5. Final consensus fields are completed.
    6. The completed coding artifact is released with the repository.

    The script does not compute inter-coder agreement because an
    inductively developed qualitative coding scheme over a small sample
    should not automatically be treated as a conventional fixed-category
    annotation task.
    """

    template = pd.DataFrame()

    base_columns = [
        "instance_id",
        "text",
        "domain",
        "gold",
    ]

    for column in base_columns:
        if column in unanimous_errors.columns:
            template[column] = unanimous_errors[column]

    for model in MODEL_ORDER:
        prediction_column = f"{model}_prediction"

        if prediction_column in unanimous_errors.columns:
            template[prediction_column] = unanimous_errors[
                prediction_column
            ]

    template["coder_1_primary_phenomenon"] = ""
    template["coder_1_secondary_phenomena"] = ""
    template["coder_1_analytical_rationale"] = ""

    template["coder_2_primary_phenomenon"] = ""
    template["coder_2_secondary_phenomena"] = ""
    template["coder_2_analytical_rationale"] = ""

    template["primary_category_agreement"] = ""

    template["final_primary_phenomenon"] = ""
    template["final_secondary_phenomena"] = ""
    template["final_analytical_rationale"] = ""

    template["adjudication_notes"] = ""

    return template


# =============================================================================
# OUTPUT
# =============================================================================


def save_results(
    repository_root: Path,
    model_agreement: pd.DataFrame,
    agreement_matrix: pd.DataFrame,
    instance_difficulty: pd.DataFrame,
    difficulty_by_domain: pd.DataFrame,
    unanimous_errors: pd.DataFrame,
    qualitative_template: pd.DataFrame,
) -> None:
    output_paths = [
        repository_root / OUTPUT_MODEL_AGREEMENT,
        repository_root / OUTPUT_MODEL_AGREEMENT_MATRIX,
        repository_root / OUTPUT_INSTANCE_DIFFICULTY,
        repository_root / OUTPUT_INSTANCE_DIFFICULTY_BY_DOMAIN,
        repository_root / OUTPUT_UNANIMOUS_ERRORS,
        repository_root / OUTPUT_QUALITATIVE_TEMPLATE,
    ]

    for output_path in output_paths:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    model_agreement.to_csv(
        repository_root / OUTPUT_MODEL_AGREEMENT,
        index=False,
    )

    agreement_matrix.to_csv(
        repository_root / OUTPUT_MODEL_AGREEMENT_MATRIX,
        index=True,
    )

    instance_difficulty.to_csv(
        repository_root / OUTPUT_INSTANCE_DIFFICULTY,
        index=False,
    )

    difficulty_by_domain.to_csv(
        repository_root / OUTPUT_INSTANCE_DIFFICULTY_BY_DOMAIN,
        index=False,
    )

    unanimous_errors.to_csv(
        repository_root / OUTPUT_UNANIMOUS_ERRORS,
        index=False,
    )

    qualitative_template.to_csv(
        repository_root / OUTPUT_QUALITATIVE_TEMPLATE,
        index=False,
    )


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    repository_root = find_repository_root()

    print_section("RESOLVING PREDICTION FILES")

    prediction_files = resolve_prediction_files(repository_root)

    for model in MODEL_ORDER:
        relative_path = prediction_files[model].relative_to(
            repository_root
        )
        print(f"{model}: {relative_path}")

    print_section("LOADING PREDICTIONS")

    predictions: Dict[str, pd.DataFrame] = {}

    for model in MODEL_ORDER:
        predictions[model] = load_prediction_file(
            model,
            prediction_files[model],
        )

        print(
            f"{model}: "
            f"{len(predictions[model])} predictions loaded"
        )

    print_section("VALIDATING COMMON TEST INSTANCES")

    validate_common_test_instances(predictions)

    print(
        f"All models contain {EXPECTED_N_INSTANCES} predictions."
    )
    print("Instance ordering is identical across models.")
    print("Gold-label ordering is identical across models.")

    reference = predictions[MODEL_ORDER[0]]

    if "domain" in reference.columns:
        print("Domain ordering is identical across models.")

    if "text" in reference.columns:
        print("Text ordering is identical across models.")

    metadata = select_metadata_source(predictions)

    print_section("ANALYSIS CONFIGURATION")

    print(f"Number of models: {len(MODEL_ORDER)}")
    print(f"Number of test instances: {EXPECTED_N_INSTANCES}")
    print(
        "Number of pairwise model comparisons: "
        f"{len(list(combinations(MODEL_ORDER, 2)))}"
    )
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

    print_section("COMPUTING PAIRWISE MODEL AGREEMENT")

    model_agreement = compute_pairwise_model_agreement(
        predictions
    )

    agreement_matrix = compute_agreement_matrix(
        predictions
    )

    print(model_agreement.to_string(index=False))

    print_section("RAW AGREEMENT MATRIX")

    print(agreement_matrix.to_string())

    print_section("COMPUTING MODEL-RELATIVE INSTANCE DIFFICULTY")

    instance_difficulty = compute_instance_difficulty(
        predictions,
        metadata,
    )

    difficulty_counts = (
        instance_difficulty["model_relative_difficulty"]
        .value_counts()
        .reindex(["easy", "moderate", "hard"], fill_value=0)
    )

    difficulty_percentages = (
        difficulty_counts / len(instance_difficulty) * 100
    )

    difficulty_summary = pd.DataFrame(
        {
            "difficulty_group": difficulty_counts.index,
            "n_instances": difficulty_counts.values,
            "percentage": difficulty_percentages.values,
        }
    )

    print(difficulty_summary.to_string(index=False))

    print_section("COMPUTING MODEL-RELATIVE DIFFICULTY BY DOMAIN")

    difficulty_by_domain = compute_instance_difficulty_by_domain(
        instance_difficulty
    )

    print(difficulty_by_domain.to_string(index=False))

    print_section("EXTRACTING UNANIMOUS ERRORS")

    unanimous_errors = extract_unanimous_errors(
        instance_difficulty
    )

    print(
        "Number of instances misclassified by all "
        f"{len(MODEL_ORDER)} model-level prediction vectors: "
        f"{len(unanimous_errors)}"
    )

    if "domain" in unanimous_errors.columns:
        unanimous_by_domain = (
            unanimous_errors["domain"]
            .value_counts()
            .rename_axis("domain")
            .reset_index(name="n_unanimous_errors")
        )

        print()
        print(unanimous_by_domain.to_string(index=False))

    print_section("CREATING QUALITATIVE CODING TEMPLATE")

    qualitative_template = create_qualitative_coding_template(
        unanimous_errors
    )

    print(
        f"Qualitative coding template created with "
        f"{len(qualitative_template)} instances."
    )
    print(
        "The coding fields are intentionally empty and must be "
        "completed through manual qualitative analysis."
    )

    save_results(
        repository_root=repository_root,
        model_agreement=model_agreement,
        agreement_matrix=agreement_matrix,
        instance_difficulty=instance_difficulty,
        difficulty_by_domain=difficulty_by_domain,
        unanimous_errors=unanimous_errors,
        qualitative_template=qualitative_template,
    )

    print_section("SAVED RESULTS")

    print(OUTPUT_MODEL_AGREEMENT)
    print(OUTPUT_MODEL_AGREEMENT_MATRIX)
    print(OUTPUT_INSTANCE_DIFFICULTY)
    print(OUTPUT_INSTANCE_DIFFICULTY_BY_DOMAIN)
    print(OUTPUT_UNANIMOUS_ERRORS)
    print(OUTPUT_QUALITATIVE_TEMPLATE)

    print_section("METHODOLOGICAL REMINDER")

    print(
        "The easy/moderate/hard categories are model-relative "
        "empirical groupings and must not be described as intrinsic "
        "instance difficulty."
    )
    print(
        "The qualitative coding template is an input to manual "
        "qualitative analysis. Generating the template does not by "
        "itself constitute the qualitative error analysis."
    )

    print_section("Done.")


if __name__ == "__main__":
    main()
    