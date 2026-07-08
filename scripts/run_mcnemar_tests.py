#!/usr/bin/env python3
"""
run_mcnemar_tests.py

Pairwise exact McNemar tests for the SpanishHopeMultidomain benchmark.

This script:

1. Loads the fixed prediction files for the six evaluated models.
2. Validates that every model contains exactly 400 predictions.
3. Validates that gold-label ordering is identical across models.
4. Computes the paired correctness contingency table for every model pair.
5. Performs a two-sided exact McNemar test using the binomial distribution.
6. Applies Holm's family-wise error-rate correction across all pairwise tests.
7. Saves complete results to:
       results/mcnemar_pairwise_tests.csv

Important methodological note
-----------------------------
McNemar's test evaluates whether two classifiers differ in their paired
correctness/error patterns on the same test instances. It does NOT test
differences in Macro-F1. Pairwise Macro-F1 differences are analyzed
separately using the shared paired bootstrap procedure implemented in
run_bootstrap_confidence_intervals.py.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest


# =============================================================================
# CONFIGURATION
# =============================================================================

EXPECTED_N_INSTANCES = 400
ALPHA = 0.05

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = REPO_ROOT / "results"

OUTPUT_FILE = RESULTS_DIR / "mcnemar_pairwise_tests.csv"


# Exact prediction files successfully used by the bootstrap script.
MODEL_FILES = {
    "SVM": RESULTS_DIR / "svm_predictions.csv",
    "BETO": RESULTS_DIR / "beto_majority_predictions.csv",
    "RoBERTuito": RESULTS_DIR / "robertuito_majority_predictions.csv",
    "Qwen2.5-7B": RESULTS_DIR / "qwen25_7b_predictions.csv",
    "GPT-4o-mini": RESULTS_DIR / "gpt4o_mini_predictions.csv",
    "GPT-4.1-mini": RESULTS_DIR / "gpt41_mini_predictions.csv",
}


# Candidate column names used to identify gold labels and predictions.
GOLD_COLUMN_CANDIDATES = [
    "gold",
    "gold_label",
    "gold_labels",
    "true_label",
    "true_labels",
    "label",
    "labels",
    "y_true",
    "target",
    "ground_truth",
]

PREDICTION_COLUMN_CANDIDATES = [
    "prediction",
    "predictions",
    "predicted_label",
    "predicted_labels",
    "pred_label",
    "pred",
    "y_pred",
    "output",
    "parsed_prediction",
    "majority_prediction",
]


# =============================================================================
# PRINTING UTILITIES
# =============================================================================

def print_header(title: str) -> None:
    """Print a section header."""

    print()
    print("=" * 40)
    print(title)
    print("=" * 40)


# =============================================================================
# COLUMN DETECTION
# =============================================================================

def normalize_column_name(column_name: str) -> str:
    """
    Normalize a column name for robust comparison.
    """

    return str(column_name).strip().lower()


def find_column(
    dataframe: pd.DataFrame,
    candidates: List[str],
    column_type: str,
    file_path: Path,
) -> str:
    """
    Find a dataframe column from a list of candidate names.

    Matching is case-insensitive after stripping whitespace.
    """

    normalized_to_original = {
        normalize_column_name(column): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        normalized_candidate = normalize_column_name(candidate)

        if normalized_candidate in normalized_to_original:
            return normalized_to_original[normalized_candidate]

    raise ValueError(
        f"Could not identify the {column_type} column in:\n"
        f"  {file_path}\n\n"
        f"Available columns:\n"
        f"  {list(dataframe.columns)}\n\n"
        f"Accepted candidate names:\n"
        f"  {candidates}"
    )


# =============================================================================
# LABEL NORMALIZATION
# =============================================================================

def normalize_label(value) -> str:
    """
    Normalize benchmark labels to canonical binary labels:

        HS
        NHS

    The function accepts common textual and numeric representations.
    """

    if pd.isna(value):
        raise ValueError("Encountered a missing label value.")

    # Handle integer values.
    if isinstance(value, (int, np.integer)):
        if int(value) == 1:
            return "HS"

        if int(value) == 0:
            return "NHS"

    # Handle floating-point values representing integers.
    if isinstance(value, (float, np.floating)):
        if float(value) == 1.0:
            return "HS"

        if float(value) == 0.0:
            return "NHS"

    text = str(value).strip().lower()

    # Remove common surrounding quotation marks.
    text = text.strip("\"'")
    text = text.strip()

    # Normalize separators and repeated whitespace.
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = " ".join(text.split())

    hope_labels = {
        "hs",
        "hope",
        "hope speech",
        "hopespeech",
        "1",
        "true",
    }

    non_hope_labels = {
        "nhs",
        "non hope",
        "non hope speech",
        "nonhopespeech",
        "0",
        "false",
    }

    if text in hope_labels:
        return "HS"

    if text in non_hope_labels:
        return "NHS"

    raise ValueError(
        f"Unrecognized label value: {value!r}"
    )


def normalize_label_series(
    series: pd.Series,
    model_name: str,
    column_name: str,
    file_path: Path,
) -> np.ndarray:
    """
    Normalize all labels in a pandas Series.
    """

    normalized_labels = []

    for row_index, value in series.items():
        try:
            normalized_labels.append(normalize_label(value))

        except ValueError as error:
            raise ValueError(
                f"Invalid label in model '{model_name}'.\n"
                f"File: {file_path}\n"
                f"Column: {column_name}\n"
                f"Row index: {row_index}\n"
                f"Value: {value!r}\n"
                f"Original error: {error}"
            ) from error

    return np.asarray(normalized_labels, dtype=object)


# =============================================================================
# PREDICTION LOADING
# =============================================================================

def load_prediction_file(
    model_name: str,
    file_path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and validate one model prediction file.

    Returns
    -------
    gold_labels:
        Canonical gold labels in test-set order.

    predictions:
        Canonical model predictions in test-set order.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Prediction file for {model_name} not found:\n"
            f"  {file_path}"
        )

    dataframe = pd.read_csv(file_path)

    if len(dataframe) != EXPECTED_N_INSTANCES:
        raise ValueError(
            f"Prediction file for {model_name} contains "
            f"{len(dataframe)} rows, but "
            f"{EXPECTED_N_INSTANCES} were expected.\n"
            f"File: {file_path}"
        )

    gold_column = find_column(
        dataframe=dataframe,
        candidates=GOLD_COLUMN_CANDIDATES,
        column_type="gold-label",
        file_path=file_path,
    )

    prediction_column = find_column(
        dataframe=dataframe,
        candidates=PREDICTION_COLUMN_CANDIDATES,
        column_type="prediction",
        file_path=file_path,
    )

    gold_labels = normalize_label_series(
        series=dataframe[gold_column],
        model_name=model_name,
        column_name=gold_column,
        file_path=file_path,
    )

    predictions = normalize_label_series(
        series=dataframe[prediction_column],
        model_name=model_name,
        column_name=prediction_column,
        file_path=file_path,
    )

    if len(gold_labels) != EXPECTED_N_INSTANCES:
        raise ValueError(
            f"{model_name}: unexpected number of gold labels."
        )

    if len(predictions) != EXPECTED_N_INSTANCES:
        raise ValueError(
            f"{model_name}: unexpected number of predictions."
        )

    return gold_labels, predictions


# =============================================================================
# COMMON TEST-ORDER VALIDATION
# =============================================================================

def validate_common_gold_order(
    gold_labels_by_model: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Verify that every model uses exactly the same gold-label ordering.

    Returns the common gold-label vector.
    """

    model_names = list(gold_labels_by_model.keys())

    reference_model = model_names[0]
    reference_gold = gold_labels_by_model[reference_model]

    for model_name in model_names[1:]:
        current_gold = gold_labels_by_model[model_name]

        if len(current_gold) != len(reference_gold):
            raise ValueError(
                f"Gold-label length mismatch between "
                f"{reference_model} and {model_name}."
            )

        if not np.array_equal(reference_gold, current_gold):
            differing_indices = np.flatnonzero(
                reference_gold != current_gold
            )

            preview = differing_indices[:20]

            raise ValueError(
                f"Gold-label ordering differs between "
                f"{reference_model} and {model_name}.\n"
                f"Number of differing positions: "
                f"{len(differing_indices)}\n"
                f"First differing indices: "
                f"{preview.tolist()}"
            )

    return reference_gold


# =============================================================================
# EXACT McNEMAR TEST
# =============================================================================

def compute_contingency_counts(
    gold_labels: np.ndarray,
    predictions_model_1: np.ndarray,
    predictions_model_2: np.ndarray,
) -> Tuple[int, int, int, int]:
    """
    Compute the paired correctness contingency table.

    Returns
    -------
    both_correct:
        Both models predict correctly.

    model_1_only_correct:
        Model 1 is correct and Model 2 is wrong.

    model_2_only_correct:
        Model 1 is wrong and Model 2 is correct.

    both_wrong:
        Both models predict incorrectly.
    """

    correct_1 = predictions_model_1 == gold_labels
    correct_2 = predictions_model_2 == gold_labels

    both_correct = int(
        np.sum(correct_1 & correct_2)
    )

    model_1_only_correct = int(
        np.sum(correct_1 & ~correct_2)
    )

    model_2_only_correct = int(
        np.sum(~correct_1 & correct_2)
    )

    both_wrong = int(
        np.sum(~correct_1 & ~correct_2)
    )

    total = (
        both_correct
        + model_1_only_correct
        + model_2_only_correct
        + both_wrong
    )

    if total != len(gold_labels):
        raise RuntimeError(
            f"Invalid contingency-table total: "
            f"{total} != {len(gold_labels)}"
        )

    return (
        both_correct,
        model_1_only_correct,
        model_2_only_correct,
        both_wrong,
    )


def exact_mcnemar_p_value(
    model_1_only_correct: int,
    model_2_only_correct: int,
) -> float:
    """
    Compute the two-sided exact McNemar p-value.

    Under the null hypothesis, conditional on the number of discordant
    pairs, each direction of disagreement has probability 0.5.

    scipy.stats.binomtest implements the exact two-sided binomial test.
    """

    n_discordant = (
        model_1_only_correct
        + model_2_only_correct
    )

    if n_discordant == 0:
        return 1.0

    result = binomtest(
        k=model_1_only_correct,
        n=n_discordant,
        p=0.5,
        alternative="two-sided",
    )

    return float(result.pvalue)


# =============================================================================
# HOLM CORRECTION
# =============================================================================

def holm_adjust_pvalues(
    p_values: np.ndarray,
) -> np.ndarray:
    """
    Apply Holm's step-down family-wise error-rate correction.

    The adjusted p-values are computed as:

        adjusted_(i) = max_{j <= i} [(m - j + 1) * p_(j)]

    after sorting raw p-values in ascending order.

    Adjusted values are capped at 1.0 and returned in the original order.
    """

    p_values = np.asarray(p_values, dtype=float)

    if p_values.ndim != 1:
        raise ValueError(
            "p_values must be a one-dimensional array."
        )

    if len(p_values) == 0:
        return np.asarray([], dtype=float)

    if np.any(np.isnan(p_values)):
        raise ValueError(
            "Cannot apply Holm correction to NaN p-values."
        )

    if np.any((p_values < 0.0) | (p_values > 1.0)):
        raise ValueError(
            "All p-values must lie in [0, 1]."
        )

    number_of_tests = len(p_values)

    sorted_indices = np.argsort(
        p_values,
        kind="stable",
    )

    sorted_p_values = p_values[sorted_indices]

    adjusted_sorted = np.empty(
        number_of_tests,
        dtype=float,
    )

    running_maximum = 0.0

    for rank, p_value in enumerate(
        sorted_p_values,
        start=1,
    ):
        multiplier = number_of_tests - rank + 1

        adjusted_value = min(
            1.0,
            multiplier * p_value,
        )

        running_maximum = max(
            running_maximum,
            adjusted_value,
        )

        adjusted_sorted[rank - 1] = running_maximum

    adjusted_original_order = np.empty(
        number_of_tests,
        dtype=float,
    )

    adjusted_original_order[sorted_indices] = adjusted_sorted

    return adjusted_original_order


# =============================================================================
# PAIRWISE ANALYSIS
# =============================================================================

def run_pairwise_mcnemar_tests(
    gold_labels: np.ndarray,
    predictions_by_model: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Run all pairwise exact McNemar tests.
    """

    rows = []

    model_names = list(predictions_by_model.keys())

    for model_1, model_2 in combinations(model_names, 2):

        predictions_1 = predictions_by_model[model_1]
        predictions_2 = predictions_by_model[model_2]

        (
            both_correct,
            model_1_only_correct,
            model_2_only_correct,
            both_wrong,
        ) = compute_contingency_counts(
            gold_labels=gold_labels,
            predictions_model_1=predictions_1,
            predictions_model_2=predictions_2,
        )

        n_discordant = (
            model_1_only_correct
            + model_2_only_correct
        )

        raw_p_value = exact_mcnemar_p_value(
            model_1_only_correct=model_1_only_correct,
            model_2_only_correct=model_2_only_correct,
        )

        accuracy_model_1 = float(
            np.mean(predictions_1 == gold_labels)
        )

        accuracy_model_2 = float(
            np.mean(predictions_2 == gold_labels)
        )

        accuracy_difference = (
            accuracy_model_1
            - accuracy_model_2
        )

        if model_1_only_correct > model_2_only_correct:
            direction = f"{model_1} > {model_2}"

        elif model_2_only_correct > model_1_only_correct:
            direction = f"{model_2} > {model_1}"

        else:
            direction = "equal discordant counts"

        rows.append(
            {
                "model_1": model_1,
                "model_2": model_2,
                "n_test_instances": len(gold_labels),
                "both_correct": both_correct,
                "model_1_only_correct": model_1_only_correct,
                "model_2_only_correct": model_2_only_correct,
                "both_wrong": both_wrong,
                "n_discordant": n_discordant,
                "accuracy_model_1": accuracy_model_1,
                "accuracy_model_2": accuracy_model_2,
                "accuracy_difference": accuracy_difference,
                "direction": direction,
                "raw_p_value": raw_p_value,
                "test": "two-sided exact McNemar",
            }
        )

    results = pd.DataFrame(rows)

    if len(results) != 15:
        raise RuntimeError(
            f"Expected 15 pairwise comparisons for 6 models, "
            f"but obtained {len(results)}."
        )

    adjusted_p_values = holm_adjust_pvalues(
        results["raw_p_value"].to_numpy()
    )

    results["holm_adjusted_p_value"] = adjusted_p_values

    results["significant_raw_0.05"] = (
        results["raw_p_value"] < ALPHA
    )

    results["significant_holm_0.05"] = (
        results["holm_adjusted_p_value"] < ALPHA
    )

    return results


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def format_results_for_console(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return a readable subset of columns for console output.
    """

    columns = [
        "model_1",
        "model_2",
        "both_correct",
        "model_1_only_correct",
        "model_2_only_correct",
        "both_wrong",
        "n_discordant",
        "raw_p_value",
        "holm_adjusted_p_value",
        "significant_holm_0.05",
    ]

    console_results = results[columns].copy()

    return console_results.sort_values(
        by=[
            "holm_adjusted_p_value",
            "raw_p_value",
        ],
        ascending=True,
        kind="stable",
    ).reset_index(drop=True)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Resolve prediction files
    # -------------------------------------------------------------------------

    print_header("RESOLVING PREDICTION FILES")

    for model_name, file_path in MODEL_FILES.items():

        if not file_path.exists():
            raise FileNotFoundError(
                f"Prediction file for {model_name} not found:\n"
                f"  {file_path}"
            )

        relative_path = file_path.relative_to(REPO_ROOT)

        print(
            f"{model_name}: {relative_path}"
        )

    # -------------------------------------------------------------------------
    # Load predictions
    # -------------------------------------------------------------------------

    print_header("LOADING PREDICTIONS")

    gold_labels_by_model: Dict[str, np.ndarray] = {}
    predictions_by_model: Dict[str, np.ndarray] = {}

    for model_name, file_path in MODEL_FILES.items():

        gold_labels, predictions = load_prediction_file(
            model_name=model_name,
            file_path=file_path,
        )

        gold_labels_by_model[model_name] = gold_labels
        predictions_by_model[model_name] = predictions

        print(
            f"{model_name}: "
            f"{len(predictions)} predictions loaded"
        )

    # -------------------------------------------------------------------------
    # Validate common test order
    # -------------------------------------------------------------------------

    print_header("VALIDATING COMMON TEST ORDER")

    common_gold_labels = validate_common_gold_order(
        gold_labels_by_model
    )

    print(
        f"All models contain "
        f"{len(common_gold_labels)} predictions."
    )

    print(
        "Gold-label ordering is identical across models."
    )

    # -------------------------------------------------------------------------
    # Analysis configuration
    # -------------------------------------------------------------------------

    print_header("ANALYSIS CONFIGURATION")

    number_of_models = len(predictions_by_model)

    number_of_pairwise_comparisons = (
        number_of_models
        * (number_of_models - 1)
        // 2
    )

    print(
        f"Number of models: "
        f"{number_of_models}"
    )

    print(
        f"Number of test instances: "
        f"{len(common_gold_labels)}"
    )

    print(
        f"Number of pairwise comparisons: "
        f"{number_of_pairwise_comparisons}"
    )

    print(
        "McNemar test: two-sided exact binomial test"
    )

    print(
        "Multiple-comparison correction: Holm"
    )

    print(
        f"Family-wise alpha: {ALPHA}"
    )

    # -------------------------------------------------------------------------
    # Run tests
    # -------------------------------------------------------------------------

    print_header("RUNNING PAIRWISE EXACT McNEMAR TESTS")

    results = run_pairwise_mcnemar_tests(
        gold_labels=common_gold_labels,
        predictions_by_model=predictions_by_model,
    )

    # -------------------------------------------------------------------------
    # Print results
    # -------------------------------------------------------------------------

    print_header("PAIRWISE McNEMAR RESULTS")

    console_results = format_results_for_console(
        results
    )

    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        220,
        "display.max_colwidth",
        40,
        "display.float_format",
        lambda value: f"{value:.10g}",
    ):
        print(
            console_results.to_string(index=False)
        )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    number_raw_significant = int(
        results["significant_raw_0.05"].sum()
    )

    number_holm_significant = int(
        results["significant_holm_0.05"].sum()
    )

    print_header("SIGNIFICANCE SUMMARY")

    print(
        f"Raw p < {ALPHA}: "
        f"{number_raw_significant} / "
        f"{len(results)} comparisons"
    )

    print(
        f"Holm-adjusted p < {ALPHA}: "
        f"{number_holm_significant} / "
        f"{len(results)} comparisons"
    )

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------

    results_to_save = results.sort_values(
        by=[
            "holm_adjusted_p_value",
            "raw_p_value",
        ],
        ascending=True,
        kind="stable",
    ).reset_index(drop=True)

    results_to_save.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print_header("SAVED RESULTS")

    print(
        OUTPUT_FILE.relative_to(REPO_ROOT)
    )

    print_header("Done.")


if __name__ == "__main__":
    main()
    