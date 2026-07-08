#!/usr/bin/env python3

"""
Bootstrap confidence intervals and paired Macro-F1 differences for
SpanishHopeMultidomain.

This script reproduces the bootstrap analysis used in the study.

Models
------
- SVM
- BETO majority-vote predictions
- RoBERTuito majority-vote predictions
- Qwen2.5-7B-Instruct
- GPT-4o-mini
- GPT-4.1-mini

Method
------
- 400 common test instances.
- 10,000 bootstrap replicates.
- Shared bootstrap samples across all models.
- Sampling with replacement from the complete test set.
- Macro-F1 computed for every model on every bootstrap sample.
- 95% percentile confidence intervals.
- Pairwise Macro-F1 differences computed from the same shared bootstrap
  replicates.
- Random seed: 42.

Outputs
-------
results/bootstrap_confidence_intervals.csv
results/bootstrap_pairwise_macro_f1_differences.csv
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

N_BOOTSTRAPS = 10_000
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 42
EXPECTED_N_INSTANCES = 400


MODEL_ORDER = [
    "SVM",
    "BETO",
    "RoBERTuito",
    "Qwen2.5-7B",
    "GPT-4o-mini",
    "GPT-4.1-mini",
]


# Exact files already confirmed by your terminal output.
PREDICTION_FILES = {
    "SVM": RESULTS_DIR / "svm_predictions.csv",

    "BETO":
        RESULTS_DIR / "beto_majority_predictions.csv",

    "RoBERTuito":
        RESULTS_DIR / "robertuito_majority_predictions.csv",

    "Qwen2.5-7B":
        RESULTS_DIR / "qwen25_7b_predictions.csv",

    "GPT-4o-mini":
        RESULTS_DIR / "gpt4o_mini_predictions.csv",
}


# Possible GPT-4.1-mini filenames.
#
# The script first checks these exact paths. If none exists, it safely
# searches all repository CSV files and identifies the GPT-4.1 prediction
# file from filename/path evidence plus CSV structure.
GPT41_EXACT_CANDIDATES = [

    RESULTS_DIR / "gpt41_mini_predictions.csv",

    RESULTS_DIR / "gpt4_1_mini_predictions.csv",

    RESULTS_DIR / "gpt4.1_mini_predictions.csv",

    RESULTS_DIR / "gpt4.1-mini_predictions.csv",

    RESULTS_DIR / "gpt-4.1-mini_predictions.csv",

    RESULTS_DIR / "gpt_4_1_mini_predictions.csv",

    RESULTS_DIR / "gpt41_predictions.csv",

    RESULTS_DIR / "gpt4_1_predictions.csv",

    RESULTS_DIR / "gpt4.1_predictions.csv",

    RESULTS_DIR / "gpt_4_1_predictions.csv",
]


# =============================================================================
# COLUMN NAMES
# =============================================================================

GOLD_COLUMN_CANDIDATES = [
    "gold",
    "gold_label",
    "true_label",
    "label",
    "y_true",
    "target",
    "expected_label",
    "actual_label",
]


PREDICTION_COLUMN_CANDIDATES = [
    "prediction",
    "predicted_label",
    "pred_label",
    "y_pred",
    "pred",
    "parsed_prediction",
    "majority_vote",
    "majority_prediction",
    "majority_label",
    "output_label",
    "response_label",
]


# =============================================================================
# LABEL NORMALIZATION
# =============================================================================

def normalize_label(value) -> int:
    """
    Convert benchmark labels to binary integers.

    Returns
    -------
    1 : Hope Speech
    0 : Non Hope Speech
    """

    if pd.isna(value):
        raise ValueError("Missing label encountered.")

    if isinstance(value, (int, np.integer)):

        value = int(value)

        if value in (0, 1):
            return value


    if isinstance(value, (float, np.floating)):

        if value in (0.0, 1.0):
            return int(value)


    text = str(value).strip().lower()

    text = text.replace("_", " ")
    text = text.replace("-", " ")

    text = " ".join(text.split())


    hope_labels = {
        "1",
        "hs",
        "hope",
        "hope speech",
        "hopespeech",
    }


    non_hope_labels = {
        "0",
        "nhs",
        "non hope",
        "nonhope",
        "non hope speech",
        "nonhope speech",
        "nonhopespeech",
    }


    if text in hope_labels:
        return 1


    if text in non_hope_labels:
        return 0


    raise ValueError(
        f"Unrecognized label value: {value!r}"
    )


# =============================================================================
# DATAFRAME UTILITIES
# =============================================================================

def find_column(
    df: pd.DataFrame,
    candidates: List[str],
) -> Optional[str]:

    """
    Find the first matching column name, case-insensitively.
    """

    lower_to_original = {
        str(column).lower(): column
        for column in df.columns
    }


    for candidate in candidates:

        if candidate in df.columns:
            return candidate


        candidate_lower = candidate.lower()

        if candidate_lower in lower_to_original:
            return lower_to_original[candidate_lower]


    return None


def safe_read_csv(
    path: Path,
) -> Optional[pd.DataFrame]:

    try:
        return pd.read_csv(path)

    except Exception:
        return None


# =============================================================================
# GPT-4.1-MINI FILE DISCOVERY
# =============================================================================

def normalize_path(path: Path) -> str:

    return (
        str(path)
        .lower()
        .replace("\\", "/")
        .replace("-", "_")
        .replace(".", "_")
    )


def looks_like_gpt41_path(
    path: Path,
) -> bool:

    """
    Determine whether a path plausibly refers to GPT-4.1-mini.

    This deliberately supports many filename conventions.
    """

    text = normalize_path(path)


    positive_patterns = [

        "gpt41",

        "gpt4_1",

        "gpt_4_1",

        "4_1_mini",

        "4_1",

        "gpt41mini",

        "gpt4_1mini",
    ]


    return any(
        pattern in text
        for pattern in positive_patterns
    )


def is_analysis_output(
    path: Path,
) -> bool:

    """
    Exclude downstream analysis CSV files.
    """

    text = normalize_path(path)


    excluded_patterns = [

        "bootstrap",

        "mcnemar",

        "agreement",

        "confusion",

        "classification_report",

        "metrics",

        "pairwise",

        "statistical",

        "statistics",

        "error_analysis",

        "errors",

        "summary",
    ]


    return any(
        pattern in text
        for pattern in excluded_patterns
    )


def validate_prediction_dataframe(
    df: pd.DataFrame,
) -> bool:

    """
    Validate the minimum structure required from a prediction file.
    """

    if len(df) != EXPECTED_N_INSTANCES:
        return False


    gold_column = find_column(
        df,
        GOLD_COLUMN_CANDIDATES,
    )


    prediction_column = find_column(
        df,
        PREDICTION_COLUMN_CANDIDATES,
    )


    if gold_column is None:
        return False


    if prediction_column is None:
        return False


    try:

        df[gold_column].map(normalize_label)

        df[prediction_column].map(normalize_label)

    except ValueError:

        return False


    return True


def gpt41_candidate_score(
    path: Path,
) -> int:

    """
    Rank plausible GPT-4.1-mini prediction files.
    """

    text = normalize_path(path)

    filename = normalize_path(Path(path.name))


    score = 0


    # Strong model evidence.

    if "gpt41_mini" in text:
        score += 300

    if "gpt4_1_mini" in text:
        score += 300

    if "gpt_4_1_mini" in text:
        score += 300

    if "gpt41mini" in text:
        score += 250

    if "gpt41" in text:
        score += 150

    if "gpt4_1" in text:
        score += 150

    if "gpt_4_1" in text:
        score += 150


    # Prediction-file evidence.

    if "prediction" in filename:
        score += 100

    if "predictions" in filename:
        score += 50


    # Prefer results and predictions directories.

    if "/results/" in f"/{text}":
        score += 40

    if "/predictions/" in f"/{text}":
        score += 40


    # Prefer "mini" endpoints.

    if "mini" in text:
        score += 30


    # Explicitly reject GPT-4o.

    if "gpt4o" in text:
        score -= 1000

    if "gpt_4o" in text:
        score -= 1000


    return score


def discover_gpt41_prediction_file() -> Path:

    """
    Find the GPT-4.1-mini prediction file.

    Procedure:
    1. Check known exact filenames.
    2. Recursively inspect model-related CSV files.
    3. Validate CSV structure and row count.
    4. Rank candidates.
    5. Require a unique best candidate.
    """

    # -------------------------------------------------------------------------
    # STEP 1: EXACT CANDIDATES
    # -------------------------------------------------------------------------

    for path in GPT41_EXACT_CANDIDATES:

        if not path.exists():
            continue


        df = safe_read_csv(path)


        if df is None:
            continue


        if validate_prediction_dataframe(df):
            return path


    # -------------------------------------------------------------------------
    # STEP 2: RECURSIVE SEARCH
    # -------------------------------------------------------------------------

    valid_candidates = []


    for path in PROJECT_ROOT.rglob("*.csv"):

        if is_analysis_output(path):
            continue


        if not looks_like_gpt41_path(path):
            continue


        df = safe_read_csv(path)


        if df is None:
            continue


        if not validate_prediction_dataframe(df):
            continue


        score = gpt41_candidate_score(path)


        valid_candidates.append(
            (score, path)
        )


    # -------------------------------------------------------------------------
    # STEP 3: FAILURE REPORT
    # -------------------------------------------------------------------------

    if not valid_candidates:

        all_csv_files = sorted(
            PROJECT_ROOT.rglob("*.csv")
        )


        prediction_like_files = [

            path.relative_to(PROJECT_ROOT)

            for path in all_csv_files

            if (
                "prediction"
                in normalize_path(path)
                or
                "predictions"
                in normalize_path(path)
            )

            and not is_analysis_output(path)
        ]


        prediction_file_text = "\n".join(

            f"  - {path}"

            for path in prediction_like_files
        )


        raise FileNotFoundError(

            "Could not locate the GPT-4.1-mini prediction file.\n\n"

            "Prediction-like CSV files currently present in the "
            "repository are:\n"

            f"{prediction_file_text}\n\n"

            "The script requires the fixed 400-instance prediction "
            "vector produced by GPT-4.1-mini."
        )


    # -------------------------------------------------------------------------
    # STEP 4: RANK CANDIDATES
    # -------------------------------------------------------------------------

    valid_candidates.sort(

        key=lambda item: (
            -item[0],
            str(item[1]),
        )
    )


    best_score = valid_candidates[0][0]


    best_candidates = [

        path

        for score, path in valid_candidates

        if score == best_score
    ]


    # -------------------------------------------------------------------------
    # STEP 5: REQUIRE UNIQUE BEST FILE
    # -------------------------------------------------------------------------

    if len(best_candidates) > 1:

        candidate_text = "\n".join(

            f"  - {path.relative_to(PROJECT_ROOT)}"

            for path in best_candidates
        )


        raise RuntimeError(

            "Multiple equally ranked GPT-4.1-mini prediction files "
            "were found:\n"

            f"{candidate_text}\n\n"

            "The script refuses to select one silently."
        )


    return best_candidates[0]


# =============================================================================
# PREDICTION FILE LOADING
# =============================================================================

def load_prediction_file(
    model_name: str,
    path: Path,
) -> Tuple[np.ndarray, np.ndarray]:

    """
    Load one prediction CSV and return gold labels and predictions.
    """

    if not path.exists():

        raise FileNotFoundError(

            f"Prediction file for {model_name} does not exist:\n"
            f"{path}"
        )


    df = pd.read_csv(path)


    if len(df) != EXPECTED_N_INSTANCES:

        raise ValueError(

            f"{model_name}: expected "
            f"{EXPECTED_N_INSTANCES} rows, "
            f"found {len(df)} in:\n"
            f"{path}"
        )


    gold_column = find_column(
        df,
        GOLD_COLUMN_CANDIDATES,
    )


    prediction_column = find_column(
        df,
        PREDICTION_COLUMN_CANDIDATES,
    )


    if gold_column is None:

        raise ValueError(

            f"{model_name}: no gold-label column found in:\n"
            f"{path}\n\n"
            f"Available columns: {list(df.columns)}"
        )


    if prediction_column is None:

        raise ValueError(

            f"{model_name}: no prediction column found in:\n"
            f"{path}\n\n"
            f"Available columns: {list(df.columns)}"
        )


    gold = (
        df[gold_column]
        .map(normalize_label)
        .to_numpy(dtype=np.int64)
    )


    predictions = (
        df[prediction_column]
        .map(normalize_label)
        .to_numpy(dtype=np.int64)
    )


    return gold, predictions


# =============================================================================
# VALIDATE COMMON TEST INSTANCES
# =============================================================================

def validate_alignment(
    gold_vectors: Dict[str, np.ndarray],
) -> np.ndarray:

    """
    Verify identical gold-label ordering across all model files.
    """

    reference_model = MODEL_ORDER[0]

    reference_gold = gold_vectors[reference_model]


    if len(reference_gold) != EXPECTED_N_INSTANCES:

        raise ValueError(

            f"Expected {EXPECTED_N_INSTANCES} test instances, "
            f"found {len(reference_gold)}."
        )


    for model_name in MODEL_ORDER[1:]:

        model_gold = gold_vectors[model_name]


        if len(model_gold) != len(reference_gold):

            raise ValueError(

                f"Gold-label vector length mismatch for "
                f"{model_name}."
            )


        if not np.array_equal(
            reference_gold,
            model_gold,
        ):

            mismatches = np.flatnonzero(
                reference_gold != model_gold
            )


            raise ValueError(

                f"Gold-label ordering mismatch between "
                f"{reference_model} and {model_name}.\n"

                f"First mismatching row indices: "
                f"{mismatches[:20].tolist()}"
            )


    return reference_gold


# =============================================================================
# SHARED PAIRED BOOTSTRAP
# =============================================================================

def run_shared_paired_bootstrap(
    gold: np.ndarray,
    predictions: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:

    """
    Run the shared paired bootstrap.

    One bootstrap sample of test-instance indices is generated for each
    replicate. The same sampled indices are applied to all six models.
    """

    rng = np.random.default_rng(
        RANDOM_SEED
    )


    n_instances = len(gold)


    bootstrap_scores = {

        model_name:
            np.empty(
                N_BOOTSTRAPS,
                dtype=np.float64,
            )

        for model_name in MODEL_ORDER
    }


    for bootstrap_index in range(N_BOOTSTRAPS):

        sampled_indices = rng.integers(

            low=0,

            high=n_instances,

            size=n_instances,
        )


        sampled_gold = gold[
            sampled_indices
        ]


        for model_name in MODEL_ORDER:

            sampled_predictions = (

                predictions[model_name][
                    sampled_indices
                ]
            )


            bootstrap_scores[
                model_name
            ][bootstrap_index] = f1_score(

                sampled_gold,

                sampled_predictions,

                average="macro",

                zero_division=0,
            )


        completed = bootstrap_index + 1


        if completed % 1000 == 0:

            print(

                f"Completed {completed} / "
                f"{N_BOOTSTRAPS} replicates"
            )


    return bootstrap_scores


# =============================================================================
# CONFIDENCE INTERVAL
# =============================================================================

def percentile_confidence_interval(
    values: np.ndarray,
) -> Tuple[float, float]:

    """
    Compute the two-sided percentile confidence interval.
    """

    alpha = 1.0 - CONFIDENCE_LEVEL


    lower_percentile = (
        100.0 * alpha / 2.0
    )


    upper_percentile = (
        100.0 * (1.0 - alpha / 2.0)
    )


    lower = float(
        np.percentile(
            values,
            lower_percentile,
        )
    )


    upper = float(
        np.percentile(
            values,
            upper_percentile,
        )
    )


    return lower, upper


# =============================================================================
# MODEL-LEVEL RESULTS
# =============================================================================

def build_model_results(
    gold: np.ndarray,
    predictions: Dict[str, np.ndarray],
    bootstrap_scores: Dict[str, np.ndarray],
) -> pd.DataFrame:

    rows = []


    for model_name in MODEL_ORDER:

        observed_macro_f1 = f1_score(

            gold,

            predictions[model_name],

            average="macro",

            zero_division=0,
        )


        scores = bootstrap_scores[
            model_name
        ]


        ci_lower, ci_upper = (
            percentile_confidence_interval(
                scores
            )
        )


        rows.append({

            "model":
                model_name,

            "macro_f1":
                observed_macro_f1,

            "bootstrap_mean":
                float(np.mean(scores)),

            "ci_95_lower":
                ci_lower,

            "ci_95_upper":
                ci_upper,

            "n_bootstraps":
                N_BOOTSTRAPS,

            "random_seed":
                RANDOM_SEED,
        })


    results = pd.DataFrame(rows)


    results = results.sort_values(

        "macro_f1",

        ascending=False,

    ).reset_index(drop=True)


    return results


# =============================================================================
# PAIRWISE DIFFERENCES
# =============================================================================

def build_pairwise_results(
    gold: np.ndarray,
    predictions: Dict[str, np.ndarray],
    bootstrap_scores: Dict[str, np.ndarray],
) -> pd.DataFrame:

    observed_scores = {

        model_name: f1_score(

            gold,

            predictions[model_name],

            average="macro",

            zero_division=0,
        )

        for model_name in MODEL_ORDER
    }


    rows = []


    for model_1, model_2 in combinations(
        MODEL_ORDER,
        2,
    ):

        observed_difference = (

            observed_scores[model_1]

            -

            observed_scores[model_2]
        )


        bootstrap_differences = (

            bootstrap_scores[model_1]

            -

            bootstrap_scores[model_2]
        )


        ci_lower, ci_upper = (

            percentile_confidence_interval(

                bootstrap_differences
            )
        )


        rows.append({

            "model_1":
                model_1,

            "model_2":
                model_2,

            "macro_f1_model_1":
                observed_scores[model_1],

            "macro_f1_model_2":
                observed_scores[model_2],

            "macro_f1_difference":
                observed_difference,

            "bootstrap_mean_difference":
                float(
                    np.mean(
                        bootstrap_differences
                    )
                ),

            "ci_95_lower":
                ci_lower,

            "ci_95_upper":
                ci_upper,

            "n_bootstraps":
                N_BOOTSTRAPS,

            "random_seed":
                RANDOM_SEED,
        })


    results = pd.DataFrame(rows)


    results["_abs_difference"] = (

        results[
            "macro_f1_difference"
        ].abs()
    )


    results = (

        results

        .sort_values(

            "_abs_difference",

            ascending=False,
        )

        .drop(
            columns="_abs_difference"
        )

        .reset_index(drop=True)
    )


    return results


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print()
    print("========================================")
    print("RESOLVING PREDICTION FILES")
    print("========================================")


    prediction_files = dict(
        PREDICTION_FILES
    )


    prediction_files[
        "GPT-4.1-mini"
    ] = discover_gpt41_prediction_file()


    for model_name in MODEL_ORDER:

        print(

            f"{model_name}: "

            f"{prediction_files[model_name].relative_to(PROJECT_ROOT)}"
        )


    print()
    print("========================================")
    print("LOADING PREDICTIONS")
    print("========================================")


    gold_vectors = {}

    predictions = {}


    for model_name in MODEL_ORDER:

        gold, model_predictions = (

            load_prediction_file(

                model_name=model_name,

                path=prediction_files[
                    model_name
                ],
            )
        )


        gold_vectors[
            model_name
        ] = gold


        predictions[
            model_name
        ] = model_predictions


        print(

            f"{model_name}: "
            f"{len(model_predictions)} predictions loaded"
        )


    print()
    print("========================================")
    print("VALIDATING COMMON TEST ORDER")
    print("========================================")


    gold = validate_alignment(
        gold_vectors
    )


    print(
        "All models contain 400 predictions."
    )

    print(
        "Gold-label ordering is identical across models."
    )


    print()
    print("========================================")
    print("ANALYSIS CONFIGURATION")
    print("========================================")


    print(
        f"Number of models: "
        f"{len(MODEL_ORDER)}"
    )

    print(
        f"Number of test instances: "
        f"{len(gold)}"
    )

    print(
        f"Number of bootstrap replicates: "
        f"{N_BOOTSTRAPS}"
    )

    print(
        f"Confidence level: "
        f"{CONFIDENCE_LEVEL}"
    )

    print(
        f"Random seed: "
        f"{RANDOM_SEED}"
    )


    print()
    print("========================================")
    print("RUNNING SHARED PAIRED BOOTSTRAP")
    print("========================================")


    bootstrap_scores = (

        run_shared_paired_bootstrap(

            gold=gold,

            predictions=predictions,
        )
    )


    print()
    print("========================================")
    print("MODEL-LEVEL BOOTSTRAP RESULTS")
    print("========================================")


    model_results = build_model_results(

        gold=gold,

        predictions=predictions,

        bootstrap_scores=bootstrap_scores,
    )


    print(
        model_results.to_string(
            index=False
        )
    )


    print()
    print("========================================")
    print("PAIRWISE MACRO-F1 DIFFERENCES")
    print("========================================")


    pairwise_results = build_pairwise_results(

        gold=gold,

        predictions=predictions,

        bootstrap_scores=bootstrap_scores,
    )


    print(
        pairwise_results.to_string(
            index=False
        )
    )


    model_output = (

        RESULTS_DIR

        / "bootstrap_confidence_intervals.csv"
    )


    pairwise_output = (

        RESULTS_DIR

        / "bootstrap_pairwise_macro_f1_differences.csv"
    )


    model_results.to_csv(

        model_output,

        index=False,
    )


    pairwise_results.to_csv(

        pairwise_output,

        index=False,
    )


    print()
    print("========================================")
    print("SAVED RESULTS")
    print("========================================")


    print(
        model_output.relative_to(
            PROJECT_ROOT
        )
    )


    print(
        pairwise_output.relative_to(
            PROJECT_ROOT
        )
    )


    print()
    print("========================================")
    print("Done.")
    print("========================================")
    print()


if __name__ == "__main__":
    main()
    