#!/usr/bin/env python3
"""
Validate and summarize the manually coded unanimous-error instances.

This script is part of the SpanishHopeMultidomain benchmark framework.

Purpose
-------
The script validates the completed qualitative coding file produced from the
instances misclassified by all evaluated model-level prediction vectors and
generates exact descriptive summaries for manuscript reporting.

The qualitative analysis is based on a deliberately selected subset:
instances misclassified by all six model-level prediction vectors.

Accordingly, the generated summaries characterize recurrent phenomena within
the unanimous-error subset. They must not be interpreted as:

    * frequencies among all benchmark errors;
    * prevalence estimates for the complete benchmark;
    * intrinsic properties of the social domains;
    * a quantitative taxonomy validated by multiple coders.

The coding was performed by one qualitative coder.

Inputs
------
results/qualitative_error_coding_template.csv

Expected number of coded instances
----------------------------------
16

Outputs
-------
results/qualitative_primary_phenomenon_summary.csv
results/qualitative_secondary_phenomenon_summary.csv
results/qualitative_boolean_diagnostic_summary.csv
results/qualitative_primary_by_domain.csv
results/qualitative_secondary_by_domain.csv
results/qualitative_boolean_by_domain.csv
results/qualitative_annotation_ambiguity_cases.csv
results/qualitative_primary_secondary_cooccurrence.csv
results/qualitative_boolean_cooccurrence.csv
results/qualitative_coding_summary.csv

No qualitative labels are generated automatically. The script only validates
and summarizes coding decisions already recorded in the input file.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "qualitative_error_coding_template.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"

EXPECTED_N_INSTANCES = 16
EXPECTED_N_MODELS = 6

EXPECTED_DOMAINS = {
    "lgbt",
    "obesity",
    "racism",
}

REQUIRED_COLUMNS = [
    "id",
    "text",
    "gold",
    "domain",
    "n_models_correct",
    "n_model_errors",
    "model_error_rate",
    "difficulty_group",
    "primary_error_phenomenon",
    "secondary_error_phenomenon",
    "implicit_supportive_intent",
    "mixed_communicative_stance",
    "informational_or_context_dependent_discourse",
    "irony_or_sarcasm",
    "sociocultural_reference",
    "annotation_ambiguity",
    "qualitative_notes",
]

BOOLEAN_COLUMNS = [
    "implicit_supportive_intent",
    "mixed_communicative_stance",
    "informational_or_context_dependent_discourse",
    "irony_or_sarcasm",
    "sociocultural_reference",
    "annotation_ambiguity",
]

NONEMPTY_CODING_COLUMNS = [
    "primary_error_phenomenon",
    "qualitative_notes",
]

ALLOWED_GOLD_LABELS = {
    "hs",
    "nhs",
}

ALLOWED_DIFFICULTY_GROUPS = {
    "hard",
}

TRUE_STRINGS = {
    "true",
    "1",
    "yes",
    "y",
}

FALSE_STRINGS = {
    "false",
    "0",
    "no",
    "n",
}


# =============================================================================
# OUTPUT FILES
# =============================================================================

PRIMARY_SUMMARY_FILE = (
    RESULTS_DIR
    / "qualitative_primary_phenomenon_summary.csv"
)

SECONDARY_SUMMARY_FILE = (
    RESULTS_DIR
    / "qualitative_secondary_phenomenon_summary.csv"
)

BOOLEAN_SUMMARY_FILE = (
    RESULTS_DIR
    / "qualitative_boolean_diagnostic_summary.csv"
)

PRIMARY_BY_DOMAIN_FILE = (
    RESULTS_DIR
    / "qualitative_primary_by_domain.csv"
)

SECONDARY_BY_DOMAIN_FILE = (
    RESULTS_DIR
    / "qualitative_secondary_by_domain.csv"
)

BOOLEAN_BY_DOMAIN_FILE = (
    RESULTS_DIR
    / "qualitative_boolean_by_domain.csv"
)

AMBIGUITY_CASES_FILE = (
    RESULTS_DIR
    / "qualitative_annotation_ambiguity_cases.csv"
)

PRIMARY_SECONDARY_COOCCURRENCE_FILE = (
    RESULTS_DIR
    / "qualitative_primary_secondary_cooccurrence.csv"
)

BOOLEAN_COOCCURRENCE_FILE = (
    RESULTS_DIR
    / "qualitative_boolean_cooccurrence.csv"
)

CODING_SUMMARY_FILE = (
    RESULTS_DIR
    / "qualitative_coding_summary.csv"
)


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def print_saved_file(path: Path) -> None:
    print(path.relative_to(PROJECT_ROOT))


# =============================================================================
# NORMALIZATION HELPERS
# =============================================================================

def normalize_string_series(series: pd.Series) -> pd.Series:
    """
    Convert a series to pandas string dtype and strip surrounding whitespace.

    Missing values remain missing.
    """
    return series.astype("string").str.strip()


def normalize_lower_string_series(series: pd.Series) -> pd.Series:
    """
    Normalize strings and convert non-missing values to lowercase.
    """
    return normalize_string_series(series).str.lower()


def parse_boolean_value(value, column_name: str, row_id) -> bool:
    """
    Convert an explicitly coded Boolean value to bool.

    Missing or unexpected values raise ValueError.
    """
    if pd.isna(value):
        raise ValueError(
            f"Missing Boolean coding in column '{column_name}' "
            f"for instance ID {row_id}."
        )

    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()

    if normalized in TRUE_STRINGS:
        return True

    if normalized in FALSE_STRINGS:
        return False

    raise ValueError(
        f"Invalid Boolean coding '{value}' in column "
        f"'{column_name}' for instance ID {row_id}. "
        f"Expected an explicit TRUE/FALSE value."
    )


def parse_boolean_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Parse all Boolean diagnostic columns strictly.
    """
    dataframe = dataframe.copy()

    for column in BOOLEAN_COLUMNS:
        dataframe[column] = [
            parse_boolean_value(
                value=value,
                column_name=column,
                row_id=row_id,
            )
            for value, row_id in zip(
                dataframe[column],
                dataframe["id"],
            )
        ]

    return dataframe


# =============================================================================
# VALIDATION
# =============================================================================

def load_coding_file(path: Path) -> pd.DataFrame:
    """
    Load the qualitative coding file while preserving IDs exactly.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Qualitative coding file not found:\n  {path}"
        )

    dataframe = pd.read_csv(
        path,
        dtype={
            "id": "string",
            "text": "string",
            "gold": "string",
            "domain": "string",
            "difficulty_group": "string",
            "primary_error_phenomenon": "string",
            "secondary_error_phenomenon": "string",
            "qualitative_notes": "string",
        },
        keep_default_na=True,
    )

    return dataframe


def validate_required_columns(dataframe: pd.DataFrame) -> None:
    """
    Validate the exact required schema.

    Additional columns are allowed, but every required column must exist.
    """
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "The qualitative coding file is missing required columns:\n  "
            + "\n  ".join(missing_columns)
        )


def normalize_coding_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize textual coding fields before validation and summarization.
    """
    dataframe = dataframe.copy()

    dataframe["id"] = normalize_string_series(dataframe["id"])
    dataframe["text"] = normalize_string_series(dataframe["text"])
    dataframe["gold"] = normalize_lower_string_series(dataframe["gold"])
    dataframe["domain"] = normalize_lower_string_series(
        dataframe["domain"]
    )
    dataframe["difficulty_group"] = normalize_lower_string_series(
        dataframe["difficulty_group"]
    )

    dataframe["primary_error_phenomenon"] = (
        normalize_lower_string_series(
            dataframe["primary_error_phenomenon"]
        )
    )

    dataframe["secondary_error_phenomenon"] = (
        normalize_lower_string_series(
            dataframe["secondary_error_phenomenon"]
        )
    )

    dataframe["qualitative_notes"] = normalize_string_series(
        dataframe["qualitative_notes"]
    )

    return dataframe


def validate_instance_count(dataframe: pd.DataFrame) -> None:
    if len(dataframe) != EXPECTED_N_INSTANCES:
        raise ValueError(
            f"Expected exactly {EXPECTED_N_INSTANCES} qualitative instances, "
            f"but found {len(dataframe)}."
        )


def validate_unique_ids(dataframe: pd.DataFrame) -> None:
    if dataframe["id"].isna().any():
        raise ValueError(
            "At least one qualitative instance has a missing ID."
        )

    duplicated_mask = dataframe["id"].duplicated(keep=False)

    if duplicated_mask.any():
        duplicated_ids = sorted(
            dataframe.loc[duplicated_mask, "id"]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Duplicate instance IDs found in the qualitative coding file:\n  "
            + "\n  ".join(duplicated_ids)
        )


def validate_nonempty_text(dataframe: pd.DataFrame) -> None:
    missing_mask = (
        dataframe["text"].isna()
        | dataframe["text"].eq("")
    )

    if missing_mask.any():
        ids = dataframe.loc[missing_mask, "id"].tolist()

        raise ValueError(
            "Missing original text for instance IDs:\n  "
            + "\n  ".join(map(str, ids))
        )


def validate_gold_labels(dataframe: pd.DataFrame) -> None:
    observed = set(
        dataframe["gold"].dropna().astype(str).unique()
    )

    unexpected = observed - ALLOWED_GOLD_LABELS

    if unexpected:
        raise ValueError(
            "Unexpected gold labels found:\n  "
            + "\n  ".join(sorted(unexpected))
        )

    if dataframe["gold"].isna().any():
        raise ValueError(
            "At least one qualitative instance has a missing gold label."
        )


def validate_domains(dataframe: pd.DataFrame) -> None:
    observed = set(
        dataframe["domain"].dropna().astype(str).unique()
    )

    unexpected = observed - EXPECTED_DOMAINS
    missing = EXPECTED_DOMAINS - observed

    if unexpected:
        raise ValueError(
            "Unexpected domains found:\n  "
            + "\n  ".join(sorted(unexpected))
        )

    if missing:
        raise ValueError(
            "Expected benchmark domains absent from qualitative coding file:\n  "
            + "\n  ".join(sorted(missing))
        )

    if dataframe["domain"].isna().any():
        raise ValueError(
            "At least one qualitative instance has a missing domain."
        )


def validate_unanimous_error_subset(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate that every row still represents a unanimous model error.
    """
    numeric_columns = [
        "n_models_correct",
        "n_model_errors",
        "model_error_rate",
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="raise",
        )

    invalid_models_correct = (
        dataframe["n_models_correct"] != 0
    )

    invalid_model_errors = (
        dataframe["n_model_errors"] != EXPECTED_N_MODELS
    )

    invalid_error_rate = (
        (dataframe["model_error_rate"] - 1.0).abs() > 1e-12
    )

    if invalid_models_correct.any():
        ids = dataframe.loc[
            invalid_models_correct,
            "id",
        ].tolist()

        raise ValueError(
            "The following rows are not unanimous errors because "
            "n_models_correct != 0:\n  "
            + "\n  ".join(map(str, ids))
        )

    if invalid_model_errors.any():
        ids = dataframe.loc[
            invalid_model_errors,
            "id",
        ].tolist()

        raise ValueError(
            "The following rows are not unanimous errors because "
            f"n_model_errors != {EXPECTED_N_MODELS}:\n  "
            + "\n  ".join(map(str, ids))
        )

    if invalid_error_rate.any():
        ids = dataframe.loc[
            invalid_error_rate,
            "id",
        ].tolist()

        raise ValueError(
            "The following rows are not unanimous errors because "
            "model_error_rate != 1.0:\n  "
            + "\n  ".join(map(str, ids))
        )


def validate_difficulty_group(dataframe: pd.DataFrame) -> None:
    observed = set(
        dataframe["difficulty_group"]
        .dropna()
        .astype(str)
        .unique()
    )

    unexpected = observed - ALLOWED_DIFFICULTY_GROUPS

    if unexpected:
        raise ValueError(
            "Unexpected difficulty groups found:\n  "
            + "\n  ".join(sorted(unexpected))
        )

    if dataframe["difficulty_group"].isna().any():
        raise ValueError(
            "At least one qualitative instance has a missing "
            "difficulty_group."
        )


def validate_completed_coding(dataframe: pd.DataFrame) -> None:
    """
    Validate that required manual coding fields are complete.

    Secondary phenomena are optional and may be missing.
    """
    for column in NONEMPTY_CODING_COLUMNS:
        missing_mask = (
            dataframe[column].isna()
            | dataframe[column].eq("")
        )

        if missing_mask.any():
            ids = dataframe.loc[missing_mask, "id"].tolist()

            raise ValueError(
                f"Manual coding is incomplete in column '{column}' "
                f"for instance IDs:\n  "
                + "\n  ".join(map(str, ids))
            )


def validate_no_category_case_collisions(
    original_dataframe: pd.DataFrame,
) -> None:
    """
    Detect category labels that differ only by capitalization or whitespace.

    This validation is performed before normalized category labels are used
    for summarization.
    """
    for column in [
        "primary_error_phenomenon",
        "secondary_error_phenomenon",
    ]:
        nonmissing = original_dataframe[column].dropna().astype(str)

        groups = {}

        for value in nonmissing:
            stripped = value.strip()

            if not stripped:
                continue

            normalized = stripped.lower()
            groups.setdefault(normalized, set()).add(value)

        collisions = {
            normalized: values
            for normalized, values in groups.items()
            if len(values) > 1
        }

        if collisions:
            details = []

            for normalized, values in sorted(collisions.items()):
                details.append(
                    f"{normalized}: {sorted(values)}"
                )

            raise ValueError(
                f"Category labels in '{column}' differ only by "
                "capitalization or whitespace:\n  "
                + "\n  ".join(details)
            )


def validate_boolean_columns(dataframe: pd.DataFrame) -> None:
    """
    Ensure all parsed Boolean diagnostic fields are actual bool values.
    """
    for column in BOOLEAN_COLUMNS:
        invalid_mask = ~dataframe[column].map(
            lambda value: isinstance(value, bool)
        )

        if invalid_mask.any():
            ids = dataframe.loc[invalid_mask, "id"].tolist()

            raise ValueError(
                f"Invalid parsed Boolean values in column '{column}' "
                f"for instance IDs:\n  "
                + "\n  ".join(map(str, ids))
            )


def validate_coding_file(
    original_dataframe: pd.DataFrame,
    dataframe: pd.DataFrame,
) -> None:
    """
    Run all validation checks.
    """
    validate_required_columns(original_dataframe)
    validate_instance_count(dataframe)
    validate_unique_ids(dataframe)
    validate_nonempty_text(dataframe)
    validate_gold_labels(dataframe)
    validate_domains(dataframe)
    validate_unanimous_error_subset(dataframe)
    validate_difficulty_group(dataframe)
    validate_completed_coding(dataframe)
    validate_no_category_case_collisions(original_dataframe)
    validate_boolean_columns(dataframe)


# =============================================================================
# SUMMARY HELPERS
# =============================================================================

def add_count_percentages(
    dataframe: pd.DataFrame,
    count_column: str,
    denominator: int,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    dataframe["percentage"] = (
        dataframe[count_column]
        / denominator
        * 100.0
    )

    return dataframe


def summarize_primary_phenomena(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        dataframe["primary_error_phenomenon"]
        .value_counts(dropna=False)
        .rename_axis("primary_error_phenomenon")
        .reset_index(name="n_instances")
    )

    summary = add_count_percentages(
        dataframe=summary,
        count_column="n_instances",
        denominator=len(dataframe),
    )

    return summary


def summarize_secondary_phenomena(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    secondary = dataframe[
        dataframe["secondary_error_phenomenon"].notna()
        & dataframe["secondary_error_phenomenon"].ne("")
    ].copy()

    summary = (
        secondary["secondary_error_phenomenon"]
        .value_counts()
        .rename_axis("secondary_error_phenomenon")
        .reset_index(name="n_instances")
    )

    summary["percentage_of_all_instances"] = (
        summary["n_instances"]
        / len(dataframe)
        * 100.0
    )

    n_with_secondary = len(secondary)

    if n_with_secondary > 0:
        summary["percentage_of_instances_with_secondary"] = (
            summary["n_instances"]
            / n_with_secondary
            * 100.0
        )
    else:
        summary["percentage_of_instances_with_secondary"] = pd.Series(
            dtype=float
        )

    return summary


def summarize_boolean_diagnostics(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for column in BOOLEAN_COLUMNS:
        n_true = int(dataframe[column].sum())
        n_false = int((~dataframe[column]).sum())

        rows.append(
            {
                "diagnostic_dimension": column,
                "n_true": n_true,
                "n_false": n_false,
                "percentage_true": (
                    n_true / len(dataframe) * 100.0
                ),
            }
        )

    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        by=[
            "n_true",
            "diagnostic_dimension",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)

    return summary


def summarize_category_by_domain(
    dataframe: pd.DataFrame,
    category_column: str,
) -> pd.DataFrame:
    subset = dataframe[
        dataframe[category_column].notna()
        & dataframe[category_column].ne("")
    ].copy()

    summary = (
        subset.groupby(
            ["domain", category_column],
            dropna=False,
        )
        .size()
        .reset_index(name="n_instances")
    )

    domain_totals = (
        dataframe.groupby("domain")
        .size()
        .rename("n_domain_unanimous_errors")
        .reset_index()
    )

    summary = summary.merge(
        domain_totals,
        on="domain",
        how="left",
        validate="many_to_one",
    )

    summary["percentage_of_domain_unanimous_errors"] = (
        summary["n_instances"]
        / summary["n_domain_unanimous_errors"]
        * 100.0
    )

    summary = summary.sort_values(
        by=[
            "domain",
            "n_instances",
            category_column,
        ],
        ascending=[
            True,
            False,
            True,
        ],
    ).reset_index(drop=True)

    return summary


def summarize_boolean_by_domain(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    domain_totals = dataframe.groupby("domain").size().to_dict()

    for domain in sorted(dataframe["domain"].unique()):
        domain_subset = dataframe[
            dataframe["domain"] == domain
        ]

        denominator = int(domain_totals[domain])

        for column in BOOLEAN_COLUMNS:
            n_true = int(domain_subset[column].sum())

            rows.append(
                {
                    "domain": domain,
                    "diagnostic_dimension": column,
                    "n_true": n_true,
                    "n_domain_unanimous_errors": denominator,
                    "percentage_of_domain_unanimous_errors": (
                        n_true / denominator * 100.0
                    ),
                }
            )

    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        by=[
            "domain",
            "n_true",
            "diagnostic_dimension",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    ).reset_index(drop=True)

    return summary


def extract_annotation_ambiguity_cases(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "id",
        "text",
        "gold",
        "domain",
        "primary_error_phenomenon",
        "secondary_error_phenomenon",
        "qualitative_notes",
    ]

    cases = dataframe.loc[
        dataframe["annotation_ambiguity"],
        columns,
    ].copy()

    cases = cases.sort_values(
        by=[
            "domain",
            "id",
        ]
    ).reset_index(drop=True)

    return cases


def summarize_primary_secondary_cooccurrence(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    subset = dataframe[
        dataframe["secondary_error_phenomenon"].notna()
        & dataframe["secondary_error_phenomenon"].ne("")
    ].copy()

    summary = (
        subset.groupby(
            [
                "primary_error_phenomenon",
                "secondary_error_phenomenon",
            ]
        )
        .size()
        .reset_index(name="n_instances")
    )

    summary["percentage_of_all_instances"] = (
        summary["n_instances"]
        / len(dataframe)
        * 100.0
    )

    n_with_secondary = len(subset)

    if n_with_secondary > 0:
        summary["percentage_of_instances_with_secondary"] = (
            summary["n_instances"]
            / n_with_secondary
            * 100.0
        )
    else:
        summary["percentage_of_instances_with_secondary"] = pd.Series(
            dtype=float
        )

    summary = summary.sort_values(
        by=[
            "n_instances",
            "primary_error_phenomenon",
            "secondary_error_phenomenon",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    ).reset_index(drop=True)

    return summary


def summarize_boolean_cooccurrence(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for first, second in combinations(BOOLEAN_COLUMNS, 2):
        n_both_true = int(
            (dataframe[first] & dataframe[second]).sum()
        )

        if n_both_true == 0:
            continue

        rows.append(
            {
                "diagnostic_dimension_1": first,
                "diagnostic_dimension_2": second,
                "n_both_true": n_both_true,
                "percentage_of_all_instances": (
                    n_both_true
                    / len(dataframe)
                    * 100.0
                ),
            }
        )

    summary = pd.DataFrame(
        rows,
        columns=[
            "diagnostic_dimension_1",
            "diagnostic_dimension_2",
            "n_both_true",
            "percentage_of_all_instances",
        ],
    )

    if not summary.empty:
        summary = summary.sort_values(
            by=[
                "n_both_true",
                "diagnostic_dimension_1",
                "diagnostic_dimension_2",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        ).reset_index(drop=True)

    return summary


def create_overall_coding_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    secondary_mask = (
        dataframe["secondary_error_phenomenon"].notna()
        & dataframe["secondary_error_phenomenon"].ne("")
    )

    rows = [
        {
            "statistic": "n_coded_unanimous_errors",
            "value": len(dataframe),
        },
        {
            "statistic": "n_unique_primary_phenomena",
            "value": dataframe[
                "primary_error_phenomenon"
            ].nunique(),
        },
        {
            "statistic": "n_instances_with_secondary_phenomenon",
            "value": int(secondary_mask.sum()),
        },
        {
            "statistic": "n_unique_secondary_phenomena",
            "value": dataframe.loc[
                secondary_mask,
                "secondary_error_phenomenon",
            ].nunique(),
        },
        {
            "statistic": "n_annotation_ambiguity_cases",
            "value": int(
                dataframe["annotation_ambiguity"].sum()
            ),
        },
        {
            "statistic": "n_lgbt_unanimous_errors",
            "value": int(
                (dataframe["domain"] == "lgbt").sum()
            ),
        },
        {
            "statistic": "n_obesity_unanimous_errors",
            "value": int(
                (dataframe["domain"] == "obesity").sum()
            ),
        },
        {
            "statistic": "n_racism_unanimous_errors",
            "value": int(
                (dataframe["domain"] == "racism").sum()
            ),
        },
    ]

    return pd.DataFrame(rows)


# =============================================================================
# CONSISTENCY CHECKS FOR GENERATED SUMMARIES
# =============================================================================

def validate_generated_summaries(
    dataframe: pd.DataFrame,
    primary_summary: pd.DataFrame,
    secondary_summary: pd.DataFrame,
    boolean_summary: pd.DataFrame,
    primary_by_domain: pd.DataFrame,
    secondary_by_domain: pd.DataFrame,
    boolean_by_domain: pd.DataFrame,
    ambiguity_cases: pd.DataFrame,
    primary_secondary_cooccurrence: pd.DataFrame,
) -> None:
    """
    Validate internal consistency of all generated descriptive summaries.
    """
    if primary_summary["n_instances"].sum() != len(dataframe):
        raise RuntimeError(
            "Primary phenomenon counts do not sum to the number "
            "of coded instances."
        )

    n_with_secondary = int(
        (
            dataframe["secondary_error_phenomenon"].notna()
            & dataframe["secondary_error_phenomenon"].ne("")
        ).sum()
    )

    if secondary_summary["n_instances"].sum() != n_with_secondary:
        raise RuntimeError(
            "Secondary phenomenon counts do not sum to the number "
            "of instances with secondary coding."
        )

    if len(boolean_summary) != len(BOOLEAN_COLUMNS):
        raise RuntimeError(
            "Boolean diagnostic summary does not contain exactly one row "
            "for every diagnostic dimension."
        )

    if not (
        boolean_summary["n_true"]
        + boolean_summary["n_false"]
        == len(dataframe)
    ).all():
        raise RuntimeError(
            "Boolean diagnostic counts are internally inconsistent."
        )

    if primary_by_domain["n_instances"].sum() != len(dataframe):
        raise RuntimeError(
            "Primary-by-domain counts do not sum to the number "
            "of coded instances."
        )

    if secondary_by_domain["n_instances"].sum() != n_with_secondary:
        raise RuntimeError(
            "Secondary-by-domain counts do not sum to the number "
            "of instances with secondary coding."
        )

    expected_boolean_domain_rows = (
        len(EXPECTED_DOMAINS)
        * len(BOOLEAN_COLUMNS)
    )

    if len(boolean_by_domain) != expected_boolean_domain_rows:
        raise RuntimeError(
            "Boolean-by-domain summary has an unexpected number of rows."
        )

    if len(ambiguity_cases) != int(
        dataframe["annotation_ambiguity"].sum()
    ):
        raise RuntimeError(
            "Annotation-ambiguity case extraction is inconsistent "
            "with the Boolean coding."
        )

    if (
        primary_secondary_cooccurrence["n_instances"].sum()
        != n_with_secondary
    ):
        raise RuntimeError(
            "Primary-secondary co-occurrence counts do not sum to the "
            "number of instances with secondary coding."
        )


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(
    primary_summary: pd.DataFrame,
    secondary_summary: pd.DataFrame,
    boolean_summary: pd.DataFrame,
    primary_by_domain: pd.DataFrame,
    secondary_by_domain: pd.DataFrame,
    boolean_by_domain: pd.DataFrame,
    ambiguity_cases: pd.DataFrame,
    primary_secondary_cooccurrence: pd.DataFrame,
    boolean_cooccurrence: pd.DataFrame,
    coding_summary: pd.DataFrame,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    primary_summary.to_csv(
        PRIMARY_SUMMARY_FILE,
        index=False,
    )

    secondary_summary.to_csv(
        SECONDARY_SUMMARY_FILE,
        index=False,
    )

    boolean_summary.to_csv(
        BOOLEAN_SUMMARY_FILE,
        index=False,
    )

    primary_by_domain.to_csv(
        PRIMARY_BY_DOMAIN_FILE,
        index=False,
    )

    secondary_by_domain.to_csv(
        SECONDARY_BY_DOMAIN_FILE,
        index=False,
    )

    boolean_by_domain.to_csv(
        BOOLEAN_BY_DOMAIN_FILE,
        index=False,
    )

    ambiguity_cases.to_csv(
        AMBIGUITY_CASES_FILE,
        index=False,
    )

    primary_secondary_cooccurrence.to_csv(
        PRIMARY_SECONDARY_COOCCURRENCE_FILE,
        index=False,
    )

    boolean_cooccurrence.to_csv(
        BOOLEAN_COOCCURRENCE_FILE,
        index=False,
    )

    coding_summary.to_csv(
        CODING_SUMMARY_FILE,
        index=False,
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print_header("LOADING QUALITATIVE CODING FILE")

    original_dataframe = load_coding_file(INPUT_FILE)

    print(
        f"Loaded {len(original_dataframe)} rows from "
        f"{INPUT_FILE.relative_to(PROJECT_ROOT)}"
    )

    print_header("VALIDATING FILE SCHEMA")

    validate_required_columns(original_dataframe)

    print("All required columns are present.")

    print_header("NORMALIZING CODING FIELDS")

    dataframe = normalize_coding_dataframe(
        original_dataframe
    )

    dataframe = parse_boolean_columns(dataframe)

    print("Textual coding fields normalized.")
    print("Boolean diagnostic fields parsed.")

    print_header("VALIDATING COMPLETED MANUAL CODING")

    validate_coding_file(
        original_dataframe=original_dataframe,
        dataframe=dataframe,
    )

    print(
        f"Exactly {EXPECTED_N_INSTANCES} unique qualitative instances "
        "are present."
    )
    print("All instances are unanimous model errors.")
    print("All required manual coding fields are complete.")
    print("All Boolean diagnostic fields contain explicit valid values.")
    print("Gold labels, domains, and difficulty groups are valid.")
    print("No duplicate instance IDs were found.")
    print("No category-label capitalization/whitespace collisions were found.")

    print_header("COMPUTING PRIMARY PHENOMENON SUMMARY")

    primary_summary = summarize_primary_phenomena(dataframe)

    print(primary_summary.to_string(index=False))

    print_header("COMPUTING SECONDARY PHENOMENON SUMMARY")

    secondary_summary = summarize_secondary_phenomena(dataframe)

    if secondary_summary.empty:
        print("No secondary phenomena were coded.")
    else:
        print(secondary_summary.to_string(index=False))

    print_header("COMPUTING BOOLEAN DIAGNOSTIC SUMMARY")

    boolean_summary = summarize_boolean_diagnostics(dataframe)

    print(boolean_summary.to_string(index=False))

    print_header("COMPUTING PRIMARY PHENOMENA BY DOMAIN")

    primary_by_domain = summarize_category_by_domain(
        dataframe=dataframe,
        category_column="primary_error_phenomenon",
    )

    print(primary_by_domain.to_string(index=False))

    print_header("COMPUTING SECONDARY PHENOMENA BY DOMAIN")

    secondary_by_domain = summarize_category_by_domain(
        dataframe=dataframe,
        category_column="secondary_error_phenomenon",
    )

    if secondary_by_domain.empty:
        print("No secondary phenomena were coded.")
    else:
        print(secondary_by_domain.to_string(index=False))

    print_header("COMPUTING BOOLEAN DIAGNOSTICS BY DOMAIN")

    boolean_by_domain = summarize_boolean_by_domain(dataframe)

    print(boolean_by_domain.to_string(index=False))

    print_header("EXTRACTING ANNOTATION-AMBIGUITY CASES")

    ambiguity_cases = extract_annotation_ambiguity_cases(dataframe)

    print(
        "Number of instances coded with annotation_ambiguity = TRUE: "
        f"{len(ambiguity_cases)}"
    )

    if not ambiguity_cases.empty:
        print()
        print(
            ambiguity_cases[
                [
                    "id",
                    "domain",
                    "gold",
                    "primary_error_phenomenon",
                    "secondary_error_phenomenon",
                ]
            ].to_string(index=False)
        )

    print_header("COMPUTING PRIMARY-SECONDARY CO-OCCURRENCE")

    primary_secondary_cooccurrence = (
        summarize_primary_secondary_cooccurrence(dataframe)
    )

    if primary_secondary_cooccurrence.empty:
        print("No primary-secondary co-occurrences are available.")
    else:
        print(
            primary_secondary_cooccurrence.to_string(index=False)
        )

    print_header("COMPUTING BOOLEAN DIAGNOSTIC CO-OCCURRENCE")

    boolean_cooccurrence = summarize_boolean_cooccurrence(dataframe)

    if boolean_cooccurrence.empty:
        print("No Boolean diagnostic co-occurrences are present.")
    else:
        print(boolean_cooccurrence.to_string(index=False))

    print_header("CREATING OVERALL CODING SUMMARY")

    coding_summary = create_overall_coding_summary(dataframe)

    print(coding_summary.to_string(index=False))

    print_header("VALIDATING GENERATED SUMMARIES")

    validate_generated_summaries(
        dataframe=dataframe,
        primary_summary=primary_summary,
        secondary_summary=secondary_summary,
        boolean_summary=boolean_summary,
        primary_by_domain=primary_by_domain,
        secondary_by_domain=secondary_by_domain,
        boolean_by_domain=boolean_by_domain,
        ambiguity_cases=ambiguity_cases,
        primary_secondary_cooccurrence=(
            primary_secondary_cooccurrence
        ),
    )

    print("All generated summaries passed internal consistency checks.")

    print_header("SAVING RESULTS")

    save_results(
        primary_summary=primary_summary,
        secondary_summary=secondary_summary,
        boolean_summary=boolean_summary,
        primary_by_domain=primary_by_domain,
        secondary_by_domain=secondary_by_domain,
        boolean_by_domain=boolean_by_domain,
        ambiguity_cases=ambiguity_cases,
        primary_secondary_cooccurrence=(
            primary_secondary_cooccurrence
        ),
        boolean_cooccurrence=boolean_cooccurrence,
        coding_summary=coding_summary,
    )

    for path in [
        PRIMARY_SUMMARY_FILE,
        SECONDARY_SUMMARY_FILE,
        BOOLEAN_SUMMARY_FILE,
        PRIMARY_BY_DOMAIN_FILE,
        SECONDARY_BY_DOMAIN_FILE,
        BOOLEAN_BY_DOMAIN_FILE,
        AMBIGUITY_CASES_FILE,
        PRIMARY_SECONDARY_COOCCURRENCE_FILE,
        BOOLEAN_COOCCURRENCE_FILE,
        CODING_SUMMARY_FILE,
    ]:
        print_saved_file(path)

    print_header("METHODOLOGICAL REMINDER")

    print(
        "The generated frequencies describe only the 16 instances "
        "misclassified by all six model-level prediction vectors."
    )
    print(
        "They must not be reported as frequencies among all benchmark "
        "errors or as prevalence estimates for the complete benchmark."
    )
    print(
        "The qualitative coding was performed by one coder; no "
        "inter-coder agreement statistic is applicable to this analysis."
    )
    print(
        "Annotation-ambiguity coding identifies potentially contestable "
        "or construct-boundary cases and does not automatically invalidate "
        "the benchmark gold labels."
    )

    print_header("Done.")


if __name__ == "__main__":
    main()
    