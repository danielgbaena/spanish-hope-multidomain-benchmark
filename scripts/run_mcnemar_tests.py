import os
import pandas as pd

from statsmodels.stats.contingency_tables import mcnemar

# =========================================================
# CONFIG
# =========================================================

RESULTS_DIR = "results"

# =========================================================
# LOAD PREDICTIONS
# =========================================================

beto = pd.read_csv(
    os.path.join(
        RESULTS_DIR,
        "beto_majority_predictions.csv"
    )
)

robertuito = pd.read_csv(
    os.path.join(
        RESULTS_DIR,
        "robertuito_majority_predictions.csv"
    )
)

gpt4o = pd.read_csv(
    os.path.join(
        RESULTS_DIR,
        "gpt4o_mini_predictions.csv"
    )
)

gpt41 = pd.read_csv(
    os.path.join(
        RESULTS_DIR,
        "gpt41_mini_predictions.csv"
    )
)

qwen = pd.read_csv(
    os.path.join(
        RESULTS_DIR,
        "qwen25_7b_predictions.csv"
    )
)

# =========================================================
# HELPER FUNCTION
# =========================================================

def compute_mcnemar(
    df1,
    df2,
    model1_name,
    model2_name
):

    correct1 = (
        df1["prediction"] == df1["gold"]
    )

    correct2 = (
        df2["prediction"] == df2["gold"]
    )

    both_correct = (
        (correct1 == True) &
        (correct2 == True)
    ).sum()

    model1_correct_only = (
        (correct1 == True) &
        (correct2 == False)
    ).sum()

    model2_correct_only = (
        (correct1 == False) &
        (correct2 == True)
    ).sum()

    both_wrong = (
        (correct1 == False) &
        (correct2 == False)
    ).sum()

    table = [
        [both_correct, model1_correct_only],
        [model2_correct_only, both_wrong]
    ]

    result = mcnemar(
        table,
        exact=False,
        correction=True
    )

    return {
        "comparison":
            f"{model1_name} vs {model2_name}",

        "model1_correct_only":
            model1_correct_only,

        "model2_correct_only":
            model2_correct_only,

        "p_value":
            result.pvalue,
    }

# =========================================================
# RUN TESTS
# =========================================================

comparisons = []

comparisons.append(
    compute_mcnemar(
        gpt41,
        beto,
        "GPT-4.1-mini",
        "BETO"
    )
)

comparisons.append(
    compute_mcnemar(
        gpt41,
        robertuito,
        "GPT-4.1-mini",
        "RoBERTuito"
    )
)

comparisons.append(
    compute_mcnemar(
        gpt41,
        gpt4o,
        "GPT-4.1-mini",
        "GPT-4o-mini"
    )
)

comparisons.append(
    compute_mcnemar(
        beto,
        robertuito,
        "BETO",
        "RoBERTuito"
    )
)

comparisons.append(
    compute_mcnemar(
        gpt4o,
        qwen,
        "GPT-4o-mini",
        "Qwen2.5-7B"
    )
)

# =========================================================
# RESULTS DATAFRAME
# =========================================================

results_df = pd.DataFrame(comparisons)

# =========================================================
# SAVE CSV
# =========================================================

output_path = os.path.join(
    RESULTS_DIR,
    "mcnemar_results.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

# =========================================================
# DISPLAY
# =========================================================

print("\n========================================")
print("MCNEMAR SIGNIFICANCE TESTS")
print("========================================")

for _, row in results_df.iterrows():

    p = row["p_value"]

    significance = (
        "SIGNIFICANT"
        if p < 0.05
        else "NOT SIGNIFICANT"
    )

    print(
        f"\n{row['comparison']}"
    )

    print(
        f"p-value: {p:.6f}"
    )

    print(
        f"Result: {significance}"
    )

print("\n========================================")
print("Saved results to:")
print(output_path)

print("\n========================================")
print("Done.")
print("========================================")
