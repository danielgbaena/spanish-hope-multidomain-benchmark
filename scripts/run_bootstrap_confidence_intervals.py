import os
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score

# =========================================================
# CONFIG
# =========================================================

RESULTS_DIR = "results"

N_BOOTSTRAPS = 1000
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# =========================================================
# MODELS TO ANALYZE
# =========================================================

MODELS = {
    "SVM":
        "svm_predictions.csv",

    "BETO":
        "beto_majority_predictions.csv",

    "RoBERTuito":
        "robertuito_majority_predictions.csv",

    "Qwen2.5-7B":
        "qwen25_7b_predictions.csv",

    "GPT-4o-mini":
        "gpt4o_mini_predictions.csv",

    "GPT-4.1-mini":
        "gpt41_mini_predictions.csv",
}

# =========================================================
# BOOTSTRAP FUNCTION
# =========================================================

def bootstrap_macro_f1(
    gold_labels,
    pred_labels,
    n_bootstraps=1000
):

    scores = []

    n_samples = len(gold_labels)

    gold_labels = np.array(gold_labels)
    pred_labels = np.array(pred_labels)

    for _ in range(n_bootstraps):

        indices = np.random.choice(
            np.arange(n_samples),
            size=n_samples,
            replace=True
        )

        gold_sample = gold_labels[indices]
        pred_sample = pred_labels[indices]

        score = f1_score(
            gold_sample,
            pred_sample,
            average="macro"
        )

        scores.append(score)

    scores = np.array(scores)

    mean_score = np.mean(scores)

    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)

    return mean_score, lower, upper


# =========================================================
# RUN ANALYSIS
# =========================================================

results = []

print("\n========================================")
print("BOOTSTRAP CONFIDENCE INTERVALS")
print("========================================")

for model_name, filename in MODELS.items():

    print(f"\nProcessing {model_name}...")

    file_path = os.path.join(
        RESULTS_DIR,
        filename
    )

    if not os.path.exists(file_path):
        print(f"WARNING: File not found -> {file_path}")
        continue

    df = pd.read_csv(file_path)

    gold = df["gold"]
    preds = df["prediction"]

    macro_f1 = f1_score(
        gold,
        preds,
        average="macro"
    )

    mean_score, lower_ci, upper_ci = (
        bootstrap_macro_f1(
            gold,
            preds,
            n_bootstraps=N_BOOTSTRAPS
        )
    )

    results.append({
        "model": model_name,
        "macro_f1": macro_f1,
        "bootstrap_mean": mean_score,
        "ci_95_lower": lower_ci,
        "ci_95_upper": upper_ci,
    })

    print(
        f"Macro-F1: {macro_f1:.4f}"
    )

    print(
        f"95% CI: "
        f"[{lower_ci:.4f}, {upper_ci:.4f}]"
    )

# =========================================================
# SAVE RESULTS
# =========================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="macro_f1",
    ascending=False
)

output_path = os.path.join(
    RESULTS_DIR,
    "bootstrap_confidence_intervals.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

# =========================================================
# DISPLAY
# =========================================================

print("\n========================================")
print("FINAL RESULTS")
print("========================================")

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nSaved to:")
print(output_path)

print("\n========================================")
print("Done.")
print("========================================")
