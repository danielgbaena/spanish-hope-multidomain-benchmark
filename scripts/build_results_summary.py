import os
import json
import pandas as pd

# =========================================================
# CONFIG
# =========================================================

RESULTS_DIR = "results"

OUTPUT_CSV = os.path.join(
    RESULTS_DIR,
    "final_results_summary.csv"
)

# =========================================================
# LOAD RESULTS
# =========================================================

rows = []

# ---------------------------------------------------------
# SVM
# ---------------------------------------------------------

with open(
    os.path.join(RESULTS_DIR, "svm_results.json")
) as f:

    svm = json.load(f)

rows.append({
    "model": "SVM",
    "overall_macro_f1_mean": svm["macro_f1"],
    "overall_macro_f1_std": 0.0,
    "lgbt_mean": svm["per_domain"]["lgbt"],
    "lgbt_std": 0.0,
    "obesity_mean": svm["per_domain"]["obesity"],
    "obesity_std": 0.0,
    "racism_mean": svm["per_domain"]["racism"],
    "racism_std": 0.0,
})

# ---------------------------------------------------------
# BETO MULTISEED
# ---------------------------------------------------------

with open(
    os.path.join(RESULTS_DIR, "beto_multiseed_results.json")
) as f:

    beto = json.load(f)

rows.append({
    "model": "BETO",
    "overall_macro_f1_mean":
        beto["overall_macro_f1_mean"],

    "overall_macro_f1_std":
        beto["overall_macro_f1_std"],

    "lgbt_mean":
        beto["per_domain"]["lgbt"]["mean"],

    "lgbt_std":
        beto["per_domain"]["lgbt"]["std"],

    "obesity_mean":
        beto["per_domain"]["obesity"]["mean"],

    "obesity_std":
        beto["per_domain"]["obesity"]["std"],

    "racism_mean":
        beto["per_domain"]["racism"]["mean"],

    "racism_std":
        beto["per_domain"]["racism"]["std"],
})

# ---------------------------------------------------------
# ROBERTUITO MULTISEED
# ---------------------------------------------------------

with open(
    os.path.join(
        RESULTS_DIR,
        "robertuito_multiseed_results.json"
    )
) as f:

    robertuito = json.load(f)

rows.append({
    "model": "RoBERTuito",

    "overall_macro_f1_mean":
        robertuito["overall_macro_f1_mean"],

    "overall_macro_f1_std":
        robertuito["overall_macro_f1_std"],

    "lgbt_mean":
        robertuito["per_domain"]["lgbt"]["mean"],

    "lgbt_std":
        robertuito["per_domain"]["lgbt"]["std"],

    "obesity_mean":
        robertuito["per_domain"]["obesity"]["mean"],

    "obesity_std":
        robertuito["per_domain"]["obesity"]["std"],

    "racism_mean":
        robertuito["per_domain"]["racism"]["mean"],

    "racism_std":
        robertuito["per_domain"]["racism"]["std"],
})

# ---------------------------------------------------------
# GPT-4o-mini
# ---------------------------------------------------------

with open(
    os.path.join(
        RESULTS_DIR,
        "gpt4o_mini_results.json"
    )
) as f:

    gpt4o = json.load(f)

rows.append({
    "model": "GPT-4o-mini",

    "overall_macro_f1_mean":
        gpt4o["macro_f1"],

    "overall_macro_f1_std": 0.0,

    "lgbt_mean":
        gpt4o["per_domain"]["lgbt"],

    "lgbt_std": 0.0,

    "obesity_mean":
        gpt4o["per_domain"]["obesity"],

    "obesity_std": 0.0,

    "racism_mean":
        gpt4o["per_domain"]["racism"],

    "racism_std": 0.0,
})

# ---------------------------------------------------------
# GPT-4.1-mini
# ---------------------------------------------------------

with open(
    os.path.join(
        RESULTS_DIR,
        "gpt41_mini_results.json"
    )
) as f:

    gpt41 = json.load(f)

rows.append({
    "model": "GPT-4.1-mini",

    "overall_macro_f1_mean":
        gpt41["macro_f1"],

    "overall_macro_f1_std": 0.0,

    "lgbt_mean":
        gpt41["per_domain"]["lgbt"],

    "lgbt_std": 0.0,

    "obesity_mean":
        gpt41["per_domain"]["obesity"],

    "obesity_std": 0.0,

    "racism_mean":
        gpt41["per_domain"]["racism"],

    "racism_std": 0.0,
})

# ---------------------------------------------------------
# QWEN2.5-7B
# ---------------------------------------------------------

with open(
    os.path.join(
        RESULTS_DIR,
        "qwen25_7b_results.json"
    )
) as f:

    qwen = json.load(f)

rows.append({
    "model": "Qwen2.5-7B",

    "overall_macro_f1_mean":
        qwen["macro_f1"],

    "overall_macro_f1_std": 0.0,

    "lgbt_mean":
        qwen["per_domain"]["lgbt"],

    "lgbt_std": 0.0,

    "obesity_mean":
        qwen["per_domain"]["obesity"],

    "obesity_std": 0.0,

    "racism_mean":
        qwen["per_domain"]["racism"],

    "racism_std": 0.0,
})

# =========================================================
# BUILD DATAFRAME
# =========================================================

df = pd.DataFrame(rows)

# =========================================================
# SORT BY PERFORMANCE
# =========================================================

df = df.sort_values(
    by="overall_macro_f1_mean",
    ascending=False,
)

# =========================================================
# SAVE CSV
# =========================================================

df.to_csv(
    OUTPUT_CSV,
    index=False,
)

# =========================================================
# DISPLAY
# =========================================================

print("\n========================================")
print("FINAL RESULTS SUMMARY")
print("========================================")

print(df)

print("\nSaved to:")
print(OUTPUT_CSV)

print("\n========================================")
print("Done.")
print("========================================")
