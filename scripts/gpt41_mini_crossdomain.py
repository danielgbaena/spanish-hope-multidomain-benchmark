import os
import json
import time
import pandas as pd

from collections import Counter

from sklearn.metrics import (
    classification_report,
    f1_score,
)

from openai import OpenAI

# ========================================
# CONFIG
# ========================================

MODEL_NAME = "gpt-4.1-mini"

TEST_PATH = "data/test.csv"
GOLD_PATH = "data/test_gold.csv"

OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================================
# OPENAI CLIENT
# ========================================

client = OpenAI()

# ========================================
# LOAD DATA
# ========================================

print("\n========================================")
print("Loading datasets...")
print("========================================")

test_df = pd.read_csv(TEST_PATH)
gold_df = pd.read_csv(GOLD_PATH)

print(f"Test samples: {len(test_df)}")

# ========================================
# PROMPT
# ========================================

SYSTEM_PROMPT = """
You are an expert annotator for Hope Speech detection.

Classify the given Spanish social media text into ONE label only:

- "Hope Speech"
- "Non Hope Speech"

A text is considered "Hope Speech" if it:
1. Explicitly supports the social integration of minorities
2. Positively encourages vulnerable communities
3. Promotes empathy, solidarity, tolerance, or inclusion
4. Provides emotional support or encouragement

A text is considered "Non Hope Speech" if it:
1. Does not express supportive intent
2. Promotes hostility or discrimination
3. Uses insults or offensive language
4. Is neutral/informational without supportive meaning

Return ONLY:
- Hope Speech
OR
- Non Hope Speech
"""

# ========================================
# PREDICTION FUNCTION
# ========================================

def predict(text):

    try:

        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
        )

        output = (
            response.choices[0]
            .message.content
            .strip()
        )

        if "Hope Speech" in output and "Non" not in output:
            return "hs"

        return "nhs"

    except Exception as e:

        print(f"Error: {e}")

        time.sleep(5)

        return "nhs"

# ========================================
# RUN INFERENCE
# ========================================

print("\n========================================")
print("Running inference...")
print("========================================")

predictions = []

for idx, row in test_df.iterrows():

    if idx % 25 == 0:
        print(f"Processed {idx}/{len(test_df)}")

    text = str(row["text"])

    pred = predict(text)

    predictions.append(pred)

# ========================================
# EVALUATION
# ========================================

gold_labels = gold_df["category"].tolist()

macro_f1 = f1_score(
    gold_labels,
    predictions,
    average="macro",
)

print("\n========================================")
print("FINAL RESULTS")
print("========================================")

print(f"\nMacro-F1: {macro_f1:.4f}")

print("\nClassification report:\n")

print(
    classification_report(
        gold_labels,
        predictions,
        digits=4,
    )
)

print("\nPrediction distribution:")

print(
    Counter(predictions)
)

# ========================================
# PER-DOMAIN RESULTS
# ========================================

print("\n========================================")
print("Per-domain results")
print("========================================")

domain_results = {}

for domain in ["lgbt", "obesity", "racism"]:

    subset = gold_df[
        gold_df["topic"] == domain
    ]

    subset_preds = [
        predictions[i]
        for i in subset.index
    ]

    domain_f1 = f1_score(
        subset["category"],
        subset_preds,
        average="macro",
    )

    domain_results[domain] = domain_f1

    print(f"{domain}: {domain_f1:.4f}")

# ========================================
# SAVE PREDICTIONS
# ========================================

predictions_df = pd.DataFrame({
    "id": gold_df["id"],
    "gold": gold_df["category"],
    "prediction": predictions,
    "topic": gold_df["topic"],
})

predictions_path = os.path.join(
    OUTPUT_DIR,
    "gpt41_mini_predictions.csv",
)

predictions_df.to_csv(
    predictions_path,
    index=False,
)

# ========================================
# SAVE RESULTS
# ========================================

results = {
    "macro_f1": float(macro_f1),
    "per_domain": {
        k: float(v)
        for k, v in domain_results.items()
    },
}

results_path = os.path.join(
    OUTPUT_DIR,
    "gpt41_mini_results.json",
)

with open(results_path, "w") as f:
    json.dump(results, f, indent=4)

# ========================================
# DONE
# ========================================

print("\n========================================")
print("Experiment completed successfully.")
print("========================================")
