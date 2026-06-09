import os
import json
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report,
    f1_score,
)

# ========================================
# CONFIG
# ========================================

TRAIN_PATH = "data/train.csv"
DEV_PATH = "data/dev.csv"
TEST_PATH = "data/test.csv"
GOLD_PATH = "data/test_gold.csv"

OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================================
# LOAD DATA
# ========================================

print("\n========================================")
print("Loading datasets...")
print("========================================")

train_df = pd.read_csv(TRAIN_PATH)
dev_df = pd.read_csv(DEV_PATH)
test_df = pd.read_csv(TEST_PATH)
gold_df = pd.read_csv(GOLD_PATH)

print(f"Train samples: {len(train_df)}")
print(f"Dev samples:   {len(dev_df)}")
print(f"Test samples:  {len(test_df)}")

# ========================================
# PREPARE DATA
# ========================================

X_train = train_df["text"].astype(str)
y_train = train_df["category"]

X_dev = dev_df["text"].astype(str)
y_dev = dev_df["category"]

X_test = test_df["text"].astype(str)
y_test = gold_df["category"]

test_domains = gold_df["topic"]

# ========================================
# MODEL
# ========================================

print("\n========================================")
print("Training SVM...")
print("========================================")

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=20000,
        ),
    ),
    (
        "clf",
        LinearSVC(),
    ),
])

model.fit(X_train, y_train)

# ========================================
# PREDICTIONS
# ========================================

print("\n========================================")
print("Evaluating...")
print("========================================")

predictions = model.predict(X_test)

# ========================================
# OVERALL METRICS
# ========================================

macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro",
)

print(f"\nMacro-F1: {macro_f1:.4f}")

print("\nClassification report:\n")
print(classification_report(y_test, predictions))

# ========================================
# PER-DOMAIN RESULTS
# ========================================

print("\n========================================")
print("Per-domain results")
print("========================================")

domain_results = {}

for domain in sorted(test_domains.unique()):

    mask = test_domains == domain

    domain_f1 = f1_score(
        y_test[mask],
        predictions[mask],
        average="macro",
    )

    domain_results[domain] = domain_f1

    print(f"{domain}: {domain_f1:.4f}")

# ========================================
# SAVE PREDICTIONS
# ========================================

predictions_df = pd.DataFrame({
    "id": test_df["id"],
    "gold": y_test,
    "prediction": predictions,
    "topic": test_domains,
})

predictions_path = os.path.join(
    OUTPUT_DIR,
    "svm_predictions.csv",
)

predictions_df.to_csv(
    predictions_path,
    index=False,
)

# ========================================
# SAVE METRICS
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
    "svm_results.json",
)

with open(results_path, "w") as f:
    json.dump(results, f, indent=4)

# ========================================
# DONE
# ========================================

print("\n========================================")
print("Experiment completed successfully.")
print("========================================")
