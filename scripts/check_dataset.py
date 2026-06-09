import pandas as pd

print("\n========================================")
print("Loading datasets...")
print("========================================")

train_df = pd.read_csv("data/train.csv")
dev_df = pd.read_csv("data/dev.csv")
test_df = pd.read_csv("data/test.csv")
gold_df = pd.read_csv("data/test_gold.csv")

# ========================================
# BASIC INFO
# ========================================

print("\nTRAIN")
print("----------------------------------------")
print("Shape:", train_df.shape)
print("Columns:", train_df.columns.tolist())

print("\nCategory distribution:")
print(train_df["category"].value_counts())

print("\nTopic distribution:")
print(train_df["topic"].value_counts())

# ========================================

print("\nDEV")
print("----------------------------------------")
print("Shape:", dev_df.shape)
print("Columns:", dev_df.columns.tolist())

print("\nCategory distribution:")
print(dev_df["category"].value_counts())

print("\nTopic distribution:")
print(dev_df["topic"].value_counts())

# ========================================

print("\nTEST (blind)")
print("----------------------------------------")
print("Shape:", test_df.shape)
print("Columns:", test_df.columns.tolist())

# ========================================

print("\nTEST GOLD")
print("----------------------------------------")
print("Shape:", gold_df.shape)
print("Columns:", gold_df.columns.tolist())

print("\nCategory distribution:")
print(gold_df["category"].value_counts())

print("\nTopic distribution:")
print(gold_df["topic"].value_counts())

# ========================================
# DATA INTEGRITY CHECKS
# ========================================

print("\n========================================")
print("Integrity checks")
print("========================================")

assert len(test_df) == len(gold_df), \
    "Mismatch between test.csv and test_gold.csv"

assert all(test_df["id"] == gold_df["id"]), \
    "IDs do not match between test.csv and test_gold.csv"

print("✓ test.csv and test_gold.csv are aligned")

# ========================================

print("\n========================================")
print("Dataset check completed successfully.")
print("========================================")
