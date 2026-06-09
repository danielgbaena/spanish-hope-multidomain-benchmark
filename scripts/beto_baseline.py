import os
import json
import random
import numpy as np
import pandas as pd
import torch

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from sklearn.metrics import (
    f1_score,
    classification_report,
)

# =========================================================
# CONFIG
# =========================================================

MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"

TRAIN_PATH = "data/train.csv"
DEV_PATH = "data/dev.csv"
TEST_PATH = "data/test.csv"
GOLD_PATH = "data/test_gold.csv"

OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

SEEDS = [42, 52, 62]

MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5

LABEL2ID = {
    "nhs": 0,
    "hs": 1,
}

ID2LABEL = {
    0: "nhs",
    1: "hs",
}

# =========================================================
# REPRODUCIBILITY
# =========================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =========================================================
# LOAD DATA
# =========================================================

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

# =========================================================
# LABELS
# =========================================================

train_df["label"] = train_df["category"].map(LABEL2ID)
dev_df["label"] = dev_df["category"].map(LABEL2ID)
gold_df["label"] = gold_df["category"].map(LABEL2ID)

# =========================================================
# TOKENIZER
# =========================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):

    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=MAX_LENGTH,
    )

# =========================================================
# METRICS
# =========================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
    )

    return {
        "macro_f1": macro_f1
    }

# =========================================================
# MULTI-SEED TRAINING
# =========================================================

all_predictions = []

all_macro_f1 = []

all_domain_scores = {
    "lgbt": [],
    "obesity": [],
    "racism": [],
}

for seed in SEEDS:

    print("\n========================================")
    print(f"Running seed {seed}")
    print("========================================")

    set_seed(seed)

    train_dataset = Dataset.from_pandas(
        train_df[["text", "label"]]
    )

    dev_dataset = Dataset.from_pandas(
        dev_df[["text", "label"]]
    )

    test_dataset = Dataset.from_pandas(
        pd.DataFrame({
            "text": test_df["text"],
            "label": gold_df["label"],
        })
    )

    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
    )

    dev_dataset = dev_dataset.map(
        tokenize_function,
        batched=True,
    )

    test_dataset = test_dataset.map(
        tokenize_function,
        batched=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir=f"tmp/beto_seed_{seed}",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    predictions_output = trainer.predict(test_dataset)

    logits = predictions_output.predictions

    preds = np.argmax(logits, axis=-1)

    all_predictions.append(preds)

    pred_labels = [
        ID2LABEL[p]
        for p in preds
    ]

    gold_labels = gold_df["category"].tolist()

    macro_f1 = f1_score(
        gold_labels,
        pred_labels,
        average="macro",
    )

    all_macro_f1.append(macro_f1)

    print(f"\nMacro-F1: {macro_f1:.4f}")

    print("\nClassification report:\n")

    print(
        classification_report(
            gold_labels,
            pred_labels,
            digits=4,
        )
    )

    print("\nPer-domain results")

    for domain in ["lgbt", "obesity", "racism"]:

        subset = gold_df[
            gold_df["topic"] == domain
        ]

        subset_preds = [
            pred_labels[i]
            for i in subset.index
        ]

        domain_f1 = f1_score(
            subset["category"],
            subset_preds,
            average="macro",
        )

        all_domain_scores[domain].append(domain_f1)

        print(f"{domain}: {domain_f1:.4f}")

# =========================================================
# MAJORITY VOTING
# =========================================================

print("\n========================================")
print("Majority voting...")
print("========================================")

all_predictions = np.array(all_predictions)

final_predictions = []

for i in range(len(test_df)):

    votes = all_predictions[:, i]

    majority_vote = np.bincount(votes).argmax()

    final_predictions.append(majority_vote)

final_labels = [
    ID2LABEL[p]
    for p in final_predictions
]

# =========================================================
# FINAL METRICS
# =========================================================

final_macro_f1 = f1_score(
    gold_df["category"],
    final_labels,
    average="macro",
)

print("\n========================================")
print("FINAL MULTI-SEED RESULTS")
print("========================================")

print(
    f"\nOverall Macro-F1: "
    f"{np.mean(all_macro_f1):.4f} "
    f"± {np.std(all_macro_f1):.4f}"
)

print("\nPer-domain Macro-F1")
print("-------------------")

for domain in ["lgbt", "obesity", "racism"]:

    scores = all_domain_scores[domain]

    print(
        f"{domain}: "
        f"{np.mean(scores):.4f} "
        f"± {np.std(scores):.4f}"
    )

# =========================================================
# SAVE PREDICTIONS
# =========================================================

predictions_df = pd.DataFrame({
    "id": gold_df["id"],
    "gold": gold_df["category"],
    "prediction": final_labels,
    "topic": gold_df["topic"],
})

predictions_path = os.path.join(
    OUTPUT_DIR,
    "beto_predictions.csv",
)

predictions_df.to_csv(
    predictions_path,
    index=False,
)

# =========================================================
# SAVE RESULTS
# =========================================================

results = {
    "overall_macro_f1_mean": float(np.mean(all_macro_f1)),
    "overall_macro_f1_std": float(np.std(all_macro_f1)),
    "per_domain": {
        domain: {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
        }
        for domain, scores in all_domain_scores.items()
    },
}

results_path = os.path.join(
    OUTPUT_DIR,
    "beto_results.json",
)

with open(results_path, "w") as f:
    json.dump(results, f, indent=4)

# =========================================================
# DONE
# =========================================================

print("\n========================================")
print("Experiment completed successfully.")
print("========================================")
