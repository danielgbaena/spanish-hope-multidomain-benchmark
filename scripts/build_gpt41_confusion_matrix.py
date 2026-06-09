import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

# =========================================================
# LOAD PREDICTIONS
# =========================================================

df = pd.read_csv(
    "results/gpt41_mini_predictions.csv"
)

# =========================================================
# CONFUSION MATRIX
# =========================================================

labels = ["nhs", "hs"]

cm = confusion_matrix(
    df["gold"],
    df["prediction"],
    labels=labels
)

# =========================================================
# FIGURE
# =========================================================

fig, ax = plt.subplots(figsize=(6, 6))

im = ax.imshow(cm)

# =========================================================
# AXES
# =========================================================

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(["NHS", "HS"])
ax.set_yticklabels(["NHS", "HS"])

ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")

ax.set_title(
    "GPT-4.1-mini Overall Confusion Matrix"
)

# =========================================================
# CELL VALUES
# =========================================================

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):

        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            fontsize=18
        )

# =========================================================
# COLORBAR
# =========================================================

plt.colorbar(im)

# =========================================================
# LAYOUT
# =========================================================

plt.tight_layout()

# =========================================================
# SAVE FIGURE
# =========================================================

output_path = (
    "results/gpt41_confusion_matrix.pdf"
)

plt.savefig(
    output_path,
    bbox_inches="tight"
)

print("\n========================================")
print("Confusion matrix generated successfully.")
print("========================================")

print(f"\nSaved to:\n{output_path}")

print("\n========================================")
print("Done.")
print("========================================")
