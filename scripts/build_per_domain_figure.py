import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# LOAD RESULTS
# =========================================================

df = pd.read_csv(
    "results/final_results_summary.csv"
)

# =========================================================
# SORT MODELS
# =========================================================

df = df.sort_values(
    by="overall_macro_f1_mean",
    ascending=False
)

# =========================================================
# DATA
# =========================================================

models = df["model"].tolist()

lgbt_scores = df["lgbt_mean"].tolist()
obesity_scores = df["obesity_mean"].tolist()
racism_scores = df["racism_mean"].tolist()

# =========================================================
# POSITIONS
# =========================================================

x = np.arange(len(models))

width = 0.25

# =========================================================
# FIGURE
# =========================================================

plt.figure(figsize=(12, 6))

plt.bar(
    x - width,
    lgbt_scores,
    width,
    label="LGBT"
)

plt.bar(
    x,
    obesity_scores,
    width,
    label="Obesity"
)

plt.bar(
    x + width,
    racism_scores,
    width,
    label="Racism"
)

# =========================================================
# AXES
# =========================================================

plt.ylabel("Macro-F1")
plt.xlabel("Model")

plt.title(
    "Per-domain Macro-F1 Performance Across Models"
)

plt.xticks(
    x,
    models,
    rotation=0
)

# =========================================================
# Y LIMITS
# =========================================================

plt.ylim(0.0, 0.9)

# =========================================================
# LEGEND
# =========================================================

plt.legend()

# =========================================================
# GRID
# =========================================================

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

# =========================================================
# LAYOUT
# =========================================================

plt.tight_layout()

# =========================================================
# SAVE FIGURE
# =========================================================

output_path = (
    "results/per_domain_results_figure.pdf"
)

plt.savefig(
    output_path,
    bbox_inches="tight"
)

print("\n========================================")
print("Figure generated successfully.")
print("========================================")

print(f"\nSaved to:\n{output_path}")

print("\n========================================")
print("Done.")
print("========================================")
