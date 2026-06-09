import os
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Data
# -----------------------------

models = [
    "SVM",
    "BETO",
    "RoBERTuito",
    "Qwen2.5-7B",
    "GPT-4o-mini",
    "GPT-4.1-mini"
]

# In-domain performance (LGBT)
lgbt_scores = np.array([
    0.595,
    0.653,
    0.695,
    0.731,
    0.707,
    0.756
])

# Unseen-domain performance
# Average of Obesity and Racism
unseen_scores = np.array([
    (0.480 + 0.594) / 2,  # SVM
    (0.531 + 0.616) / 2,  # BETO
    (0.555 + 0.596) / 2,  # RoBERTuito
    (0.559 + 0.479) / 2,  # Qwen2.5-7B
    (0.546 + 0.621) / 2,  # GPT-4o-mini
    (0.656 + 0.734) / 2   # GPT-4.1-mini
])

# Cross-domain variation
drops = lgbt_scores - unseen_scores

# -----------------------------
# Matplotlib configuration
# -----------------------------

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

# -----------------------------
# Figure setup
# -----------------------------

fig, ax = plt.subplots(figsize=(8.5, 5.8))

# Consistent color palette
colors = [
    "#7f7f7f",  # SVM
    "#1f77b4",  # BETO
    "#17becf",  # RoBERTuito
    "#ff7f0e",  # Qwen2.5-7B
    "#9467bd",  # GPT-4o-mini
    "#d62728"   # GPT-4.1-mini
]

# X positions
x_positions = [0, 1]

# -----------------------------
# Plot slope lines
# -----------------------------

for i, model in enumerate(models):

    ax.plot(
        x_positions,
        [lgbt_scores[i], unseen_scores[i]],
        marker='o',
        linewidth=2.5,
        markersize=8,
        color=colors[i],
        alpha=0.95
    )

    # Add model labels on the right side
    ax.text(
        1.03,
        unseen_scores[i],
        f"{model} ({drops[i]:+.3f})",
        va='center',
        fontsize=10,
        color=colors[i]
    )

# -----------------------------
# Axis formatting
# -----------------------------

ax.set_xticks(x_positions)

ax.set_xticklabels([
    "LGBT Domain",
    "Obesity + Racism\nDomain Average"
])

ax.set_ylabel("Macro-F1")

ax.set_ylim(0.45, 0.80)

# Minimal grid
ax.grid(
    axis='y',
    linestyle='--',
    alpha=0.3
)

# Cleaner borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Updated title
ax.set_title(
    "Cross-Domain Performance Variation Across Evaluated Models",
    pad=15
)

# Extra right margin for labels
plt.subplots_adjust(right=0.78)

# -----------------------------
# Create results directory
# -----------------------------

os.makedirs("results", exist_ok=True)

# -----------------------------
# Save figure
# -----------------------------

output_path = "results/robustness_degradation.pdf"

plt.savefig(
    output_path,
    bbox_inches='tight'
)

plt.close()

print(f"Figure saved as {output_path}")
