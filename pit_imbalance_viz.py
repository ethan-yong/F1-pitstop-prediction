"""PitNextLap class distribution — matches model_comparison_viz style.

Proportions from temp/analysis.ipynb train[\"PitNextLap\"].value_counts(normalize=True).
"""

import matplotlib.pyplot as plt
import numpy as np

# Shared palette with model_comparison_viz.py
COLOR_NO_PIT = "#2E86AB"  # steel blue (Random Forest)
COLOR_PIT = "#F18F01"      # orange (LightGBM)

labels = ["No pit next lap (0)", "Pit next lap (1)"]
proportions = [0.801018, 0.198982]  # ~80.1% / ~19.9%
colors = [COLOR_NO_PIT, COLOR_PIT]

fig, ax = plt.subplots(figsize=(12, 6.5))

x = np.arange(len(labels))
bars = ax.bar(
    x,
    proportions,
    width=0.55,
    color=colors,
    edgecolor="white",
    linewidth=0.8,
)

for bar, prop in zip(bars, proportions):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{prop * 100:.1f}%",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

ax.set_ylabel("Proportion of laps", fontsize=11)
ax.set_title(
    "F1 Pit-Stop Prediction — Target Distribution (Train Set)",
    fontsize=13,
    fontweight="bold",
    pad=12,
)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_yticks(np.arange(0, 1.01, 0.2))

ax.axhline(
    0.8,
    color="#888888",
    linestyle="--",
    linewidth=1.2,
    alpha=0.9,
    label="80% baseline (always-no-pit accuracy)",
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.35)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.08),
    ncol=2,
    frameon=True,
    framealpha=0.95,
    fontsize=10,
)

fig.text(
    0.5,
    0.01,
    "~4× more non-pit laps — high accuracy can hide poor pit-stop detection",
    ha="center",
    fontsize=9,
    color="#555555",
    style="italic",
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.2)

out_path = "pit_imbalance.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")
