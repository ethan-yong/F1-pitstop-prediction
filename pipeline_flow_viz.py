"""Summarized pipeline flowchart — temp/analysis.ipynb."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

COLOR_FE = "#2E86AB"
COLOR_PREP = "#A23B72"
COLOR_MODEL = "#F18F01"

STEPS = [
    ("Feature engineering\n→ 947 cols", COLOR_FE),
    ("Stratified train/val split", COLOR_PREP),
    ("5-fold OOF evaluation", COLOR_PREP),
    ("OOF vs holdout comparison", COLOR_PREP),
    ("Train RF, XGBoost, LightGBM", COLOR_MODEL),
    ("Metrics and feature importance", COLOR_MODEL),
]

BOX_W = 0.72
BOX_H = 0.09
X_CENTER = 0.5
Y_START = 0.88
Y_STEP = 0.135

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

for i, (label, color) in enumerate(STEPS):
    y = Y_START - i * Y_STEP
    x = X_CENTER - BOX_W / 2
    box = FancyBboxPatch(
        (x, y - BOX_H),
        BOX_W,
        BOX_H,
        boxstyle="round,pad=0.02,rounding_size=0.015",
        facecolor=color,
        edgecolor="white",
        linewidth=2,
        alpha=0.95,
    )
    ax.add_patch(box)
    ax.text(
        X_CENTER,
        y - BOX_H / 2,
        label,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="white",
    )

    if i < len(STEPS) - 1:
        y_next = Y_START - (i + 1) * Y_STEP
        arrow = FancyArrowPatch(
            (X_CENTER, y - BOX_H - 0.008),
            (X_CENTER, y_next + 0.008),
            arrowstyle="-|>",
            mutation_scale=18,
            color="#555555",
            linewidth=1.8,
        )
        ax.add_patch(arrow)

ax.set_title(
    "F1 Pit-Stop Prediction — Pipeline Overview",
    fontsize=14,
    fontweight="bold",
    pad=16,
)

fig.text(
    0.5,
    0.02,
    "Stratified validation + 5-fold OOF on training set (temp/analysis.ipynb)",
    ha="center",
    fontsize=9,
    color="#555555",
    style="italic",
)

plt.tight_layout()
plt.subplots_adjust(top=0.92, bottom=0.06)

out_path = "pipeline_flow.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")
