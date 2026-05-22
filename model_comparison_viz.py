"""Generate model comparison chart from validation metrics.

Holdout metrics from temp/analysis.ipynb (stratified 80/20 validation split).
"""

import matplotlib.pyplot as plt
import numpy as np

models = ["Random Forest", "XGBoost", "LightGBM"]
colors = ["#2E86AB", "#A23B72", "#F18F01"]

# Holdout validation (classification_report / roc_auc_score on X_valid)
metrics = {
    "Accuracy": [0.89, 0.86, 0.86],
    "Pit Recall": [0.67, 0.90, 0.91],
    "Pit F1": [0.71, 0.73, 0.72],
    "ROC-AUC": [np.nan, 0.9440943164661366, 0.9439334835459454],
}


def _format_bar_label(value: float, metric_name: str) -> str:
    if metric_name == "ROC-AUC":
        return f"{value:.3f}"
    return f"{value:.2f}"

Y_MIN = 0.55
Y_MAX = 1.02

metric_names = list(metrics.keys())
x = np.arange(len(metric_names))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6.5))

for i, (model, color) in enumerate(zip(models, colors)):
    values = [metrics[m][i] for m in metric_names]
    offset = (i - 1) * width
    bars = ax.bar(
        x + offset,
        values,
        width,
        label=model,
        color=color,
        edgecolor="white",
        linewidth=0.8,
    )
    for bar, val, metric_name in zip(bars, values, metric_names):
        if np.isnan(val):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.008,
            _format_bar_label(val, metric_name),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

# Mark RF missing ROC-AUC
roc_idx = metric_names.index("ROC-AUC")
rf_offset = -width
ax.text(
    x[roc_idx] + rf_offset,
    Y_MIN + 0.02,
    "N/A",
    ha="center",
    va="bottom",
    fontsize=10,
    color="#666666",
    style="italic",
)

ax.set_ylabel("Score", fontsize=11)
ax.set_title(
    "F1 Pit-Stop Prediction — Model Comparison (Validation Set)",
    fontsize=13,
    fontweight="bold",
    pad=12,
)
ax.set_xticks(x)
ax.set_xticklabels(metric_names, fontsize=11)
ax.set_ylim(Y_MIN, Y_MAX)

# Truncation indicator on y-axis
ax.set_yticks(np.arange(0.6, 1.01, 0.1))
ax.text(
    -0.08,
    Y_MIN,
    f"▼ {Y_MIN}",
    transform=ax.get_yaxis_transform(),
    ha="right",
    va="center",
    fontsize=9,
    color="#666666",
)

ax.axhline(
    0.8,
    color="#888888",
    linestyle="--",
    linewidth=1.2,
    alpha=0.9,
    label="80% baseline (always-no-pit)",
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.35)

# Legend below chart — no overlap with bars
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.08),
    ncol=4,
    frameon=True,
    framealpha=0.95,
    fontsize=10,
)

fig.text(
    0.5,
    0.01,
    "Bars start at 0.55 (truncated axis) to highlight differences between models",
    ha="center",
    fontsize=9,
    color="#555555",
    style="italic",
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.22)

out_path = "model_comparison.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")
