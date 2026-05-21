# XGBoost OOF + Threshold Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a self-contained notebook section below the existing XGBoost evaluation that produces out-of-fold (OOF) probabilities on the training split, tunes the classification threshold on those OOF scores, and reports validation metrics using the tuned threshold—without modifying any existing cells.

**Architecture:** Use 5-fold `StratifiedKFold` on `(X_train, y_train)` only. Each fold trains a fresh `XGBClassifier` with the same hyperparameters as `xgb_model` in cell 20. Concatenate fold validation probabilities into `oof_probs`. Scan thresholds on OOF predictions to maximize F1 for class 1 (pit-stop laps). Apply `best_threshold` to the already-fitted `xgb_model` predictions on `X_valid` from the existing cell—no retraining of `xgb_model`. All logic lives in new cells inserted immediately after cell 21 (XGBoost evaluation).

**Tech Stack:** Python 3.11, Jupyter (`analysis.ipynb`), pandas, numpy, scikit-learn (`StratifiedKFold`, metrics), XGBoost (`XGBClassifier`)

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `analysis.ipynb` | Modify (append only) | Insert 2 new cells after cell 21; zero edits to existing cells |
| `docs/superpowers/plans/2026-05-20-xgboost-oof-threshold-tuning.md` | Reference | This plan |

**Hard constraint:** Do not edit cells 0–21 or any cell after the insertion point except by inserting new cells at index 22 (pushing later cells down).

**Prerequisites (must already be executed):** Cells through 15 (`X_train`, `X_valid`, `y_train`, `y_valid`), cell 20 (`xgb_model.fit`), cell 21 (optional baseline prints).

---

### Task 1: Insert markdown header cell

**Files:**
- Modify: `analysis.ipynb` (insert new cell at index **22**)

- [ ] **Step 1: Add markdown cell**

Insert a new markdown cell at index 22 with this exact content:

```markdown
## XGBoost OOF + Threshold Tuning

Out-of-fold (OOF) predictions are generated on `X_train` with 5-fold stratified CV using the same XGBoost hyperparameters as above. The decision threshold is tuned on OOF probabilities to maximize **F1 (class 1)**. The tuned threshold is then applied to `xgb_model` probabilities on `X_valid` (model unchanged).
```

- [ ] **Step 2: Verify notebook structure**

Run in repo root:

```bash
python -c "
import json
nb = json.load(open('analysis.ipynb', encoding='utf-8'))
assert 'OOF + Threshold Tuning' in ''.join(nb['cells'][22]['source'])
print('OK: markdown cell at index 22')
"
```

Expected: `OK: markdown cell at index 22`

---

### Task 2: Insert OOF + threshold tuning code cell

**Files:**
- Modify: `analysis.ipynb` (insert new code cell at index **23**, immediately below Task 1 markdown)

- [ ] **Step 1: Add the code cell**

Insert a new code cell at index 23 with this exact content:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

# --- Config (matches existing xgb_model in cell 20) ---
N_SPLITS = 5
RANDOM_STATE = 42
XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=4,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# --- 1) Out-of-fold probabilities on training split ---
skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

oof_probs = np.zeros(len(y_train), dtype=float)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
    fold_model = XGBClassifier(**XGB_PARAMS)
    fold_model.fit(
        X_train.iloc[tr_idx],
        y_train.iloc[tr_idx],
    )
    oof_probs[val_idx] = fold_model.predict_proba(X_train.iloc[val_idx])[:, 1]
    print(f"Fold {fold}/{N_SPLITS} complete")

print(f"OOF coverage: {(oof_probs > 0).sum()} / {len(oof_probs)} rows with predictions")

# --- 2) Threshold tuning on OOF (optimize F1 for class 1) ---
threshold_grid = np.arange(0.05, 0.96, 0.01)
f1_scores = [
    f1_score(y_train, oof_probs >= t, zero_division=0)
    for t in threshold_grid
]

best_idx = int(np.argmax(f1_scores))
best_threshold = float(threshold_grid[best_idx])
best_oof_f1 = float(f1_scores[best_idx])

oof_pred_tuned = (oof_probs >= best_threshold).astype(int)

print(f"\nBest threshold (OOF, F1 class 1): {best_threshold:.2f}")
print(f"OOF F1 (class 1) at best threshold: {best_oof_f1:.4f}")
print(
    "OOF metrics at best threshold:",
    f"precision={precision_score(y_train, oof_pred_tuned, zero_division=0):.4f},",
    f"recall={recall_score(y_train, oof_pred_tuned, zero_division=0):.4f},",
    f"roc_auc={roc_auc_score(y_train, oof_probs):.4f},",
    f"pr_auc={average_precision_score(y_train, oof_probs):.4f}",
)

# --- 3) Threshold leaderboard (top 5) ---
threshold_results = pd.DataFrame({
    "threshold": threshold_grid,
    "f1_class_1": f1_scores,
    "precision_class_1": [
        precision_score(y_train, oof_probs >= t, zero_division=0)
        for t in threshold_grid
    ],
    "recall_class_1": [
        recall_score(y_train, oof_probs >= t, zero_division=0)
        for t in threshold_grid
    ],
}).sort_values("f1_class_1", ascending=False)

print("\nTop 5 thresholds by OOF F1 (class 1):")
display(threshold_results.head(5).round(4))

# --- 4) Apply tuned threshold to existing xgb_model on validation split ---
valid_probs = xgb_model.predict_proba(X_valid)[:, 1]
valid_pred_default = xgb_model.predict(X_valid)
valid_pred_tuned = (valid_probs >= best_threshold).astype(int)

print(f"\nValidation @ default threshold (0.50):")
print(classification_report(y_valid, valid_pred_default))
print(confusion_matrix(y_valid, valid_pred_default))

print(f"Validation @ tuned threshold ({best_threshold:.2f}):")
print(classification_report(y_valid, valid_pred_tuned))
print(confusion_matrix(y_valid, valid_pred_tuned))

valid_summary = pd.DataFrame(
    [
        {
            "threshold": 0.50,
            "accuracy": accuracy_score(y_valid, valid_pred_default),
            "precision_class_1": precision_score(
                y_valid, valid_pred_default, zero_division=0
            ),
            "recall_class_1": recall_score(
                y_valid, valid_pred_default, zero_division=0
            ),
            "f1_class_1": f1_score(y_valid, valid_pred_default, zero_division=0),
            "roc_auc": roc_auc_score(y_valid, valid_probs),
            "pr_auc": average_precision_score(y_valid, valid_probs),
        },
        {
            "threshold": best_threshold,
            "accuracy": accuracy_score(y_valid, valid_pred_tuned),
            "precision_class_1": precision_score(
                y_valid, valid_pred_tuned, zero_division=0
            ),
            "recall_class_1": recall_score(
                y_valid, valid_pred_tuned, zero_division=0
            ),
            "f1_class_1": f1_score(y_valid, valid_pred_tuned, zero_division=0),
            "roc_auc": roc_auc_score(y_valid, valid_probs),
            "pr_auc": average_precision_score(y_valid, valid_probs),
        },
    ]
).round(4)

print("\nValidation summary (default vs tuned threshold):")
display(valid_summary)
```

- [ ] **Step 2: Run the new cell in Jupyter**

1. Restart kernel (clean state).
2. Run all cells through cell 21.
3. Run new cells 22 (markdown) and 23 (code).

Expected outputs (approximate, from prior runs):
- Five lines: `Fold 1/5 complete` … `Fold 5/5 complete`
- `OOF coverage: 351312 / 351312` (or your `len(y_train)` count)
- `Best threshold` between `0.30` and `0.60` (data-dependent)
- `OOF F1 (class 1)` printed
- Top-5 threshold table displayed
- Two `classification_report` blocks for validation (default vs tuned)
- `valid_summary` DataFrame with 2 rows

- [ ] **Step 3: Sanity checks**

After running cell 23, verify in a scratch cell or inline:

```python
assert len(oof_probs) == len(y_train)
assert np.isfinite(oof_probs).all()
assert 0.0 < best_threshold < 1.0
assert valid_summary.loc[1, "threshold"] == best_threshold
print("Sanity checks passed")
```

Expected: `Sanity checks passed`

- [ ] **Step 4: Commit (notebook only)**

```bash
git add analysis.ipynb
git commit -m "feat(notebook): add XGBoost OOF and threshold tuning cells"
```

---

## Design Notes

### Why StratifiedKFold on `X_train` only?

- OOF must not see validation data when tuning the threshold (avoids leakage).
- `StratifiedKFold` preserves ~20% pit-stop rate per fold, matching the imbalanced target.
- The notebook’s markdown (section 4) mentions race-based splits, but the implemented split (cell 15) is `train_test_split(..., stratify=y)`. This plan **matches implemented code** without changing cell 15. A future plan can swap in `GroupKFold` by race if cell 15 is updated.

### Why reuse `xgb_model` for validation?

- Cell 20 already fits `xgb_model` on full `X_train`.
- OOF is used only to pick `best_threshold`.
- Applying that threshold to `xgb_model.predict_proba(X_valid)` gives a fair “same model, better decision rule” comparison vs default 0.5.

### Metric choice for threshold tuning

- Optimize **F1 (class 1)** per project evaluation focus.
- To prioritize recall instead, change the `argmax` line to use `recall_score(y_train, oof_probs >= t, zero_division=0)`.

---

## Self-Review

| Requirement | Task |
|-------------|------|
| OOF pipeline for XGBoost | Task 2, section 1 |
| Threshold tuning | Task 2, section 2 |
| No edits to existing cells | File structure constraint |
| New cell(s) below XGBoost eval | Insert at index 22–23 after cell 21 |
| Same XGB hyperparameters as cell 20 | `XGB_PARAMS` dict |
| Validation comparison | Task 2, section 4 |
| No placeholders | All code provided inline |

**Gaps:** None for stated scope. Race-based GroupKFold is documented as a follow-up, not in scope.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-20-xgboost-oof-threshold-tuning.md`. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — implement tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
