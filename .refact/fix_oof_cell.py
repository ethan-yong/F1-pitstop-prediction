import json
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "analysis.ipynb"
nb = json.loads(path.read_text(encoding="utf-8"))

code_source = r'''import numpy as np
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

print()
print(f"Best threshold (OOF, F1 class 1): {best_threshold:.2f}")
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

print()
print("Top 5 thresholds by OOF F1 (class 1):")
display(threshold_results.head(5).round(4))

# --- 4) Apply tuned threshold to existing xgb_model on validation split ---
valid_probs = xgb_model.predict_proba(X_valid)[:, 1]
valid_pred_default = xgb_model.predict(X_valid)
valid_pred_tuned = (valid_probs >= best_threshold).astype(int)

print()
print("Validation @ default threshold (0.50):")
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

print()
print("Validation summary (default vs tuned threshold):")
display(valid_summary)
'''

# Use raw string and split on real newlines only (no \n in f-strings for blank lines)
nb["cells"][23]["source"] = [line + "\n" for line in code_source.split("\n")]

path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

# Verify syntax
compile(code_source, "<oof_cell>", "exec")
print("OK: cell 23 fixed and syntax valid")
