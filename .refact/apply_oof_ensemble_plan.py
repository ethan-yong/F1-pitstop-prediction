"""Apply OOF accuracy tuning + 3-model ensemble plan to analysis.ipynb."""
import json
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "analysis.ipynb"
nb = json.loads(path.read_text(encoding="utf-8"))


def set_source(cell: dict, source: str) -> None:
    cell["source"] = [line + "\n" for line in source.splitlines()]
    cell["outputs"] = []
    cell["execution_count"] = None


def make_cell(cell_type: str, cell_id: str, source: str) -> dict:
    cell = {
        "cell_type": cell_type,
        "id": cell_id,
        "metadata": {},
        "source": [line + "\n" for line in source.splitlines()],
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


HELPERS_SOURCE = """from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score

THRESHOLD_GRID = np.arange(0.05, 0.96, 0.01)
SUBMISSION_THRESHOLD_OBJECTIVE = "accuracy"  # or "f1"
N_SPLITS = 5
RANDOM_STATE = 42

OOF_SKF = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)


def tune_threshold(y_true, probs, grid, metric="f1"):
    if metric == "f1":
        metric_fn = lambda yt, pred: f1_score(yt, pred, zero_division=0)
    elif metric == "accuracy":
        metric_fn = lambda yt, pred: accuracy_score(yt, pred)
    else:
        raise ValueError(f"Unknown metric: {metric!r}")
    scores = [metric_fn(y_true, (probs >= t).astype(int)) for t in grid]
    best_idx = int(np.argmax(scores))
    return float(grid[best_idx]), float(scores[best_idx]), scores


def generate_oof_proba(model_factory, X, y, skf, label=""):
    oof = np.zeros(len(y), dtype=float)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        fold_model = model_factory()
        fold_model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof[val_idx] = fold_model.predict_proba(X.iloc[val_idx])[:, 1]
        print(f"{label} fold {fold}/{skf.n_splits} complete")
    return oof


def build_threshold_leaderboard(y_true, probs, grid, f1_scores, acc_scores):
    return pd.DataFrame(
        {
            "threshold": grid,
            "accuracy": acc_scores,
            "f1_class_1": f1_scores,
            "precision_class_1": [
                precision_score(y_true, probs >= t, zero_division=0)
                for t in grid
            ],
            "recall_class_1": [
                recall_score(y_true, probs >= t, zero_division=0)
                for t in grid
            ],
        }
    )


def print_oof_threshold_comparison(model_label, oof_probs, y_true, t_f1, f1_at_t, t_acc, acc_at_t):
    pred_f1 = (oof_probs >= t_f1).astype(int)
    pred_acc = (oof_probs >= t_acc).astype(int)
    print()
    print(f"=== {model_label}: OOF threshold comparison ===")
    print(f"F1 threshold:       {t_f1:.2f}  |  OOF F1: {f1_at_t:.4f}  |  OOF accuracy: {accuracy_score(y_true, pred_f1):.4f}")
    print(f"Accuracy threshold: {t_acc:.2f}  |  OOF accuracy: {acc_at_t:.4f}  |  OOF F1: {f1_score(y_true, pred_acc, zero_division=0):.4f}")
    print(
        f"roc_auc={roc_auc_score(y_true, oof_probs):.4f}, "
        f"pr_auc={average_precision_score(y_true, oof_probs):.4f}"
    )


def validation_threshold_summary(y_valid, probs, t_f1, t_acc):
    rows = []
    for objective, t in [("default", 0.50), ("f1", t_f1), ("accuracy", t_acc)]:
        pred = (probs >= t).astype(int)
        rows.append(
            {
                "objective": objective,
                "threshold": t,
                "accuracy": accuracy_score(y_valid, pred),
                "precision_class_1": precision_score(y_valid, pred, zero_division=0),
                "recall_class_1": recall_score(y_valid, pred, zero_division=0),
                "f1_class_1": f1_score(y_valid, pred, zero_division=0),
                "roc_auc": roc_auc_score(y_valid, probs),
                "pr_auc": average_precision_score(y_valid, probs),
            }
        )
    return pd.DataFrame(rows).round(4)


def replace_model_metrics(name, y_true, y_pred, y_prob):
    for i, row in enumerate(model_results):
        if row["Model"] == name:
            model_results[i] = {
                "Model": name,
                "Accuracy": float(accuracy_score(y_true, y_pred)),
                "Precision (class 1)": float(
                    precision_score(y_true, y_pred, pos_label=1, zero_division=0)
                ),
                "Recall (class 1)": float(
                    recall_score(y_true, y_pred, pos_label=1, zero_division=0)
                ),
                "F1 (class 1)": float(
                    f1_score(y_true, y_pred, pos_label=1, zero_division=0)
                ),
                "ROC-AUC": float(roc_auc_score(y_true, y_prob)),
                "PR-AUC": float(average_precision_score(y_true, y_prob)),
            }
            return
    record_model_metrics(name, y_true, y_pred, y_prob)
"""

OOF_MD = """## Multi-Model OOF + Threshold Tuning

Out-of-fold (OOF) probabilities are generated on `X_train` with 5-fold stratified CV using Optuna-tuned hyperparameters for **Random Forest**, **XGBoost**, and **LightGBM**.

Decision thresholds are tuned on OOF predictions two ways: **F1 (class 1)** and **accuracy**. Accuracy thresholds align with the Kaggle scoring metric; F1 thresholds remain for pit-stop recall analysis. Tuned thresholds are applied to full-fit models on `X_valid` without retraining.
"""

OOF_CODE = """RF_PARAMS = {
    **rf_best_params,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}
XGB_PARAMS = {
    **xgb_best_params,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}
LGBM_PARAMS = {
    **lgbm_best_params,
    "scale_pos_weight": scale_pos_weight,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}

print("Generating OOF probabilities (5-fold stratified CV)...")
rf_oof_probs = generate_oof_proba(
    lambda: RandomForestClassifier(**RF_PARAMS),
    X_train,
    y_train,
    OOF_SKF,
    label="Random Forest",
)
xgb_oof_probs = generate_oof_proba(
    lambda: XGBClassifier(**XGB_PARAMS),
    X_train,
    y_train,
    OOF_SKF,
    label="XGBoost",
)
lgbm_oof_probs = generate_oof_proba(
    lambda: LGBMClassifier(**LGBM_PARAMS),
    X_train,
    y_train,
    OOF_SKF,
    label="LightGBM",
)

for label, oof in [
    ("Random Forest", rf_oof_probs),
    ("XGBoost", xgb_oof_probs),
    ("LightGBM", lgbm_oof_probs),
]:
    covered = int(np.isfinite(oof).sum())
    print(f"{label} OOF coverage: {covered} / {len(oof)}")

# --- Threshold tuning per model ---
(
    rf_best_threshold_f1,
    rf_best_oof_f1,
    rf_f1_scores,
) = tune_threshold(y_train, rf_oof_probs, THRESHOLD_GRID, metric="f1")
(
    rf_best_threshold_acc,
    rf_best_oof_acc,
    rf_acc_scores,
) = tune_threshold(y_train, rf_oof_probs, THRESHOLD_GRID, metric="accuracy")

(
    xgb_best_threshold_f1,
    xgb_best_oof_f1,
    xgb_f1_scores,
) = tune_threshold(y_train, xgb_oof_probs, THRESHOLD_GRID, metric="f1")
(
    xgb_best_threshold_acc,
    xgb_best_oof_acc,
    xgb_acc_scores,
) = tune_threshold(y_train, xgb_oof_probs, THRESHOLD_GRID, metric="accuracy")

(
    lgbm_best_threshold_f1,
    lgbm_best_oof_f1,
    lgbm_f1_scores,
) = tune_threshold(y_train, lgbm_oof_probs, THRESHOLD_GRID, metric="f1")
(
    lgbm_best_threshold_acc,
    lgbm_best_oof_acc,
    lgbm_acc_scores,
) = tune_threshold(y_train, lgbm_oof_probs, THRESHOLD_GRID, metric="accuracy")

for model_label, oof_probs, t_f1, f1_at_t, t_acc, acc_at_t in [
    ("Random Forest", rf_oof_probs, rf_best_threshold_f1, rf_best_oof_f1, rf_best_threshold_acc, rf_best_oof_acc),
    ("XGBoost", xgb_oof_probs, xgb_best_threshold_f1, xgb_best_oof_f1, xgb_best_threshold_acc, xgb_best_oof_acc),
    ("LightGBM", lgbm_oof_probs, lgbm_best_threshold_f1, lgbm_best_oof_f1, lgbm_best_threshold_acc, lgbm_best_oof_acc),
]:
    print_oof_threshold_comparison(model_label, oof_probs, y_train, t_f1, f1_at_t, t_acc, acc_at_t)

rf_valid_probs = model.predict_proba(X_valid)[:, 1]
xgb_valid_probs = xgb_model.predict_proba(X_valid)[:, 1]
lgbm_valid_probs = lgbm_model.predict_proba(X_valid)[:, 1]

single_model_thresholds = [
    ("Random Forest", rf_valid_probs, rf_best_threshold_f1, rf_best_threshold_acc),
    ("XGBoost", xgb_valid_probs, xgb_best_threshold_f1, xgb_best_threshold_acc),
    ("LightGBM", lgbm_valid_probs, lgbm_best_threshold_f1, lgbm_best_threshold_acc),
]

for model_label, valid_probs, t_f1, t_acc in single_model_thresholds:
    print()
    print(f"=== {model_label}: validation threshold summary ===")
    display(validation_threshold_summary(y_valid, valid_probs, t_f1, t_acc))

    pred_acc = (valid_probs >= t_acc).astype(int)
    replace_model_metrics(model_label, y_valid, pred_acc, valid_probs)

OOF_THRESHOLDS_F1 = {
    "Random Forest": rf_best_threshold_f1,
    "XGBoost": xgb_best_threshold_f1,
    "LightGBM": lgbm_best_threshold_f1,
}
OOF_THRESHOLDS_ACC = {
    "Random Forest": rf_best_threshold_acc,
    "XGBoost": xgb_best_threshold_acc,
    "LightGBM": lgbm_best_threshold_acc,
}
"""

ENSEMBLE_MD = """## Ensemble (Blend + Stack)

Combine OOF probabilities from all three tuned models via **equal blend**, **Optuna-weighted blend** (weights tuned on OOF accuracy), and **LogisticRegression stacking**.

Thresholds on ensemble OOF probabilities are tuned for both F1 and accuracy. Validation metrics use accuracy-optimal thresholds for fair comparison with the Kaggle metric.
"""

ENSEMBLE_CODE = """from sklearn.linear_model import LogisticRegression

# --- Equal blend ---
blend_equal_oof = (rf_oof_probs + xgb_oof_probs + lgbm_oof_probs) / 3.0

blend_equal_valid_probs = (rf_valid_probs + xgb_valid_probs + lgbm_valid_probs) / 3.0


def weighted_blend_probs(probs_list, weights):
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    stacked = np.column_stack(probs_list)
    return (stacked * w).sum(axis=1)


# --- Optuna weighted blend (maximize OOF accuracy with tuned threshold) ---
def objective_blend(trial):
    w_rf = trial.suggest_float("w_rf", 0.1, 1.0)
    w_xgb = trial.suggest_float("w_xgb", 0.1, 1.0)
    w_lgbm = trial.suggest_float("w_lgbm", 0.1, 1.0)
    blended = weighted_blend_probs(
        [rf_oof_probs, xgb_oof_probs, lgbm_oof_probs],
        [w_rf, w_xgb, w_lgbm],
    )
    _, best_acc, _ = tune_threshold(y_train, blended, THRESHOLD_GRID, metric="accuracy")
    return best_acc


study_blend = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    study_name="blend_weights",
)
study_blend.optimize(objective_blend, n_trials=30, show_progress_bar=True)

blend_weights = [
    study_blend.best_params["w_rf"],
    study_blend.best_params["w_xgb"],
    study_blend.best_params["w_lgbm"],
]
blend_weighted_oof = weighted_blend_probs(
    [rf_oof_probs, xgb_oof_probs, lgbm_oof_probs],
    blend_weights,
)
blend_weighted_valid_probs = weighted_blend_probs(
    [rf_valid_probs, xgb_valid_probs, lgbm_valid_probs],
    blend_weights,
)

print("Blend weights (RF, XGB, LGBM):", [round(w, 4) for w in blend_weights])
print(f"Blend OOF accuracy @ Optuna weights: {study_blend.best_value:.4f}")

# --- Stacking ---
meta_X_train = np.column_stack([rf_oof_probs, xgb_oof_probs, lgbm_oof_probs])
stack_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=RANDOM_STATE,
)
stack_model.fit(meta_X_train, y_train)

meta_X_valid = np.column_stack([rf_valid_probs, xgb_valid_probs, lgbm_valid_probs])
stack_oof_probs = stack_model.predict_proba(meta_X_train)[:, 1]
stack_valid_probs = stack_model.predict_proba(meta_X_valid)[:, 1]

ensemble_configs = [
    ("Blend (equal)", blend_equal_oof, blend_equal_valid_probs),
    ("Blend (weighted)", blend_weighted_oof, blend_weighted_valid_probs),
    ("Stack (LR)", stack_oof_probs, stack_valid_probs),
]

ensemble_thresholds_f1 = {}
ensemble_thresholds_acc = {}

for name, oof_probs, valid_probs in ensemble_configs:
    t_f1, oof_f1, _ = tune_threshold(y_train, oof_probs, THRESHOLD_GRID, metric="f1")
    t_acc, oof_acc, _ = tune_threshold(y_train, oof_probs, THRESHOLD_GRID, metric="accuracy")
    ensemble_thresholds_f1[name] = t_f1
    ensemble_thresholds_acc[name] = t_acc
    print_oof_threshold_comparison(name, oof_probs, y_train, t_f1, oof_f1, t_acc, oof_acc)

    pred_acc = (valid_probs >= t_acc).astype(int)
    record_model_metrics(name, y_valid, pred_acc, valid_probs)

OOF_THRESHOLDS_F1.update(ensemble_thresholds_f1)
OOF_THRESHOLDS_ACC.update(ensemble_thresholds_acc)


def blend_equal_predict_proba(X):
    probs = np.column_stack([
        model.predict_proba(X)[:, 1],
        xgb_model.predict_proba(X)[:, 1],
        lgbm_model.predict_proba(X)[:, 1],
    ])
    return probs.mean(axis=1)


def blend_weighted_predict_proba(X):
    return weighted_blend_probs(
        [
            model.predict_proba(X)[:, 1],
            xgb_model.predict_proba(X)[:, 1],
            lgbm_model.predict_proba(X)[:, 1],
        ],
        blend_weights,
    )


def stack_predict_proba(X):
    meta_X = np.column_stack([
        model.predict_proba(X)[:, 1],
        xgb_model.predict_proba(X)[:, 1],
        lgbm_model.predict_proba(X)[:, 1],
    ])
    return stack_model.predict_proba(meta_X)[:, 1]


PREDICTORS = {
    "Random Forest": lambda X: model.predict_proba(X)[:, 1],
    "XGBoost": lambda X: xgb_model.predict_proba(X)[:, 1],
    "LightGBM": lambda X: lgbm_model.predict_proba(X)[:, 1],
    "Blend (equal)": blend_equal_predict_proba,
    "Blend (weighted)": blend_weighted_predict_proba,
    "Stack (LR)": stack_predict_proba,
}
"""

COMPARISON_MD = """## Model Comparison

Models and ensembles are ranked by **validation accuracy** (Kaggle metric). Metrics use **accuracy-optimal OOF thresholds** on validation probabilities.

For imbalanced pit-stop prediction (~20% positive laps), also review **Recall (class 1)** and **F1 (class 1)** when pit-stop detection matters more than overall accuracy.
"""

SUBMISSION_MD = """## Generate Submission

After comparing tuned models and ensembles on the validation split, predict `PitNextLap` on the test set with the **best validation accuracy** approach, using the **accuracy-optimal OOF threshold**, then write `submission.csv` for Kaggle.
"""

SUBMISSION_CODE = """# Features for test rows (same preprocessing as training)
X_test = transform_preprocessed(test_fe)

# Best model/ensemble by validation accuracy from model_comparison
winner_name = model_comparison.loc[0, "Model"]

if SUBMISSION_THRESHOLD_OBJECTIVE == "accuracy":
    threshold_map = OOF_THRESHOLDS_ACC
elif SUBMISSION_THRESHOLD_OBJECTIVE == "f1":
    threshold_map = OOF_THRESHOLDS_F1
else:
    raise ValueError(
        f"SUBMISSION_THRESHOLD_OBJECTIVE must be 'accuracy' or 'f1', "
        f"got {SUBMISSION_THRESHOLD_OBJECTIVE!r}"
    )

if winner_name not in PREDICTORS:
    raise ValueError(f"Unknown winner model: {winner_name}")

submission_threshold = threshold_map.get(winner_name, 0.5)
test_probs = PREDICTORS[winner_name](X_test)
test_pred = (test_probs >= submission_threshold).astype(int)

submission = pd.DataFrame({
    "id": test["id"].values,
    "PitNextLap": test_pred,
})

submission.to_csv("submission.csv", index=False)

print(f"Submission model: {winner_name}")
print(f"Threshold objective: {SUBMISSION_THRESHOLD_OBJECTIVE}")
print(f"Saved submission.csv ({len(submission):,} rows)")
print(f"Threshold used: {submission_threshold:.2f}")
print(submission["PitNextLap"].value_counts().rename("count"))
print(submission.head())
"""

COMPARISON_CODE = """metric_cols = [
    "Accuracy",
    "Precision (class 1)",
    "Recall (class 1)",
    "F1 (class 1)",
    "ROC-AUC",
    "PR-AUC",
]

model_comparison = pd.DataFrame(model_results)

for col in metric_cols:
    model_comparison[col] = pd.to_numeric(model_comparison[col], errors="coerce")

model_comparison = (
    model_comparison.sort_values("Accuracy", ascending=False).reset_index(drop=True)
)

model_comparison_display = model_comparison.copy()
for col in metric_cols:
    model_comparison_display[col] = model_comparison_display[col].map(
        lambda x: f"{x:.4f}"
    )

model_comparison_display
"""

OPTUNA_MD = """## Hyperparameter Tuning (Optuna)

Three separate Optuna studies (`direction="maximize"`) tune **Random Forest**, **XGBoost**, and **LightGBM** on `X_train` using stratified 5-fold CV. The objective is mean **F1 (class 1)** so models learn pit-stop signal under imbalance. **Decision thresholds** are tuned separately on OOF predictions for F1 and accuracy later. `X_valid` is held out until final model evaluation.

Random Forest uses a **light search** (only `n_estimators` and `max_depth`) because full RF tuning is slow with 5-fold CV.
"""

# Remove old XGB-only OOF cells
nb["cells"] = [
    c
    for c in nb["cells"]
    if c.get("id") not in {"xgb-oof-threshold-md", "xgb-oof-threshold-code"}
]

# Find insert point: after LightGBM eval
if any(c.get("id") == "oof-threshold-helpers" for c in nb["cells"]):
    print("OK: OOF/ensemble cells already present; skipping insert")
else:
    lgbm_eval_idx = next(
        i
        for i, c in enumerate(nb["cells"])
        if c["cell_type"] == "code"
        and 'record_model_metrics("LightGBM"' in "".join(c.get("source", []))
    )

    new_cells = [
        make_cell("code", "oof-threshold-helpers", HELPERS_SOURCE),
        make_cell("markdown", "multi-model-oof-md", OOF_MD),
        make_cell("code", "multi-model-oof-code", OOF_CODE),
        make_cell("markdown", "ensemble-md", ENSEMBLE_MD),
        make_cell("code", "ensemble-code", ENSEMBLE_CODE),
    ]

    for offset, cell in enumerate(new_cells):
        nb["cells"].insert(lgbm_eval_idx + 1 + offset, cell)

# Patch markdown and code cells
for cell in nb["cells"]:
    cid = cell.get("id")
    if cid == "optuna-hp-tuning-md":
        set_source(cell, OPTUNA_MD)
    elif cid == "6c6f5f30":
        set_source(cell, COMPARISON_MD)
    elif cid == "6fa204b4":
        set_source(cell, COMPARISON_CODE)
    elif cid == "913ea433":
        set_source(cell, SUBMISSION_MD)
    elif cid == "a61cccb7":
        set_source(cell, SUBMISSION_CODE)

path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("OK: analysis.ipynb updated")
print("Total cells:", len(nb["cells"]))
