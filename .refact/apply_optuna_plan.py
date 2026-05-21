"""Apply Optuna HP tuning plan to analysis.ipynb."""
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "analysis.ipynb"

OPTUNA_MD = {
    "cell_type": "markdown",
    "id": "optuna-hp-tuning-md",
    "metadata": {},
    "source": [
        "## Hyperparameter Tuning (Optuna)\n",
        "\n",
        "Three separate Optuna studies (`direction=\"maximize\"`) tune **Random Forest**, **XGBoost**, and **LightGBM** on `X_train` using stratified 5-fold CV. The objective is mean **F1 (class 1)**. `X_valid` is held out until final model evaluation.\n",
    ],
}

OPTUNA_CODE = r'''from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

N_SPLITS = 5
RANDOM_STATE = 42
N_TRIALS = 40  # use 5-10 for quick dry-runs

CV_FOLDS = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)


def cv_f1_class1(model, X, y) -> float:
    scores = []
    for tr_idx, val_idx in CV_FOLDS.split(X, y):
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        pred = model.predict(X.iloc[val_idx])
        scores.append(
            f1_score(y.iloc[val_idx], pred, pos_label=1, zero_division=0)
        )
    return float(np.mean(scores))


def _max_features(trial):
    choice = trial.suggest_categorical("max_features", ["sqrt", "log2", "none"])
    return None if choice == "none" else choice


# --- Random Forest ---
def objective_rf(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "max_depth": trial.suggest_int("max_depth", 6, 24),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        "max_features": _max_features(trial),
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    model = RandomForestClassifier(**params)
    return cv_f1_class1(model, X_train, y_train)


study_rf = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    study_name="random_forest",
)
study_rf.optimize(objective_rf, n_trials=N_TRIALS, show_progress_bar=True)
rf_best_params = study_rf.best_params.copy()
rf_best_params["max_features"] = (
    None if rf_best_params.get("max_features") == "none" else rf_best_params["max_features"]
)
print("Random Forest best CV F1:", study_rf.best_value)
print("Random Forest best params:", rf_best_params)


# --- XGBoost ---
def objective_xgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 2.0, 6.0),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    model = XGBClassifier(**params)
    return cv_f1_class1(model, X_train, y_train)


study_xgb = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    pruner=optuna.pruners.MedianPruner(),
    study_name="xgboost",
)
study_xgb.optimize(objective_xgb, n_trials=N_TRIALS, show_progress_bar=True)
xgb_best_params = study_xgb.best_params.copy()
print("XGBoost best CV F1:", study_xgb.best_value)
print("XGBoost best params:", xgb_best_params)


# --- LightGBM ---
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])


def objective_lgbm(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "num_leaves": trial.suggest_int("num_leaves", 16, 127),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "scale_pos_weight": scale_pos_weight,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
    }
    model = LGBMClassifier(**params)
    return cv_f1_class1(model, X_train, y_train)


study_lgbm = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    pruner=optuna.pruners.MedianPruner(),
    study_name="lightgbm",
)
study_lgbm.optimize(objective_lgbm, n_trials=N_TRIALS, show_progress_bar=True)
lgbm_best_params = study_lgbm.best_params.copy()
print("LightGBM best CV F1:", study_lgbm.best_value)
print("LightGBM best params:", lgbm_best_params)

optuna_results = pd.DataFrame(
    [
        {"Model": "Random Forest", "CV F1 (class 1)": study_rf.best_value, **rf_best_params},
        {"Model": "XGBoost", "CV F1 (class 1)": study_xgb.best_value, **xgb_best_params},
        {"Model": "LightGBM", "CV F1 (class 1)": study_lgbm.best_value, **lgbm_best_params},
    ]
)
optuna_results
'''

RF_TRAIN = """model = RandomForestClassifier(
    **rf_best_params,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

model.fit(X_train, y_train)"""

XGB_TRAIN = """# Create XGBoost model (Optuna-tuned hyperparameters)
xgb_model = XGBClassifier(
    **xgb_best_params,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# Train model
xgb_model.fit(X_train, y_train)"""

LGBM_TRAIN = """# Handle imbalance (fixed from training split)
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# Create model (Optuna-tuned hyperparameters)
lgbm_model = LGBMClassifier(
    **lgbm_best_params,
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)

# Train
lgbm_model.fit(X_train, y_train)"""

LGBM_EVAL = """# Predict
y_pred = lgbm_model.predict(X_valid)
y_prob = lgbm_model.predict_proba(X_valid)[:, 1]

record_model_metrics("LightGBM", y_valid, y_pred, y_prob)

# Evaluation
print(classification_report(y_valid, y_pred))
print(confusion_matrix(y_valid, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_valid, y_prob))"""

SUBMISSION_MD = """## Generate Submission

After comparing tuned models on the validation split, predict `PitNextLap` on the test set with the **best validation F1 (class 1)** model, then write `submission.csv` for Kaggle."""

SUBMISSION_CODE = """# Features for test rows (same preprocessing as training)
X_test = transform_preprocessed(test_fe)

# Best model by validation F1 (class 1) from model_comparison
winner_name = model_comparison.loc[0, "Model"]
submission_threshold = 0.5

if winner_name == "XGBoost":
    winner_model = xgb_model
    submission_threshold = best_threshold
elif winner_name == "LightGBM":
    winner_model = lgbm_model
elif winner_name == "Random Forest":
    winner_model = model
else:
    raise ValueError(f"Unknown winner model: {winner_name}")

test_probs = winner_model.predict_proba(X_test)[:, 1]
test_pred = (test_probs >= submission_threshold).astype(int)

submission = pd.DataFrame({
    "id": test["id"].values,
    "PitNextLap": test_pred,
})

submission.to_csv("submission.csv", index=False)

print(f"Submission model: {winner_name}")
print(f"Saved submission.csv ({len(submission):,} rows)")
print(f"Threshold used: {submission_threshold:.2f}")
print(submission["PitNextLap"].value_counts().rename("count"))
print(submission.head())"""

XGB_PARAMS_REPLACEMENT = """XGB_PARAMS = {
    **xgb_best_params,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}"""


def set_source(cell, text: str):
    cell["source"] = [line + "\n" for line in text.split("\n")]
    if cell["source"] and cell["source"][-1] == "\n":
        cell["source"].pop()
    # ensure last line has newline per notebook convention
    if cell["source"] and not cell["source"][-1].endswith("\n"):
        cell["source"][-1] += "\n"


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # 1) Insert Optuna cells after preprocessing (index 20)
    if not any("optuna-hp-tuning-md" == c.get("id") for c in cells):
        optuna_code_cell = {
            "cell_type": "code",
            "id": "optuna-hp-tuning-code",
            "metadata": {},
            "outputs": [],
            "execution_count": None,
            "source": [],
        }
        set_source(optuna_code_cell, OPTUNA_CODE.strip())
        cells.insert(20, optuna_code_cell)
        cells.insert(20, OPTUNA_MD)

    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))

        if c.get("id") == "optuna-hp-tuning-code" and "study_rf" not in src:
            set_source(c, OPTUNA_CODE.strip())

        if "model = RandomForestClassifier(" in src and "n_estimators=200" in src:
            set_source(c, RF_TRAIN)

        if "xgb_model = XGBClassifier(" in src and "n_estimators=300" in src:
            set_source(c, XGB_TRAIN)

        if "XGB_PARAMS = dict(" in src:
            set_source(
                c,
                src.replace(
                    """XGB_PARAMS = dict(
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
)""",
                    XGB_PARAMS_REPLACEMENT.strip(),
                ).strip(),
            )
            if "XGB_PARAMS = dict(" in "".join(c.get("source", [])):
                # fallback: replace whole block up to closing paren before skf
                lines = src.splitlines()
                new_lines = []
                skip = False
                for line in lines:
                    if line.strip().startswith("XGB_PARAMS = dict("):
                        skip = True
                        new_lines.extend(XGB_PARAMS_REPLACEMENT.strip().splitlines())
                        continue
                    if skip:
                        if line.strip() == ")":
                            skip = False
                        continue
                    new_lines.append(line)
                set_source(c, "\n".join(new_lines))

        if "model = LGBMClassifier(" in src and "num_leaves=31" in src:
            set_source(c, LGBM_TRAIN)

        if (
            'record_model_metrics("LightGBM"' in src
            and "lgbm_model" not in src
            and "model.predict(X_valid)" in src
        ):
            set_source(c, LGBM_EVAL)

    # 2) Move submission cells after model comparison
    sub_md_idx = None
    sub_code_idx = None
    comparison_idx = None
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        if "## Generate Submission" in src and sub_md_idx is None:
            sub_md_idx = i
        if "submission.to_csv" in src and "winner_name" not in src:
            sub_code_idx = i
        if "model_comparison_display" in src:
            comparison_idx = i

    if sub_md_idx is not None and sub_code_idx is not None and comparison_idx is not None:
        md_cell = cells.pop(sub_md_idx if sub_md_idx < sub_code_idx else sub_md_idx - 1)
        # re-find code idx after pop
        sub_code_idx = next(
            i for i, c in enumerate(cells) if "submission.to_csv" in "".join(c.get("source", []))
        )
        code_cell = cells.pop(sub_code_idx)
        comparison_idx = next(
            i for i, c in enumerate(cells) if "model_comparison_display" in "".join(c.get("source", []))
        )
        set_source(md_cell, SUBMISSION_MD.strip())
        set_source(code_cell, SUBMISSION_CODE.strip())
        cells.insert(comparison_idx + 1, md_cell)
        cells.insert(comparison_idx + 2, code_cell)
    else:
        # update in place if already moved
        for c in cells:
            src = "".join(c.get("source", []))
            if "submission.to_csv" in src and "winner_name" not in src:
                set_source(c, SUBMISSION_CODE.strip())
            if "## Generate Submission" in src and "best validation F1" not in src:
                set_source(c, SUBMISSION_MD.strip())

    NB_PATH.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Updated", NB_PATH)


if __name__ == "__main__":
    main()
