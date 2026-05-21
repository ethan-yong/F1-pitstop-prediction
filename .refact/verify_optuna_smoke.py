"""Smoke-test Optuna objectives on a small stratified sample."""
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

N_SPLITS = 3
RANDOM_STATE = 42
N_TRIALS = 2
SAMPLE_SIZE = 8000

path = Path.home() / ".cache" / "kagglehub" / "competitions" / "playground-series-s6e5"
train = pd.read_csv(path / "train.csv")
y = train["PitNextLap"]
X = train.drop(columns=["PitNextLap", "id"], errors="ignore").select_dtypes(include=[np.number])
X = X.fillna(0)

rng = np.random.default_rng(RANDOM_STATE)
idx = rng.choice(len(X), size=min(SAMPLE_SIZE, len(X)), replace=False)
X = X.iloc[idx].reset_index(drop=True)
y = y.iloc[idx].reset_index(drop=True)

CV_FOLDS = StratifiedKFold(
    n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
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


def objective_rf(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 120),
        "max_depth": trial.suggest_int("max_depth", 6, 12),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
        "max_features": _max_features(trial),
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    return cv_f1_class1(RandomForestClassifier(**params), X, y)


def objective_xgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 120),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 2.0, 6.0),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    return cv_f1_class1(XGBClassifier(**params), X, y)


scale_pos_weight = len(y[y == 0]) / len(y[y == 1])


def objective_lgbm(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 120),
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "num_leaves": trial.suggest_int("num_leaves", 16, 64),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "scale_pos_weight": scale_pos_weight,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
    }
    return cv_f1_class1(LGBMClassifier(**params), X, y)


for name, fn in [
    ("rf", objective_rf),
    ("xgb", objective_xgb),
    ("lgbm", objective_lgbm),
]:
    study = optuna.create_study(direction="maximize")
    study.optimize(fn, n_trials=N_TRIALS, show_progress_bar=False)
    assert study.best_value > 0, name
    print(f"OK {name}: best_value={study.best_value:.4f}")

print("All Optuna smoke tests passed.")
