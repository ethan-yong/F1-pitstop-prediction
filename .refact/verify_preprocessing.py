"""Verify preprocessing logic on notebook data if CSVs are available."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]


def find_data_dir():
    for p in ROOT.rglob("train.csv"):
        if p.parent.name != ".refact":
            return p.parent
    return None


def main():
    data_dir = find_data_dir()
    if data_dir is None:
        print("SKIP: train.csv not found (run notebook download first)")
        return 0

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    print(f"Data: {data_dir}")

    # Minimal path: use raw numeric + target only for smoke test of helpers
    drop_cols = ["id", "PitNextLap"]
    cat_cols = ["Driver", "Compound", "Race"]
    X = train.drop(columns=["PitNextLap"])
    y = train["PitNextLap"]

    for c in cat_cols:
        if c in X.columns:
            le = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X[c + "_le"] = le.fit_transform(X[[c]])
            X = X.drop(columns=[c])

    X = X.select_dtypes(include=[np.number])
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    VAR_THRESHOLD = 0.0
    MIN_TARGET_CORR = 0.01
    MAX_FEATURE_CORR = 0.95

    vt = VarianceThreshold(threshold=VAR_THRESHOLD)
    vt.fit(X_train)
    var_cols = X_train.columns[vt.get_support()].tolist()

    target_corr = X_train[var_cols].corrwith(y_train).abs()
    corr_cols = target_corr[target_corr >= MIN_TARGET_CORR].index.tolist()

    remaining = list(corr_cols)
    target_corr = X_train[remaining].corrwith(y_train).abs()
    while True:
        if len(remaining) <= 1:
            break
        corr = X_train[remaining].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        pairs = (
            upper.stack()
            .reset_index()
            .rename(columns={"level_0": "a", "level_1": "b", 0: "corr"})
        )
        high = pairs[pairs["corr"] > MAX_FEATURE_CORR]
        if high.empty:
            break
        row = high.loc[high["corr"].idxmax()]
        drop_col = (
            row["a"]
            if target_corr[row["a"]] < target_corr[row["b"]]
            else row["b"]
        )
        remaining.remove(drop_col)
        target_corr = target_corr.drop(drop_col)

    selected = remaining
    assert selected, "no features left"

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train[selected])
    X_va = scaler.transform(X_valid[selected])
    X_te = scaler.transform(test.drop(columns=["id"], errors="ignore").reindex(columns=X_train.columns, fill_value=0)[selected])

    print(f"OK: {X_train.shape[1]} -> {len(selected)} features")
    print(f"OK: train {X_tr.shape}, valid {X_va.shape}, test {X_te.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
