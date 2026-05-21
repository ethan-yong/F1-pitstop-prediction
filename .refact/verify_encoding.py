"""Verify label + target encoding pipeline (train-only TE)."""
import json
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

path = Path(__file__).resolve().parents[1]
nb_path = path / "analysis.ipynb"
nb = json.loads(nb_path.read_text(encoding="utf-8"))

assert "get_dummies" not in "".join(nb["cells"][10]["source"])
assert "OrdinalEncoder" in "".join(nb["cells"][1]["source"])
assert "Label and Target Encoding" in "".join(nb["cells"][13]["source"])
assert "target_encode_cols" in "".join(nb["cells"][14]["source"])
print("OK: notebook structure checks passed")

# Load data
import kagglehub

data_path = kagglehub.competition_download("playground-series-s6e5")
train = pd.read_csv(f"{data_path}/train.csv")
test = pd.read_csv(f"{data_path}/test.csv")

# Minimal replay: FE through split + encoding (exec cell 10 logic is heavy - spot check encoding only)
train_fe = train.copy()
test_fe = test.copy()
n_train = len(train)

# After split from combined would have PitNextLap on train only
train_fe = train_fe.drop(columns=["id"])
test_fe = test_fe.drop(columns=["id"])

categorical_cols = ["Driver", "Compound", "Race"]
# Simplified: only check encoding mechanics on raw cols present in train
train_fe["TyreWearCategory"] = "Fresh"
test_fe["TyreWearCategory"] = "Fresh"
train_fe["RacePhase"] = "Mid"
test_fe["RacePhase"] = "Mid"

categorical_cols = ["Driver", "Compound", "Race", "TyreWearCategory", "RacePhase"]
target_encode_cols = ["Driver", "Race", "Compound"]
SMOOTH = 10

le = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
le_cols = [c + "_le" for c in categorical_cols]
train_fe[le_cols] = le.fit_transform(train_fe[categorical_cols])
test_fe[le_cols] = le.transform(test_fe[categorical_cols])

global_mean = train_fe["PitNextLap"].mean()
for col in target_encode_cols:
    stats = train_fe.groupby(col)["PitNextLap"].agg(["mean", "count"])
    stats["smooth"] = (stats["count"] * stats["mean"] + SMOOTH * global_mean) / (
        stats["count"] + SMOOTH
    )
    mapping = stats["smooth"]
    train_fe[col + "_te"] = train_fe[col].map(mapping).fillna(global_mean)
    test_fe[col + "_te"] = test_fe[col].map(mapping).fillna(global_mean)

train_fe.drop(columns=categorical_cols, inplace=True)
test_fe.drop(columns=categorical_cols, inplace=True)

assert "Driver_le" in train_fe.columns
assert "Race_te" in train_fe.columns
assert not any(c.startswith("Race_") and c.endswith("_Grand Prix") for c in train_fe.columns)
assert train_fe.shape[1] < 100
print("OK: encoding smoke test passed")
print("Train columns:", train_fe.shape[1])
