"""Verify OOF accuracy + ensemble notebook changes."""
import ast
import json
import sys
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "analysis.ipynb"
nb = json.loads(path.read_text(encoding="utf-8"))

required_ids = {
    "oof-threshold-helpers",
    "multi-model-oof-md",
    "multi-model-oof-code",
    "ensemble-md",
    "ensemble-code",
}
forbidden_ids = {"xgb-oof-threshold-md", "xgb-oof-threshold-code"}

ids = [c.get("id") for c in nb["cells"]]
missing = required_ids - set(ids)
forbidden = forbidden_ids & set(ids)
if missing:
    sys.exit(f"Missing cells: {missing}")
if forbidden:
    sys.exit(f"Old cells still present: {forbidden}")

for cid in required_ids:
    assert ids.count(cid) == 1, f"Duplicate cell id: {cid}"

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if cell.get("id") in required_ids | {"6fa204b4", "a61cccb7", "optuna-hp-tuning-code"}:
        ast.parse(src)

comparison_src = "".join(next(c["source"] for c in nb["cells"] if c.get("id") == "6fa204b4"))
assert 'sort_values("Accuracy"' in comparison_src

submission_src = "".join(next(c["source"] for c in nb["cells"] if c.get("id") == "a61cccb7"))
for token in ["PREDICTORS", "OOF_THRESHOLDS_ACC", "SUBMISSION_THRESHOLD_OBJECTIVE"]:
    assert token in submission_src, token

ensemble_src = "".join(next(c["source"] for c in nb["cells"] if c.get("id") == "ensemble-code"))
for token in ["Blend (equal)", "Blend (weighted)", "Stack (LR)", "LogisticRegression"]:
    assert token in ensemble_src, token

oof_src = "".join(next(c["source"] for c in nb["cells"] if c.get("id") == "multi-model-oof-code"))
for token in ["rf_oof_probs", "xgb_oof_probs", "lgbm_oof_probs", "tune_threshold", "metric=\"accuracy\""]:
    assert token in oof_src, token

print("OK: structure and syntax checks passed")
print("Total cells:", len(nb["cells"]))
