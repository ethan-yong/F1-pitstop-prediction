import json
from pathlib import Path

nb = json.loads(Path("reference_kernel/predicting-f1-pit-stops-xgboost.ipynb").read_text(encoding="utf-8"))
print("reference cells:", len(nb["cells"]))
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    preview = src.strip().split("\n", 1)[0][:90]
    print(f"{i:02d} {c['cell_type']:8} {preview}")
