import json

nb = json.load(open("analysis.ipynb", encoding="utf-8"))
keys = [
    "train_test_split",
    "Random Forest Classifier",
    "test_fe[X.columns]",
    "features = X.columns",
    '"Feature": X.columns',
]
for i, c in enumerate(nb["cells"]):
    s = "".join(c.get("source", []))
    for k in keys:
        if k in s:
            print(i, c["cell_type"], k)
            break
