import ast
import json

nb = json.load(open("analysis.ipynb", encoding="utf-8"))
src = "".join(nb["cells"][19]["source"])
ast.parse(src)
assert "transform_preprocessed" in src
assert "SELECTED_FEATURES" in src
assert "VarianceThreshold" in "".join(nb["cells"][1]["source"])
print("OK: cell 19 parses; imports and helpers present")
