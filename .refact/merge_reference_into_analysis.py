"""Append reference Kaggle kernel cells into analysis.ipynb."""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis.ipynb"
REFERENCE = ROOT / "reference_kernel" / "predicting-f1-pit-stops-xgboost.ipynb"

ref_nb = json.loads(REFERENCE.read_text(encoding="utf-8"))
analysis_nb = json.loads(ANALYSIS.read_text(encoding="utf-8"))

ref_source = "".join(ref_nb["cells"][0].get("source", []))

# Local-friendly tweaks
local_source = ref_source.replace('device="cuda"', 'device="cpu"')
local_source = local_source.replace('"TireAge"', '"TyreLife"')
local_source = local_source.replace('df["TireAge"]', 'df["TyreLife"]')
local_source = local_source.replace("TireAgeRatio", "TyreLifeRatio")
local_source = local_source.replace("TireTempInteraction", "TyreTempInteraction")

local_source = local_source.replace(
    'COMP_PATH = Path(\n        "/kaggle/input/"\n        "competitions/"\n        "playground-series-s6e5"\n    )',
    'COMP_PATH = Path(path)  # from kagglehub download cell at top of notebook',
)
local_source = local_source.replace(
    'BLEND_PATH = Path(\n        "/kaggle/input/"\n        "datasets/"\n        "anthonytherrien/"\n        "predicting-f1-pit-stops-vault"\n    )',
    'BLEND_PATH = Path("reference_kernel/vault")',
)
local_source = local_source.replace(
    '\n\n# Call the main function\nif __name__ == "__main__":\n    main()',
    '',
)

header_md = {
    "cell_type": "markdown",
    "id": "reference-kernel-imported",
    "metadata": {},
    "source": [
        "## Reference kernel (imported)\n",
        "\n",
        "Cells below were pulled from "
        "[anthonytherrien/predicting-f1-pit-stops-xgboost](https://www.kaggle.com/code/anthonytherrien/predicting-f1-pit-stops-xgboost) "
        "into `reference_kernel/` and merged here.\n",
        "\n",
        "**Note:** This kernel uses **XGBoost regression** + **weighted blending** of external submissions, "
        "not the classifier pipeline above. `TyreLife` is used where the original used `TireAge`. "
        "Run the **Setup** cell first; the **Run** cell is optional (needs the vault dataset for blending).\n",
    ],
}

setup_code = {
    "cell_type": "code",
    "id": "reference-kernel-setup",
    "metadata": {},
    "outputs": [],
    "source": local_source.splitlines(keepends=True),
}

run_code = {
    "cell_type": "code",
    "id": "reference-kernel-run",
    "metadata": {},
    "outputs": [],
    "source": [
        "# Optional: run the reference pipeline end-to-end (writes submission_reference.csv)\n",
        "# Requires reference_kernel/vault/ with submission.csv files from Kaggle dataset\n",
        "# anthonytherrien/predicting-f1-pit-stops-vault\n",
        "\n",
        "import shutil\n",
        "from pathlib import Path\n",
        "\n",
        "VAULT = Path(\"reference_kernel/vault\")\n",
        "if not VAULT.exists():\n",
        "    print(\n",
        "        \"Skipping reference main(): vault not found at\",\n",
        "        VAULT.resolve(),\n",
        "        \"\\nDownload optional dataset or copy submission files there.\"\n",
        "        \"\\nSetup functions above are still available (create_features, blend_predictions, etc.).\"\n",
        "    )\n",
        "else:\n",
        "    # Patch output so we do not overwrite your main submission.csv\n",
        "    _orig_blend = blend_predictions\n",
        "\n",
        "    def blend_predictions_local(test_ids, prediction_dict, output_path, target_col):\n",
        "        return _orig_blend(\n",
        "            test_ids, prediction_dict,\n",
        "            \"submission_reference.csv\", target_col\n",
        "        )\n",
        "\n",
        "    blend_predictions = blend_predictions_local\n",
        "    main()\n",
        "    print(\"Wrote submission_reference.csv (reference blend; main submission.csv unchanged)\")\n",
    ],
}

# Avoid duplicating if already merged
existing_ids = {c.get("id") for c in analysis_nb["cells"]}
if "reference-kernel-imported" in existing_ids:
    print("Reference section already present in analysis.ipynb — skipping.")
else:
    analysis_nb["cells"].extend([header_md, setup_code, run_code])
    ANALYSIS.write_text(json.dumps(analysis_nb, indent=1), encoding="utf-8")
    print(f"Appended 3 cells to {ANALYSIS} (now {len(analysis_nb['cells'])} cells total)")
