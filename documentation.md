# F1 Pit Stop Prediction

## Overview

This project explores Formula 1 pit stop prediction using the Kaggle Playground Series S6E5 competition data. The current implementation is a Jupyter notebook workflow in `analysis.ipynb` that downloads the competition dataset, loads the train and test CSV files, performs basic data inspection, documents feature categories, and visualizes the target distribution.

## Current Scope

The notebook currently covers:

- Downloading the Kaggle competition files with `kagglehub`.
- Loading `train.csv` and `test.csv` with `pandas`.
- Inspecting training data structure with `head()`, `info()`, and `describe()`.
- Categorizing variables such as driver, tyre compound, race, stint, lap number, tyre life, position, gap, and pit stop indicators.
- Plotting the distribution of the `PitNextLap` target.

The notebook includes preprocessing, Optuna hyperparameter tuning for Random Forest / XGBoost / LightGBM, validation metrics, OOF threshold tuning for XGBoost, model comparison, and Kaggle submission export.

## Project Structure

```text
.
|-- analysis.ipynb          # Main notebook (EDA, tuning, models, submission)
|-- documentation.md        # Project documentation
|-- requirements.txt        # Python dependencies
|-- .gitignore              # Local Python/Jupyter ignores
`-- venv/                   # Local virtual environment, ignored by git
```

## Tech Stack

- Python
- Jupyter Notebook
- pandas
- NumPy
- matplotlib
- scikit-learn
- XGBoost
- LightGBM
- Optuna
- kagglehub

## Data Source

The notebook downloads data from the Kaggle competition:

```python
kagglehub.competition_download("playground-series-s6e5")
```

If Kaggle authentication is required on a new machine, configure Kaggle credentials before running the download cell. Typically this means placing `kaggle.json` in the local Kaggle configuration directory or authenticating through the Kaggle tooling used by `kagglehub`.

## Local Setup

From the project root:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Then start Jupyter:

```powershell
jupyter notebook
```

Open `analysis.ipynb` and run the cells from top to bottom.

## Notebook Workflow

1. Import required libraries.
2. Download the Kaggle competition dataset.
3. Load training and test data.
4. Inspect schema and summary statistics.
5. Review feature categories and decide which columns are identifiers, categorical variables, ordinal variables, numeric variables, or binary targets.
6. Plot and inspect the class balance of `PitNextLap`.
7. Engineer features, encode categoricals, and preprocess train/validation splits.
8. Run Optuna studies (`direction="maximize"`, mean CV F1 for class 1) for Random Forest, XGBoost, and LightGBM.
9. Retrain models with best params, evaluate on validation, tune XGBoost threshold via OOF, compare models, and write `submission.csv` from the best validation F1 model.

## Important Columns

- `id`: Unique row identifier, not useful as a model feature.
- `Driver`: Driver identifier or name.
- `Compound`: Tyre compound.
- `Race`: Race or event name.
- `Year`: Season year.
- `PitStop`: Whether the row is associated with a pit stop.
- `LapNumber`: Current race lap.
- `Stint`: Current tyre stint.
- `TyreLife`: Age of the tyre in laps.
- `Position`: Current race position.
- `PitNextLap`: Target variable indicating whether a pit stop occurs on the next lap.

## Hyperparameter Tuning

Optuna runs three separate studies on `X_train` only (5-fold stratified CV). Each study uses `direction="maximize"` on mean **F1 (class 1)**. Set `N_TRIALS` lower in the Optuna cell for quick dry-runs (e.g. 5–10); the default is 40 trials per model.

## Next Development Steps

- Add ensemble or stacking across tuned models.
- Extend OOF threshold tuning to the submission winner when it is not XGBoost.
- Cache Optuna studies to disk for resumable long runs.

## Reproducibility Notes

The dataset is downloaded at runtime, so the notebook depends on network access and Kaggle availability. For fully reproducible offline runs, store the downloaded CSV files in a documented local data directory and update the notebook to read from that path when files already exist.
