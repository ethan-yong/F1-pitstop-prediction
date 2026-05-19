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

Model training, validation, prediction export, and submission generation have not yet been added.

## Project Structure

```text
.
|-- analysis.ipynb          # Main exploratory notebook
|-- documentation.md        # Project documentation
|-- .gitignore              # Local Python/Jupyter ignores
`-- venv/                   # Local virtual environment, ignored by git
```

## Tech Stack

- Python
- Jupyter Notebook
- pandas
- NumPy
- matplotlib
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
pip install kagglehub pandas matplotlib numpy jupyter
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

## Next Development Steps

- Add preprocessing for categorical and numeric features.
- Split the training data into train and validation sets.
- Train a baseline classifier for `PitNextLap`.
- Evaluate with metrics appropriate for class imbalance, such as ROC AUC, precision, recall, and F1 score.
- Generate predictions for `test.csv`.
- Create a Kaggle submission file.
- Add a `requirements.txt` or `pyproject.toml` so dependencies are reproducible.

## Reproducibility Notes

The dataset is downloaded at runtime, so the notebook depends on network access and Kaggle availability. For fully reproducible offline runs, store the downloaded CSV files in a documented local data directory and update the notebook to read from that path when files already exist.
