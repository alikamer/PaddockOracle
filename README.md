# Paddock Oracle

This model mainly aims to predict Formula 1 outcomes based on the provided parameters. Specifically, it is a classification model that predicts whether a driver will finish on the podium (top 3) in a race.

**Live demo:** https://paddockoracle.streamlit.app

## Results

The model was trained on seasons 2010–2025 and then tested on the 2026 season, which it had never seen during training.

| Metric | Cross-validation (2010–2023) | 2026 season (unseen) |
|---|---|---|
| ROC-AUC | 0.922 | 0.945 |
| Top-3 accuracy | — | 62% (26 of 42) |
| Accuracy | — | 0.890 |

Across the 14 races held in 2026, the model correctly identified 26 of the 42 podium finishers. It called all three podium places correctly in three races.

## Data

Race results are pulled from the [Jolpica (Ergast) F1 API](https://api.jolpi.ca/ergast/f1).

- **6,870 rows** — one row per driver per race
- **Seasons 2010–2025**, 83 drivers, 23 teams, 35 circuits
- **14.4%** of rows are podium finishes, so the target is imbalanced
- The 2026 season is stored separately and never used for training

The 2010 cut-off is deliberate: earlier seasons used different scoring systems and technical regulations, which only add noise.

## Features

Raw race results carry little predictive signal on their own, so eight features were derived from each driver's history. Every one of them is calculated with a shift, meaning a row can only see races that happened *before* it.

| Feature | Meaning |
|---|---|
| `rolling_form` | Average finishing position over the driver's last 3 races |
| `season_points` | Points accumulated so far this season |
| `team_form` | Team's average finishing position over its last 3 races |
| `circuit_history` | Driver's average finishing position at this circuit |
| `teammate_delta` | Historical gap between the driver and their teammate |
| `dnf_rate` | Share of races the driver failed to finish |
| `grid_gain` | Average positions gained between the grid and the flag |
| `career_races` | Races entered so far — separates rookies from veterans |

Starting grid position and team are used directly. Columns that only become known after the race (finishing position, points, laps, status) are dropped to prevent leakage.

Team names are unified across rebrandings, so Toro Rosso → AlphaTauri → RB is treated as one team, as are Force India → Racing Point → Aston Martin and Sauber → Audi.

## Model

Seven classifiers were compared using `TimeSeriesSplit`, which always trains on the past and validates on the future — a random split would let the model see later seasons while predicting earlier ones.

| Model | ROC-AUC |
|---|---|
| Random Forest | 0.922 |
| Logistic Regression | 0.921 |
| LightGBM | 0.918 |
| XGBoost | 0.912 |
| KNN | 0.849 |
| CART | 0.728 |

The final model is a soft-voting ensemble of the three strongest and most different performers: Logistic Regression, Random Forest and CatBoost.

## Project structure

```text
src/
  1_data_api_to_csv/     Fetching, combining and filtering raw API data
    1.4_fetch_new_season.py   Pulls a new season into a separate file
  2_eda/                 Exploratory analysis
  3_preprocessing/       Cleaning (non-starters, pit-lane starts)
  4_features/            Feature engineering
  5_models/
    5.1_research.py      Model comparison and hyperparameter search
    5.2_pipeline.py      Trains the final model and saves it
    5.3_prediction.py    Loads the saved model and predicts
  utils.py               Shared functions used by both training and the app
models/                  Saved model (f1_model.pkl)
data/data_processed/     Processed datasets
app2.py                  Streamlit interface
```

## Installation

This project was developed using **Python 3.14**. We recommend using a virtual environment to avoid version conflicts.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/alikamer/PaddockOracle.git
   cd PaddockOracle
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment:**
   - Windows:
     ```bash
     .\.venv\Scripts\activate
     ```
   - Mac/Linux:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the app from the project root:

```bash
streamlit run app2.py
```

The interface has two tabs. The first browses the 2010–2025 training data race by race. The second holds the 2026 season, which the model never saw — finished races show the prediction next to the real podium, and for races that have not been run yet you can fill in the starting grid yourself and get a prediction.

To pull newly completed races into the app:

```bash
python src/1_data_api_to_csv/1.4_fetch_new_season.py
```

To retrain the model from scratch:

```bash
python src/5_models/5.2_pipeline.py
```

## Technologies Used

- Pandas, NumPy
- Scikit-learn
- XGBoost, LightGBM, CatBoost
- Streamlit
- SHAP, Plotly, Seaborn
