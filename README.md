# Paddock Oracle

This model mainly aims to predict Formula 1 outcomes based on the provided parameters. Specifically, it is a classification model that predicts whether a driver will finish on the podium (top 3) in a race.

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
   - **Windows:**
     ```bash
     .\.venv\Scripts\activate
     ```
   - **Mac/Linux:**
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Technologies Used
- Pandas, NumPy
- Scikit-learn
- XGBoost, LightGBM, CatBoost
- Streamlit
- SHAP, Plotly, Seaborn
