# COVID-19 Mortality Risk Project

This project contains:
- `notebook.ipynb`: full homework analysis and modeling workflow.
- `app.py`: Streamlit dashboard with 4 required tabs.
- `covid.csv`: dataset used by the notebook/app.
- `models/*.pkl`: exported trained models (Decision Tree, Random Forest) and feature list.

## 1) Environment Setup

```bash
cd /Users/hdksrma/Desktop/hw--1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Run the Streamlit App

```bash
source .venv/bin/activate
streamlit run app.py
```

Notes:
- The app loads data from:
1. uploaded CSV (sidebar), else
2. local `covid.csv`, else
3. Google Drive fallback URL from the notebook.
- If `lightgbm` is unavailable, the app still runs with Decision Tree + Random Forest.

## 3) Model Pickle Files

Currently available in `models/`:
- `decision_tree.pkl`
- `random_forest.pkl`
- `feature_columns.pkl` (required input feature order)

Example load:

```python
import pickle
import pandas as pd

with open("models/random_forest.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# Build one-row input with matching columns/order
X = pd.DataFrame([your_feature_dict])[feature_columns]
pred_class = model.predict(X)[0]
pred_proba = model.predict_proba(X)[0, 1]
```

## 4) (Optional) Export Models Again

If you want to regenerate model pickle files, rerun the export command in the terminal (same training setup used in the app/notebook).
