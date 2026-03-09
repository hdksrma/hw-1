import io
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False


st.set_page_config(page_title="COVID-19 Mortality Risk Dashboard", layout="wide")
sns.set_style("whitegrid")

GOOGLE_DRIVE_FILE_ID = "1R-GDTtX0l38JYlPaG7f8eKx3D6pN-CKE"
GOOGLE_DRIVE_URL = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"


NOTEBOOK_METRICS = pd.DataFrame(
    {
        "Accuracy": [0.897667, 0.899000, 0.896333, 0.899667],
        "AUC-ROC": [0.945597, 0.950467, 0.950081, 0.950167],
        "Precision": [0.873633, 0.871239, 0.871966, 0.863529],
        "Recall": [0.935589, 0.942095, 0.934938, 0.955107],
        "F1 Score": [0.903550, 0.905283, 0.902355, 0.907013],
    },
    index=["Decision Tree", "Random Forest", "LightGBM", "Neural Network"],
)

BEST_HYPERPARAMETERS = {
    "Decision Tree": {"max_depth": 4, "min_samples_leaf": 50},
    "Random Forest": {"max_depth": 8, "n_estimators": 200},
    "LightGBM": {"learning_rate": 0.05, "max_depth": 4, "n_estimators": 50},
    "Neural Network (best tuned)": {
        "hidden_layer_size": 64,
        "learning_rate": 0.001,
        "dropout_rate": 0.3,
    },
}


@st.cache_data(show_spinner=False)
def load_data(uploaded_file):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        try:
            data = pd.read_csv("covid.csv")
        except Exception:
            data = pd.read_csv(GOOGLE_DRIVE_URL)

    if "Unnamed: 0" in data.columns:
        data = data.drop(columns=["Unnamed: 0"])
    return data


@st.cache_data(show_spinner=False)
def build_balanced_sample(data: pd.DataFrame, target_col: str = "DEATH") -> pd.DataFrame:
    death_1_count = (data[target_col] == 1).sum()
    death_0_count = (data[target_col] == 0).sum()
    n = int(min(5000, death_1_count, death_0_count))
    death_1_sample = data[data[target_col] == 1].sample(n=n, random_state=42)
    death_0_sample = data[data[target_col] == 0].sample(n=n, random_state=42)
    df = pd.concat([death_1_sample, death_0_sample], axis=0).sample(frac=1, random_state=42)
    return df


@st.cache_resource(show_spinner=False)
def train_models(df: pd.DataFrame):
    x = df.drop(columns=["DEATH"])
    y = df["DEATH"]
    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.3, random_state=42, stratify=y
    )

    models: Dict[str, object] = {}
    models["Decision Tree"] = DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, random_state=42)
    models["Random Forest"] = RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
    )
    if HAS_LIGHTGBM:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=50, max_depth=4, learning_rate=0.05, random_state=42, verbose=-1
        )

    for model in models.values():
        model.fit(train_x, train_y)

    metrics_rows = []
    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}

    for model_name, model in models.items():
        pred = model.predict(test_x)
        proba = model.predict_proba(test_x)[:, 1]
        metrics_rows.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(test_y, pred),
                "AUC-ROC": roc_auc_score(test_y, proba),
                "Precision": precision_score(test_y, pred),
                "Recall": recall_score(test_y, pred),
                "F1 Score": f1_score(test_y, pred),
            }
        )
        fpr, tpr, _ = roc_curve(test_y, proba)
        roc_data[model_name] = (fpr, tpr, roc_auc_score(test_y, proba))

    metrics_df = pd.DataFrame(metrics_rows).set_index("Model")
    return train_x, test_x, train_y, test_y, models, metrics_df, roc_data


def fig_to_st(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def select_positive_class_shap_values(raw_shap_values):
    if isinstance(raw_shap_values, list):
        if len(raw_shap_values) > 1:
            return np.asarray(raw_shap_values[1])
        return np.asarray(raw_shap_values[0])

    arr = np.asarray(raw_shap_values)
    if arr.ndim == 3:
        if arr.shape[-1] == 2:
            return arr[:, :, 1]
        if arr.shape[1] == 2:
            return arr[:, 1, :]
        if arr.shape[0] == 2:
            return arr[1, :, :]
    if arr.ndim == 2:
        if arr.shape[1] == 2:
            return arr[:, 1]
        if arr.shape[0] == 2:
            return arr[1, :]
    return arr


def select_positive_class_expected_value(raw_expected_value):
    arr = np.asarray(raw_expected_value)
    if arr.ndim == 0:
        return float(arr)
    flat = arr.reshape(-1)
    if flat.size == 1:
        return float(flat[0])
    return float(flat[1])


st.title("COVID-19 Mortality Risk Analytics and Prediction")
st.write(
    "This Streamlit app reproduces the full homework workflow: descriptive analytics, model evaluation, and explainable interactive prediction."
)

uploaded_csv = st.sidebar.file_uploader("Optional: Upload `covid.csv`", type=["csv"])
with st.spinner("Loading data..."):
    raw_data = load_data(uploaded_csv)
    df = build_balanced_sample(raw_data)

with st.spinner("Training models for interactive analysis..."):
    train_x, test_x, train_y, test_y, trained_models, computed_metrics, roc_data = train_models(df)

tabs = st.tabs(
    [
        "Executive Summary",
        "Descriptive Analytics",
        "Model Performance",
        "Explainability & Interactive Prediction",
    ]
)

with tabs[0]:
    st.subheader("Dataset and Prediction Task")
    st.write(
        "This project analyzes an anonymized COVID-19 patient dataset with clinical and demographic features such as "
        "age, sex, COVID diagnosis, hospitalization status, and comorbidities (for example diabetes, hypertension, obesity, "
        "and chronic conditions). The prediction target is `DEATH` (0 = survived, 1 = died). For modeling and visualization, "
        "the analysis uses a balanced sample of 10,000 records (5,000 deaths and 5,000 survivors) to reduce class imbalance "
        "effects and make model comparisons easier to interpret."
    )
    st.subheader("Why This Problem Matters")
    st.write(
        "Mortality risk prediction supports clinical triage and resource planning, especially during infection surges. "
        "If providers can identify high-risk patients earlier, they can prioritize monitoring, ICU readiness, and rapid intervention. "
        "At a systems level, this type of model can improve hospital operations and public health decisions by turning large patient "
        "datasets into actionable risk signals."
    )
    st.subheader("Approach and Key Findings")
    st.write(
        "The workflow compares tree-based classifiers (Decision Tree, Random Forest, and LightGBM) plus a Neural Network from the "
        "notebook. Hyperparameters were tuned in Part 2 with F1 score as the optimization metric, then models were evaluated using "
        "accuracy, precision, recall, F1, and ROC-AUC. Across models, performance was strong and tightly clustered, with AUC near 0.95, "
        "indicating good ranking of high-risk versus low-risk patients."
    )
    st.write(
        "Explainability was performed using SHAP, which helps clinicians understand feature-level contributions to each prediction. "
        "In practice, this is critical because high-stakes healthcare decisions need both predictive strength and transparent reasoning. "
        "The interactive tab in this app lets users set patient features, compare model outputs, and inspect a SHAP waterfall chart for "
        "the same custom patient profile."
    )

with tabs[1]:
    st.subheader("Target Distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x="DEATH", data=df, ax=ax, palette=sns.cubehelix_palette(2))
    ax.set_title("Death Distribution in Balanced Sample")
    ax.set_xlabel("DEATH (0 = Lived, 1 = Died)")
    fig_to_st(fig)
    st.caption(
        "This plot confirms the balanced analysis sample used for modeling (equal survivors and deaths). "
        "Balancing helps prevent a model from appearing accurate simply by over-predicting the majority class."
    )

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df["AGE"], bins=30, kde=True, ax=ax)
        ax.set_title("Age Distribution")
        fig_to_st(fig)
        st.caption(
            "The age distribution is concentrated in younger and middle-age ranges with a decreasing tail in older ages. "
            "This context matters because age is clinically associated with severe COVID outcomes and mortality risk."
        )
    with c2:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x="DEATH", y="AGE", data=df, ax=ax, palette="viridis")
        ax.set_title("Age by Mortality Status")
        ax.set_xlabel("DEATH (0 = Lived, 1 = Died)")
        fig_to_st(fig)
        st.caption(
            "Patients who died show a higher median age and generally older distribution than survivors. "
            "This supports the hypothesis that age is a major risk factor and should be retained in all predictive models."
        )

    if {"COVID_POSITIVE", "HOSPITALIZED"}.issubset(df.columns):
        mortality_rates_df = (
            df.groupby(["COVID_POSITIVE", "HOSPITALIZED"])["DEATH"].mean().reset_index(name="Mortality_Rate")
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(
            x="COVID_POSITIVE",
            y="Mortality_Rate",
            hue="HOSPITALIZED",
            data=mortality_rates_df,
            palette="viridis",
            ax=ax,
        )
        ax.set_title("Mortality Rate by COVID Status and Hospitalization")
        fig_to_st(fig)
        st.caption(
            "Hospitalization and COVID-positive status together correspond to the highest mortality groups in this sample. "
            "This interaction view helps clinicians and operations teams prioritize monitoring intensity and bed allocation."
        )

    if {"COVID_POSITIVE", "DIABETES"}.issubset(df.columns):
        covid_positive_df = df[df["COVID_POSITIVE"] == 1]
        diabetes_rates = covid_positive_df.groupby("DIABETES")["DEATH"].mean().reset_index(name="Mortality_Rate")
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(x="DIABETES", y="Mortality_Rate", data=diabetes_rates, palette="magma", ax=ax)
        ax.set_title("Mortality Rate for COVID-Positive Patients by Diabetes")
        fig_to_st(fig)
        st.caption(
            "Among COVID-positive patients, diabetes is associated with a visibly higher mortality rate. "
            "This indicates diabetes should be treated as a key comorbidity in risk stratification."
        )

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap")
    fig_to_st(fig)
    st.caption(
        "The heatmap summarizes linear relationships across all numeric features and the mortality target. "
        "While correlation does not prove causality, it highlights clusters of related risk factors for deeper model-based analysis."
    )

with tabs[2]:
    st.subheader("Section 2.7 Model Comparison Table (from notebook)")
    st.dataframe(NOTEBOOK_METRICS.style.format("{:.6f}"))

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        x=NOTEBOOK_METRICS.index,
        y=NOTEBOOK_METRICS["F1 Score"].values,
        palette="viridis",
        ax=ax,
    )
    ax.set_title("Comparison of F1 Scores Across Models")
    ax.set_xlabel("Model")
    ax.set_ylabel("F1 Score")
    fig_to_st(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        x=NOTEBOOK_METRICS.index,
        y=NOTEBOOK_METRICS["AUC-ROC"].values,
        palette="magma",
        ax=ax,
    )
    ax.set_title("Comparison of AUC-ROC Scores Across Models")
    ax.set_xlabel("Model")
    ax.set_ylabel("AUC-ROC")
    fig_to_st(fig)

    st.subheader("ROC Curves (recomputed with notebook best hyperparameters)")
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, (fpr, tpr, auc_score) in roc_data.items():
        ax.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.50)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves by Model")
    ax.legend(loc="lower right")
    fig_to_st(fig)

    st.subheader("Best Hyperparameters")
    hp_df = pd.DataFrame.from_dict(BEST_HYPERPARAMETERS, orient="index")
    st.dataframe(hp_df)

    st.subheader("Recomputed Metrics in This App Run")
    st.dataframe(computed_metrics.style.format("{:.4f}"))

with tabs[3]:
    st.subheader("SHAP Explainability")
    explainable_models = [name for name in trained_models.keys() if name in {"Decision Tree", "Random Forest", "LightGBM"}]
    selected_model_name = st.selectbox("Select model for prediction and SHAP", explainable_models)
    selected_model = trained_models[selected_model_name]

    explainer = shap.TreeExplainer(selected_model)
    shap_values_all = select_positive_class_shap_values(explainer.shap_values(test_x))

    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_all, test_x, show=False)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_all, test_x, plot_type="bar", show=False)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.subheader("Interactive Prediction")
    st.write("Set a patient profile below. Unset features are filled with dataset mean values.")

    feature_defaults = train_x.mean().to_dict()
    feature_input = feature_defaults.copy()

    left, right = st.columns(2)
    with left:
        if "AGE" in train_x.columns:
            age_min = int(np.floor(train_x["AGE"].min()))
            age_max = int(np.ceil(train_x["AGE"].max()))
            age_default = int(round(train_x["AGE"].mean()))
            feature_input["AGE"] = st.slider("AGE", min_value=age_min, max_value=age_max, value=age_default)

        for col in ["SEX", "COVID_POSITIVE", "HOSPITALIZED", "PNEUMONIA"]:
            if col in train_x.columns:
                feature_input[col] = st.selectbox(col, options=[0, 1], index=0 if feature_defaults[col] < 0.5 else 1)

    with right:
        for col in ["DIABETES", "HYPERTENSION", "OBESITY", "TOBACCO"]:
            if col in train_x.columns:
                feature_input[col] = st.selectbox(
                    col, options=[0, 1], index=0 if feature_defaults[col] < 0.5 else 1, key=f"sel_{col}"
                )

    input_df = pd.DataFrame([feature_input])[train_x.columns]
    pred_proba = float(selected_model.predict_proba(input_df)[0, 1])
    pred_class = int(pred_proba >= 0.5)

    st.metric("Predicted Class (DEATH)", pred_class)
    st.metric("Predicted Probability of DEATH=1", f"{pred_proba:.4f}")

    st.subheader("SHAP Waterfall for Custom Input")
    shap_values_one = select_positive_class_shap_values(explainer.shap_values(input_df))
    if shap_values_one.ndim == 2:
        shap_values_for_input = shap_values_one[0]
    else:
        shap_values_for_input = shap_values_one

    expected_value = select_positive_class_expected_value(explainer.expected_value)

    explanation = shap.Explanation(
        values=shap_values_for_input,
        base_values=expected_value,
        data=input_df.iloc[0].values,
        feature_names=input_df.columns.tolist(),
    )
    fig = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, show=False)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.sidebar.markdown("---")
st.sidebar.write("If `covid.csv` is not local, the app attempts to load it from Google Drive.")
if not HAS_LIGHTGBM:
    st.sidebar.warning("LightGBM is not installed. The app runs with Decision Tree and Random Forest.")
