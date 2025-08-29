"""
A Streamlit app for predicting diabetes risk using the Pima Indians dataset.

This application reproduces the data preparation and modelling pipeline from
the provided Jupyter notebook.  Upon startup the app loads the
`pima‑indians‑diabetes.csv` data set, cleans the data, performs feature
engineering, trains two models (a logistic regression and a random
forest) and exposes a simple interface for interactive predictions.

To run the app locally install the required dependencies (pandas,
numpy, scikit‑learn, matplotlib and streamlit) and start the server via

    streamlit run ints/app.py

Make sure that the file ``pima‑indians‑diabetes.csv`` is placed in the same
directory as this script.  If you wish to experiment with different
models or threshold values the relevant options are surfaced in the
sidebar.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score



@dataclass
class TrainedModels:
    """A simple container for fitted models and preprocessing artefacts."""

    logreg: LogisticRegression
    rf: RandomForestClassifier
    scaler: StandardScaler
    feature_names: List[str]
    median_values: Dict[str, float]


def load_raw_data(csv_path: pathlib.Path) -> pd.DataFrame:
    """Load the Pima Indians Diabetes CSV file.

    Parameters
    ----------
    csv_path: pathlib.Path
        The location of the CSV file.

    Returns
    -------
    DataFrame
        Raw dataset with canonical column names.
    """
    df = pd.read_csv(csv_path)
    # Ensure consistent column names as expected in the notebook
    df.columns = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
        "Outcome",
    ]
    return df


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning and feature engineering steps to the raw data.

    The transformations mirror those implemented in the provided
    Jupyter notebook:

    * Replace implausible zeros with NaN for certain physiological
      measurements.
    * Fill missing values with the median (for continuous variables) or
      create missing indicators for variables with many zero values.
    * Drop the original ``Insulin`` measurement while retaining a
      missingness indicator.
    * Derive categorical groupings for glucose, body mass index and age.
    * One‑hot encode these categories with the first level dropped to
      avoid multicollinearity.

    Parameters
    ----------
    df: DataFrame
        Raw Pima Indians data with canonical column names.

    Returns
    -------
    DataFrame
        A new data frame ready for modelling, containing both the
        engineered numeric features and one‑hot encoded categorical
        features.
    """
    # Work on a copy to avoid mutating the original data
    df_clean = df.copy()

    # Columns where zero indicates missingness
    hidden_missing = ["Glucose", "BloodPressure", "SkinThickness", "BMI"]
    df_clean[hidden_missing] = df_clean[hidden_missing].replace(0, np.nan)

    # Fill selected columns with their median values
    fill_missing_value = ["Glucose", "BloodPressure", "BMI"]
    for col in fill_missing_value:
        median_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_val)

    # Create missingness indicators for SkinThickness and Insulin; set zeros to NaN
    for col in ["SkinThickness", "Insulin"]:
        indicator = f"{col}_missing"
        # 1 if original value is missing or zero, else 0
        df_clean[indicator] = df_clean[col].apply(lambda x: int(pd.isna(x) or x == 0))
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # Drop the original Insulin measurement entirely (as in the notebook)
    if "Insulin" in df_clean.columns:
        df_clean = df_clean.drop(columns=["Insulin"])

    # Feature engineering: categorise glucose levels
    def glucose_cat(x: float) -> str:
        if x < 100:
            return "Normal"
        elif x < 126:
            return "Pre-diabetes"
        else:
            return "Diabetes"

    df_clean["Glucose_category"] = df_clean["Glucose"].apply(glucose_cat)

    # Categorise BMI according to WHO definitions
    def bmi_cat(x: float) -> str:
        if x < 25:
            return "Normal"
        elif x < 30:
            return "Overweight"
        else:
            return "Obese"

    df_clean["BMI_category"] = df_clean["BMI"].apply(bmi_cat)

    # Bin ages into meaningful groups
    def age_bin(x: float) -> str:
        if x < 30:
            return "Young"
        elif x < 50:
            return "Middle"
        else:
            return "Older"

    df_clean["Age_bins"] = df_clean["Age"].apply(age_bin)

    # One‑hot encode the categorical variables; drop the first level to avoid
    # multicollinearity. get_dummies handles new categories gracefully when
    # ``dummy_na=False`` (the default).
    df_model = pd.get_dummies(
        df_clean,
        columns=["Glucose_category", "BMI_category", "Age_bins"],
        drop_first=True,
    )

    return df_model


@st.cache_data(show_spinner=False)
def train_models(
    df_model: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> TrainedModels:
    """Train logistic regression and random forest models on the prepared data.

    This function splits the input data into training and test subsets,
    standardises the numeric predictors (for the logistic model) and fits
    both classifiers.  The fitted models, scalers and supporting
    artefacts are returned for downstream prediction.

    Caching prevents re‑training on each rerun of the app; training
    occurs only when the underlying data changes.

    Parameters
    ----------
    df_model: DataFrame
        The fully engineered model matrix with dummy variables and
        continuous predictors.
    test_size: float, optional
        Proportion of the data used for testing.  The remainder is used
        for training.  Defaults to 0.2.
    random_state: int, optional
        Seed to ensure reproducible splits and model initialisation.

    Returns
    -------
    TrainedModels
        A container holding the logistic regression, random forest,
        standardiser, final feature ordering and median values used for
        imputation.
    """
    # Separate predictors and target
    X = df_model.drop(columns=["Outcome"])
    y = df_model["Outcome"]

    # Compute medians for imputing missing values in new samples
    median_values = {
        "Glucose": df_model["Glucose"].median(),
        "BloodPressure": df_model["BloodPressure"].median(),
        "BMI": df_model["BMI"].median(),
        "SkinThickness": df_model["SkinThickness"].median(),
    }

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Standardiser for logistic regression (only on training data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Logistic regression classifier
    logreg = LogisticRegression(max_iter=500, random_state=random_state)
    logreg.fit(X_train_scaled, y_train)

    # Random forest classifier
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)

    feature_names = list(X_train.columns)
    

    return TrainedModels(
        logreg=logreg,
        rf=rf,
        scaler=scaler,
        feature_names=feature_names,
        median_values=median_values,
    )


def preprocess_user_input(
    user_input: Dict[str, float], models: TrainedModels
) -> pd.DataFrame:
    """Transform raw user input into the model's feature space.

    The incoming dictionary contains the eight basic measurements:
    Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
    BMI, DiabetesPedigreeFunction and Age.  This function mirrors
    the feature engineering applied to the training data so that
    predictions can be made on a single observation.

    Parameters
    ----------
    user_input: dict
        A dictionary keyed by feature names with numeric values as
        provided by the user.
    models: TrainedModels
        The artefacts from training including medians and feature order.

    Returns
    -------
    DataFrame
        A single‑row frame with columns aligned to the training feature
        order.  One‑hot encoded categorical variables are created on
        demand and missing columns are filled with zeros.
    """
    # Convert dictionary to DataFrame
    df = pd.DataFrame([user_input])

    # Handle missingness similarly to training: treat zeros as missing
    # for the specified columns and fill with median values
    for col in ["Glucose", "BloodPressure", "BMI"]:
        if df.at[0, col] == 0:
            df.at[0, col] = models.median_values[col]

    # Missing indicators for SkinThickness and Insulin
    df["SkinThickness_missing"] = 1 if df.at[0, "SkinThickness"] == 0 else 0
    df["Insulin_missing"] = 1 if df.at[0, "Insulin"] == 0 else 0

    # Fill SkinThickness zeros with median
    if df.at[0, "SkinThickness"] == 0:
        df.at[0, "SkinThickness"] = models.median_values["SkinThickness"]

    # Drop original Insulin value (mirror training step)
    df = df.drop(columns=["Insulin"])

    # Feature engineering: categorisation for glucose, BMI and age
    glucose_value = df.at[0, "Glucose"]
    if glucose_value < 100:
        glu_cat = "Normal"
    elif glucose_value < 126:
        glu_cat = "Pre-diabetes"
    else:
        glu_cat = "Diabetes"

    bmi_value = df.at[0, "BMI"]
    if bmi_value < 25:
        bmi_cat = "Normal"
    elif bmi_value < 30:
        bmi_cat = "Overweight"
    else:
        bmi_cat = "Obese"

    age_value = df.at[0, "Age"]
    if age_value < 30:
        age_bin = "Young"
    elif age_value < 50:
        age_bin = "Middle"
    else:
        age_bin = "Older"

    df["Glucose_category"] = glu_cat
    df["BMI_category"] = bmi_cat
    df["Age_bins"] = age_bin

    # One‑hot encode the new categorical variables (dropping the first level)
    df_enc = pd.get_dummies(
        df,
        columns=["Glucose_category", "BMI_category", "Age_bins"],
        drop_first=True,
    )

    # Align the columns to match the order expected by the model
    X = df_enc.reindex(columns=models.feature_names, fill_value=0)
    return X
    


def main() -> None:
    st.set_page_config(
        page_title="Diabetes Risk Predictor",
        page_icon="🩺",
        layout="centered",
    )
    st.title("🩺 Diabetes Risk Predictor")
    st.write(
        """
        Enter patient information in the sidebar to estimate the
        probability of developing diabetes.  The app reproduces the
        feature engineering and modelling pipeline of the attached
        notebook.  You can choose between a logistic regression or a
        random forest model and adjust the decision threshold for the
        classification.  The probability and predicted class are
        displayed below.
        """
    )
    
    
    
    # Sidebar for feature input
    st.sidebar.header("Patient Parameters")
    user_input = {
        "Pregnancies": st.sidebar.number_input(
            "Pregnancies",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
            help="Number of times the patient was pregnant",
        ),
        "Glucose": st.sidebar.number_input(
            "Glucose (mg/dL)",
            min_value=0,
            max_value=300,
            value=120,
            step=1,
            help="Plasma glucose concentration a 2 hours in an oral glucose tolerance test",
        ),
        "BloodPressure": st.sidebar.number_input(
            "Blood Pressure (mm Hg)",
            min_value=0,
            max_value=200,
            value=70,
            step=1,
            help="Diastolic blood pressure in mm Hg",
        ),
        "SkinThickness": st.sidebar.number_input(
            "Skin Thickness (mm)",
            min_value=0,
            max_value=100,
            value=20,
            step=1,
            help="Triceps skin fold thickness in mm",
        ),
        "Insulin": st.sidebar.number_input(
            "Insulin (IU/mL)",
            min_value=0,
            max_value=900,
            value=80,
            step=1,
            help="2-Hour serum insulin in µU/mL. Enter 0 if unknown.",
        ),
        "BMI": st.sidebar.number_input(
            "Body Mass Index (kg/m²)",
            min_value=0.0,
            max_value=80.0,
            value=30.0,
            step=0.1,
            help="Body mass index (weight in kg/(height in m)²)",
        ),
        "DiabetesPedigreeFunction": st.sidebar.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.5,
            step=0.01,
            help="A function which scores likelihood of diabetes based on family history",
        ),
        "Age": st.sidebar.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=33,
            step=1,
            help="Age in years",
        ),
    }
    
    user_input_data_frame = pd.DataFrame.from_dict(user_input, orient='index').T
    user_input_data_frame
    # Path to the CSV file; adjust if necessary
    csv_file = pathlib.Path(__file__).with_name("pima-indians-diabetes.csv")

    # Load and preprocess data for training; note that this executes only
    # once per session due to caching
    try:
        raw_df = load_raw_data(csv_file)
        prepared_df = preprocess_dataframe(raw_df)
    except FileNotFoundError:
        st.error(
            f"Unable to locate {csv_file}. Please place the data file in "
            f"the same directory as this script."
        )
        st.stop()

    models = train_models(prepared_df)

    # Model selection
    model_choice = st.sidebar.radio(
        "Choose a model", ["Logistic Regression", "Random Forest"], index=0
    )

    # Decision threshold slider
    threshold = st.sidebar.slider(
        "Decision Threshold", 0.05, 0.95, 0.5, 0.01,
        help="Probability above which the model will classify a patient as diabetic."
    )

    # When the user clicks the predict button
    if st.sidebar.button("Predict"):
        # Preprocess the user input to align with training feature space
        X_user = preprocess_user_input(user_input, models)

        if model_choice == "Logistic Regression":
            # Scale features using the training scaler
            X_scaled = models.scaler.transform(X_user)
            prob = models.logreg.predict_proba(X_scaled)[0, 1]
        else:
            prob = models.rf.predict_proba(X_user)[0, 1]

        # Determine the class based on threshold
        prediction = int(prob >= threshold)
        st.subheader("Predicted Probability")
        st.metric(
            label="Risk of Diabetes", value=f"{prob*100:.1f}%",
            delta=None
        )
        st.subheader("Predicted Class")
        if prediction == 1:
            st.success(
                f"The model predicts **Positive** for diabetes at the chosen threshold "
                f"({threshold:.2f})."
            )
        else:
            st.info(
                f"The model predicts **Negative** for diabetes at the chosen threshold "
                f"({threshold:.2f})."
            )
          
       

    # Display feature importance when the random forest is selected
    st.sidebar.subheader("Feature Importance (Random Forest)")
    # Plot only when models have been trained and the tree model is selected
    importances = models.rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = 10
    top_features = [models.feature_names[i] for i in indices[:top_n]]
    top_importances = importances[indices[:top_n]]
    
    st.write(f"@ threshold = {threshold:.2f}")

    # Create a simple bar chart using Streamlit's builtin chart support
    importance_df = pd.DataFrame({
        "Feature": top_features,
        "Importance": top_importances,
    })
    st.sidebar.bar_chart(
        importance_df.set_index("Feature")
    )


if __name__ == "__main__":
    main()