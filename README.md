# -Indian-Diabetes-Prediction
📖 Project Overview  This project predicts the risk of diabetes using the Pima Indians Diabetes dataset. It includes data cleaning, feature engineering, machine learning models (Logistic Regression and Random Forest), and a fully interactive Streamlit dashboard.
The dashboard provides:

Single patient prediction (with gauge chart and SHAP explanations).

Batch prediction from CSV/Excel files.

Download of batch results as CSV.

Model evaluation tools: Confusion Matrix, ROC-AUC, Calibration Curve, and Decision Curve Analysis (DCA).

Global interpretability with SHAP summary plots.

✨ Features -->

  - Data preprocessing & cleaning

       - Handle missing values (zeros treated as missing).
       - Median imputation + missing indicators.
       - Remove unreliable Insulin column.

  - Feature Engineering

       - Glucose categories: Normal / Pre-diabetes / Diabetes.
       - BMI categories: Normal / Overweight / Obese.
       - Age bins: Young / Middle / Older.

     - Machine Learning Models

        - Logistic Regression (scaled inputs).
        - Random Forest (class-balanced).

   - Interactive Streamlit App

       - Probability Gauge Chart instead of plain numbers.
       - SHAP explanations (per-patient and global).
       - Batch input (CSV/Excel upload).
       - Download results to CSV.

      - Evaluation Tab:
       - Confusion Matrix (with threshold slider).
       - ROC Curve & AUC.
       - Calibration Curve & Brier Score.
       - Decision Curve Analysis (DCA).
       - SHAP Summary (global feature impact).

   - ⚙️ Requirements
        - numpy
        - pandas
        - scikit-learn
        - streamlit
        - matplotlib
        - plotly
        - shap

     - Install:
          - pip install -r requirements.txt

-  🚀 How to Run

Place the dataset pima-indians-diabetes.csv in the same directory as app.py.

Run:

streamlit run app.py


Use the sidebar to select model, threshold, and enter patient features.

Explore the Batch and Evaluation tabs for group predictions and model analysis.

📂 Project Structure
project/
│── app.py                        # Streamlit app
│── indian diabetes.ipynb         # Original notebook with preprocessing & experiments
│── pima-indians-diabetes.csv     # Dataset
│── requirements.txt
│── README.md

⚠️ Notes

This app is for educational purposes only and must not be used for medical diagnosis.

Results are based on the limited Pima Indians dataset and may not generalize to other populations.

