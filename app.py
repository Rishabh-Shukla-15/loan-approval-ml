import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Loan Prediction", layout="wide")
st.title("🏦 Loan Approval Prediction – ML Model Comparison")

# ---------------- LOAD MODELS ----------------
models = {
    "Logistic Regression": joblib.load(os.path.join(BASE_DIR, "saved_models", "logistic.pkl")),
    "Decision Tree": joblib.load(os.path.join(BASE_DIR, "saved_models", "decision_tree.pkl")),
    "Random Forest": joblib.load(os.path.join(BASE_DIR, "saved_models", "random_forest.pkl")),
    "XGBoost (Best)": joblib.load(os.path.join(BASE_DIR, "saved_models", "xgboost.pkl")),
}

model_name = st.selectbox("Select Model", list(models.keys()))
model = models[model_name]

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data")
    st.dataframe(df.head())

    # ---------------- PREPROCESSING (MATCH TRAINING) ----------------

    # Drop non-feature columns
    if "Loan_ID" in df.columns:
        df = df.drop("Loan_ID", axis=1)

    if "Loan_Status" in df.columns:
        df = df.drop("Loan_Status", axis=1)

    # Separate categorical & numerical columns
    cat_cols = df.select_dtypes(include="object").columns
    num_cols = df.select_dtypes(exclude="object").columns

    # Impute missing values
    df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])
    df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])

    # Encode categorical columns
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Scale numerical columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # ---------------- PREDICTION ----------------
    preds = model.predict(df)

    df["Prediction"] = preds
    df["Prediction"] = df["Prediction"].map({1: "Approved", 0: "Rejected"})

    st.subheader("✅ Predictions")
    st.dataframe(df)
