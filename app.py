
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ===============================
# LOAD SAVED MODEL & FILES
# ===============================
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")

# ===============================
# STREAMLIT PAGE
# ===============================
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction App")
st.write("Predict whether a telecom customer is likely to churn.")

# ===============================
# USER INPUTS
# ===============================

# ---- Numeric Inputs (VERY IMPORTANT ORDER)
SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0, value=600.0)

# ---- Categorical Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["NO_INTERNET", "No", "Yes"])
OnlineBackup = st.selectbox("Online Backup", ["NO_INTERNET", "No", "Yes"])
DeviceProtection = st.selectbox("Device Protection", ["NO_INTERNET", "No", "Yes"])
TechSupport = st.selectbox("Tech Support", ["NO_INTERNET", "No", "Yes"])
StreamingTV = st.selectbox("Streaming TV", ["NO_INTERNET", "No", "Yes"])
StreamingMovies = st.selectbox("Streaming Movies", ["NO_INTERNET", "No", "Yes"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

# ===============================
# BUILD INPUT DATAFRAME
# ===============================
input_dict = {
    "SeniorCitizen": 1 if SeniorCitizen == "Yes" else 0,
    "tenure": tenure,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges,

    "gender_Male": 1 if gender == "Male" else 0,
    "Partner_Yes": 1 if Partner == "Yes" else 0,
    "Dependents_Yes": 1 if Dependents == "Yes" else 0,
    "PhoneService_Yes": 1 if PhoneService == "Yes" else 0,

    "MultipleLines_No phone service": 1 if MultipleLines == "No phone service" else 0,
    "MultipleLines_Yes": 1 if MultipleLines == "Yes" else 0,

    "InternetService_DSL": 1 if InternetService == "DSL" else 0,
    "InternetService_Fiber optic": 1 if InternetService == "Fiber optic" else 0,

    "OnlineSecurity_No internet service": 1 if OnlineSecurity == "NO_INTERNET" else 0,
    "OnlineSecurity_Yes": 1 if OnlineSecurity == "Yes" else 0,

    "OnlineBackup_No internet service": 1 if OnlineBackup == "NO_INTERNET" else 0,
    "OnlineBackup_Yes": 1 if OnlineBackup == "Yes" else 0,

    "DeviceProtection_No internet service": 1 if DeviceProtection == "NO_INTERNET" else 0,
    "DeviceProtection_Yes": 1 if DeviceProtection == "Yes" else 0,

    "TechSupport_No internet service": 1 if TechSupport == "NO_INTERNET" else 0,
    "TechSupport_Yes": 1 if TechSupport == "Yes" else 0,

    "StreamingTV_No internet service": 1 if StreamingTV == "NO_INTERNET" else 0,
    "StreamingTV_Yes": 1 if StreamingTV == "Yes" else 0,

    "StreamingMovies_No internet service": 1 if StreamingMovies == "NO_INTERNET" else 0,
    "StreamingMovies_Yes": 1 if StreamingMovies == "Yes" else 0,

    "Contract_One year": 1 if Contract == "One year" else 0,
    "Contract_Two year": 1 if Contract == "Two year" else 0,

    "PaperlessBilling_Yes": 1 if PaperlessBilling == "Yes" else 0,

    "PaymentMethod_Bank transfer (automatic)": 1 if PaymentMethod == "Bank transfer (automatic)" else 0,
    "PaymentMethod_Credit card (automatic)": 1 if PaymentMethod == "Credit card (automatic)" else 0,
    "PaymentMethod_Electronic check": 1 if PaymentMethod == "Electronic check" else 0,
    "PaymentMethod_Mailed check": 1 if PaymentMethod == "Mailed check" else 0
}

input_df = pd.DataFrame([input_dict])

# ===============================
# ALIGN WITH TRAINING FEATURES
# ===============================
input_df = input_df.reindex(columns=model_columns, fill_value=0)

# ===============================
# SCALE NUMERIC FEATURES (FIXED)
# ===============================
numeric_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# ===============================
# PREDICTION
# ===============================
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)[0][1]

st.subheader("🔍 Prediction Result")

if prediction == 1:
    st.error("⚠️ This customer is likely to churn")
else:
    st.success("✅ This customer is likely to stay")

st.write(f"*Churn Probability:* {prediction_proba:.2%}")

# ===============================
# FEATURE IMPORTANCE (OPTIONAL)
# ===============================
if st.checkbox("Show Feature Importance"):
    importances = model.feature_importances_
    top_features = pd.Series(importances, index=model_columns).sort_values(ascending=False)[:10]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=top_features.values, y=top_features.index, ax=ax)
    ax.set_title("Top 10 Features Influencing Churn")
    st.pyplot(fig)
