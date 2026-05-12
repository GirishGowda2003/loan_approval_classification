import pickle
import streamlit as st
import numpy as np
import pandas as pd

# Load model
with open("ann_model (1).pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Loan Prediction Dashboard", layout="centered")

st.title("🏦 Loan Approval Prediction")
st.write("Fill applicant details to predict loan status")

# ---------------- INPUT FIELDS ---------------- #

col1, col2 = st.columns(2)

with col1:
    ApplicantIncome = st.number_input("Applicant Income", min_value=0.0, value=5000.0)
    CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0.0, value=0.0)
    LoanAmount = st.number_input("Loan Amount", min_value=0.0, value=120.0)
    Loan_Amount_Term = st.number_input("Loan Term (in months)", min_value=0.0, value=360.0)

with col2:
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Gender_Male = 1 if Gender == "Male" else 0

    married = st.selectbox("Married", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    dependents = st.selectbox("Dependents", [0, 1, 2, 3])
    education = st.selectbox("Education", [0, 1], format_func=lambda x: "Not Graduate" if x == 0 else "Graduate")
    self_employed = st.selectbox("Self Employed", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    Credit_History = st.selectbox("Credit History", [0, 1], format_func=lambda x: "Bad" if x == 0 else "Good")
    property_area = st.selectbox("Property Area", [0, 1, 2], format_func=lambda x: ["Rural", "Semiurban", "Urban"][x])

# ---------------- PREDICTION ---------------- #

if st.button("Predict Loan Status", use_container_width=True):

    # ✅ Correct feature order (VERY IMPORTANT)
    input_df = pd.DataFrame({
        'Gender': [Gender_Male],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],
        'ApplicantIncome': [ApplicantIncome],
        'CoapplicantIncome': [CoapplicantIncome],
        'LoanAmount': [LoanAmount],
        'Loan_Amount_Term': [Loan_Amount_Term],
        'Credit_History': [Credit_History],
        'Property_Area': [property_area]
    })

    # Convert to numpy (no scaling applied)
    input_data = input_df.values

    # Prediction
    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result:")

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    # Probability
    try:
        prob = model.predict_proba(input_data)[0][1]
        st.info(f"Approval Probability: {prob:.2%}")
    except:
        st.info("Probability not available for this model.")

st.info("💡 Note: This version works without scaler. For better accuracy, retrain model using StandardScaler and Pipeline.")