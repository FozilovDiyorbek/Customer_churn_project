import streamlit as st
import joblib
import pandas as pd

st.title("Customer Churn Prediction")

# Load model
model = joblib.load("models/best_model.pkl")

# Inputs
credit_score = st.slider("Credit Score", 300, 850, 650)
age = st.slider("Age", 18, 100, 35)
tenure = st.slider("Tenure (Years)", 0, 10, 3)
balance = st.number_input("Balance", value=10000.0)
num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", value=50000.0)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])

# Preparing for one-hot encoding
data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary],
    "Geography_France": [1 if geography == "France" else 0],
    "Geography_Germany": [1 if geography == "Germany" else 0],
    "Geography_Spain": [1 if geography == "Spain" else 0],
    "Gender_Female": [1 if gender == "Female" else 0],
    "Gender_Male": [1 if gender == "Male" else 0],
})

if st.button("Predict"):
    prediction = model.predict(data)
    if prediction[0] == 1:
        st.write("🟥 Natija: **The client leaves**")
    else:
        st.write("🟩 Natija: **The client will not leave.**")

