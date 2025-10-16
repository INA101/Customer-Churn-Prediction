# Gender -> 1 Female, 0 Male
# Chrun -> 1 Yes, 0 No  
# Scaler is exported as scaler.pkl
# model is exported as model.pkl
# Order of the X 'Age', 'Gender', 'Tenure', 'MonthlyCharges'

import streamlit as st
import joblib
import numpy as np

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

st.title("Customer Churn Prediction App")

st.divider()

st.write("This app predicts whether a customer will churn or not based on their details.")
st.divider()
st.write("Please enter the values below and hit the predict button ")
st.divider()

age = st.number_input("Enter age of the customer", min_value=10, max_value=100, value=30)

tenure = st.number_input("Enter the tenure of the customer (in months)", min_value=0, max_value=130, value= 130)

monthlycharge = st.number_input("Enter the monthly charge of the customer", min_value=30, max_value=200, value=70)

gender = st.selectbox("Enter the Gender of the customer", options=["Male", "Female"])

st.divider()

predictbutton = st.button("Predict!")

st.divider()

if predictbutton:

    gender_selected = 1 if gender == 'Female' else 0

    X =[age, gender_selected, tenure, monthlycharge]

    X1 = np.array(X)

    X_array = scaler.transform([X1])

    prediction = model.predict(X_array)[0]

    predicted = "Churn" if prediction == 1 else "Not Churn"

    st.balloons()

    st.write(f'The customer is likely to: {predicted}')

else:
    st.write("Please enter all the values and hit the predict button")

