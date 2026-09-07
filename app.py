import streamlit as st
import pandas as pd
import numpy as np
import joblib

#Load the trained model and scaler
heart_model = joblib.load('heart_failure_model.pkl')
heart_scaler = joblib.load('scaler.pkl')

st.title("Heart Failure Prediction App")
st.image("https://cdn.britannica.com/05/85505-050-A6EC2994/Cross-section-human-heart.jpg", caption="Heart",width=400)

age = st.slider("Age", 20, 100, 50)
anaemia = st.selectbox("Anaemia (0: No, 1: Yes)", [0, 1])
creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase (mcg/L)", min_value=0, max_value=10000, value=600)
diabetes = st.radio("Diabetes", (0, 1))
ejection_fraction = st.slider("Ejection Fraction (%)", 10, 80, 30)
high_blood_pressure = st.radio("High Blood Pressure", (0, 1))
platelets = st.number_input("Platelets (kiloplatelets/mL)", min_value=0, max_value=1000000, value=250000)
serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.0, max_value=10.0, value=1.0)
serum_sodium = st.number_input("Serum Sodium (mEq/L)", min_value=100, max_value=150, value=135)
sex = st.radio("Sex (0: Female, 1: Male)", (0, 1))
smoking = st.radio("Smoking (0: No, 1: Yes)", (0, 1))
time = st.number_input("Time (days)", min_value=0, max_value=500, value=100)

if st.button("Heart Failure Prediction"):
    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        'age': [age],
        'anaemia': [anaemia],
        'creatinine_phosphokinase': [creatinine_phosphokinase],
        'diabetes': [diabetes],
        'ejection_fraction': [ejection_fraction],
        'high_blood_pressure': [high_blood_pressure],
        'platelets': [platelets],
        'serum_creatinine': [serum_creatinine],
        'serum_sodium': [serum_sodium],
        'sex': [sex],
        'smoking': [smoking],
        'time': [time]
    })
    scaled_data = heart_scaler.transform(input_data)
    prediction = heart_model.predict(scaled_data)

    if prediction[0] == 1:
        st.error("The patient is at risk of heart failure.")
    else:
        st.success("The patient is not at risk of heart failure.")
