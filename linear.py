import streamlit as st
import joblib
import numpy as np

#load model
model = joblib.load("student_model.pkl")

st.title("Student Pass/fail Predictor")

st.write("Enter student details")
study_hours = st.number_input("Study Hours", 0,12)



if st.button("Predict"):
    input_data = np.array([[study_hours]])

    prediction = model.predict(input_data)

    st.success(prediction)

