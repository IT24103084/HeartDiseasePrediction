import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, and selected features
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")
selected_features = joblib.load("selected_features.pkl")

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction System")
st.write("This app predicts the possibility of heart disease using selected important health features.")

st.warning("This project is for educational purposes only. It is not a real medical diagnosis tool.")

st.subheader("Enter Patient Details")

age = st.number_input("Age", min_value=1, max_value=120, value=30)

cp = st.selectbox(
    "Chest Pain Type",
    [0, 1, 2, 3],
    format_func=lambda x: {
        0: "Typical Angina",
        1: "Atypical Angina",
        2: "Non-anginal Pain",
        3: "Asymptomatic"
    }[x]
)

thalach = st.number_input(
    "Maximum Heart Rate Achieved",
    min_value=60,
    max_value=250,
    value=150
)

exang = st.selectbox(
    "Exercise Induced Angina",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

oldpeak = st.number_input(
    "Oldpeak / ST Depression",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)

ca = st.selectbox(
    "Number of Major Vessels",
    [0, 1, 2, 3, 4]
)

thal = st.selectbox(
    "Thalassemia",
    [0, 1, 2, 3],
    format_func=lambda x: {
        0: "Unknown",
        1: "Normal",
        2: "Fixed Defect",
        3: "Reversible Defect"
    }[x]
)

if st.button("Predict"):
    input_data = pd.DataFrame({
        "age": [age],
        "cp": [cp],
        "thalach": [thalach],
        "exang": [exang],
        "oldpeak": [oldpeak],
        "ca": [ca],
        "thal": [thal]
    })

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("Higher possibility of heart disease")
    else:
        st.success("Lower possibility of heart disease")

    st.write("Prediction confidence:")
    st.write(f"Lower possibility: {probability[0][0] * 100:.2f}%")
    st.write(f"Higher possibility: {probability[0][1] * 100:.2f}%")