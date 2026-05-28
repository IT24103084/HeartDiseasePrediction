import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


# =========================
# Load model, scaler, and selected features
# =========================
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")
selected_features = joblib.load("selected_features.pkl")


# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)


# =========================
# Session state
# =========================
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "latest_pdf_report" not in st.session_state:
    st.session_state.latest_pdf_report = None

if "latest_csv_report" not in st.session_state:
    st.session_state.latest_csv_report = None

if "latest_report_available" not in st.session_state:
    st.session_state.latest_report_available = False


# =========================
# Helper functions
# =========================
def get_risk_level(higher_probability):
    if higher_probability < 40:
        return "Low"
    elif higher_probability < 70:
        return "Medium"
    else:
        return "High"


def get_health_tips(risk_level):
    if risk_level == "Low":
        return """
        ✅ Low Risk Health Tips

        - Maintain a balanced diet.
        - Exercise regularly.
        - Drink enough water.
        - Avoid smoking and excessive alcohol.
        - Continue regular health checkups.
        """
    elif risk_level == "Medium":
        return """
        ⚠️ Medium Risk Health Tips

        - Monitor blood pressure and cholesterol levels.
        - Reduce oily, salty, and high-sugar foods.
        - Do regular walking, jogging, or light exercise.
        - Manage stress and get enough sleep.
        - Consider consulting a doctor if symptoms continue.
        """
    else:
        return """
        🚨 High Risk Health Tips

        - Please consult a qualified doctor as soon as possible.
        - Do not ignore chest pain, breathing difficulty, or unusual tiredness.
        - Avoid heavy physical activity until medical advice is taken.
        - Monitor blood pressure, cholesterol, and sugar levels.
        - Follow professional medical guidance.
        """


def clean_text_for_pdf(text):
    return (
        text.replace("✅", "")
        .replace("⚠️", "")
        .replace("🚨", "")
        .replace("**", "")
        .replace("\n", "<br/>")
    )


def generate_pdf_report(report_data, health_tips):
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    elements = []

    row = report_data.iloc[0]

    # Title
    title = Paragraph("Heart Disease Prediction Report", styles["Title"])
    subtitle = Paragraph("Machine Learning Based Prediction Summary", styles["Normal"])

    elements.append(title)
    elements.append(Spacer(1, 8))
    elements.append(subtitle)
    elements.append(Spacer(1, 20))

    # Patient input details
    elements.append(Paragraph("Patient Input Details", styles["Heading2"]))

    patient_data = [
        ["Feature", "Value"],
        ["Date & Time", str(row["Date & Time"])],
        ["Age", str(row["Age"])],
        ["Chest Pain Type", str(row["Chest Pain Type"])],
        ["Maximum Heart Rate", str(row["Maximum Heart Rate"])],
        ["Exercise Induced Angina", str(row["Exercise Induced Angina"])],
        ["Oldpeak", str(row["Oldpeak"])],
        ["Major Vessels", str(row["Major Vessels"])],
        ["Thalassemia", str(row["Thalassemia"])]
    ]

    patient_table = Table(patient_data, colWidths=[220, 250])
    patient_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0ea5e9")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))

    elements.append(patient_table)
    elements.append(Spacer(1, 20))

    # Prediction result
    elements.append(Paragraph("Prediction Result", styles["Heading2"]))

    prediction_data = [
        ["Result", "Value"],
        ["Prediction", str(row["Prediction Result"])],
        ["Lower Possibility", f'{row["Lower Possibility (%)"]}%'],
        ["Higher Possibility", f'{row["Higher Possibility (%)"]}%'],
        ["Risk Level", str(row["Risk Level"])]
    ]

    prediction_table = Table(prediction_data, colWidths=[220, 250])
    prediction_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#6366f1")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#eef2ff")),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))

    elements.append(prediction_table)
    elements.append(Spacer(1, 20))

    # Health guidance
    elements.append(Paragraph("Health Guidance", styles["Heading2"]))

    cleaned_tips = clean_text_for_pdf(health_tips)
    elements.append(Paragraph(cleaned_tips, styles["Normal"]))
    elements.append(Spacer(1, 20))

    # Disclaimer
    disclaimer = """
    <b>Disclaimer:</b> This report is generated for educational purposes only.
    It should not be used as a real medical diagnosis.
    Please consult a qualified medical professional for proper medical advice.
    """

    elements.append(Paragraph(disclaimer, styles["Normal"]))
    elements.append(Spacer(1, 20))

    footer = Paragraph(
        "Developed using Python, Scikit-learn, Streamlit, and ReportLab",
        styles["Italic"]
    )
    elements.append(footer)

    doc.build(elements)

    pdf = buffer.getvalue()
    buffer.close()

    return pdf


# =========================
# Sidebar
# =========================
st.sidebar.title("❤️ Heart Disease Prediction")

st.sidebar.write("""
This machine learning app predicts the possibility of heart disease using selected important health features.
""")

st.sidebar.info("""
Selected Features:
- Age
- Chest Pain Type
- Maximum Heart Rate
- Exercise Induced Angina
- Oldpeak
- Major Vessels
- Thalassemia
""")

st.sidebar.markdown("### 🛠️ Technologies Used")
st.sidebar.write("""
- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib
- ReportLab
""")

st.sidebar.warning("""
This is an educational project only.  
It is not a medical diagnosis tool.
""")


# =========================
# Main title
# =========================
st.title("❤️ Heart Disease Prediction System")
st.write("Enter patient health details below and click **Predict** to get the result.")

st.divider()


# =========================
# Information box
# =========================
st.info("""
This app uses a trained machine learning model to estimate whether a patient has a lower or higher possibility of heart disease.
Please enter realistic values for better prediction.
""")


# =========================
# Input Guide
# =========================
with st.expander("📘 Input Guide - What do these fields mean?"):
    st.write("""
    **Age:** Patient's age.

    **Chest Pain Type:** Type of chest pain experienced by the patient.
    - Typical Angina
    - Atypical Angina
    - Non-anginal Pain
    - Asymptomatic

    **Maximum Heart Rate Achieved:** Highest heart rate achieved during exercise.

    **Exercise Induced Angina:** Whether exercise causes chest pain.

    **Oldpeak / ST Depression:** ST depression caused by exercise compared to rest.

    **Number of Major Vessels:** Number of major blood vessels colored by fluoroscopy.

    **Thalassemia:** Blood disorder test result related to thalassemia.
    """)


# =========================
# Model Information
# =========================
with st.expander("🤖 Model Information"):
    st.write("""
    **Project Type:** Machine Learning Classification

    **Algorithm Used:** Random Forest Classifier

    **Number of Selected Features:** 7

    **Output:** Lower or Higher possibility of heart disease

    **Selected Features Used by the Model:**
    - Age
    - Chest Pain Type
    - Maximum Heart Rate Achieved
    - Exercise Induced Angina
    - Oldpeak / ST Depression
    - Number of Major Vessels
    - Thalassemia
    """)


# =========================
# Input section
# =========================
st.subheader("📝 Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30,
        help="Enter the age of the patient"
    )

    cp = st.selectbox(
        "Chest Pain Type",
        [0, 1, 2, 3],
        format_func=lambda x: {
            0: "Typical Angina",
            1: "Atypical Angina",
            2: "Non-anginal Pain",
            3: "Asymptomatic"
        }[x],
        help="Type of chest pain experienced by the patient"
    )

    thalach = st.number_input(
        "Maximum Heart Rate Achieved",
        min_value=60,
        max_value=250,
        value=150,
        help="Maximum heart rate achieved during exercise"
    )

    exang = st.selectbox(
        "Exercise Induced Angina",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        help="Chest pain caused by exercise"
    )

with col2:
    oldpeak = st.number_input(
        "Oldpeak / ST Depression",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="ST depression induced by exercise relative to rest"
    )

    ca = st.selectbox(
        "Number of Major Vessels",
        [0, 1, 2, 3, 4],
        help="Number of major vessels colored by fluoroscopy"
    )

    thal = st.selectbox(
        "Thalassemia",
        [0, 1, 2, 3],
        format_func=lambda x: {
            0: "Unknown",
            1: "Normal",
            2: "Fixed Defect",
            3: "Reversible Defect"
        }[x],
        help="Thalassemia blood disorder result"
    )


# =========================
# Convert selected values to readable text
# =========================
cp_text = {
    0: "Typical Angina",
    1: "Atypical Angina",
    2: "Non-anginal Pain",
    3: "Asymptomatic"
}[cp]

exang_text = "No" if exang == 0 else "Yes"

thal_text = {
    0: "Unknown",
    1: "Normal",
    2: "Fixed Defect",
    3: "Reversible Defect"
}[thal]

st.divider()


# =========================
# Prediction button
# =========================
st.subheader("🔍 Prediction")

if st.button("Predict Heart Disease Possibility", use_container_width=True):
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

    lower_probability = probability[0][0] * 100
    higher_probability = probability[0][1] * 100

    prediction_result = (
        "Higher Possibility of Heart Disease"
        if prediction[0] == 1
        else "Lower Possibility of Heart Disease"
    )

    risk_level = get_risk_level(higher_probability)
    health_tips = get_health_tips(risk_level)

    # Prediction result
    st.subheader("📊 Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        if prediction[0] == 1:
            st.error("⚠️ Higher Possibility of Heart Disease")
            st.write("""
            The model predicts that the patient may have a higher possibility of heart disease.
            Please consult a qualified medical professional for proper diagnosis.
            """)
        else:
            st.success("✅ Lower Possibility of Heart Disease")
            st.write("""
            The model predicts that the patient may have a lower possibility of heart disease.
            However, regular health checkups are still important.
            """)

    with result_col2:
        st.metric("Lower Possibility", f"{lower_probability:.2f}%")
        st.metric("Higher Possibility", f"{higher_probability:.2f}%")

        st.write("### Risk Level")
        st.progress(int(higher_probability))

        if risk_level == "Low":
            st.success("Risk Level: Low")
        elif risk_level == "Medium":
            st.warning("Risk Level: Medium")
        else:
            st.error("Risk Level: High")

    st.divider()

    # Health tips
    st.subheader("💡 Health Tips Based on Result")
    st.info(health_tips)
    st.warning("This advice is for educational purposes only. Please consult a doctor for real medical advice.")

    st.divider()

    # Report data with Date & Time kept
    report_data = pd.DataFrame({
        "Date & Time": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Age": [age],
        "Chest Pain Type": [cp_text],
        "Maximum Heart Rate": [thalach],
        "Exercise Induced Angina": [exang_text],
        "Oldpeak": [oldpeak],
        "Major Vessels": [ca],
        "Thalassemia": [thal_text],
        "Prediction Result": [prediction_result],
        "Lower Possibility (%)": [round(lower_probability, 2)],
        "Higher Possibility (%)": [round(higher_probability, 2)],
        "Risk Level": [risk_level]
    })

    # Input Summary
    st.write("### 📋 Input Summary")
    st.dataframe(report_data, use_container_width=True, hide_index=True)

    # Generate reports
    pdf_report = generate_pdf_report(report_data, health_tips)
    csv_report = report_data.to_csv(index=False).encode("utf-8")

    # Save reports in session state so download buttons do not disappear
    st.session_state.latest_pdf_report = pdf_report
    st.session_state.latest_csv_report = csv_report
    st.session_state.latest_report_available = True

    st.success("✅ Report generated successfully. You can download it from the section below.")

    # Save to prediction history
    st.session_state.prediction_history.append(report_data.iloc[0].to_dict())

else:
    st.write("Click the prediction button after entering patient details.")


# =========================
# Persistent Report Download Section
# =========================
st.divider()
st.subheader("📄 Download Latest Prediction Report")

if st.session_state.latest_report_available:
    st.download_button(
        label="📄 Download Prediction Report as PDF",
        data=st.session_state.latest_pdf_report,
        file_name="heart_disease_prediction_report.pdf",
        mime="application/pdf",
        use_container_width=True
    )

    st.download_button(
        label="📥 Download Report Data as CSV",
        data=st.session_state.latest_csv_report,
        file_name="heart_disease_prediction_data.csv",
        mime="text/csv",
        use_container_width=True
    )
else:
    st.info("No report generated yet. Make a prediction first to download the report.")


# =========================
# Prediction History
# =========================
st.divider()
st.subheader("🕘 Prediction History")

if len(st.session_state.prediction_history) > 0:
    history_df = pd.DataFrame(st.session_state.prediction_history)
    st.dataframe(history_df, use_container_width=True, hide_index=True)

    history_csv = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Full Prediction History",
        data=history_csv,
        file_name="heart_disease_prediction_history.csv",
        mime="text/csv",
        use_container_width=True
    )

    if st.button("🗑️ Clear Prediction History", use_container_width=True):
        st.session_state.prediction_history = []
        st.success("Prediction history cleared successfully.")
        st.rerun()
else:
    st.info("No predictions made yet. Prediction history will appear here after you make a prediction.")


st.divider()


# =========================
# Footer
# =========================
st.caption("""
Developed as a machine learning project using Python, Scikit-learn, Streamlit, and ReportLab.
This application is for educational purposes only and should not be used as a real medical diagnosis system.
""")