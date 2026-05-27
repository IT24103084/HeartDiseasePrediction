import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, and selected features
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")
selected_features = joblib.load("selected_features.pkl")

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Card styling */
    .custom-card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        margin: 15px 0;
    }
    
    /* Header section */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #00bcd4 100%);
        color: white;
        padding: 40px 30px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }
    
    .hero-section h1 {
        font-size: 2.5em;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .hero-section p {
        font-size: 1.1em;
        opacity: 0.95;
    }
    
    /* Input section card */
    .input-card {
        background: white;
        border-radius: 15px;
        padding: 30px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
    }
    
    /* Result cards */
    .result-card-success {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        border-radius: 15px;
        padding: 30px;
        color: #1a5f34;
        box-shadow: 0 8px 16px rgba(132, 250, 176, 0.3);
    }
    
    .result-card-danger {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        border-radius: 15px;
        padding: 30px;
        color: #8b0000;
        box-shadow: 0 8px 16px rgba(250, 112, 154, 0.3);
    }
    
    /* Button styling */
    .predict-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 40px;
        border-radius: 10px;
        font-weight: bold;
        font-size: 1.1em;
        border: none;
        cursor: pointer;
        transition: transform 0.2s;
    }
    
    .predict-button:hover {
        transform: scale(1.02);
    }
    
    /* Sidebar styling */
    .sidebar-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Table styling */
    table {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 20px;
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.9em;
        margin-top: 40px;
        border-top: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Risk level indicator */
    .risk-low {
        color: #2ecc71;
        font-weight: bold;
        font-size: 1.2em;
    }
    
    .risk-high {
        color: #e74c3c;
        font-weight: bold;
        font-size: 1.2em;
    }
    
    /* Metric styling */
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📋 About This App")
    st.markdown("""
    This is a **Heart Disease Prediction System** that uses machine learning 
    to estimate the possibility of heart disease based on important health indicators.
    """)
    
    st.markdown("---")
    st.markdown("### 🔬 Features Used")
    st.markdown("""
    1. **Age** - Patient's age in years
    2. **Chest Pain Type** - Classification of chest pain
    3. **Maximum Heart Rate** - Peak heart rate during exercise
    4. **Exercise Induced Angina** - Angina triggered by exercise
    5. **Oldpeak (ST Depression)** - Exercise-induced ST segment depression
    6. **Major Vessels** - Number of colored vessels
    7. **Thalassemia** - Blood disorder classification
    """)
    
    st.markdown("---")
    st.markdown("### 🛠️ Technologies")
    st.markdown("""
    - **Python** - Programming language
    - **Scikit-learn** - ML model training
    - **Streamlit** - Web framework
    - **Pandas** - Data handling
    - **Joblib** - Model persistence
    """)
    
    st.markdown("---")
    st.markdown("### ⚠️ Disclaimer")
    st.warning("""
    **Educational Purpose Only**
    
    This application is designed for educational purposes and should NOT be used as 
    a replacement for professional medical diagnosis. Always consult a qualified 
    healthcare professional for accurate medical advice.
    """)

# Hero Section
st.markdown("""
<div class="hero-section">
    <h1>❤️ Heart Disease Prediction System</h1>
    <p>Powered by Machine Learning | Enter Your Health Details Below</p>
</div>
""", unsafe_allow_html=True)

# Instructions
st.markdown("""
<div style="background: rgba(255, 255, 255, 0.95); border-radius: 15px; padding: 20px; margin: 20px 0;">
    <h3>📖 How to Use</h3>
    <ul style="font-size: 1em;">
        <li>Enter your health metrics in the form below</li>
        <li>Use realistic values based on medical check-ups</li>
        <li>Click the <b>Predict</b> button to get an estimate</li>
        <li>Review the risk assessment and consult a doctor for confirmation</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Input section
st.markdown("### 📝 Patient Health Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider(
        "👤 Age (years)",
        min_value=18,
        max_value=120,
        value=50,
        step=1,
        help="Select your age in years"
    )

    cp = st.selectbox(
        "💔 Chest Pain Type",
        [0, 1, 2, 3],
        format_func=lambda x: {
            0: "Typical Angina",
            1: "Atypical Angina",
            2: "Non-anginal Pain",
            3: "Asymptomatic"
        }[x],
        help="Select the type of chest pain you experience"
    )

    thalach = st.slider(
        "💓 Maximum Heart Rate Achieved (bpm)",
        min_value=60,
        max_value=220,
        value=150,
        step=1,
        help="Your peak heart rate during exercise"
    )

    exang = st.radio(
        "🏃 Exercise Induced Angina",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        horizontal=True,
        help="Do you experience chest pain during exercise?"
    )

with col2:
    oldpeak = st.slider(
        "📉 ST Depression (Oldpeak)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="ST segment depression induced by exercise"
    )

    ca = st.select_slider(
        "🔴 Number of Major Vessels",
        options=[0, 1, 2, 3, 4],
        value=0,
        help="Number of major vessels colored by fluoroscopy"
    )

    thal = st.selectbox(
        "🩸 Thalassemia",
        [0, 1, 2, 3],
        format_func=lambda x: {
            0: "Unknown",
            1: "Normal",
            2: "Fixed Defect",
            3: "Reversible Defect"
        }[x],
        help="Blood disorder classification"
    )

# Prediction button
st.markdown("---")
col_button1, col_button2, col_button3 = st.columns([1, 2, 1])

with col_button2:
    predict_clicked = st.button(
        "🔍 Predict Heart Disease Possibility",
        use_container_width=True,
        type="primary"
    )

if predict_clicked:
    # Create input dataframe with correct feature order
    input_data = pd.DataFrame({
        "age": [age],
        "cp": [cp],
        "thalach": [thalach],
        "exang": [exang],
        "oldpeak": [oldpeak],
        "ca": [ca],
        "thal": [thal]
    })

    # Scale and predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    lower_probability = probability[0][0] * 100
    higher_probability = probability[0][1] * 100

    # Determine risk level
    if higher_probability > 70:
        risk_level = "🔴 Very High Risk"
        risk_color = "danger"
    elif higher_probability > 50:
        risk_level = "🟠 High Risk"
        risk_color = "danger"
    elif higher_probability > 30:
        risk_level = "🟡 Moderate Risk"
        risk_color = "warning"
    else:
        risk_level = "🟢 Low Risk"
        risk_color = "success"

    st.markdown("---")
    st.markdown("### 📊 Prediction Results")

    # Result cards
    result_col1, result_col2 = st.columns(2)

    with result_col1:
        if prediction[0] == 1:
            st.markdown("""
            <div class="result-card-danger">
                <h3 style="margin-top: 0;">⚠️ Higher Possibility of Heart Disease</h3>
                <p>The model indicates a <b>higher possibility</b> of heart disease based on the provided health metrics.</p>
                <p><b>⚕️ Recommendation:</b> Please consult a qualified medical professional immediately for proper diagnosis and treatment.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-card-success">
                <h3 style="margin-top: 0;">✅ Lower Possibility of Heart Disease</h3>
                <p>The model indicates a <b>lower possibility</b> of heart disease based on the provided health metrics.</p>
                <p><b>💚 Recommendation:</b> Continue maintaining a healthy lifestyle with regular exercise and health checkups.</p>
            </div>
            """, unsafe_allow_html=True)

    with result_col2:
        col_metric1, col_metric2 = st.columns(2)
        
        with col_metric1:
            st.metric(
                "🟢 Lower Possibility",
                f"{lower_probability:.1f}%",
                help="Probability of not having heart disease"
            )
        
        with col_metric2:
            st.metric(
                "🔴 Higher Possibility",
                f"{higher_probability:.1f}%",
                help="Probability of having heart disease"
            )

    # Risk Level Indicator
    st.markdown(f"### Risk Level: {risk_level}")
    st.progress(int(higher_probability) / 100)

    # Input Summary Table
    st.markdown("### 📋 Input Summary")
    summary_data = pd.DataFrame({
        "Feature": [
            "👤 Age",
            "💔 Chest Pain Type",
            "💓 Maximum Heart Rate",
            "🏃 Exercise Induced Angina",
            "📉 ST Depression",
            "🔴 Major Vessels",
            "🩸 Thalassemia"
        ],
        "Value": [
            f"{age} years",
            {
                0: "Typical Angina",
                1: "Atypical Angina",
                2: "Non-anginal Pain",
                3: "Asymptomatic"
            }[cp],
            f"{thalach} bpm",
            "No" if exang == 0 else "Yes",
            f"{oldpeak}",
            f"{ca}",
            {
                0: "Unknown",
                1: "Normal",
                2: "Fixed Defect",
                3: "Reversible Defect"
            }[thal]
        ]
    })

    st.table(summary_data)

else:
    st.markdown("""
    <div style="background: rgba(255, 255, 255, 0.95); border-radius: 15px; padding: 20px; text-align: center; margin: 40px 0;">
        <p style="font-size: 1.1em; color: #666;">
            👉 <b>Enter your health details above and click the "Predict" button to get started!</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>❤️ <b>Heart Disease Prediction System</b> | Educational Project Only</p>
    <p>Built with Python • Scikit-learn • Streamlit | Not a substitute for professional medical diagnosis</p>
    <p>© 2024 - All Rights Reserved | For educational purposes only</p>
</div>
""", unsafe_allow_html=True)