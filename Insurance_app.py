import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('xgb_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Custom styling
st.markdown("""
    <style>
    .stApp {background-color: #f4f4f4;}
    .title {color: #2E4053; text-align: center; font-size: 28px;}
    .subheader {color: #566573; text-align: center; font-size: 18px;}
    .info-box {background-color: #EAEDED; padding: 15px; border-radius: 10px; margin-bottom: 15px;}
    </style>
    """, unsafe_allow_html=True)

# App Header
st.markdown("<h1 class='title'>🚗 Insurance Premium Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Enter your details below to estimate your insurance premium.</p>", unsafe_allow_html=True)

# Form for User Input
with st.form("insurance_form"):
    col1, col2 = st.columns(2)  # Create two columns
    
    with col1:
        age = st.number_input("🔢 Age", min_value=18, max_value=100, step=1)
        gender = st.selectbox("⚥ Gender", ["Male", "Female", "Other"])
        marital_status = st.selectbox("💍 Marital Status", ["Single", "Married", "Divorced", "Widowed"])
        dependents = st.number_input("👨‍👩‍👧‍👦 Number of Dependents", min_value=0, max_value=10, step=1)
        education = st.selectbox("🎓 Education Level", ["High School", "Bachelor's", "Master's", "Doctorate"])
        health_score = st.slider("💪 Health Score", min_value=0, max_value=100, step=1)
        policy_type = st.selectbox("📜 Policy Type", ["Basic", "Standard", "Premium"])
    
    with col2:
        prev_claims = st.number_input("📝 Previous Claims", min_value=0, max_value=20, step=1)
        vehicle_age = st.number_input("🚗 Vehicle Age (years)", min_value=0, max_value=30, step=1)
        credit_score = st.slider("💳 Credit Score", min_value=300, max_value=850, step=1)
        insurance_duration = st.number_input("📆 Insurance Duration (years)", min_value=1, max_value=50, step=1)
        exercise_freq = st.selectbox("🏃 Exercise Frequency", ["Never", "Occasionally", "Regularly"])
        annual_income = st.number_input("💰 Annual Income ($)", min_value=10000, max_value=500000, step=1000)
    
    submit = st.form_submit_button("📊 Predict Premium")

# Prediction Logic
if submit:
    # Encoding categorical variables
    gender_dict = {"Male": 0, "Female": 1, "Other": 2}
    marital_dict = {"Single": 0, "Married": 1, "Divorced": 2, "Widowed": 3}
    education_dict = {"High School": 0, "Bachelor's": 1, "Master's": 2, "Doctorate": 3}
    policy_dict = {"Basic": 0, "Standard": 1, "Premium": 2}
    exercise_dict = {"Never": 0, "Occasionally": 1, "Regularly": 2}

    input_data = np.array([
        age, gender_dict[gender], marital_dict[marital_status], dependents,
        education_dict[education], health_score, policy_dict[policy_type],
        prev_claims, vehicle_age, credit_score, insurance_duration,
        exercise_dict[exercise_freq], annual_income
    ]).reshape(1, -1)
    
    # Predict the insurance premium amount
    predicted_premium = model.predict(input_data)[0]
    
    # Display the result beautifully
    st.markdown(f"""
        <div class="info-box">
        <h3>💰 Estimated Insurance Premium: <span style="color:#2E86C1">${predicted_premium:,.2f}</span></h3>
        </div>
        """, unsafe_allow_html=True)
