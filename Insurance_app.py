import mlflow
import xgboost as xgb
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Page Configuration
st.set_page_config(
    page_title="Insurance Premium Predictor",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stNumberInput>div>div>input {
        background-color: #fff;
    }
    .stSuccess {
        font-size: 18px !important;
    }
</style>
""", unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    train_data = pd.read_csv("train.csv")
    X = train_data[['Age', 'Annual Income', 'Health Score']].dropna()
    y = train_data['Premium Amount'].loc[X.index]
    return X, y

X, y = load_data()

# Sidebar for Model Info
with st.sidebar:
    st.header("Model Information")
    st.markdown("""
    - **Algorithm**: XGBoost Regressor
    - **Features**: Age, Annual Income, Health Score
    - **Target**: Insurance Premium Amount
    """)
    
    if st.checkbox("Show Data Summary"):
        st.subheader("Data Statistics")
        st.dataframe(X.describe())
    
    if st.checkbox("Show Feature Distributions"):
        st.subheader("Feature Distributions")
        fig, ax = plt.subplots()
        X.hist(ax=ax, bins=20)
        plt.tight_layout()
        st.pyplot(fig)

# Train Model
@st.cache_resource
def train_model():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    
    # Log model with MLflow
    with mlflow.start_run() as run:
        mlflow.xgboost.log_model(model, "model", signature=infer_signature(X_train, y_train))
    
    return model, X_test, y_test

model, X_test, y_test = train_model()

# Main Content
st.title("🏥 Insurance Premium Prediction")
st.markdown("Predict your health insurance premium based on key factors")

# Prediction Section
col1, col2 = st.columns(2)

with col1:
    st.subheader("Enter Your Details")
    with st.form("prediction_form"):
        age = st.slider("Age", 18, 100, 30)
        income = st.number_input("Annual Income (₹)", min_value=1000, value=500000, step=1000)
        health_score = st.slider("Health Score (0-100)", 0.0, 100.0, 75.0, step=0.5)
        
        submitted = st.form_submit_button("Calculate Premium")
        
        if submitted:
            input_data = pd.DataFrame([[age, income, health_score]], 
                                    columns=['Age', 'Annual Income', 'Health Score'])
            prediction = model.predict(input_data)[0]
            
            st.session_state.prediction = prediction

with col2:
    st.subheader("Prediction Result")
    
    if 'prediction' in st.session_state:
        st.markdown(f"""
        <div style="background-color:#e6f3ff; padding:20px; border-radius:10px;">
            <h3 style="color:#0066cc;">Estimated Premium</h3>
            <h1 style="color:#0066cc;">₹{st.session_state.prediction:,.2f}</h1>
            <p>Based on your input:</p>
            <ul>
                <li>Age: {age} years</li>
                <li>Income: ₹{income:,.0f}</li>
                <li>Health Score: {health_score}/100</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Show comparison to average
        avg_premium = y.mean()
        diff = st.session_state.prediction - avg_premium
        st.metric("Comparison to Average Premium", 
                 f"₹{st.session_state.prediction:,.2f}", 
                 f"{'Above' if diff > 0 else 'Below'} average by ₹{abs(diff):,.2f}")

# Model Performance Section
st.subheader("Model Performance Metrics")
col1, col2, col3 = st.columns(3)

with col1:
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    st.metric("Mean Absolute Error", f"₹{mae:,.2f}")

with col2:
    r2 = r2_score(y_test, y_pred)
    st.metric("R² Score", f"{r2:.2%}")

with col3:
    error_dist = y_test - y_pred
    st.metric("Error Distribution", f"±₹{np.std(error_dist):,.2f}")

# Feature Importance Visualization
st.subheader("Feature Importance")
importance = model.feature_importances_
features = X.columns
fig, ax = plt.subplots()
ax.barh(features, importance)
ax.set_xlabel('Importance Score')
ax.set_title('What Factors Affect Premium Most?')
st.pyplot(fig)

# Data Explorer
expander = st.expander("Explore Sample Data")
expander.dataframe(X.join(y).sample(10, random_state=42))