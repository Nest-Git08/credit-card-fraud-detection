import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Credit Card Fraud Detector",
    layout="centered",
    initial_sidebar_state="collapsed",
    page_icon="💳"
)

with st.sidebar:
    st.title("What's This?")
    st.markdown("""
    This fraud detection tool uses a trained ensemble model, that combines XGBoost, Logistic Regression, and LightGBM models,
    to evaluate the likelihood that a credit card transaction is fraudulent.
    Enter the Time, V1-V28 features, and Amount, and click 'Detect' to get a result.
    """)
    st.markdown("---")
    st.markdown(""":blue[Did you know?]  
                Fraudulent transactions tend to spike around :blue[2 AM] in the morning! """)
    st.markdown("---")
    st.caption("Optimized for mobile and desktop use")

st.title("Credit Card Fraud Detector")
st.markdown("Use this tool to evaluate the risk of fraud for a transaction.")


@st.cache_resource
def load_model():
    model_path = (r"C:\Users\Ernest\Python Projects\CODAR_CAPSTONE_PROJECTS\Credit_Fraud_Detection\models\ensemble_model\ensemble.pkl")
    scaler_path = (r"C:\Users\Ernest\Python Projects\CODAR_CAPSTONE_PROJECTS\Credit_Fraud_Detection\models\scalers\MinMax-Scaler.pkl")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


model, scaler = load_model()

with st.form("txn_form"):
    st.subheader("Enter Transaction Features")
    cols = [f"V{i}" for i in range(1, 29)] + ["Amount"] + ["Time"]
    inputs = []

    for i in range(0, len(cols), 3):
        col_group = st.columns(3)
        for j in range(3):
            if i + j < len(cols):
                val = col_group[j].number_input(f"{cols[i + j]}", value=0.0, step=0.01)
                inputs.append(val)

    submitted = st.form_submit_button("Detect Fraud"),

if submitted:
    X_input = np.array([inputs])
    X_scaled = scaler.transform(X_input)
    prediction = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]

    st.markdown("---")
    if prediction == 1:
        st.error(f"Fraud Detected! (Confidence: {prob:.2%})")
        st.markdown("Consider investigating this transaction further.")
    else:
        st.success(f"Legit Transaction (Confidence: {1 - prob:.2%})")
        st.markdown("No immediate action required.")


@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\Ernest\Python Projects\CODAR_CAPSTONE_PROJECTS\Credit_Fraud_Detection\data\creditcard_data.csv")
    return df


df = load_data()
st.subheader("Fraud Frequency by Hour of Day")

# --- Fraud Frequency Chart ---
df['Hour'] = (df['Time'] // 3600) % 24
fraud_by_hour = df[df['Class'] == 1]['Hour'].value_counts().reset_index()
fraud_by_hour.columns = ['Hour', 'FraudCount']
fraud_by_hour = fraud_by_hour.sort_values('Hour')

fig = px.bar(
    fraud_by_hour,
    x='Hour',
    y='FraudCount',
    title='Fraud Frequency by Hour of Day',
    color='FraudCount',
    color_continuous_scale='Reds'
)
st.plotly_chart(fig, use_container_width=True)
