import streamlit as st
import numpy as np
import joblib

# Set page config
st.set_page_config(page_title="Insurance Response Predictor", page_icon="🧠", layout="centered")

# Load model
model = joblib.load("InsuranceCustomerResponse_Logreg_Model.pkl")

# App title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🧠 Insurance Response Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict whether a customer will respond positively to an insurance offer 📩</p>", unsafe_allow_html=True)
st.markdown("---")

# Layout in columns
col1, col2 = st.columns(2)

with col1:
    age = st.slider("👤 Age", 18, 80, 35)
    gender = st.selectbox("🚻 Gender", ["Male", "Female"])
    previously_insured = st.selectbox("🛡️ Previously Insured", ["Yes", "No"])

with col2:
    annual_premium = st.number_input("💰 Annual Premium", 1000, 100000, 25000, step=500)
    vehicle_damage = st.selectbox("🚗 Vehicle Damage History", ["Yes", "No"])

# Encode categorical fields
gender_encoded = 1 if gender == "Male" else 0
vehicle_damage_encoded = 1 if vehicle_damage == "Yes" else 0
previously_insured_encoded = 1 if previously_insured == "Yes" else 0

# Prepare input array
input_data = np.array([[age, gender_encoded, annual_premium, vehicle_damage_encoded, previously_insured_encoded]])

st.markdown("---")
if st.button("🔍 Predict Response"):
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    # Display result
    if pred == 1:
        st.success("✅ The customer is likely to respond positively to the offer!")
        st.markdown(f"**Probability:** `{prob:.2%}`")
        st.progress(min(int(prob * 100), 100))
    else:
        st.error("❌ The customer is unlikely to respond to the offer.")
        st.markdown(f"**Probability:** `{prob:.2%}`")
        st.progress(min(int(prob * 100), 100))

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; font-size: 13px;'>Built with ❤️ using Streamlit | Model: Logistic Regression</div>",
        unsafe_allow_html=True
    )
