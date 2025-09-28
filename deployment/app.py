import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(
    repo_id="Suvidhya/tourism-package-model",
    filename="best_tourism_model.joblib"
)
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Prediction App")
st.write("""
This application predicts the likelihood of a customer taking a tourism package based on their profile and preferences.
Please enter the customer details below to get a prediction.
""")

# User input
TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry", "Employee Referral"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
Occupation = st.selectbox("Occupation", ["Salaried", "Self Employed", "Business", "Student", "Other"])
Gender = st.selectbox("Gender", ["Male", "Female"])
MaritalStatus = st.selectbox("Marital Status", ["Married", "Single"])
Designation = st.text_input("Designation")
ProductPitched = st.text_input("Product Pitched")
Passport = st.selectbox("Passport", ["Yes", "No"])
OwnCar = st.selectbox("Own Car", ["Yes", "No"])
Age = st.number_input("Age", min_value=18, max_value=80, value=30)
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
PreferredPropertyStar = st.number_input("Preferred Property Star", min_value=1, max_value=5, value=3)
NumberOfTrips = st.number_input("Number of Trips per Year", min_value=0, max_value=20, value=1)
MonthlyIncome = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=5000)
DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=120, value=30)
NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=20, value=1)
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=10, value=5)
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'Occupation': Occupation,
    'Gender': Gender,
    'MaritalStatus': MaritalStatus,
    'Designation': Designation,
    'ProductPitched': ProductPitched,
    'Passport': Passport,
    'OwnCar': OwnCar,
    'Age': Age,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'MonthlyIncome': MonthlyIncome,
    'DurationOfPitch': DurationOfPitch,
    'NumberOfFollowups': NumberOfFollowups,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting
}])

# Make prediction
prediction_proba = model.predict_proba(input_data)[:, 1][0]
prediction = int(prediction_proba >= 0.45)  # threshold as used in training

st.write(f"Predicted Probability of taking the package: {prediction_proba:.2f}")
st.write(f"Prediction (1 = Will take package, 0 = Will not take package): {prediction}")

