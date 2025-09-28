import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# ===== Load Model from Hugging Face Hub =====
model_path = hf_hub_download(
    repo_id="Suvidhya/tourism-package-model",
    filename="best_tourism_model.joblib"
)
model = joblib.load(model_path)

# ===== Streamlit UI =====
st.title("🏝 Tourism Package Prediction App")
st.write("""
This app predicts whether a customer is likely to take a tourism package 
based on their profile and preferences.
""")

# ===== User Input =====
TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry", "Employee Referral"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
Occupation = st.selectbox("Occupation", ["Salaried", "Self Employed", "Business", "Student", "Other"])
Gender = st.selectbox("Gender", ["Male", "Female"])
MaritalStatus = st.selectbox("Marital Status", ["Married", "Single"])

# Replace free text with fixed categories (adjust these to match your training dataset)
Designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP"])
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])

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

# ===== Data Formatting =====
# Map binary categorical values to match training (0/1)
Passport = 1 if Passport == "Yes" else 0
OwnCar = 1 if OwnCar == "Yes" else 0

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

# ===== Prediction =====
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[:, 1][0]
    prediction = int(prediction_proba >= 0.45)  # use 0.5 if you didn’t tune threshold

    st.subheader("Prediction Result")
    st.write(f"**Probability of taking the package:** {prediction_proba:.2f}")
    st.write(f"**Prediction:** {'Will take package' if prediction == 1 else 'Will not take package'}")
