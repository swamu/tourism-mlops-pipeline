import streamlit as st
import pandas as pd
import pickle
from huggingface_hub import hf_hub_download
import os

# Page configuration
st.set_page_config(
page_title="Tourism Package Predictor",
page_icon="",
layout="wide"
)

# Title and description
st.title(" Wellness Tourism Package Prediction")
st.markdown("""
This application predicts whether a customer will purchase the Wellness Tourism Package.
Enter customer details below to get a prediction.
""")

@st.cache_resource
def load_model():
    """Load model from Hugging Face Model Hub"""
try:
    model_file = hf_hub_download(
repo_id="swamu/tourism-prediction-model",
filename="model_artifacts.pkl",
token=os.environ.get("HF_TOKEN")
)
with open(model_file, 'rb') as f:
artifacts = pickle.load(f)
return artifacts
except Exception as e:
    st.error(f"Error loading model: {e}")
return None

# Load model
artifacts = load_model()

if artifacts:
model = artifacts['model']
scaler = artifacts['scaler']
label_encoders = artifacts['label_encoders']
feature_names = artifacts['feature_names']
categorical_cols = artifacts['categorical_cols']
numerical_cols = artifacts['numerical_cols']

st.success(" Model loaded successfully!")

# Create input form
st.header("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
age = st.number_input("Age", min_value=18, max_value=100, value=35)
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, value=15.0)
number_of_persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)

with col2:
type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
gender = st.selectbox("Gender", ["Male", "Female"])
number_of_followups = st.number_input("Number of Followups", min_value=0, max_value=10, value=3)

with col3:
preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
number_of_trips = st.number_input("Number of Trips per Year", min_value=0.0, value=3.0)
passport = st.selectbox("Has Passport", ["Yes", "No"])

col4, col5, col6 = st.columns(3)

with col4:
pitch_satisfaction = st.slider("Pitch Satisfaction Score", 1, 5, 3)
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])

with col5:
own_car = st.selectbox("Owns Car", ["Yes", "No"])
number_of_children = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)

with col6:
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthly_income = st.number_input("Monthly Income", min_value=0.0, value=25000.0)

# Predict button
if st.button(" Predict Purchase Probability", type="primary"):
try:
# Create input dataframe
input_data = {
'Age': age,
'TypeofContact': type_of_contact,
'CityTier': city_tier,
'DurationOfPitch': duration_of_pitch,
'Occupation': occupation,
'Gender': gender,
'NumberOfPersonVisiting': number_of_persons,
'NumberOfFollowups': number_of_followups,
'ProductPitched': product_pitched,
'PreferredPropertyStar': preferred_property_star,
'MaritalStatus': marital_status,
'NumberOfTrips': number_of_trips,
'Passport': 1 if passport == "Yes" else 0,
'PitchSatisfactionScore': pitch_satisfaction,
'OwnCar': 1 if own_car == "Yes" else 0,
'NumberOfChildrenVisiting': number_of_children,
'Designation': designation,
'MonthlyIncome': monthly_income
}

input_df = pd.DataFrame([input_data])

# Ensure correct column order
input_df = input_df[feature_names]

# Preprocess
for col in categorical_cols:
if col in label_encoders:
input_df[col] = label_encoders[col].transform(input_df[col].astype(str))

input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

# Make prediction
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)[0]

# Display results
st.success(" Prediction Complete!")

col_a, col_b = st.columns(2)

with col_a:
if prediction == 1:
st.success("### Likely to Purchase!")
st.metric("Purchase Probability", f"{prediction_proba[1]*100:.2f}%")
else:
st.warning("### Unlikely to Purchase")
st.metric("Purchase Probability", f"{prediction_proba[1]*100:.2f}%")

with col_b:
st.info("### Recommendation")
if prediction == 1:
st.write(" High potential customer - Proceed with offer")
else:
st.write("Consider personalized engagement strategy")

except Exception as e:
    st.error(f"Error making prediction: {e}")
else:
st.error("Failed to load model. Please check the configuration.")
