# Import Libraries
import streamlit as st
import pandas as pd
import joblib

# Load the Pre-trained Model and Label Encoder
model = joblib.load('house_price_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Streamlit UI
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://img.freepik.com/free-photo/luxury-pool-villa-spectacular-contemporary-design-digital-art-real-estate-home-house-property-ge_1258-150749.jpg");
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Housing Prices in India")

st.header("Make a Prediction")

# User Inputs
area = st.number_input("Area (in sq ft)", min_value=0)
bedrooms = st.number_input("Number of Bedrooms", min_value=0, max_value=10)
gymnasium = st.number_input("Gymnasium", min_value=0, max_value=1)
swimming_pool = st.number_input("Swimming Pool", min_value=0, max_value=1)
school = st.number_input("Nearby Schools", min_value=0, max_value=10)
car_parking = st.number_input("Car Parking Spaces", min_value=0, max_value=10)
hospital = st.number_input("Nearby Hospitals ", min_value=0, max_value=10)
location = st.text_input("Location")

# Encode Location
if location:
    encoded_location = label_encoder.transform([location])[0] if location in label_encoder.classes_ else -1
else:
    encoded_location = -1

# Prepare Input Data
input_data = pd.DataFrame([[area, bedrooms, gymnasium, swimming_pool, school, car_parking, hospital, encoded_location]],
                          columns=['Area', 'No. of Bedrooms', 'Gymnasium', 'SwimmingPool', 'School', 'CarParking', 'Hospital', 'Location'])

# Predict Button
if st.button("Predict"):
    if encoded_location == -1:
        st.error("Invalid location. Please enter a valid location from the dataset.")
    else:
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Price (in lakhs): {prediction:.2f}")
