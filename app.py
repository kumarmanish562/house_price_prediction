# Import Libraries
import streamlit as st
import pandas as pd
import joblib

# Load the Pre-trained Model and Label Encoder
model_path = r"C:\Users\qmani\OneDrive\Documents\GitHub\house_price_prediction\house_price_model.pkl"
encoder_path = r"C:\Users\qmani\OneDrive\Documents\GitHub\house_price_prediction\label_encoder.pkl"

model = joblib.load(model_path)
label_encoder = joblib.load(encoder_path)

# Get the list of valid locations
valid_locations = list(label_encoder.classes_)

# Custom CSS for Unique Design
st.markdown(
    """
    <style>
    /* Global Background */
    .stApp {
        background: linear-gradient(to right, #b92b27, #1565C0);
        background-size: cover;
    }

    /* Sidebar Design */
    [data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #1e3c72, #2a5298);
        color: white;
    }

    /* Title and Header Styling */
    h1 {
        font-family: 'Arial Black', sans-serif;
        color: #ffffff;
        text-align: center;
        margin-top: -20px;
    }
    
    h2 {
        font-family: 'Arial', sans-serif;
        color: #d1e8ff;
        text-align: center;
    }

    /* Cards for Input Sections */
    .input-card {
        background: rgba(255, 255, 255, 0.8);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
    }

    /* Predict Button Styling */
    .stButton>button {
        background-color: #f39c12;
        color: white;
        padding: 12px 24px;
        font-size: 18px;
        border-radius: 10px;
        transition: background-color 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #e67e22;
        transform: scale(1.05);
    }

    /* Sidebar Text */
    .sidebar-title {
        font-size: 20px;
        font-weight: bold;
        color: #ffffff;
    }

    .sidebar-info {
        font-size: 14px;
        color: #e3f2fd;
    }

    /* Footer */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #0c0c0c;
        color: #ffffff;
        text-align: center;
        padding: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and Header
st.markdown("<h1>🏡 House Price Prediction in India</h1>", unsafe_allow_html=True)
st.markdown("<h2>📊 Get Accurate Predictions for Your Dream Home</h2>", unsafe_allow_html=True)

# Sidebar Info
st.sidebar.markdown('<p class="sidebar-title">ℹ️ About the Project</p>', unsafe_allow_html=True)
st.sidebar.markdown(
    """
    <p class="sidebar-info">
    This house price prediction model is trained on datasets from six major cities in India:
    </p>
    <ul class="sidebar-info">
        <li>🏙️ Bangalore</li>
        <li>🌆 Mumbai</li>
        <li>🏖️ Chennai</li>
        <li>🏰 Hyderabad</li>
        <li>🏗️ Delhi</li>
        <li>🌉 Kolkata</li>
    </ul>
    """,
    unsafe_allow_html=True,
)

# Show valid locations in the sidebar
st.sidebar.markdown('<p class="sidebar-title">📍 Available Locations</p>', unsafe_allow_html=True)
st.sidebar.write(", ".join(valid_locations))

# Create Input Section
st.markdown('<div class="input-card">', unsafe_allow_html=True)

# User Inputs
area = st.number_input("📏 Area (in sq ft)", min_value=0, step=50, format="%d")
bedrooms = st.number_input("🛏️ Number of Bedrooms", min_value=0, max_value=10, step=1)
gymnasium = st.selectbox("🏋️ Gymnasium", ["No", "Yes"])
swimming_pool = st.selectbox("🏊 Swimming Pool", ["No", "Yes"])
school = st.number_input("🏫 Nearby Schools", min_value=0, max_value=10, step=1)
car_parking = st.number_input("🚗 Car Parking Spaces", min_value=0, max_value=10, step=1)
hospital = st.number_input("🏥 Nearby Hospitals", min_value=0, max_value=10, step=1)

# Location Selection
location = st.text_input("📍 Location (Enter from the available list)").strip()

# Encode Location
if location in valid_locations:
    encoded_location = label_encoder.transform([location])[0]
else:
    encoded_location = -1

# Encode Binary Features (Gym & Pool)
gymnasium_encoded = 1 if gymnasium == "Yes" else 0
swimming_pool_encoded = 1 if swimming_pool == "Yes" else 0

# Prepare Input Data for Model
input_data = pd.DataFrame(
    [[area, bedrooms, gymnasium_encoded, swimming_pool_encoded, school, car_parking, hospital, encoded_location]],
    columns=['Area', 'No. of Bedrooms', 'Gymnasium', 'SwimmingPool', 'School', 'CarParking', 'Hospital', 'Location']
)

st.markdown('</div>', unsafe_allow_html=True)

# Predict Button
if st.button("🔍 Predict"):
    if encoded_location == -1:
        st.error("❗ Invalid location. Please select a valid location from the sidebar.")
    else:
        try:
            prediction = model.predict(input_data)[0]
            st.success(f"🏠 Predicted Price (in lakhs): ₹{prediction:.2f}")
        except Exception as e:
            st.error(f"⚠️ Error in prediction: {e}")

# Footer Section
st.markdown(
    """
    <div class="footer">
        Developed with ❤️ by Manish Kumar
    </div>
    """,
    unsafe_allow_html=True,
)
