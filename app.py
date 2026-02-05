import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1. Load the saved model and scaler
# Make sure these files are in the same folder as app.py on GitHub
model = joblib.load('house_model.joblib')
scaler = joblib.load('scaler.joblib')

# 2. UI Header
st.set_page_config(page_title="House Price Predictor", page_icon="🏡")
st.title("🏡 King County House Price Predictor")
st.write("Adjust the sliders to estimate the market value of a house.")

# 3. Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    sqft_living = st.slider("Living Area (sqft)", 500, 10000, 2000)
    bedrooms = st.number_input("Bedrooms", 1, 10, 3)
    bathrooms = st.number_input("Bathrooms", 1, 8, 2)
    floors = st.selectbox("Floors", [1, 1.5, 2, 2.5, 3, 3.5])
    grade = st.slider("Construction Grade (1-13)", 1, 13, 7)

with col2:
    waterfront = st.checkbox("Waterfront View?")
    view = st.slider("View Quality (0-4)", 0, 4, 0)
    condition = st.slider("House Condition (1-5)", 1, 5, 3)
    yr_built = st.number_input("Year Built", 1900, 2015, 1990)
    zipcode = st.number_input("Zipcode", 98001, 98199, 98103)

# 4. Feature Engineering Logic (Matching your Training Code)
house_age = 2015 - yr_built
years_since_renovation = house_age # Simplified for the app

# 5. Prepare the Input Array (18 features total)
# The order must match the X_train columns exactly
# Index mapping based on your code:
# ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 
# 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'zipcode', 
# 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'house_age', 'years_since_renovation']

features = np.array([[
    bedrooms, bathrooms, sqft_living, 5000, floors, int(waterfront),
    view, condition, grade, sqft_living, 0, zipcode,
    47.5, -122.2, 1800, 5000, house_age, years_since_renovation
]])

# 6. Scaling and Prediction
if st.button("Calculate Estimated Price"):
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    
    st.success(f"### Estimated Price: ${prediction[0]:,.2f}")
    st.balloons()
