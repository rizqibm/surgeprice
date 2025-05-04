
import streamlit as st
import pandas as pd
import numpy as np
from surgepricing_xgboost import preprocess_data, xgb_model

st.set_page_config(page_title="Surge Pricing Type Prediction", layout="centered")

st.title("🚕 Predict Surge Pricing Type")

# Input form
st.subheader("Input Trip Data")
with st.form("input_form"):
    distance = st.number_input("Trip Distance (km)", min_value=0.0, format="%.2f")
    cab_type = st.selectbox("Cab Type", ["Uber", "Lyft"])
    destination = st.selectbox("Destination", ["Back Bay", "Beacon Hill", "Boston University", 
                                               "Fenway", "Financial District", "Haymarket Square",
                                               "North End", "North Station", "Northeastern University",
                                               "South Station", "Theatre District", "West End"])
    price = st.number_input("Price ($)", min_value=0.0, format="%.2f")
    surge_multiplier = st.selectbox("Surge Multiplier", [1.0, 1.25, 1.5, 2.0])
    name = st.selectbox("Cab Name", ["UberPool", "UberX", "UberXL", "UberBlack", "UberSUV", 
                                     "Lyft", "Lyft XL", "Lyft Black", "Lyft Black XL"])
    weekday = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    rating = st.slider("User Rating", 1.0, 5.0, 4.5, 0.1)
    lifestyle_index = st.slider("Lifestyle Index", 0.0, 1.0, 0.5, 0.01)
    
    submitted = st.form_submit_button("Predict")

if submitted:
    # Buat dataframe input
    user_input = pd.DataFrame({
        'distance': [distance],
        'cab_type': [cab_type],
        'destination': [destination],
        'price': [price],
        'surge_multiplier': [surge_multiplier],
        'name': [name],
        'weekday': [weekday],
        'rating': [rating],
        'user_lifestyle_index': [lifestyle_index]
    })

    # Preprocessing & predict
    try:
        processed = preprocess_data(user_input)
        pred = xgb_model.predict(processed)
        surge_map = {0: "No Surge", 1: "Low Surge", 2: "High Surge"}
        st.success(f"Predicted Surge Pricing Type: **{surge_map.get(pred[0])}**")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
