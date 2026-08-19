import streamlit as st
import pickle
import json
import numpy as np

# --- 1. Page Configuration ---
st.set_page_config(page_title="Bengaluru House Price Predictor", page_icon="🏡", layout="centered")

# --- 2. Load Artifacts (Cached for Performance) ---
# st.cache_resource ensures the model and data are loaded only once, not every time the user interacts.
@st.cache_resource
def load_artifacts():
    # Load columns
    with open("./artifacts/columns.json", "r") as f:
        data_columns = json.load(f)['data_columns']
        locations = data_columns[3:]  # The first 3 columns are sqft, bath, bhk

    # Load model
    with open('./artifacts/banglore_home_prices_model.pickle', 'rb') as f:
        model = pickle.load(f)
        
    return model, data_columns, locations

model, data_columns, locations = load_artifacts()

# --- 3. Prediction Logic ---
def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = data_columns.index(location.lower())
    except ValueError:
        loc_index = -1

    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    
    if loc_index >= 0:
        x[loc_index] = 1

    return round(model.predict([x])[0], 2)

# --- 4. Streamlit UI Design ---
st.title("🏡 Bengaluru House Price Predictor")
st.write("Enter the details of the property below to estimate its market price.")

# Form inputs mapping to the variables expected by your model
location = st.selectbox("Choose a Location", options=["Other"] + [loc.title() for loc in locations])
sqft = st.number_input("Total Area (Square Feet)", min_value=300.0, max_value=50000.0, value=1000.0, step=100.0)

col1, col2 = st.columns(2)
with col1:
    bhk = st.number_input("BHK (Bedrooms)", min_value=1, max_value=20, value=2, step=1)
with col2:
    bath = st.number_input("Bathrooms", min_value=1, max_value=20, value=2, step=1)

# Action button to trigger prediction
if st.button("Estimate Price", type="primary"):
    price = get_estimated_price(location, sqft, bhk, bath)
    st.success(f"The estimated price is **₹ {price} Lakhs**")