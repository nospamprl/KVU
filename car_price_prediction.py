# Import necessary libraries
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model_path = 'best_xgboost_model.pkl'
model = joblib.load(model_path)

# Load the data to get dropdown options
file_path = 'ML.csv'
data = pd.read_csv(file_path)

# Extract unique options for dropdowns
makes = sorted(data['Marca.1.Índice'].unique())
models = sorted(data['Modelo.1.Índice'].unique())
versions = sorted(data['Version.1.Índice'].unique())

# Streamlit UI
st.title("Car Price Prediction App")
st.write("Enter the car details to predict its price.")

# User inputs
age = st.number_input("Car Age (Years):", min_value=0, max_value=30, value=5)
mileage = st.number_input("Mileage (KM):", min_value=0, max_value=500000, value=50000)
list_price = st.number_input("List Price:", min_value=0, max_value=2000000, value=500000)
make = st.selectbox("Make:", makes)
car_model = st.selectbox("Model:", models)
version = st.selectbox("Version:", versions)

# Prepare data for prediction as DataFrame
input_data = pd.DataFrame([[2023 - age, mileage, make, car_model, version, list_price]],
                          columns=["Año", "Kilometraje", "Marca.1.Índice", "Modelo.1.Índice", "Version.1.Índice", "Precios Lista.Precio de Lista"]).astype(float)

# Predict price
if st.button("Predict Price"):
    predicted_price = model.predict(input_data)[0]
    st.write(f"### Predicted Price: ${predicted_price:,.2f}")
