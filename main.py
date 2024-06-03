import streamlit as st
import pandas as pd
import pickle

# Load the saved model
with open('gold_price_model.pkl', 'rb') as f:
  model = pickle.load(f)

# Title and User Input
st.title('Gold Price Prediction App')

spx = st.number_input('Enter SPX Value:')
oil_price = st.number_input('Enter US Oil Price:')
slv = st.number_input('Enter SLV Value:')
eur_usd = st.number_input('Enter EUR/USD Exchange Rate:')
day = st.selectbox('Select Day:', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
month = st.selectbox('Select Month:', ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])

# Convert Day and Month to numerical format for prediction
day_mapping = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
month_mapping = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}

# Prepare user input data
data = {'SPX': [spx], 'USO': [oil_price], 'SLV': [slv], 'EUR/USD': [eur_usd], 'Day': [day_mapping[day]], 'Month': [month_mapping[month]]}
df = pd.DataFrame(data)

# Button to trigger prediction
predict_button = st.button("Predict Gold Price")

# Make prediction on button click
if predict_button:
  prediction = model.predict(df)[0]
  st.write(f"Predicted Gold Price: ${prediction:.2f}")

# Display additional information (Optional)
st.sidebar.header("Model Information")
st.sidebar.write("This model is a Random Forest Regressor trained to predict gold prices based on economic factors.")
# Add more details about model performance here (e.g., R-squared)
