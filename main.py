import streamlit as st
import pandas as pd
import pickle

# Load the saved model
@st.cache
def load_model():
    with open('gold_price_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# Function to predict gold price
def predict_gold_price(SPX, USO, SLV, EUR_USD, Day, Month):
    input_data = pd.DataFrame({'SPX': [SPX], 'USO': [USO], 'SLV': [SLV], 'EUR/USD': [EUR_USD], 'Day': [Day], 'Month': [Month]})
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit UI
st.title('Gold Price Predictor')

st.write("""
## Enter the input parameters to predict the Gold Price
""")

# User inputs
SPX = st.number_input('Enter SPX')
USO = st.number_input('Enter USO')
SLV = st.number_input('Enter SLV')
EUR_USD = st.number_input('Enter EUR/USD')
Day = st.number_input('Enter Day (1-7)')
Month = st.number_input('Enter Month (1-12)')

# Predicting gold price
if st.button('Predict'):
    gold_price_prediction = predict_gold_price(SPX, USO, SLV, EUR_USD, Day, Month)
    st.write(f"Predicted Gold Price: {gold_price_prediction}")
