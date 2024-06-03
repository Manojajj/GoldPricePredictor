import streamlit as st
import pandas as pd
import numpy as np
import pickle

def main():
    st.title("Gold Price Prediction")

    # Load the saved model
    with open('gold_price_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Sidebar for user inputs
    st.sidebar.title("Gold Price Prediction Inputs")
    SPX = st.sidebar.number_input("SPX")
    USO = st.sidebar.number_input("USO")
    SLV = st.sidebar.number_input("SLV")
    EUR_USD = st.sidebar.number_input("EUR/USD")
    day = st.sidebar.selectbox("Day", list(range(1, 8)))
    month = st.sidebar.selectbox("Month", list(range(1, 13)))

    # Make prediction
    if st.sidebar.button("Predict"):
        input_data = pd.DataFrame([[SPX, USO, SLV, EUR_USD, day, month]], columns=['SPX', 'USO', 'SLV', 'EUR/USD', 'Day', 'Month'])
        prediction = model.predict(input_data)
        st.write(f"Predicted Gold Price: {prediction[0]}")

if __name__ == '__main__':
    main()
