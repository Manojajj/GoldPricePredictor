import streamlit as st
import pandas as pd
import numpy as np
import pickle

def main():
    st.title("Gold Price Prediction")

    # Load the saved model
    with open('gold_price_model.pkl', 'rb') as file:
        model = pickle.load(file)

    # Sidebar for user inputs
    st.sidebar.title("Gold Price Prediction Inputs")
    day = st.sidebar.selectbox("Day", list(range(1, 32)))
    month = st.sidebar.selectbox("Month", list(range(1, 13)))
    spx = st.sidebar.number_input("S&P 500 (SPX)")
    uso = st.sidebar.number_input("US Oil Rate (USO)")
    slv = st.sidebar.number_input("Silver Rate (SLV)")
    eurusd = st.sidebar.number_input("EUR/USD Exchange Rate")

    # Make prediction
    if st.sidebar.button("Predict"):
        input_data = pd.DataFrame([[spx, uso, slv, eurusd, day, month]], columns=['SPX', 'USO', 'SLV', 'EUR/USD', 'Day', 'Month'])
        prediction = model.predict(input_data)
        st.write(f"Predicted Gold Price: {prediction[0]}")

if __name__ == '__main__':
    main()
