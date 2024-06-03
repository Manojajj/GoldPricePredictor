import streamlit as st
import pandas as pd
import numpy as np
import pickle

def main():
    st.title("Gold Price Prediction")

    # Load the saved model
    try:
        with open('gold_price_model.pkl', 'rb') as file:
            model = pickle.load(file)
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'gold_price_model.pkl' is in the correct directory.")
        return
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return

    # Sidebar for user inputs
    st.sidebar.title("Gold Price Prediction Inputs")
    day = st.sidebar.selectbox("Day", list(range(1, 32)))
    month = st.sidebar.selectbox("Month", list(range(1, 13)))
    spx = st.sidebar.number_input("S&P 500 (SPX)", value=0.0)
    uso = st.sidebar.number_input("US Oil Rate (USO)", value=0.0)
    slv = st.sidebar.number_input("Silver Rate (SLV)", value=0.0)
    eurusd = st.sidebar.number_input("EUR/USD Exchange Rate", value=0.0)

    # Make prediction
    if st.sidebar.button("Predict"):
        try:
            input_data = pd.DataFrame([[spx, uso, slv, eurusd, day, month]], columns=['SPX', 'USO', 'SLV', 'EUR/USD', 'Day', 'Month'])
            prediction = model.predict(input_data)
            st.success(f"Predicted Gold Price: {prediction[0]:.2f}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

if __name__ == '__main__':
    main()
