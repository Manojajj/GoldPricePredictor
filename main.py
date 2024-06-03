import streamlit as st
import pandas as pd
import numpy as np
import joblib

def main():
    st.title("Gold Price Prediction")

    # Load the saved model
    model = joblib.load('gold_price_model.pkl')

    # Sidebar for user inputs
    st.sidebar.title("Gold Price Prediction Inputs")
    day = st.sidebar.selectbox("Day", list(range(1, 8)))
    month = st.sidebar.selectbox("Month", list(range(1, 13)))
    slv = st.sidebar.number_input("Silver Rate (SLV)")
    uso = st.sidebar.number_input("US Oil Rate (USO)")

    # Make prediction
    if st.sidebar.button("Predict"):
        input_data = pd.DataFrame([[day, month, slv, uso]], columns=['day', 'month', 'SLV', 'USO'])
        prediction = model.predict(input_data)
        st.write(f"Predicted Gold Price: {prediction[0]}")

if __name__ == '__main__':
    main()
