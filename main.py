import streamlit as st
import pandas as pd
import numpy as np
import pickle

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def predict_price(model, input_data):
    return model.predict(input_data)

def main():
    # Load the model
    model_path = 'gold_price_model.pkl'
    model = load_model(model_path)

    # Title
    st.title('Gold Price Predictor')

    # Input form for user input
    st.sidebar.header('Input Features')
    date = st.sidebar.date_input('Date', value=pd.to_datetime('today'))

    day_of_week = date.dayofweek + 1
    month = date.month

    slv = st.sidebar.number_input('Silver Price', value=25.0)
    uso = st.sidebar.number_input('US Oil Price', value=75.0)

    input_data = np.array([[day_of_week, month, slv, uso]])

    # Predicting price
    if st.sidebar.button('Predict'):
        prediction = predict_price(model, input_data)
        st.success(f'Predicted Gold Price: ${prediction[0]:.2f}')

if __name__ == '__main__':
    main()
