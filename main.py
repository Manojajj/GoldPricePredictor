import streamlit as st
import pandas as pd
import pickle

@st.cache_resource
def load_model():
    with open('gold_price_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# Title
st.title("Gold Price Predictor")

# Sidebar for input features
st.sidebar.header("Input Features")

def user_input_features():
    SPX = st.sidebar.number_input("SPX", value=0.0)
    USO = st.sidebar.number_input("USO", value=0.0)
    SLV = st.sidebar.number_input("SLV", value=0.0)
    EUR_USD = st.sidebar.number_input("EUR/USD", value=0.0)
    Day = st.sidebar.selectbox("Day", [1, 2, 3, 4, 5, 6, 7])
    Month = st.sidebar.selectbox("Month", list(range(1, 13)))
    data = {
        "SPX": SPX,
        "USO": USO,
        "SLV": SLV,
        "EUR/USD": EUR_USD,
        "Day": Day,
        "Month": Month
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Main panel
st.subheader("User Input features")
st.write(input_df)

# Prediction
prediction = model.predict(input_df)

st.subheader("Predicted Gold Price")
st.write(prediction[0])
