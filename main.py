import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load and preprocess data
data = pd.read_csv('/mnt/data/gld_price_data.csv')

# Display the data
st.write("Gold Price Data")
st.write(data.head())

# Feature and target variables
X = data.drop(columns=['Date', 'GLD'])
y = data['GLD']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write("Model Performance")
st.write(f"Mean Absolute Error: {mae}")
st.write(f"Mean Squared Error: {mse}")
st.write(f"R-squared: {r2}")

# Streamlit app for prediction
st.write("### Predict Gold Price")
st.write("Input the values to predict the gold price")

input_data = {}
for column in X.columns:
    input_data[column] = st.number_input(f"Input {column}", value=0.0)

input_df = pd.DataFrame([input_data])
prediction = model.predict(input_df)

st.write(f"Predicted Gold Price: {prediction[0]}")
