import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

st.title("Stock Price Predictor App")

# User inputs the stock ticker symbol
stock = st.text_input("Enter the Stock ID", "GOOG")

# Define date range (last 20 years)
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

# Download stock data from Yahoo Finance
# google_data = yf.download(stock, start, end)
google_data = yf.download(stock, start, end, auto_adjust=False)

# Load the pre-trained model
model = load_model("Latest_stock_price_model.keras")

# Display raw stock data
st.subheader("Stock Data")
st.write(google_data)

# Split the dataset into training and test segments
splitting_len = int(len(google_data) * 0.7)
x_test = google_data[['Close']].iloc[splitting_len:]

# Helper function for plotting
def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(full_data['Close'], label='Original Close', color='blue')
    plt.plot(values, label='Moving Average', color='orange')
    if extra_data:
        plt.plot(extra_dataset, label='Extra Data', color='green')
    plt.legend()
    return fig

# Plot moving averages
st.subheader('Original Close Price and MA for 250 days')
google_data['MA_for_250_days'] = google_data['Close'].rolling(250).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_250_days'], google_data))

st.subheader('Original Close Price and MA for 200 days')
google_data['MA_for_200_days'] = google_data['Close'].rolling(200).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_200_days'], google_data))

st.subheader('Original Close Price and MA for 100 days')
google_data['MA_for_100_days'] = google_data['Close'].rolling(100).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'], google_data))

st.subheader('MA for 100 days and MA for 250 days')
fig = plt.figure(figsize=(15,6))
plt.plot(google_data['MA_for_100_days'], label='MA for 100 days', color='orange')
plt.plot(google_data['MA_for_250_days'], label='MA for 250 days', color='green')
plt.legend()
st.pyplot(fig)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test)

# Prepare sequences for LSTM model
x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# Get predictions from the model
predictions = model.predict(x_data)

# Inverse scaling to get original stock price values
inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

# Prepare DataFrame for comparison
ploting_data = pd.DataFrame({
    'Original Test Data': inv_y_test.flatten(),
    'Predictions': inv_pre.flatten()
}, index=google_data.index[splitting_len + 100:])

# Show comparison DataFrame
st.subheader("Original values vs Predicted values")
st.write(ploting_data)

# Plot original vs predicted stock prices
st.subheader('Original Close Price vs Predicted Close Price')
fig = plt.figure(figsize=(15,6))
plt.plot(google_data['Close'][:splitting_len + 100], label='Train Data', color='blue')
plt.plot(ploting_data['Original Test Data'], label='Original Test Data', color='orange')
plt.plot(ploting_data['Predictions'], label='Predicted Test Data', color='green')
plt.legend()
st.pyplot(fig)
