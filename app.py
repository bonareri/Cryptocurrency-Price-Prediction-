import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tensorflow.keras.models import load_model
from matplotlib.dates import DateFormatter
from datetime import timedelta

# Set the Seaborn style for all plots
sns.set_style("darkgrid")

# Define the coins and their tickers
COINS = {
    "bitcoin": "BTC-USD",
    "ethereum": "ETH-USD",
    "solana": "SOL-USD",
    "tether": "USDT-USD",
    "xrp": "XRP-USD",
    "binancecoin": "BNB-USD"
}

# Define correct scaler file names
SCALER_FILES = {
    "bitcoin": "btc_scaler.pkl",
    "ethereum": "eth_scaler.pkl",
    "solana": "sol_scaler.pkl",
    "tether": "usdt_scaler.pkl",
    "xrp": "xrp_scaler.pkl",
    "binancecoin": "bnb_scaler.pkl"
}

# Function to fetch historical price data
def fetch_crypto_data(ticker):
    try:
        crypto = yf.Ticker(ticker)
        df = crypto.history(period="max")
        if df.empty:
            return None
        df.reset_index(inplace=True)
        df = df.rename(columns={"Date": "date"})
        df["date"] = pd.to_datetime(df["date"])
        
        # Filter data to start from 2023
        df = df[df["date"] >= "2023-01-01"]
        df.set_index("date", inplace=True)
        
        return df[["Close"]]
    except Exception as e:
        st.error(f"❌ Error fetching data for {ticker}: {e}")
        return None

# Load data for selected cryptocurrency
def load_crypto_data():
    return {coin: fetch_crypto_data(ticker) for coin, ticker in COINS.items()}

# Load trained model and scaler dynamically
def load_model_and_scaler(coin):
    model_path = f"{coin}_lstm_model.keras"
    scaler_path = SCALER_FILES.get(coin, f"{coin}_scaler.pkl")

    try:
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except FileNotFoundError:
        st.error(f"❌ Model or scaler file not found for {coin}. Expected: {scaler_path}")
    except Exception as e:
        st.error(f"⚠️ Error loading model or scaler for {coin}: {e}")
    
    return None, None

# Function to create LSTM sequences
def create_sequences(dataset, look_back=5):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i + look_back, 0])
        y.append(dataset[i + look_back, 0])
    return np.array(X).reshape(-1, look_back, 1), np.array(y)

# Function to forecast next month's prices
def forecast_next_month(model, scaler, last_window, look_back=5, days=30):
    future_predictions = []
    input_seq = last_window.reshape(1, look_back, 1)
    
    for _ in range(days):
        predicted_price = model.predict(input_seq)[0][0]
        future_predictions.append(predicted_price)
        
        input_seq = np.append(input_seq[:, 1:, :], [[[predicted_price]]], axis=1)
    
    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Streamlit UI
st.title("Cryptocurrency Price Prediction")
st.sidebar.header("Select Cryptocurrency")
selected_coin = st.sidebar.selectbox("Choose a cryptocurrency:", list(COINS.keys()))

# Load data
data = load_crypto_data()
df = data.get(selected_coin)

if df is None:
    st.error(f"❌ No data found for {selected_coin}. Try again later.")
    st.stop()

# Load model and scaler
model, scaler = load_model_and_scaler(selected_coin)
if model is None or scaler is None:
    st.stop()

# Scale the data
scaled_data = scaler.transform(df["Close"].values.reshape(-1, 1))
X, y = create_sequences(scaled_data, look_back=5)

# Make predictions
predictions = scaler.inverse_transform(model.predict(X))
actual_prices = scaler.inverse_transform(y.reshape(-1, 1))

# Create a DataFrame for comparison
dates = df.index[5:]
comparison_df = pd.DataFrame({
    "Date": dates.strftime("%Y-%m-%d"),
    "Actual Price (USD)": actual_prices.flatten(),
    "Predicted Price (USD)": predictions.flatten()
})

st.subheader(f"{selected_coin.capitalize()} - Actual vs. Predicted Prices")
st.write(comparison_df.tail())

# Forecast the next 30 days
last_window = scaled_data[-5:]
future_prices = forecast_next_month(model, scaler, last_window)
future_dates = [dates[-1] + timedelta(days=i) for i in range(1, 31)]

future_df = pd.DataFrame({
    "Date": [d.strftime("%Y-%m-%d") for d in future_dates],
    "Predicted Price (USD)": future_prices.flatten()
})

st.subheader(f"{selected_coin.capitalize()} - Forecast for Next 30 Days")
st.write(future_df)

# Plot results
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(dates, actual_prices, label="Actual Price", color="blue", linewidth=2)
ax.plot(dates, predictions, label="Predicted Price", color="red", linestyle="dashed", linewidth=2)
ax.plot(future_dates, future_prices, label="Forecasted Price", color="green", linestyle="dotted", linewidth=2)

# Format x-axis dates as YYYY-MM-DD
date_formatter = DateFormatter("%Y-%m-%d")
ax.xaxis.set_major_formatter(date_formatter)
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.title(f"{selected_coin.capitalize()} Price Prediction")
plt.legend()
plt.grid(True)

st.pyplot(fig)
