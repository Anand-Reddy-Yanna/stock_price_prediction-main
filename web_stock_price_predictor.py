# app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pandas.tseries.offsets import BDay
import os
import tempfile

# TensorFlow / Keras with fallback for environments where tensorflow.keras isn't resolved by linter
try:
    from tensorflow.keras.models import Sequential, load_model  # type: ignore
    from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
    from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
except Exception:  # fallback to standalone keras
    from keras.models import Sequential, load_model
    from keras.layers import LSTM, Dense, Dropout
    from keras.callbacks import EarlyStopping

st.set_page_config(layout="wide", page_title="Stock Predictor")
st.title("📈 Stock Price Predictor")

# ---------------------------
# Helper functions: indicators
# ---------------------------
def add_moving_averages(df, windows=[20,50,100,200]):
    for w in windows:
        df[f"MA_{w}"] = df['Close'].rolling(window=w).mean()
    return df

def add_bollinger_bands(df, window=20, num_std=2):
    ma = df['Close'].rolling(window=window).mean()
    std = df['Close'].rolling(window=window).std()
    df['BB_upper'] = ma + num_std * std
    df['BB_lower'] = ma - num_std * std
    return df

def add_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def add_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    return df

# ---------------------------
# Data loader with caching
# ---------------------------
@st.cache_data(ttl=3600)
def load_data(ticker, start_date, end_date, auto_adjust=False):
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=auto_adjust)
    if df.empty:
        return df
    expected_cols = ['Open','High','Low','Close','Adj Close','Volume']
    available_cols = [c for c in expected_cols if c in df.columns]
    df = df[available_cols].copy()
    if "Adj Close" in df.columns:
        df.rename(columns={"Adj Close": "Adj_Close"}, inplace=True)
    df.index = pd.to_datetime(df.index)
    return df

# ---------------------------
# LSTM utilities
# ---------------------------
def create_sequences(values, lookback=100):
    # values expected shape: (n, 1) or (n,)
    vals = np.array(values)
    if vals.ndim == 1:
        vals = vals.reshape(-1, 1)
    X, y = [], []
    for i in range(lookback, len(vals)):
        X.append(vals[i-lookback:i])
        y.append(vals[i])  # yields shape (1,) or (1,1)
    X = np.array(X)
    y = np.array(y)
    # if y has extra dim, flatten to (n_samples, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    return X, y

def build_lstm_model(input_shape, units1=128, units2=64, dropout=0.2):
    model = Sequential()
    model.add(LSTM(units1, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(LSTM(units2, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# ---------------------------
# Sidebar UI controls
# ---------------------------
st.sidebar.header("Settings")
popular = ["GOOG","AAPL","MSFT","TSLA","AMZN","NVDA","META"]
ticker = st.sidebar.selectbox("Select ticker", popular + ["Other"], index=0)
if ticker == "Other":
    ticker = st.sidebar.text_input("Enter ticker symbol", value="GOOG").upper()

end_date = st.sidebar.date_input("End date", value=datetime.now().date())
start_limit = datetime.now().date() - timedelta(days=365*20)
start_date = st.sidebar.date_input("Start date", value=start_limit, min_value=start_limit, max_value=end_date)
auto_adjust = st.sidebar.checkbox("Auto adjust data (use Adj Close)", value=False)

st.sidebar.subheader("Indicators")
show_ma = st.sidebar.checkbox("Show Moving Averages (20,50,100,200)", value=True)
show_bb = st.sidebar.checkbox("Show Bollinger Bands", value=True)
show_rsi = st.sidebar.checkbox("Show RSI (14)", value=True)
show_macd = st.sidebar.checkbox("Show MACD", value=True)

st.sidebar.subheader("Model & Forecast")
lookback = int(st.sidebar.number_input("LSTM lookback (days)", min_value=10, max_value=500, value=100, step=10))
train_epochs = int(st.sidebar.number_input("Train epochs", min_value=1, max_value=200, value=10))
basis_batch = 16
batch_size = int(st.sidebar.number_input("Batch size", min_value=1, max_value=256, value=basis_batch))
forecast_days = int(st.sidebar.number_input("Forecast days (multi-step)", min_value=1, max_value=60, value=7))
compare_hw = st.sidebar.checkbox("Compare with Holt-Winters", value=True)
retrain = st.sidebar.button("Retrain LSTM (use current settings)")

# ---------------------------
# Load data
# ---------------------------
data_load_state = st.text("Downloading data...")
df = load_data(ticker, start_date, end_date + timedelta(days=1), auto_adjust=auto_adjust)
if df.empty:
    st.error("No data found for ticker. Check symbol or date range.")
    st.stop()
data_load_state.text("Data downloaded ✅")

# Add indicators
if show_ma: df = add_moving_averages(df)
if show_bb: df = add_bollinger_bands(df)
if show_rsi: df = add_rsi(df)
if show_macd: df = add_macd(df)

# ---------------------------
# Main Layout
# ---------------------------
left, right = st.columns([3,1])
with left:
    st.subheader(f"{ticker} — Closing Price & Indicators")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlestick'
    ))
    if show_ma:
        for w in [20,50,100,200]:
            col = f"MA_{w}"
            if col in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=f"MA {w}", opacity=0.8))
    if show_bb and 'BB_upper' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], name='BB Upper', line=dict(width=1), opacity=0.6))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name='BB Lower', line=dict(width=1), opacity=0.6))
    fig.update_layout(height=600, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    if show_rsi and 'RSI' in df.columns:
        st.subheader("RSI (14)")
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'))
        rsi_fig.add_hline(y=70, line_dash="dash", annotation_text="Overbought")
        rsi_fig.add_hline(y=30, line_dash="dash", annotation_text="Oversold")
        st.plotly_chart(rsi_fig, use_container_width=True)

    if show_macd and 'MACD' in df.columns:
        st.subheader("MACD")
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'))
        macd_fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], name='Signal'))
        st.plotly_chart(macd_fig, use_container_width=True)

with right:
    st.subheader("Summary Stats")
    latest = float(df['Close'].iloc[-1])
    change_series = df['Close'].pct_change().iloc[-1:]
    change = float(change_series.iloc[0] * 100) if not change_series.empty else 0.0
    st.metric("Latest Close", f"{latest:.2f}", delta=f"{change:.2f}%")
    
    st.markdown("**Key stats (selected range)**")
    stats = df['Close'].describe().rename(index={'50%':'median'})
    st.write(stats)

# ---------------------------
# Forecasting Section (Fixed)
# ---------------------------
st.subheader("🔮 Forecasting")

# Scale data
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(df[['Close']].values)  # shape (n,1)

# Adjust lookback if data is too short
orig_lookback = int(lookback)
if len(scaled) <= lookback:
    lookback = max(10, min(orig_lookback, max(2, len(scaled)//2)))
    st.warning(f"Dataset is short for lookback {orig_lookback}. Using lookback={lookback}.")

# Create sequences
X, y = create_sequences(scaled, lookback)
if X.size == 0 or X.shape[0] == 0:
    st.error("Not enough data to create sequences for LSTM. Try reducing lookback or expanding the date range.")
    st.stop()

# Ensure X has shape (samples, timesteps, 1)
if X.ndim == 2:
    X = X.reshape(X.shape[0], X.shape[1], 1)
elif X.ndim == 3 and X.shape[2] != 1:
    X = X.reshape(X.shape[0], X.shape[1], -1)

# Model save/load
model_dir = tempfile.gettempdir()
model_path = os.path.join(model_dir, f"{ticker}_lstm.h5")

try:
    if retrain or not os.path.exists(model_path):
        model = build_lstm_model((X.shape[1], 1))
        es = EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)
        model.fit(X, y, epochs=int(train_epochs), batch_size=int(batch_size), verbose=0, callbacks=[es])
        # Save model - requires h5py for .h5 saving; if issues occur consider .keras or SavedModel format
        model.save(model_path)
    else:
        model = load_model(model_path)
except Exception as e:
    st.error(f"Model training/loading failed: {e}")
    st.stop()

# Forecast generation (robust)
forecast = None
forecast_dates = None
try:
    # Prepare last sequence for forecast: shape (1, lookback, 1)
    last_seq = scaled[-lookback:].reshape(1, lookback, 1)

    forecast_scaled = []
    current_seq = last_seq.copy()  # shape (1, lookback, 1)

    for _ in range(int(forecast_days)):
        pred = model.predict(current_seq, verbose=0)  # shape (1,1)
        pred_val = float(pred[0,0])
        forecast_scaled.append([pred_val])  # keep as 2D item for inverse transform

        # shift window: drop first timestep and append new prediction at the end
        # concatenate along timestep axis
        new_step = np.array(pred).reshape(1,1,1)  # shape (1,1,1)
        current_seq = np.concatenate([current_seq[:,1:,:], new_step], axis=1)

    # Inverse transform forecast (ensure 2D array)
    forecast_arr = np.array(forecast_scaled).reshape(-1, 1)
    try:
        forecast = scaler.inverse_transform(forecast_arr).flatten()
    except Exception as e:
        # fallback if scaler cannot inverse (shouldn't happen if scaler was fit)
        st.warning(f"Scaler inverse transform failed: {e}. Returning raw scaled values.")
        forecast = forecast_arr.flatten()

    # Forecast dates: use business days to avoid gaps on weekends
    try:
        forecast_dates = pd.bdate_range(df.index[-1] + BDay(1), periods=int(forecast_days))
    except Exception:
        forecast_dates = pd.date_range(df.index[-1] + timedelta(days=1), periods=int(forecast_days))

    # If dates length mismatches for any reason, pad/trim
    if len(forecast_dates) != len(forecast):
        # align lengths
        min_len = min(len(forecast_dates), len(forecast))
        forecast = forecast[:min_len]
        forecast_dates = forecast_dates[:min_len]

except Exception as e:
    st.error(f"Forecast generation failed: {e}")
    forecast = None

# Plot forecast
forecast_fig = go.Figure()
forecast_fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Historical"))
if forecast is not None and forecast_dates is not None and len(forecast) > 0:
    forecast_fig.add_trace(go.Scatter(x=forecast_dates, y=forecast, name="LSTM Forecast", line=dict(dash="dot")))
else:
    st.warning("LSTM forecast not available.")

# Holt-Winters comparison
if compare_hw:
    try:
        # Choose a safe seasonal period relative to data length
        max_sp = max(2, len(df) // 20)
        seasonal_periods = min(30, max_sp)
        if len(df) > seasonal_periods * 2:
            hw_model = ExponentialSmoothing(df['Close'], trend="add", seasonal="add", seasonal_periods=seasonal_periods)
            hw_fit = hw_model.fit()
            hw_forecast = hw_fit.forecast(int(forecast_days))
            forecast_fig.add_trace(go.Scatter(x=hw_forecast.index, y=hw_forecast, name="Holt-Winters Forecast"))
        else:
            st.info("Not enough data for Holt-Winters seasonal model; skipping.")
    except Exception as e:
        st.warning(f"Holt-Winters forecast failed: {e}")

st.plotly_chart(forecast_fig, use_container_width=True)

# Evaluation metrics on training set
try:
    pred_train = model.predict(X, verbose=0)
    # pred_train is shape (n,1)
    pred_train_rescaled = scaler.inverse_transform(pred_train.reshape(-1, 1))
    y_rescaled = scaler.inverse_transform(y.reshape(-1, 1))

    mse = mean_squared_error(y_rescaled, pred_train_rescaled)
    mae = mean_absolute_error(y_rescaled, pred_train_rescaled)
    r2 = r2_score(y_rescaled, pred_train_rescaled)

    st.markdown("### 📊 Model Evaluation (Training Data)")
    st.write({"MSE": float(mse), "MAE": float(mae), "R2": float(r2)})
except Exception as e:
    st.warning(f"Evaluation failed: {e}")
