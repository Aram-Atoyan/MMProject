import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # hide low-level TF C++ logs

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import tensorflow as tf

# Hide high-level TF Python warnings (like retracing)
tf.get_logger().setLevel(logging.ERROR)


def read_stock_series(ticker: str):
    """
    Read data/<ticker>.csv and return:
        dates      - numpy array of datetime64
        real_vals  - numpy array of floats (Close prices)

    Expected CSV format:
        Price,Close
        Ticker,AAPL
        Date,
        2020-11-09,116.09
        2020-11-16,114.22
        ...

    Real data starts from row 4 in the file -> index 2 in pandas.
    """
    path = f"data/{ticker}.csv"
    df = pd.read_csv(path)

    # real rows start from index 2 (skip 'Ticker' and 'Date' meta rows)
    data = df.iloc[2:].copy()

    # first column = date, second column = price
    dates = pd.to_datetime(data.iloc[:, 0].values, errors="coerce")
    real_vals = pd.to_numeric(data.iloc[:, 1].values, errors="coerce").astype(float)

    # drop any NaNs just in case
    mask = ~np.isnan(real_vals)
    dates = dates[mask]
    real_vals = real_vals[mask]

    return dates, real_vals


def get_lstm_predictions(ticker: str = "AAPL", min_history: int = 4):
    """
    Use ALL previous REAL values to predict the next one.

    Returns:
        dates  : np.ndarray of dates
        y_true : np.ndarray of real Close prices (length N)
        y_pred : np.ndarray of predictions (length N)
                 y_pred[0..min_history-1] = np.nan
                 for t >= min_history, prediction at index t uses
                 ALL real values up to t-1 (not predicted ones).
    """
    dates, y_true = read_stock_series(ticker)
    N = len(y_true)
    if N <= min_history:
        raise ValueError("Not enough data points for the given min_history.")

    # Scale full series
    scaler = MinMaxScaler()
    y_scaled = scaler.fit_transform(y_true.reshape(-1, 1))  # shape (N, 1)

    # Build training sequence:
    #   X_seq: [y0..y_{N-2}]
    #   y_seq: [y1..y_{N-1}]
    X_seq = y_scaled[:-1].reshape(1, N - 1, 1)  # (batch=1, timesteps=N-1, features=1)
    y_seq = y_scaled[1:].reshape(1, N - 1, 1)

    # Build model: allow variable time dimension (None) to reduce retracing
    model = Sequential()
    model.add(Input(shape=(None, 1)))      # <--- key change: None instead of (N - 1)
    model.add(LSTM(50, return_sequences=True))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    # Train
    model.fit(X_seq, y_seq, epochs=50, batch_size=1, verbose=0)

    # Predict for each time step
    pred_scaled = model.predict(X_seq, verbose=0)[0, :, 0]  # shape (N-1,)
    pred_unscaled = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

    # Build full prediction array
    y_pred = np.full(N, np.nan, dtype=float)
    # pred_unscaled[j] is prediction for y_true[j+1]
    for t in range(min_history, N):
        y_pred[t] = pred_unscaled[t - 1]

    return dates, y_true, y_pred


if __name__ == "__main__":
    tickers = ["AAPL", "AMZN", "GOOGL", "MSFT", "TSLA"]

    for ticker in tickers:
        dates, y_true, y_pred = get_lstm_predictions(ticker=ticker, min_history=4)

        print(f"===== {ticker} =====")
        for d, yt, yp in zip(dates, y_true, y_pred):
            print((d, yt, yp))
        print()  # blank line between tickers
