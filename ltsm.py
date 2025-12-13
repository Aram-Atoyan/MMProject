import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import logging
import numpy as np
import pandas as pd
import sklearn.preprocessing as skpre
import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)

np.random.seed(42)
tf.random.set_seed(42)


def read_stock_series(ticker: str):
    """
    Load the time series (dates, Close prices) for a single stock ticker.

    The CSV is expected to be at: data/<ticker>.csv

    Expected structure:

        Price,Close
        Ticker,AAPL
        Date,
        2020-11-09,116.09
        2020-11-16,114.22
        ...

    - First row: column names
    - Second row: metadata about ticker ("Ticker,<symbol>")
    - Third row: "Date," (header for the data region)
    - From row index 2 onward: actual (date, close_price) pairs

    Returns:
        dates     : numpy array of datetime64
        real_vals : numpy array of float (Close prices)
    """
    path = f"data/{ticker}.csv"

    df = pd.read_csv(path)

    # Skip the first 2 rows and keep rows from index 2 onwards.
    data = df.iloc[2:].copy()

    # Convert first column to datetime objects
    dates = pd.to_datetime(data.iloc[:, 0].values, errors="coerce")

    # Convert second column (price strings) to floats
    real_vals = pd.to_numeric(data.iloc[:, 1].values, errors="coerce").astype(float)

    mask = ~np.isnan(real_vals)
    dates = dates[mask]
    real_vals = real_vals[mask]

    return dates, real_vals


def build_lstm_model():
    """
    Build and compile a stacked LSTM model for next-step prediction.

    Model structure:
        Input:  variable-length sequence of shape (timesteps, 1)
        LSTM(64, return_sequences=True)
        Dropout(0.2)
        LSTM(32, return_sequences=True)
        Dropout(0.2)
        Dense(16, activation="relu")
        Dense(1)

    - Input shape uses 'None' for timesteps so the model can accept sequences
      of any length at runtime.
    - return_sequences=True ensures that we output a prediction at every time
      step, not just the final one.
    """
    model = tf.keras.models.Sequential()

    # Input layer describes the shape of each sample: (timesteps, 1 feature)
    model.add(tf.keras.layers.Input(shape=(None, 1)))

    #  LSTMs maintain an internal hidden state that carries information
    #  along the sequence. 'return_sequences=True' means we keep the output at
    #  each timestep.
    model.add(tf.keras.layers.LSTM(64, return_sequences=True))

    #  Dropout(0.2) randomly zeroes out 20% of elements
    #  in the layer’s output. This forces the network to not rely on specific
    #  neurons too much and usually improves generalization on unseen data.
    model.add(tf.keras.layers.Dropout(0.2))


    # Second LSTM with 32 units, again returning the full sequence
    model.add(tf.keras.layers.LSTM(32, return_sequences=True))

    # Another Dropout layer
    model.add(tf.keras.layers.Dropout(0.2))

    # after the LSTM layers, we have a sequence of 32-dimensional
    # feature vectors. This dense layer applies a learned linear transform +
    # ReLU activation to each timestep’s features, giving the model more
    # expressive power.
    model.add(tf.keras.layers.Dense(16, activation="relu"))

    # this is the regression output. For each timestep j in the input,
    # the model outputs one value that is meant to approximate the next
    # price y[j+1] during training.
    model.add(tf.keras.layers.Dense(1))

    # Compile the model with Adam optimizer, which is widely used for adapting the learning rate
    # for each parameter. MSE (mean squared error) penalizes large errors more
    # heavily and is standard for continuous regression tasks like price prediction.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse"
    )

    return model


def get_lstm_predictions(
    ticker: str,
    dates,
    y_true,
    min_history: int = 4
):
    """
    LSTM next-step predictions for a given ticker using externally provided data.

    IMPORTANT: This function does NOT load data internally.
    You must pass:
        dates : array-like of datetime64 / datetime objects (length N)
        y_true: array-like of floats (Close prices) (length N)

    Returns:
        y_true (np.ndarray), y_pred (np.ndarray), dates (np.ndarray)
    """
    if dates is None or y_true is None:
        raise ValueError("get_lstm_predictions requires dates and y_true (no internal loading).")

    t = str(ticker).strip().upper()
    dates = np.asarray(dates)
    y_true = np.asarray(y_true, dtype=float)

    if len(dates) != len(y_true):
        raise ValueError(f"[{t}] dates and y_true lengths differ: {len(dates)} vs {len(y_true)}")

    N = len(y_true)
    if N <= min_history:
        raise ValueError(f"[{t}] Not enough data points (got {N}) for min_history={min_history}.")

    # Scale prices
    scaler = skpre.MinMaxScaler()
    y_scaled = scaler.fit_transform(y_true.reshape(-1, 1))  # (N, 1)

    # Teacher forcing sequences:
    # input:  y[0..N-2]
    # target: y[1..N-1]
    X_seq = y_scaled[:-1].reshape(1, N - 1, 1)  # (1, N-1, 1)
    y_seq = y_scaled[1:].reshape(1, N - 1, 1)   # (1, N-1, 1)

    model = build_lstm_model()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="loss",
        patience=20,
        restore_best_weights=True,
        verbose=0
    )

    model.fit(
        X_seq,
        y_seq,
        epochs=500,
        batch_size=1,
        verbose=0,
        callbacks=[early_stop]
    )

    # Predict next-step values for each timestep
    pred_scaled = model.predict(X_seq, verbose=0)[0, :, 0]  # (N-1,)
    pred_unscaled = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

    # Build full-length prediction array
    y_pred = np.empty_like(y_true, dtype=float)

    # First min_history points = real values
    y_pred[:min_history] = y_true[:min_history]

    # For t >= min_history:
    # pred_unscaled[k] corresponds to prediction for y_true[k+1]
    for i in range(min_history, N):
        y_pred[i] = pred_unscaled[i - 1]

    return y_true, y_pred, dates



  # This method is made to run the LSTM prediction pipeline for a list of
  # stock tickers, print the (date, real, predicted) values for each ticker,
  # and collect all results in a dictionary for later use.
def run_predictions_for_tickers(tickers, min_history=4):
    """
    Run get_lstm_predictions for each ticker, print the results,
    and collect them in a dictionary.

    Returns:
        results_dict: dict mapping
            ticker -> (dates, y_true, y_pred)
    """
    results_dict = {}

    for ticker in tickers:
        print("\n====================================")
        print(f"Running LSTM predictions for {ticker}")
        print("====================================\n")

        dates, y_true, y_pred = get_lstm_predictions(
            ticker=ticker,
            min_history=min_history
        )

        results_dict[ticker] = (dates, y_true, y_pred)

        print(f"===== {ticker} =====")

        # zip(dates, y_true, y_pred) combines the three arrays so we can
        # iterate over them together and print each element as a tuple:
        # (date, real_price, predicted_price)
        for d, yt, yp in zip(dates, y_true, y_pred):
            print((d, yt, yp))
        print()

    return results_dict



