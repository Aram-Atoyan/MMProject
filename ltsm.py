import os  # standard library for OS-related operations

# hides INFO and WARNING messages from the low-level runtime, but still
# allows real errors to appear. That’s why after this line you see fewer TF logs.)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import logging          # to control TensorFlow's Python-level log verbosity
import numpy as np      # numerical arrays, math operations, random numbers
import pandas as pd     # reading CSV files and simple data manipulation
import sklearn.preprocessing as skpre  # scaling utilities (we use MinMaxScaler)
import tensorflow as tf  # main deep learning framework (Keras is included via tf.keras)

# Reduce high-level TensorFlow Python warnings
tf.get_logger().setLevel(logging.ERROR)

# setting seeds makes random processes like weight initialization and
# random shuffling more consistent between runs. It does not guarantee perfect
# determinism on all hardware, but it significantly reduces variability.
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

    # Remove rows where the price is NaN to keep data clean and aligned
    # if there are malformed rows, empty cells, or weird values,
    # Removing them here ensures that both dates and real_vals arrays have the same length and contain
    #  only valid values. This is important for training and for printing
    #  pairs (date, real, pred) later.
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


def get_lstm_predictions(ticker: str = "AAPL", min_history: int = 4):
    """
    Train an LSTM for one ticker and generate next-step predictions.

    Now:
      - y_pred[0 : min_history] are just the real values y_true[0 : min_history]
      - y_pred[min_history :] are LSTM predictions
    """
    dates, y_true = read_stock_series(ticker)
    N = len(y_true)

    if N <= min_history:
        raise ValueError(f"Not enough data points for {ticker} with min_history={min_history}.")

    # Scale prices
    scaler = skpre.MinMaxScaler()
    y_scaled = scaler.fit_transform(y_true.reshape(-1, 1))  # shape: (N, 1)

    # Teacher-forcing sequences
    X_seq = y_scaled[:-1].reshape(1, N - 1, 1)  # input: y[0..N-2]
    y_seq = y_scaled[1:].reshape(1, N - 1, 1)   # target: y[1..N-1]

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

    # Predict next-step values for each timestep in the input sequence
    pred_scaled = model.predict(X_seq, verbose=0)[0, :, 0]  # shape: (N-1,)
    pred_unscaled = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

    # Allocate prediction array the same length as y_true
    y_pred = np.empty_like(y_true, dtype=float)

    # 1) First min_history points = real values (no NaNs anymore)
    y_pred[:min_history] = y_true[:min_history]

    # 2) From min_history onward use predictions
    #    pred_unscaled[k] is the model's estimate of y_true[k+1]
    for t in range(min_history, N):
        # we want prediction for y_true[t], which is pred_unscal
        y_pred[t] = pred_unscaled[t - 1]

    return dates, y_true, y_pred



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


if __name__ == "__main__":
    # List of tickers you want to run the LSTM prediction on
    tickers = ["AAPL", "AMZN", "GOOGL", "MSFT", "TSLA"]

    # Run prediction pipeline for all tickers
    all_results = run_predictions_for_tickers(tickers, min_history=4)

