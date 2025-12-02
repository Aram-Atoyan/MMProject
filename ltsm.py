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

    Core idea (teacher forcing, only REAL data as input):

    Given real prices y[0], y[1], ..., y[N-1], we build:
        X_seq = [y[0], ..., y[N-2]]  (inputs)
        y_seq = [y[1], ..., y[N-1]]  (targets)

    At time index j in X_seq, the LSTM has processed y[0..j] (all real values
    up to that point) and is trained to output an estimate of y[j+1].

    We NEVER feed the model's own predictions back into the input sequence.
    All inputs during training and prediction are real observed values.

    Returns:
        dates  : np.ndarray of length N
        y_true : np.ndarray of real prices, length N
        y_pred : np.ndarray of predicted prices, length N
                 with y_pred[0..min_history-1] = NaN
    """
    dates, y_true = read_stock_series(ticker)
    N = len(y_true)

    #  We need at least one point after min_history to have something
    #  to predict. SO Min_history should be >= 5 for at least
    #  one prediction. Otherwise, there is no future step to forecast.
    if N <= min_history:
        raise ValueError(f"Not enough data points for {ticker} with min_history={min_history}.")

    # Scale prices to [0, 1] for stable neural network training
    scaler = skpre.MinMaxScaler()
    y_scaled = scaler.fit_transform(y_true.reshape(-1, 1))  # shape: (N, 1)

    # Build teacher-forcing sequences:
    # Input: all but last point
    # Target: all but first point (shifted by 1)

    # batch_size = 1 (just one sequence),
    # timesteps = N-1 (because we dropped one point for input and one for target),
    # features = 1 (we only feed the close price, no extra indicators).
    # So the model learns given all values up to j, predict j+1.
    X_seq = y_scaled[:-1].reshape(1, N - 1, 1)
    y_seq = y_scaled[1:].reshape(1, N - 1, 1)

    model = build_lstm_model()

    # Early stopping to avoid over-training and to keep the best model
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="loss",
        patience=20,
        restore_best_weights=True,
        verbose=0
    )

    # Train the model on the full single sequence
    # 500 is just a theoretical upper bound. In practice, training
    #  will typically stop earlier due to early_stop. Using batch_size=1 means
    #  we treat this as one long time-series sample. The recurrent state flows
    #  through the whole sequence, so the model can, in principle, use very
    #  long-term patterns, not just the last few steps.
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

    # Convert scaled predictions back to original price scale
    pred_unscaled = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

    #  We want y_pred to have the same length and indexing as y_true.
    #  However, we don't produce predictions for the very first few points,
    #  because there is not enough history to justify them. Those positions
    #  remain NaN and can be ignored later when computing metrics.
    y_pred = np.full(N, np.nan, dtype=float)

    for t in range(min_history, N):
        # For index t, use prediction that was trained as "predict y[t]".
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
