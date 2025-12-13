import sys
import numpy as np
import matplotlib.pyplot as plt

from data_loading import load_series
from ARIMA import get_arima_predictions
from ltsm import get_lstm_predictions


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


def directional_accuracy(y_true, y_pred):
    true_diff = np.diff(y_true)
    pred_diff = np.diff(y_pred)
    correct = np.sign(true_diff) == np.sign(pred_diff)
    return np.mean(correct)


def compute_all_metrics(y_true, y_pred):
    return {
        "MSE": mse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "DA": directional_accuracy(y_true, y_pred),
    }


def plot_true_vs_pred(dates, y_true, y_arima, y_lstm, ticker):
    plt.figure(figsize=(12, 5))
    plt.plot(dates, y_true, label="True", linewidth=2)
    plt.plot(dates, y_arima, label="ARIMA", alpha=0.7)
    plt.plot(dates, y_lstm, label="LSTM", alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("True vs Predicted – " + ticker)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_residuals(dates, y_true, y_arima, y_lstm, ticker):
    res_arima = y_true - y_arima
    res_lstm = y_true - y_lstm

    plt.figure(figsize=(12, 4))
    plt.plot(dates, res_arima)
    plt.axhline(0, linestyle="--")
    plt.title(f"ARIMA residuals – {ticker}")
    plt.xlabel("Time")
    plt.ylabel("Error")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(dates, res_lstm)
    plt.axhline(0, linestyle="--")
    plt.title(f"LSTM residuals – {ticker}")
    plt.xlabel("Time")
    plt.ylabel("Error")
    plt.tight_layout()
    plt.show()

def main():
    default_tickers = ["AAPL", "AMZN", "GOOGL", "MSFT", "TSLA"]
    tickers = pick_tickers(default_tickers)

    for ticker in tickers:
        print("\n==============================")
        print(f"Results for {ticker}")
        print("==============================")

        # ---- LOAD ONCE HERE ----
        dates, y_true = load_series(ticker, n_weeks=262, years=5)

        # ---- Pass the SAME data into both models ----
        y_true_arima, y_arima, dates_arima = get_arima_predictions(
            ticker, dates=dates, y_true=y_true
        )
        y_true_lstm, y_lstm, dates_lstm = get_lstm_predictions(
            ticker, dates=dates, y_true=y_true
        )

        # Sanity checks now should always pass
        if not np.allclose(y_true_arima, y_true_lstm):
            raise ValueError(f"y_true from ARIMA and LSTM do not match for {ticker}!")
        if len(dates_arima) != len(dates_lstm):
            raise ValueError(f"Dates arrays have different lengths for {ticker}!")
        if not np.array_equal(np.array(dates_arima), np.array(dates_lstm)):
            raise ValueError(f"Dates from ARIMA and LSTM do not match for {ticker}!")

        # Metrics
        metrics_arima = compute_all_metrics(y_true, y_arima)
        metrics_lstm = compute_all_metrics(y_true, y_lstm)

        print("ARIMA metrics:")
        for k, v in metrics_arima.items():
            print(f"  {k}: {v:.6f}")

        print("LSTM metrics:")
        for k, v in metrics_lstm.items():
            print(f"  {k}: {v:.6f}")

        # Plots
        plot_true_vs_pred(dates, y_true, y_arima, y_lstm, ticker)
        plot_residuals(dates, y_true, y_arima, y_lstm, ticker)

def pick_tickers(default):
    """
    Priority:
    1) CLI args: python evaluations.py TSLA (or multiple tickers)
    2) Interactive prompt if no CLI args (works in IDE/Jupyter)
    3) Empty input -> default
    """
    args = [a.strip().upper() for a in sys.argv[1:] if a.strip()]
    if args:
        return args

    # No CLI args received -> ask interactively (IDE friendly)
    user_in = input(
        f"Enter ticker(s) separated by space (press Enter for default: {', '.join(default)}): "
    ).strip().upper()

    if not user_in:
        return default

    return user_in.split()


if __name__ == "__main__":
    main()
