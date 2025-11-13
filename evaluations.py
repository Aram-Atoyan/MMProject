import numpy as np
import matplotlib.pyplot as plt

from Data_And_ARIMA import get_arima_predictions
from ltsm import get_lstm_predictions


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


def directional_accuracy(y_true, y_pred):
    """
    Fraction of times the model gets the direction of change right.

    We compare sign of (y_t - y_{t-1}) for true vs predicted.
    """
    true_diff = np.diff(y_true)
    pred_diff = np.diff(y_pred)

    # Avoid division by zero etc; sign(0) = 0 is fine
    correct = np.sign(true_diff) == np.sign(pred_diff)
    return np.mean(correct)


def compute_all_metrics(y_true, y_pred):
    return {
        "MSE": mse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "DA": directional_accuracy(y_true, y_pred),
    }


def plot_true_vs_pred(dates, y_true, y_arima, y_lstm):
    plt.figure(figsize=(12, 5))
    plt.plot(dates, y_true, label="True", linewidth=2)
    plt.plot(dates, y_arima, label="ARIMA", alpha=0.7)
    plt.plot(dates, y_lstm, label="LSTM", alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("True vs Predicted – Test Set")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_residuals(dates, y_true, y_arima, y_lstm):
    res_arima = y_true - y_arima
    res_lstm = y_true - y_lstm

    plt.figure(figsize=(12, 4))
    plt.plot(dates, res_arima)
    plt.axhline(0, linestyle="--")
    plt.title("ARIMA residuals")
    plt.xlabel("Time")
    plt.ylabel("Error")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(dates, res_lstm)
    plt.axhline(0, linestyle="--")
    plt.title("LSTM residuals")
    plt.xlabel("Time")
    plt.ylabel("Error")
    plt.tight_layout()
    plt.show()


def main():
    # Get predictions from both models
    y_true_arima, y_arima, dates_arima = get_arima_predictions()
    y_true_lstm,  y_lstm,  dates_lstm  = get_lstm_predictions()

    # Basic sanity checks
    if not np.allclose(y_true_arima, y_true_lstm):
        raise ValueError("y_true from ARIMA and LSTM do not match!")

    if len(dates_arima) != len(dates_lstm):
        raise ValueError("Dates arrays have different lengths!")

    if not np.array_equal(np.array(dates_arima), np.array(dates_lstm)):
        raise ValueError("Dates from ARIMA and LSTM do not match!")

    y_true = y_true_arima
    dates = dates_arima

    # Compute metrics
    metrics_arima = compute_all_metrics(y_true, y_arima)
    metrics_lstm  = compute_all_metrics(y_true, y_lstm)

    print("=== Model Comparison Metrics ===")
    print("ARIMA:")
    for k, v in metrics_arima.items():
        print(f"  {k}: {v:.6f}")
    print("LSTM:")
    for k, v in metrics_lstm.items():
        print(f"  {k}: {v:.6f}")

    # Plots
    plot_true_vs_pred(dates, y_true, y_arima, y_lstm)
    plot_residuals(dates, y_true, y_arima, y_lstm)


if __name__ == "__main__":
    main()
