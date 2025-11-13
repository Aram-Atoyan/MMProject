import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams["figure.figsize"] = (12, 5)
stocks = ["AAPL", "MSFT", "TSLA", "AMZN", "GOOGL"]

data_dictionary = {}

for stock in stocks:
    dfStocks = yf.download(stock, period="5y", interval="1wk")
    dfStocks = dfStocks[['Close']].dropna()
    data_dictionary[stock] = dfStocks

for stock, dfStock in data_dictionary.items():
    plt.figure()
    plt.plot(dfStock.index, dfStock["Close"])
    plt.title(f"{stock} Closing Price (Last 5 Years)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.grid(True)
    plt.show()

for stock, df in data_dictionary.items():
    df.to_csv(f"data/{stock}.csv")



def get_arima_predictions():
    """
    Train ARIMA, generate predictions on the test set,
    and return (y_true, y_pred, dates) as numpy arrays.
    """
    # TODO: replace this with real ARIMA logic
    # Example dummy data:
    n = 100
    dates = np.arange(n)  # or real datetime array
    y_true = np.sin(np.linspace(0, 4 * np.pi, n))  # dummy true
    y_pred = y_true + 0.1 * np.random.randn(n)     # dummy pred

    return y_true, y_pred, dates


def get_arima_params():
    """
    Return basic ARIMA info for MM analysis.
    Example for ARIMA(p, d, q) with AR(p) part:

    Returns:
        phi : np.ndarray of shape (p,)
            AR coefficients [phi1, phi2, ..., phip].
        c   : float
            Constant term in the AR part (on the differenced series).
        d   : int
            Differencing order used.
    """
    phi,c,d =0,0,0
    return phi, c, d