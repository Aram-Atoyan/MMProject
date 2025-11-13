import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

plt.rcParams["figure.figsize"] = (12, 5)
stocks = ["AAPL", "MSFT", "TSLA", "AMZN", "GOOGL"]

data_dictionary = {}

for stock in stocks:
    dfStocks = yf.download(stock, period="5y", interval="1wk")
    dfStocks = dfStocks[['Close']].dropna()
    data_dictionary[stock] = dfStocks

def plotDataOriginal():
    for stock, dfStock in data_dictionary.items():
        plt.figure()
        plt.plot(dfStock.index, dfStock["Close"])
        plt.title(f"{stock} Closing Price (Last 5 Years)")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.grid(True)
        plt.show()

def saveData():
    for stock, df in data_dictionary.items():
        df.to_csv(f"data/{stock}.csv")

if __name__ == "__main__":
    plotDataOriginal()

def arma_residuals(params, y, p, q):
    """
    Compute residuals e_t for an ARMA(p, q) model:
        y_t = c + sum phi_i * y_{t-i} + sum theta_j * e_{t-j} + e_t
    y : 1D numpy array (differenced series)
    """
    c = params[0]
    phi = params[1:1+p]
    theta = params[1+p:1+p+q]

    n = len(y)
    e = np.zeros(n)

    for t in range(n):
        ar_term = 0.0
        ma_term = 0.0

        # Automated Regression part
        for i in range(1, p+1):
            if t - i >= 0:
                ar_term += phi[i-1] * y[t - i]

        # Moving Average part
        for j in range(1, q+1):
            if t - j >= 0:
                ma_term += theta[j-1] * e[t - j]

        y_hat = c + ar_term + ma_term
        e[t] = y[t] - y_hat

    return e

def arma_sum_of_squared_errors(params, y, p, q):
    e = arma_residuals(params, y, p, q)
    return np.sum(e**2)

def fit_arma(y, p, q):
    """
    Fit ARMA(p, q) on 1D array y using SSE minimization.
    Returns fitted parameter vector.
    params = [c, phi_1..phi_p, theta_1..theta_q]
    """
    n_params = 1 + p + q
    x0 = np.zeros(n_params)  # initial guess

    res = minimize(
        arma_sum_of_squared_errors,
        x0,
        args=(y, p, q),
        method="L-BFGS-B"
    )

    return res.x

def choose_order(y_diff, max_p=2, max_q=2):
    best_aic = np.inf
    best_order = None
    best_params = None

    n = len(y_diff)

    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            if p == 0 and q == 0:
                continue  # ARMA(0,0) is just white noise, skip

            try:
                params = fit_arma(y_diff, p, q)
                sse = arma_sum_of_squared_errors(params, y_diff, p, q)
                k = len(params)
                sigma2 = sse / n
                aic = n * np.log(sigma2) + 2 * k

                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, 1, q)  # we fix d = 1
                    best_params = params
            except Exception as e:
                # In case optimization fails for some (p, q), just skip
                continue

    return best_order, best_params


def forecast_arma_diff(y_train, y_test_len, p, q, params):
    """
    y_train      : 1D numpy array of original prices (train only)
    y_test_len   : number of steps to forecast
    params       : fitted ARMA params on differenced y_train
    Returns:
        y_pred_test : 1D numpy array of predicted prices for test period
    """
    # 1) Build differenced training series z_t = y_t - y_{t-1}
    z_train = np.diff(y_train)  # length N_train - 1

    # 2) Recompute residuals on training differenced series
    e_train = arma_residuals(params, z_train, p, q)

    # 3) Prepare arrays for future differenced predictions
    n_train_diff = len(z_train)
    total_len = n_train_diff + y_test_len

    z_full = np.zeros(total_len)
    e_full = np.zeros(total_len)

    # First part is actual training differenced series + residuals
    z_full[:n_train_diff] = z_train
    e_full[:n_train_diff] = e_train

    c = params[0]
    phi = params[1:1+p]
    theta = params[1+p:1+p+q]

    # 4) Forecast differenced series z_future
    for t in range(n_train_diff, total_len):
        ar_term = 0.0
        ma_term = 0.0

        # AR terms: use past z_full
        for i in range(1, p+1):
            if t - i >= 0:
                ar_term += phi[i-1] * z_full[t - i]

        # MA terms: use past residuals (for forecast horizon, we set future e to 0)
        for j in range(1, q+1):
            if t - j >= 0:
                ma_term += theta[j-1] * e_full[t - j]

        z_hat = c + ar_term + ma_term

        z_full[t] = z_hat
        e_full[t] = 0.0  # expected future innovations are 0

    # z_future is the forecasted differenced series
    z_future = z_full[n_train_diff:]

    # 5) Reconstruct price forecasts from differenced forecasts
    y_pred = np.zeros(y_test_len)
    last_train_value = y_train[-1]

    for i in range(y_test_len):
        if i == 0:
            y_pred[i] = last_train_value + z_future[i]
        else:
            y_pred[i] = y_pred[i-1] + z_future[i]

    return y_pred

def get_arima_predictions(
    ticker="AAPL",
    period="5y",
    train_ratio=0.8
):
    """
    Train a simple ARIMA(p,1,q) model from scratch (no auto_arima),
    choose (p,q) by AIC over a small grid, and return:
        y_true, y_pred, dates
    where:
        y_true : true test prices (numpy array)
        y_pred : predicted test prices (numpy array)
        dates  : corresponding datetime64 array
    """

    # 1) Download data
    df = yf.download(ticker, period=period, interval="1d")
    series = df["Close"].dropna()
    values = series.values.astype(float)
    dates = series.index.to_numpy()

    # 2) Train/test split
    n = len(values)
    train_size = int(n * train_ratio)

    y_train = values[:train_size]
    y_test = values[train_size:]
    test_dates = dates[train_size:]

    # 3) Build differenced train series
    z_train = np.diff(y_train)
    z_train = z_train.astype(float)

    # 4) Choose (p,1,q) by AIC
    (p, d, q), params = choose_order(z_train, max_p=2, max_q=2)
    # Note: d will be 1 by our design

    # 5) Forecast test horizon using ARIMA(p,1,q)
    y_pred = forecast_arma_diff(y_train, len(y_test), p, q, params)

    # Convert to numpy arrays (already are)
    y_true = y_test
    dates_out = test_dates

    return y_true, y_pred, dates_out


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