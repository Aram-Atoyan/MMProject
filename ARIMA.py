import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from data_loading import load_series, TICKER_TO_FILE




ARIMA_ORDER = (1, 1, 1)



def difference(series, d=1):
    y = series.copy()
    for _ in range(d):
        y = y.diff()
    return y.dropna().values

def invert_difference(last_values, diffs, d=1):
    if d != 1:
        raise NotImplementedError("Only d=1 inversion is implemented.")
    result, prev = [], last_values[-1]
    for diff in diffs:
        val = prev + diff
        result.append(val)
        prev = val
    return np.array(result)

def arma_residuals(params, y, p, q):
    c = params[0]
    phi = params[1:1+p]
    theta = params[1+p:1+p+q]
    n = len(y)
    max_lag = max(p, q)
    eps = np.zeros(n)
    for t in range(max_lag, n):
        ar_part = sum(phi[i-1] * y[t-i] for i in range(1, p+1)) if p > 0 else 0.0
        ma_part = sum(theta[j-1] * eps[t-j] for j in range(1, q+1)) if q > 0 else 0.0
        y_hat = c + ar_part + ma_part
        eps[t] = y[t] - y_hat
    return eps[max_lag:]

def arma_loss(params, y, p, q):
    eps = arma_residuals(params, y, p, q)
    return np.sum(eps**2)

class ARIMAFromScratch:
    def __init__(self, p, d, q):
        self.p, self.d, self.q = p, d, q
        self.params_ = None
        self.sigma2_ = None
        self.y_train_ = None
        self.diff_train_ = None

    def fit(self, series):
        self.y_train_ = series.values
        diff = difference(series, self.d)
        self.diff_train_ = diff
        p, q = self.p, self.q

        init_params = np.zeros(1 + p + q)
        init_params[0] = diff.mean()

        res = minimize(arma_loss, init_params, args=(diff, p, q), method="L-BFGS-B")
        self.params_ = res.x

        eps = arma_residuals(self.params_, diff, p, q)
        self.sigma2_ = np.mean(eps**2)
        return self

    def forecast_diff(self, steps=1):
        if self.params_ is None:
            raise RuntimeError("Fit first")

        diff = self.diff_train_
        p, q = self.p, self.q
        c = self.params_[0]
        phi = self.params_[1:1+p]
        theta = self.params_[1+p:1+p+q]
        n = len(diff)
        max_lag = max(p, q)

        eps_ext = np.zeros(n + steps)
        y_ext = diff.copy()

        # compute in-sample residuals for initialization
        for t in range(max_lag, n):
            ar_part = sum(phi[i-1] * y_ext[t-i] for i in range(1, p+1)) if p > 0 else 0.0
            ma_part = sum(theta[j-1] * eps_ext[t-j] for j in range(1, q+1)) if q > 0 else 0.0
            y_hat = c + ar_part + ma_part
            eps_ext[t] = y_ext[t] - y_hat

        forecasts = []
        for h in range(steps):
            t = n + h

            ar_part = 0.0
            if p > 0:
                for i in range(1, p+1):
                    idx = t - i
                    ar_part += phi[i-1] * (y_ext[idx] if idx >= 0 else diff[0])

            ma_part = 0.0
            if q > 0:
                for j in range(1, q+1):
                    idx = t - j
                    if idx >= 0:
                        ma_part += theta[j-1] * eps_ext[idx]

            y_hat = c + ar_part + ma_part
            forecasts.append(y_hat)

            y_ext = np.append(y_ext, y_hat)
            eps_ext[t] = 0.0  # expected future innovation
        return np.array(forecasts)

    def forecast_levels(self, steps=1):
        if self.d != 1:
            raise NotImplementedError("Only d=1 level forecasts are implemented.")
        diff_forecasts = self.forecast_diff(steps)
        last_values = self.y_train_[-1:]
        return invert_difference(last_values, diff_forecasts, d=1)

    def predict_in_sample_levels(self):
        if self.params_ is None:
            raise RuntimeError("Fit first")

        diff = self.diff_train_
        p, q = self.p, self.q
        c = self.params_[0]
        phi = self.params_[1:1+p]
        theta = self.params_[1+p:1+p+q]

        n = len(diff)
        max_lag = max(p, q)
        eps = np.zeros(n)
        y_hat_diff = np.zeros(n)

        for t in range(max_lag, n):
            ar_part = sum(phi[i-1] * diff[t-i] for i in range(1, p+1)) if p > 0 else 0.0
            ma_part = sum(theta[j-1] * eps[t-j] for j in range(1, q+1)) if q > 0 else 0.0
            y_hat_diff[t] = c + ar_part + ma_part
            eps[t] = diff[t] - y_hat_diff[t]

        diffs_for_recon = diff.copy()
        diffs_for_recon[max_lag:] = y_hat_diff[max_lag:]

        N = len(self.y_train_)
        log_pred = np.zeros(N)
        log_pred[0] = self.y_train_[0]
        for t in range(1, N):
            log_pred[t] = log_pred[t-1] + diffs_for_recon[t-1]
        return log_pred

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from data_loading import load_series, TICKER_TO_FILE  # or data_utils, match your filename


def plot_acf_pacf_all(tickers=None, years=5, n_weeks=262, lags=20):
    """
    Plot ACF/PACF of differenced log prices for each ticker.

    Uses data_loading.load_series() so it works for:
      - default CSV tickers (if CSV exists)
      - any other ticker via yfinance -> weekly avg -> last n_weeks

    Parameters:
        tickers : list[str] or None
        years   : int, yfinance lookback for custom tickers
        n_weeks : int, number of weekly points
        lags    : int, number of lags for ACF/PACF
    """
    if tickers is None:
        tickers = list(TICKER_TO_FILE.keys())

    for ticker in tickers:
        t = str(ticker).strip().upper()

        # Load once using unified loader
        dates, y_true = load_series(t, n_weeks=n_weeks, years=years)

        # Build log-price and difference
        log_price = np.log(np.asarray(y_true, dtype=float))
        diff = np.diff(log_price)  # 1st difference
        diff = pd.Series(diff)     # statsmodels likes Series

        if len(diff) < max(10, lags + 1):
            print(f"[{t}] Skipping ACF/PACF: not enough points after differencing (len={len(diff)})")
            continue

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        plot_acf(diff, lags=lags, ax=axes[0])
        axes[0].set_title(f"{t} ACF")

        plot_pacf(diff, lags=lags, ax=axes[1], method="ywm")
        axes[1].set_title(f"{t} PACF")

        fig.suptitle(f"{t} differenced log price (weekly avg, last {n_weeks} points)")
        plt.tight_layout()
        plt.show()


def get_arima_predictions(ticker: str, dates, y_true):
    """
    ARIMA predictions for a given ticker using externally provided data.

    IMPORTANT: This function does NOT load data internally.
    You must pass:
        dates : array-like of datetime64 / datetime objects (length N)
        y_true: array-like of floats (Close prices) (length N)

    Returns:
        y_true (np.ndarray), y_pred (np.ndarray), dates (np.ndarray)
    """
    if dates is None or y_true is None:
        raise ValueError("get_arima_predictions requires dates and y_true (no internal loading).")

    # normalize inputs
    t = str(ticker).strip().upper()
    dates = np.asarray(dates)
    y_true = np.asarray(y_true, dtype=float)

    if len(dates) != len(y_true):
        raise ValueError(f"[{t}] dates and y_true lengths differ: {len(dates)} vs {len(y_true)}")

    if len(y_true) < 30:
        raise ValueError(f"[{t}] Not enough data points for ARIMA (got {len(y_true)}).")

    # Create pandas series for differencing logic inside ARIMAFromScratch
    series_price = pd.Series(y_true, index=pd.to_datetime(dates))
    series_log = np.log(series_price)

    # Train/test split
    n = len(series_log)
    train_size = int(n * 0.8)
    test_len = n - train_size

    train_series = series_log.iloc[:train_size]

    # Fit
    p, d, q = ARIMA_ORDER
    model = ARIMAFromScratch(p, d, q).fit(train_series)

    # In-sample predictions (train)
    log_pred_train = model.predict_in_sample_levels()  # length train_size

    # Forecast for test
    log_pred_test = model.forecast_levels(steps=test_len)  # length test_len

    # Combine and invert log
    log_pred_full = np.concatenate([log_pred_train, log_pred_test])
    y_pred = np.exp(log_pred_full)

    return y_true, y_pred, dates