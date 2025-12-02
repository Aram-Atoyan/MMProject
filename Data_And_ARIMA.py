import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# -------- config --------
TICKER_TO_FILE = {
    "AAPL": "data/AAPL.csv",
    "AMZN": "data/AMZN.csv",
    "GOOGL": "data/GOOGL.csv",
    "MSFT": "data/MSFT.csv",
    "TSLA": "data/TSLA.csv",
}
TICKER = "AAPL"
ARIMA_ORDER = (1, 1, 1)

# -------- data loading --------
def load_stock_csv(path):
    df = pd.read_csv(path)
    meta_mask = df["Price"].isin(["Ticker", "Date"])
    df = df.loc[~meta_mask].copy()
    df = df.rename(columns={"Price": "Date", "Close": "ClosePrice"})
    df["Date"] = pd.to_datetime(df["Date"])
    df["ClosePrice"] = pd.to_numeric(df["ClosePrice"], errors="coerce")
    df = df.dropna(subset=["ClosePrice"]).sort_values("Date").set_index("Date")
    df["log_price"] = np.log(df["ClosePrice"])
    return df

# -------- helpers --------
def difference(series, d=1):
    y = series.copy()
    for _ in range(d):
        y = y.diff()
    return y.dropna().values

def invert_difference(last_values, diffs, d=1):
    if d != 1:
        raise NotImplementedError
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

# -------- ARIMA class --------
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
            eps_ext[t] = 0.0
        return np.array(forecasts)

    def forecast_levels(self, steps=1):
        if self.d != 1:
            raise NotImplementedError
        diff_forecasts = self.forecast_diff(steps)
        last_values = self.y_train_[-1:]
        return invert_difference(last_values, diff_forecasts, d=1)

# -------- ACF/PACF for all tickers --------
def plot_acf_pacf_all():
    for ticker, path in TICKER_TO_FILE.items():
        df = load_stock_csv(path)
        diff = df["log_price"].diff().dropna()
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        plot_acf(diff, lags=20, ax=axes[0])
        axes[0].set_title(f"{ticker} ACF")
        plot_pacf(diff, lags=20, ax=axes[1], method="ywm")
        axes[1].set_title(f"{ticker} PACF")
        fig.suptitle(f"{ticker} differenced log price")
        plt.tight_layout()
        plt.show()

def get_arima_predictions(ticker=None):
    if ticker is None:
        ticker = TICKER
    df = load_stock_csv(TICKER_TO_FILE[ticker])
    series_log = df["log_price"]
    series_price = df["ClosePrice"]
    n = len(series_log)
    train_size = int(n * 0.8)
    train_series = series_log.iloc[:train_size]
    test_price = series_price.iloc[train_size:]
    dates = test_price.index.to_numpy()
    p, d, q = ARIMA_ORDER
    model = ARIMAFromScratch(p, d, q).fit(train_series)
    steps = len(test_price)
    log_forecast = model.forecast_levels(steps=steps)
    y_pred = np.exp(log_forecast)
    y_true = test_price.values
    return y_true, y_pred, dates

def plot_predictions_for_ticker(ticker=None):
    if ticker is None:
        ticker = TICKER
    y_true, y_pred, dates = get_arima_predictions(ticker)
    plt.figure(figsize=(10, 4))
    plt.plot(dates, y_true, label="Actual")
    plt.plot(dates, y_pred, label="ARIMA", linestyle="--")
    plt.title(f"{ticker} – Actual vs ARIMA prediction")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # run this once to generate ACF/PACF plots for all tickers
    plot_acf_pacf_all()
    plot_predictions_for_ticker("AAPL")
    # example: get predictions for current TICKER
    # y_true, y_pred, dates = get_arima_predictions()