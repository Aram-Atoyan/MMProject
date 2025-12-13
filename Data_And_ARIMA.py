import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

TICKER_TO_FILE = {
    "AAPL": "data/AAPL.csv",
    "AMZN": "data/AMZN.csv",
    "GOOGL": "data/GOOGL.csv",
    "MSFT": "data/MSFT.csv",
    "TSLA": "data/TSLA.csv",
}
ARIMA_ORDER = (1, 1, 1 )

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

def get_arima_predictions(ticker):
    df = load_stock_csv(f"data/{ticker}.csv")
    series_log = df["log_price"]
    series_price = df["ClosePrice"]
    dates = df.index.to_numpy()

    n = len(series_log)
    train_size = int(n * 0.8)

    train_series = series_log.iloc[:train_size]
    test_len = n - train_size

    p, d, q = ARIMA_ORDER
    model = ARIMAFromScratch(p, d, q).fit(train_series)

    log_pred_train = model.predict_in_sample_levels()
    last_log_train = train_series.values[-1:]
    model_full = model
    model_full.y_train_ = train_series.values
    model_full.diff_train_ = difference(train_series, d)
    log_pred_test = model_full.forecast_levels(steps=test_len)

    log_pred_full = np.concatenate([log_pred_train, log_pred_test])
    y_pred = np.exp(log_pred_full)
    y_true = series_price.values

    return y_true, y_pred, dates

if __name__ == "__main__":
    # example usage
    y_true, y_pred, dates = get_arima_predictions()
    print("Total points:", len(y_pred))