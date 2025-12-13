# data_utils.py
import os
import numpy as np
import pandas as pd
import yfinance as yf

TICKER_TO_FILE = {
    "AAPL": "data/AAPL.csv",
    "AMZN": "data/AMZN.csv",
    "GOOGL": "data/GOOGL.csv",
    "MSFT": "data/MSFT.csv",
    "TSLA": "data/TSLA.csv",
}

def _load_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Price" in df.columns:
        meta_mask = df["Price"].isin(["Ticker", "Date"])
        df = df.loc[~meta_mask].copy()
        df = df.rename(columns={"Price": "Date", "Close": "ClosePrice"})
    else:
        if "Date" not in df.columns:
            raise ValueError(f"{path} missing 'Date' column.")
        if "ClosePrice" not in df.columns and "Close" in df.columns:
            df = df.rename(columns={"Close": "ClosePrice"})

    df["Date"] = pd.to_datetime(df["Date"])
    df["ClosePrice"] = pd.to_numeric(df["ClosePrice"], errors="coerce")
    df = df.dropna(subset=["ClosePrice"]).sort_values("Date").set_index("Date")
    return df

def _fetch_weekly_avg_yfinance(ticker: str, years: int = 5, n_weeks: int = 262) -> pd.DataFrame:
    t = ticker.strip().upper()

    raw = yf.download(
        t,
        period=f"{years}y",
        interval="1d",
        progress=False,
        auto_adjust=False,
        group_by="column",   # keeps columns consistent
        threads=True,
    )

    if raw is None or raw.empty:
        raise ValueError(f"No yfinance data for '{t}'.")

    # Make sure the index is datetime (required for resample)
    if not isinstance(raw.index, pd.DatetimeIndex):
        raw.index = pd.to_datetime(raw.index, errors="coerce")
        raw = raw.dropna()

    # Extract Close in a robust way:
    close = raw["Close"]
    # Sometimes close is a DataFrame (e.g., multi-index or duplicated structure)
    if isinstance(close, pd.DataFrame):
        # take the first column (single-ticker case)
        close = close.iloc[:, 0]

    # Now close must be a Series
    if not isinstance(close, pd.Series):
        raise ValueError(f"Unexpected Close type for '{t}': {type(close)}")

    weekly = close.resample("W-FRI").mean().dropna().tail(n_weeks)

    if len(weekly) == 0:
        raise ValueError(f"Weekly series is empty for '{t}' after resampling.")

    # safest: build DataFrame from Series directly
    df = weekly.to_frame(name="ClosePrice")
    return df


def load_series(ticker: str, n_weeks: int = 262, years: int = 5):

    t = ticker.strip().upper()
    if t in TICKER_TO_FILE and os.path.exists(TICKER_TO_FILE[t]):
        df = _load_from_csv(TICKER_TO_FILE[t])
        close = df["ClosePrice"]
    else:
        df = _fetch_weekly_avg_yfinance(t, years=years, n_weeks=n_weeks)
        close = df["ClosePrice"]

    close = close.dropna().tail(n_weeks)

    dates = close.index.to_numpy()
    y_true = close.values.astype(float)
    return dates, y_true
