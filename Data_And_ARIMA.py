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