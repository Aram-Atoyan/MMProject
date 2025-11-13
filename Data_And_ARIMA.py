import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12, 5)
stocks = ["AAPL", "MSFT", "TSLA", "AMZN", "GOOGL"]

data_dictionary = {}

for stock in stocks:
    dfStocks = yf.download(stock, period="5y", interval="1d")
    dfStocks = dfStocks[[]]