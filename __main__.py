import yfinance as yf
import pandas as pd

if __name__ == "__main__":

    dataframe = pd.read_csv("data/stocks_eww_45.csv")
    # dataframe = dataframe[~dataframe['stock'].isin(["ORBIA", 'AMXB', "Q"])]
    print(dataframe.head())
    tickers = dataframe["stock"].tolist()
    print(tickers)


    df = yf.download(tickers, start="2024-01-01")["Close"]
    df.to_csv("data/stocks_info02.csv")